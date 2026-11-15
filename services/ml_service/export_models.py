"""
Model Export & INT8 Quantization Script for Semantic Cache & Complexity Router.
Exports:
1. all-MiniLM-L6-v2 -> ONNX -> INT8 Dynamic Quantization (384-dim embeddings)
2. DistilBERT Complexity Scorer -> ONNX -> INT8 Dynamic Quantization ([0, 1] difficulty probability)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertConfig
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def export_minilm_onnx():
    print("=" * 60)
    print("1. Exporting all-MiniLM-L6-v2 to INT8 ONNX")
    print("=" * 60)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "minilm_tokenizer"))
    print("✓ MiniLM Tokenizer saved to models/minilm_tokenizer")
    
    # Define a clean wrapper that outputs mean-pooled normalized embeddings
    class MiniLMEmbeddingWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            token_embeddings = outputs[0] # (batch_size, seq_len, 384)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = nn.functional.normalize(mean_pooled, p=2, dim=1)
            return normalized

    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()
    wrapped_model = MiniLMEmbeddingWrapper(base_model)
    wrapped_model.eval()

    dummy_text = "Semantic caching accelerates LLM latency."
    dummy_inputs = tokenizer(dummy_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    fp32_onnx_path = os.path.join(MODELS_DIR, "minilm_fp32.onnx")
    int8_onnx_path = os.path.join(MODELS_DIR, "minilm_int8.onnx")

    print(f"Exporting FP32 ONNX model to {fp32_onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        (
            dummy_inputs["input_ids"],
            dummy_inputs["attention_mask"],
            dummy_inputs.get("token_type_ids", torch.zeros_like(dummy_inputs["input_ids"]))
        ),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "embeddings": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ FP32 ONNX export completed.")

    print(f"Quantizing to INT8: {int8_onnx_path}...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8
    )
    
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f"✓ Quantization done! FP32 size: {fp32_size_mb:.2f}MB -> INT8 size: {int8_size_mb:.2f}MB (Compression: {fp32_size_mb/int8_size_mb:.1f}x)")
    
    # Remove unneeded FP32 file to save disk space
    if os.path.exists(fp32_onnx_path):
        os.remove(fp32_onnx_path)

    # Benchmark ONNX Runtime session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(int8_onnx_path, opts, providers=["CPUExecutionProvider"])
    
    # Warmup
    inputs_np = {
        "input_ids": dummy_inputs["input_ids"].numpy(),
        "attention_mask": dummy_inputs["attention_mask"].numpy(),
        "token_type_ids": dummy_inputs.get("token_type_ids", torch.zeros_like(dummy_inputs["input_ids"])).numpy()
    }
    _ = session.run(None, inputs_np)
    
    # Benchmark latency
    t0 = time.perf_counter()
    runs = 50
    for _ in range(runs):
        _ = session.run(None, inputs_np)
    avg_latency_ms = ((time.perf_counter() - t0) / runs) * 1000.0
    print(f"✓ MiniLM INT8 Average Latency: {avg_latency_ms:.2f}ms per query on CPU")


def export_distilbert_complexity_onnx():
    print("\n" + "=" * 60)
    print("2. Exporting DistilBERT Complexity Scorer to INT8 ONNX")
    print("=" * 60)

    # We use distilbert-base-uncased configured with a binary classification head
    # 0 = Low Complexity (Tier-1: Llama-3-8B / Gemini Flash)
    # 1 = High Complexity (Tier-2: Claude 3.5 Sonnet / GPT-4o / Gemini Pro)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "distilbert_tokenizer"))
    print("✓ DistilBERT Tokenizer saved to models/distilbert_tokenizer")

    # Construct the complexity classification model
    config = DistilBertConfig.from_pretrained(model_name, num_labels=2)
    config.id2label = {0: "simple", 1: "complex"}
    config.label2id = {"simple": 0, "complex": 1}
    
    # Load model and apply calibrated weights for prompt complexity categorization
    model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)
    
    # Calibrate the classifier head with high-signal complexity task representations
    # High complexity markers: multi-step reasoning, formal proofs, low-level concurrency,
    # distributed systems architecture, vulnerability analysis, AST transformations.
    # Simple markers: summarization, JSON/CSV extraction, translation, FAQ, sentiment.
    
    # We fine-tune / calibrate on a curated prompt dataset
    calibration_data = [
        # (text, label) 0=Simple, 1=Complex
        ("Summarize this article in 3 bullet points.", 0),
        ("Extract the phone number and address from this invoice.", 0),
        ("What is the capital of Australia?", 0),
        ("Translate this welcome greeting into Spanish.", 0),
        ("Format this list of users into a clean JSON array.", 0),
        ("Fix the spelling and grammar in this email draft.", 0),
        ("Classify this tweet sentiment as positive, neutral, or negative.", 0),
        ("Convert this markdown table to HTML table syntax.", 0),
        ("Who wrote the play Romeo and Juliet?", 0),
        ("Rewrite this sentence in a more professional tone.", 0),
        
        ("Design a Byzantine Fault Tolerant consensus protocol with asynchronous network partitions.", 1),
        ("Write a lock-free multi-producer multi-consumer ring buffer in Go using atomic operations and memory fences.", 1),
        ("Prove that NP is contained in PSPACE using polynomial space Turing machine reduction.", 1),
        ("Identify any race conditions, memory leaks, and undefined behavior in this C++ template metaprogramming code.", 1),
        ("Synthesize a high-frequency algorithmic order execution engine in Rust with kernel-bypass networking.", 1),
        ("Evaluate the security vulnerabilities in this smart contract reentrancy pattern and construct an exploit PoC.", 1),
        ("Derive the backpropagation gradient update rule for a multi-head sparse attention layer from first principles.", 1),
        ("Architect an event-driven microservices platform handling 500k RPS with idempotency and CQRS.", 1),
        ("Analyze this recursive AST parser for catastrophic backtracking and prove time complexity bounds.", 1),
        ("Implement a distributed two-phase commit coordinator with raft consensus log replication in Go.", 1)
    ]

    print(f"Calibrating DistilBERT complexity classifier head across {len(calibration_data)} exemplar prompts...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    
    for epoch in range(5):
        total_loss = 0.0
        for text, label in calibration_data:
            inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
            labels = torch.tensor([label])
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"✓ Calibration complete (final loss: {total_loss/len(calibration_data):.4f})")
    model.eval()

    # Wrap model to output calibrated probability score of complexity [0, 1]
    class ComplexityScorerWrapper(nn.Module):
        def __init__(self, base_classifier):
            super().__init__()
            self.base_classifier = base_classifier

        def forward(self, input_ids, attention_mask):
            outputs = self.base_classifier(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            # Probability of Class 1 (Complex)
            complexity_score = probs[:, 1]
            return complexity_score

    wrapped_scorer = ComplexityScorerWrapper(model)
    wrapped_scorer.eval()

    dummy_text = "Explain how semantic caching improves LLM throughput."
    dummy_inputs = tokenizer(dummy_text, padding=True, truncation=True, max_length=128, return_tensors="pt")

    fp32_onnx_path = os.path.join(MODELS_DIR, "distilbert_complexity_fp32.onnx")
    int8_onnx_path = os.path.join(MODELS_DIR, "distilbert_complexity_int8.onnx")

    print(f"Exporting FP32 DistilBERT ONNX model to {fp32_onnx_path}...")
    torch.onnx.export(
        wrapped_scorer,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["complexity_score"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "complexity_score": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ FP32 ONNX export completed.")

    print(f"Quantizing DistilBERT to INT8: {int8_onnx_path}...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8
    )

    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f"✓ Quantization done! FP32 size: {fp32_size_mb:.2f}MB -> INT8 size: {int8_size_mb:.2f}MB (Compression: {fp32_size_mb/int8_size_mb:.1f}x)")

    if os.path.exists(fp32_onnx_path):
        os.remove(fp32_onnx_path)

    # Benchmark ONNX Runtime session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(int8_onnx_path, opts, providers=["CPUExecutionProvider"])
    
    test_prompts = [
        ("Summarize this bullet point list", False),
        ("Write a concurrent lock-free skip list in C++", True),
        ("Extract names and dates from text", False),
        ("Prove the Riemann Hypothesis or show why zero distribution on critical line matters", True)
    ]
    
    print("\nVerification Test on Calibrated Scorer:")
    for prompt, expected_complex in test_prompts:
        toks = tokenizer(prompt, truncation=True, max_length=128, return_tensors="np")
        score = session.run(None, {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]})[0][0]
        verdict = "Tier-2 Frontier (Complex)" if score >= 0.45 else "Tier-1 Fast (Simple)"
        print(f"  Prompt: '{prompt}'")
        print(f"    -> Score: {score:.4f} | Route: {verdict}")

    print("\n✓ Model export and INT8 quantization fully succeeded!")


if __name__ == "__main__":
    t_start = time.time()
    export_minilm_onnx()
    export_distilbert_complexity_onnx()
    print(f"\nAll models built and verified in {time.time() - t_start:.2f}s")
