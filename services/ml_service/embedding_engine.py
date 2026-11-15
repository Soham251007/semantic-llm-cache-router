"""
Embedding Engine using ONNX Runtime INT8 Quantized all-MiniLM-L6-v2.
Provides sub-8ms 384-dimensional normalized vector embeddings on CPU.
"""

import os
import time
from typing import List, Union
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class EmbeddingEngine:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            
        self.onnx_path = os.path.join(models_dir, "minilm_int8.onnx")
        self.tokenizer_path = os.path.join(models_dir, "minilm_tokenizer")
        
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"MiniLM ONNX model not found at {self.onnx_path}")
            
        # Configure high-throughput ONNX session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.session = ort.InferenceSession(self.onnx_path, opts, providers=["CPUExecutionProvider"])
        
        # Warmup
        self._warmup()

    def _warmup(self):
        dummy = "Warmup query for ONNX runtime semantic embedding engine."
        _ = self.embed(dummy)

    def embed(self, text: str, max_length: int = 128) -> List[float]:
        """
        Embed a single text string into a normalized 384-dim vector.
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        input_dict = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            input_dict["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        else:
            input_dict["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

        # Run ONNX inference
        outputs = self.session.run(None, input_dict)
        last_hidden_state = outputs[0] # (batch_size, seq_len, 384)

        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        # L2 normalize (so dot product equals cosine similarity)
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-12, a_max=None)
        normalized = (mean_pooled / norm)[0]

        return normalized.astype(float).tolist()

    def embed_batch(self, texts: List[str], max_length: int = 128) -> List[List[float]]:
        """
        Embed a batch of texts concurrently in a single ONNX forward pass.
        """
        if not texts:
            return []
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        
        input_dict = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64)
        }

        outputs = self.session.run(None, input_dict)
        last_hidden_state = outputs[0]

        mask_expanded = np.expand_dims(inputs["attention_mask"], -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask

        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-12, a_max=None)
        normalized = mean_pooled / norm

        return normalized.astype(float).tolist()
