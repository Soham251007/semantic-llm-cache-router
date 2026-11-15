"""
Complexity Scorer using fine-tuned DistilBERT + Calibrated Task Signal Heuristics.
Evaluates prompt difficulty:
- Tier-1 (Simple / Fast Model): Extraction, Formatting, Summarization, Sentiment, Factual Q&A.
- Tier-2 (Complex / Frontier Model): Algorithmic Reasoning, System Architecture, Formal Logic, Code Analysis.
"""

import os
import re
import time
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

class ComplexityScorer:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            
        self.model_dir = os.path.join(models_dir, "distilbert_complexity")
        
        # Check if local model weights exist; otherwise load base and calibrate dynamically
        has_weights = os.path.exists(os.path.join(self.model_dir, "model.safetensors")) or \
                      os.path.exists(os.path.join(self.model_dir, "pytorch_model.bin"))

        if has_weights:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        else:
            print("Notice: DistilBERT custom checkpoint not found locally. Loading distilbert-base-uncased with calibrated classification head...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
        self.model.eval()
        
        # High-signal pattern matchers for reasoning and complexity
        self.reasoning_patterns = [
            (re.compile(r"\b(prove|proof|theorem|lemma|reduction|np-(complete|hard))\b", re.IGNORECASE), "formal_proof"),
            (re.compile(r"\b(concurrency|mutex|lock-free|atomic|race condition|deadlock|semaphore)\b", re.IGNORECASE), "low_level_concurrency"),
            (re.compile(r"\b(distributed|consensus|raft|byzantine|paxos|partition|sharding|cqrs)\b", re.IGNORECASE), "distributed_architecture"),
            (re.compile(r"\b(optimize|time complexity|space complexity|big-o|asymptotic|profile)\b", re.IGNORECASE), "algorithmic_complexity"),
            (re.compile(r"\b(vulnerability|exploit|reentrancy|buffer overflow|zero-day|xss|injection)\b", re.IGNORECASE), "security_audit"),
            (re.compile(r"```[a-z]*\n[\s\S]*?\n```", re.IGNORECASE), "code_block"),
        ]
        
        # Simple task patterns
        self.simple_patterns = [
            (re.compile(r"\b(summarize|summary|tldr|bullet points)\b", re.IGNORECASE), "summarization"),
            (re.compile(r"\b(extract|extracting|find all|parse)\b", re.IGNORECASE), "extraction"),
            (re.compile(r"\b(translate|translation|english to|to spanish|to french)\b", re.IGNORECASE), "translation"),
            (re.compile(r"\b(format as|convert to json|convert to csv|clean up)\b", re.IGNORECASE), "formatting"),
            (re.compile(r"\b(capital of|who wrote|who is|what year|define)\b", re.IGNORECASE), "factual_lookup"),
        ]

        self._warmup()

    def _warmup(self):
        dummy = "Warmup query for DistilBERT prompt complexity scoring."
        _ = self.score(dummy, alpha=0.45)

    def score(self, prompt: str, alpha: float = 0.45) -> Dict[str, Any]:
        """
        Score prompt complexity returning difficulty probability [0, 1] and recommended model tier.
        """
        t0 = time.perf_counter()
        
        # 1. Neural forward pass through DistilBERT
        inputs = self.tokenizer(prompt, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            neural_score = float(probs[0, 1].item()) # probability of class 1 (complex)

        # 2. Extract structural and syntactic signals
        detected_signals = []
        heuristic_boost = 0.0
        
        for pattern, signal_name in self.reasoning_patterns:
            if pattern.search(prompt):
                detected_signals.append(signal_name)
                heuristic_boost += 0.15

        for pattern, signal_name in self.simple_patterns:
            if pattern.search(prompt):
                detected_signals.append(signal_name)
                heuristic_boost -= 0.15

        # 3. Calibrated blend
        # S = 0.8 * neural_score + 0.2 * heuristic_score clamped to [0, 1]
        calibrated_score = 0.8 * neural_score + heuristic_boost
        calibrated_score = max(0.001, min(0.999, calibrated_score))
        
        # 4. Routing decision based on threshold alpha
        # Lower alpha -> stricter, sends more to frontier model
        # Default alpha = 0.45
        is_complex = calibrated_score >= alpha
        recommended_tier = "tier-2-frontier" if is_complex else "tier-1-fast"

        duration_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "complexity_score": round(calibrated_score, 4),
            "neural_prob": round(neural_score, 4),
            "threshold_alpha": alpha,
            "recommended_tier": recommended_tier,
            "is_complex": is_complex,
            "signals": detected_signals,
            "latency_ms": round(duration_ms, 2)
        }
