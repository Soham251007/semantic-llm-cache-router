"""
Comprehensive Test Suite for Python ML Microservice.
Tests:
1. EmbeddingEngine (dimension, normalization, cosine similarity)
2. ComplexityScorer (calibration, difficulty routing)
3. FastAPI Endpoints (/health, /embed, /score-complexity, /analyze)
"""

import os
import sys
import unittest
import numpy as np

# Ensure services directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from embedding_engine import EmbeddingEngine
from complexity_scorer import ComplexityScorer
from semantic_guard import SemanticGuard

class TestMLService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        cls.engine = EmbeddingEngine(models_dir)
        cls.scorer = ComplexityScorer(models_dir)

    def test_embedding_dimension_and_normalization(self):
        text = "Semantic LLM caching accelerates inference."
        emb = self.engine.embed(text)
        
        self.assertEqual(len(emb), 384, "Embedding dimension must be 384 for all-MiniLM-L6-v2")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4, msg="Embedding vector must be L2-normalized")

    def test_semantic_similarity_clustering(self):
        # Near-identical intent / close paraphrase must exceed 0.94 threshold
        q1 = "How do I reset my password?"
        q2 = "Can you tell me how to reset my password?"
        # Unrelated question
        q3 = "Design an operating system microkernel with IPC."

        v1 = np.array(self.engine.embed(q1))
        v2 = np.array(self.engine.embed(q2))
        v3 = np.array(self.engine.embed(q3))

        sim_paraphrase = float(np.dot(v1, v2))
        sim_unrelated = float(np.dot(v1, v3))

        print(f"\n  Similarity Close Paraphrase: {sim_paraphrase:.4f}")
        print(f"  Similarity Unrelated:        {sim_unrelated:.4f}")

        self.assertGreater(sim_paraphrase, 0.94, "Close paraphrase must exceed 0.94 threshold")
        self.assertLess(sim_unrelated, 0.40, "Unrelated prompts must yield lower similarity")

    def test_bullet_constraint_divergence_defense(self):
        """
        Verify that prompts differing by numeric/quantifier constraints (e.g. 3 vs 5 bullets)
        are blocked by the SemanticGuard even if raw vector cosine similarity is high.
        """
        bullet_pairs = [
            ("Summarize this article in 3 bullets", "Summarize this article in 5 bullets"),
            ("Give me 3 reasons why Go is fast", "Give me 5 reasons why Go is fast"),
            ("Give me top 3 tips for interview", "Give me top 5 tips for interview"),
            ("Write a 100 word essay on AI", "Write a 500 word essay on AI"),
        ]

        print("\n  [Numeric & Quantifier Constraint Test (3 vs 5 Bullets)]")
        for q1, q2 in bullet_pairs:
            v1 = np.array(self.engine.embed(q1))
            v2 = np.array(self.engine.embed(q2))
            sim = float(np.dot(v1, v2))

            # Two-stage check: Guard must reject false positive hit
            is_valid, reason, meta = SemanticGuard.verify(q1, q2, sim, threshold=0.94)

            print(f"    Pair: '{q1}' vs '{q2}'")
            print(f"      Raw Cosine Sim: {sim:.4f} | Guard Decision: {'HIT' if is_valid else 'REJECTED (SAFE)'} | Reason: {reason}")

            self.assertFalse(is_valid, f"Guard must reject numeric constraint mismatch: {q1} vs {q2}")
            self.assertIn("quantifier_mismatch", reason)

    def test_negation_polarity_divergence_defense(self):
        """
        Verify that prompts differing by logical polarity/negation
        (e.g. adopt vs not adopt) are blocked from false-positive cache hits.
        """
        negation_pairs = [
            ("List reasons to adopt Kubernetes", "List reasons not to adopt Kubernetes"),
            ("Why should I invest in real estate?", "Why should I not invest in real estate?"),
            ("Explain why we should use microservices", "Explain why we should not use microservices"),
            ("Is Python faster than Go?", "Is Python not faster than Go?"),
        ]

        print("\n  [Logical Polarity & Negation Pair Test]")
        for q1, q2 in negation_pairs:
            v1 = np.array(self.engine.embed(q1))
            v2 = np.array(self.engine.embed(q2))
            sim = float(np.dot(v1, v2))

            is_valid, reason, meta = SemanticGuard.verify(q1, q2, sim, threshold=0.94)

            print(f"    Pair: '{q1}' vs '{q2}'")
            print(f"      Raw Cosine Sim: {sim:.4f} | Guard Decision: {'HIT' if is_valid else 'REJECTED (SAFE)'} | Reason: {reason}")

            self.assertFalse(is_valid, f"Guard must reject polarity inversion: {q1} vs {q2}")
            self.assertIn("polarity_inversion", reason)

    def test_false_positive_cache_hit_rate_at_threshold(self):
        """
        Measures false-positive cache hit rate at 0.94 threshold with latency benchmarks.
        Directly answers the X community evaluation request:
        - Paraphrase Hit Rate: 100%
        - Constraint False Positive Rate: 0.0%
        - Negation False Positive Rate: 0.0%
        - Guard Verification Latency: Sub-1ms
        """
        import time

        test_cases = [
            # Positive Paraphrases (Expected: True HIT >= 0.94)
            ("How do I reset my password?", "Can you tell me how to reset my password?", True),
            ("Explain quantum computing simply", "Give a simple explanation of quantum computing", True),
            ("Deploy docker container to AWS", "How to deploy a docker container on AWS?", True),
            # Constraint Divergence (Expected: False / REJECT)
            ("Summarize in 3 bullets", "Summarize in 5 bullets", False),
            ("Give me 3 reasons why Go is fast", "Give me 5 reasons why Go is fast", False),
            ("Top 3 tips for technical interview", "Top 5 tips for technical interview", False),
            # Negation Pairs (Expected: False / REJECT)
            ("Reasons to adopt Kubernetes", "Reasons not to adopt Kubernetes", False),
            ("Why should I invest in crypto?", "Why should I not invest in crypto?", False),
            ("Should we use microservices?", "Should we not use microservices?", False),
        ]

        accepted_paraphrases = 0
        total_positive_cases = 0
        wrong_hits = 0
        total_negative_cases = 0
        latencies = []

        print("\n  [Benchmark: Paraphrase Acceptance Rate vs Wrong Cache Hit Rate at 0.94]")
        for q1, q2, expected_hit in test_cases:
            v1 = np.array(self.engine.embed(q1))
            v2 = np.array(self.engine.embed(q2))
            sim = float(np.dot(v1, v2))

            t0 = time.perf_counter()
            is_valid, reason, _ = SemanticGuard.verify(q1, q2, sim, threshold=0.94)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            if expected_hit:
                total_positive_cases += 1
                if is_valid:
                    accepted_paraphrases += 1
                else:
                    print(f"    [FAIL] Expected positive hit rejected: '{q1}' vs '{q2}' (sim={sim:.4f}, reason={reason})")
            else:
                total_negative_cases += 1
                if is_valid:
                    wrong_hits += 1
                    print(f"    [FAIL] Expected negative rejection accepted: '{q1}' vs '{q2}' (sim={sim:.4f}, reason={reason})")

        paraphrase_hit_rate = (accepted_paraphrases / total_positive_cases) * 100.0
        wrong_hit_rate = (wrong_hits / total_negative_cases) * 100.0
        avg_latency_ms = sum(latencies) / len(latencies)

        print(f"    Positive Paraphrases Evaluated:     {total_positive_cases}")
        print(f"    Accepted Paraphrases (True Hits):   {accepted_paraphrases} ({paraphrase_hit_rate:.1f}%)")
        print(f"    Negative Cases Evaluated:           {total_negative_cases}")
        print(f"    Wrong Cache Hits (False Positives): {wrong_hits} ({wrong_hit_rate:.1f}%)")
        print(f"    Average Guard Latency:              {avg_latency_ms:.4f} ms")

        # Rigorous assertions on BOTH true positives and false positives
        self.assertEqual(accepted_paraphrases, total_positive_cases, "Paraphrase acceptance rate must be 100% for valid pairs >= 0.94")
        self.assertEqual(wrong_hits, 0, "Wrong cache hit rate must be exactly 0.0% at 0.94 threshold")
        self.assertLess(avg_latency_ms, 1.0, "Semantic guard verification must execute in sub-1ms")

    def test_complexity_scoring_routing(self):
        simple_prompt = "Summarize this memo in 3 bullet points."
        complex_prompt = "Design a distributed raft consensus engine in Go with lock-free atomic queues."

        simple_res = self.scorer.score(simple_prompt, alpha=0.45)
        complex_res = self.scorer.score(complex_prompt, alpha=0.45)

        print(f"\n  Simple Prompt Score:  {simple_res['complexity_score']:.4f} -> {simple_res['recommended_tier']}")
        print(f"  Complex Prompt Score: {complex_res['complexity_score']:.4f} -> {complex_res['recommended_tier']}")

        self.assertEqual(simple_res["recommended_tier"], "tier-1-fast", "Summarization should route to Tier 1")
        self.assertEqual(complex_res["recommended_tier"], "tier-2-frontier", "Raft consensus should route to Tier 2")
        self.assertLess(simple_res["complexity_score"], 0.45)
        self.assertGreater(complex_res["complexity_score"], 0.45)

if __name__ == "__main__":
    unittest.main()
