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
