"""
Semantic Guard & Constraint Verification Engine.
Provides sub-millisecond semantic validation to eliminate false-positive cache hits:
1. Universal Numerical & Quantifier Constraints (any number bound to ANY noun / entity)
2. Closed-Class Grammatical Polarity & Negation Detection
3. Directional & Ordering Constraints
"""

import re
import time
from typing import Dict, Any, List, Tuple

# Universal English numerals mapping (closed grammatical class)
NUMERAL_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000
}

# Negation function words in English (closed grammatical class)
NEGATION_FUNCTION_WORDS = {
    "not", "never", "no", "neither", "nor", "none", "nobody", "nowhere", "nothing",
    "without", "against"
}

# Directional constraints (closed grammatical class)
DIRECTIONAL_WORDS = {
    "ascending", "descending", "asc", "desc", "alphabetical", "chronological", "reverse"
}

def _normalize_noun(noun: str) -> str:
    """
    Generic morphological stemming: collapses plurals to singular form
    dynamically without any hardcoded word dictionary.
    """
    n = noun.lower()
    if n.endswith("ies") and len(n) > 4:
        return n[:-3] + "y"
    if n.endswith("es") and len(n) > 3 and n[-3] in "sxyz":
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 2:
        return n[:-1]
    return n

class SemanticGuard:
    @staticmethod
    def extract_signature(text: str) -> Dict[str, Any]:
        """
        Extracts a generalized semantic signature containing:
        - polarity: 'positive' or 'negative' based on closed-class negation syntax
        - quantifiers: generic map of any noun bound to numerical quantities
        - numbers: all standalone numerical values in the query
        - signature: compact hashable string for O(1) comparison in Go/Redis
        """
        text_lower = text.lower()
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text_lower)

        # 1. Closed-Class Grammatical Polarity Detection
        negations_found = []
        for t in tokens:
            if t in NEGATION_FUNCTION_WORDS or t.endswith("n't") or t == "cannot":
                negations_found.append(t)

        polarity = "negative" if len(negations_found) > 0 else "positive"

        # 2. Universal Quantifier Extraction (Matches ANY number bound to ANY noun)
        quantifiers = {}

        # 2a. Match digit + ANY word (e.g. "3 bullets", "5 alternatives", "10 metrics", "100 words")
        for match in re.finditer(r"\b(\d+)\s+([a-zA-Z]+)\b", text_lower):
            val = int(match.group(1))
            noun = _normalize_noun(match.group(2))
            quantifiers[noun] = val

        # 2b. Match word numeral + ANY word (e.g. "three bullets", "five alternatives")
        num_pattern = "|".join(NUMERAL_MAP.keys())
        for match in re.finditer(rf"\b({num_pattern})\s+([a-zA-Z]+)\b", text_lower):
            val = NUMERAL_MAP[match.group(1)]
            noun = _normalize_noun(match.group(2))
            if noun not in quantifiers:
                quantifiers[noun] = val

        # 2c. Collect all raw numbers (handles standalone numbers like "in 3" vs "in 5")
        all_numbers = sorted([int(n) for n in re.findall(r"\b\d+\b", text_lower)])

        # 3. Directional / Ordering Constraints
        ordering = sorted(list(set(t for t in tokens if t in DIRECTIONAL_WORDS)))

        # 4. Generate deterministic compact signature for O(1) in-memory comparison
        quant_str = ",".join(f"{k}:{v}" for k, v in sorted(quantifiers.items()))
        num_str = ",".join(str(n) for n in all_numbers)
        ord_str = ",".join(ordering)

        compact_signature = f"pol:{polarity}|q:{quant_str}|n:{num_str}|ord:{ord_str}"

        return {
            "polarity": polarity,
            "negations": negations_found,
            "quantifiers": quantifiers,
            "numbers": all_numbers,
            "ordering": ordering,
            "signature": compact_signature
        }

    @classmethod
    def verify(
        cls,
        query_text: str,
        candidate_text: str,
        similarity: float,
        threshold: float = 0.94
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Two-stage verification:
        1. Strict alignment of polarity, numeric quantifiers, and directional constraints.
        2. Cosine similarity >= threshold
        Returns (is_valid_cache_hit, reason, metadata).
        """
        t0 = time.perf_counter()

        sig_query = cls.extract_signature(query_text)
        sig_candidate = cls.extract_signature(candidate_text)

        # 1. Polarity check (affirmative vs negated contradiction)
        if sig_query["polarity"] != sig_candidate["polarity"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"polarity_inversion ({sig_query['polarity']} vs {sig_candidate['polarity']}: {sig_query['negations']} vs {sig_candidate['negations']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 2. Universal Quantifier constraint check
        if sig_query["quantifiers"] != sig_candidate["quantifiers"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"quantifier_mismatch ({sig_query['quantifiers']} != {sig_candidate['quantifiers']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 3. Raw standalone numbers check (e.g. "top 3" vs "top 5")
        if sig_query["numbers"] != sig_candidate["numbers"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"number_mismatch ({sig_query['numbers']} != {sig_candidate['numbers']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 4. Ordering check
        if sig_query["ordering"] != sig_candidate["ordering"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"ordering_mismatch ({sig_query['ordering']} != {sig_candidate['ordering']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 5. Cosine similarity threshold check
        if similarity < threshold:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            return False, f"similarity_below_threshold ({similarity:.4f} < {threshold:.4f})", {
                "similarity": similarity,
                "latency_ms": dur_ms
            }

        dur_ms = (time.perf_counter() - t0) * 1000.0
        return True, "verified_semantic_match", {
            "similarity": similarity,
            "signature": sig_query["signature"],
            "latency_ms": dur_ms
        }
