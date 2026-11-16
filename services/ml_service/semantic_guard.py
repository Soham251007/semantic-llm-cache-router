"""
Semantic Guard & Constraint Verification Engine.
Provides sub-millisecond semantic validation to eliminate false-positive cache hits:
1. Numeric & Quantifier Constraints (e.g. '3 bullets' vs '5 bullets')
2. Logical Polarity & Negation Inversion (e.g. 'adopt' vs 'not adopt')
3. Directional & Ordering Constraints (e.g. 'ascending' vs 'descending')
"""

import re
import time
from typing import Dict, Any, List, Tuple

# Number words to string digits mapping
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
    "fifty": "50", "hundred": "100"
}

# Negation and opposition patterns
NEGATION_PATTERNS = [
    r"\b(not|never|no|neither|nor|none|nobody|nowhere|nothing)\b",
    r"\b(don't|doesn't|didn't|won't|wouldn't|shouldn't|couldn't|can't|cannot|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b",
    r"\b(without|avoid|prevent|refrain|against)\b"
]
NEGATION_REGEX = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

# Numeric / quantifier constraints (e.g., "3 bullets", "top 5 tips", "100 words")
QUANTIFIER_REGEX = re.compile(
    r"\b(top\s*)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|fifty|hundred)\s*"
    r"(bullets?|bullet\s*points?|points?|reasons?|tips?|ways?|steps?|examples?|items?|ideas?|words?|sentences?|paragraphs?|questions?)\b",
    re.IGNORECASE
)

# Ordering and directional constraints
ORDERING_REGEX = re.compile(
    r"\b(ascending|descending|asc|desc|alphabetical|chronological|reverse)\b",
    re.IGNORECASE
)

class SemanticGuard:
    @staticmethod
    def extract_signature(text: str) -> Dict[str, Any]:
        """
        Extracts a structural semantic signature containing:
        - polarity: 'positive' or 'negative'
        - quantifiers: normalized list of numeric/target constraints
        - ordering: sorting/directional constraints
        - signature: compact hashable string for O(1) comparison in Go/Redis
        """
        text_lower = text.lower()

        # 1. Polarity detection
        negations = NEGATION_REGEX.findall(text_lower)
        neg_tokens = []
        for match in negations:
            if isinstance(match, tuple):
                for m in match:
                    if m:
                        neg_tokens.append(m)
            elif match:
                neg_tokens.append(match)

        has_negation = len(neg_tokens) > 0
        polarity = "negative" if has_negation else "positive"

        # 2. Quantifier & Cardinal Constraints
        quantifiers = []
        for match in QUANTIFIER_REGEX.finditer(text_lower):
            raw_num = match.group(2).lower()
            unit = match.group(3).lower()

            if "bullet" in unit:
                norm_unit = "bullets"
            elif "point" in unit:
                norm_unit = "points"
            elif "reason" in unit:
                norm_unit = "reasons"
            elif "tip" in unit:
                norm_unit = "tips"
            elif "step" in unit:
                norm_unit = "steps"
            elif "example" in unit:
                norm_unit = "examples"
            elif "word" in unit:
                norm_unit = "words"
            else:
                norm_unit = unit

            num_val = NUMBER_WORDS.get(raw_num, raw_num)
            quantifiers.append(f"{norm_unit}:{num_val}")

        quantifiers = sorted(list(set(quantifiers)))

        # 3. Ordering / Directional Constraints
        order_matches = ORDERING_REGEX.findall(text_lower)
        ordering = sorted(list(set(m.lower() for m in order_matches)))

        # 4. Generate compact deterministic signature for O(1) filter
        sig_parts = [f"pol:{polarity}"]
        if quantifiers:
            sig_parts.append(f"q:{','.join(quantifiers)}")
        if ordering:
            sig_parts.append(f"ord:{','.join(ordering)}")

        compact_signature = ";".join(sig_parts)

        return {
            "polarity": polarity,
            "negations": neg_tokens,
            "quantifiers": quantifiers,
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

        # 1. Polarity check
        if sig_query["polarity"] != sig_candidate["polarity"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"polarity_inversion ({sig_query['polarity']} vs {sig_candidate['polarity']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 2. Quantifier constraint check
        if sig_query["quantifiers"] != sig_candidate["quantifiers"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"quantifier_mismatch ({sig_query['quantifiers']} != {sig_candidate['quantifiers']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 3. Ordering check
        if sig_query["ordering"] != sig_candidate["ordering"]:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            reason = f"ordering_mismatch ({sig_query['ordering']} != {sig_candidate['ordering']})"
            return False, reason, {
                "similarity": similarity,
                "query_signature": sig_query["signature"],
                "candidate_signature": sig_candidate["signature"],
                "latency_ms": dur_ms
            }

        # 4. Cosine similarity threshold check
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
