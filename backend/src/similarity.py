"""
TokenShrink - Semantic Similarity Module
Evaluates how much meaning is preserved after compression.

Uses cosine similarity on TF-IDF vectors (always available).
Optionally uses sentence-transformers for deeper semantic scoring.
"""

import math
import re
from typing import Dict, List, Tuple
from collections import Counter


def _tfidf_vector(text: str, vocab: List[str], idf: Dict[str, float]) -> List[float]:
    """Compute a TF-IDF vector for text over a fixed vocabulary."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    tf = Counter(words)
    total = max(len(words), 1)
    vec = []
    for term in vocab:
        t = tf.get(term, 0) / total
        vec.append(t * idf.get(term, 1.0))
    return vec


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticSimilarityEvaluator:
    """
    Measures semantic similarity between original and compressed texts.
    
    Primary method: TF-IDF cosine similarity (no external dependencies).
    Optional method: Sentence-transformers embeddings (if installed).
    """

    def __init__(self):
        self._transformer_model = None
        self._has_transformers = False
        self._try_load_transformers()

    def _try_load_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Use lightweight model
            self._transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._has_transformers = True
        except Exception:
            self._has_transformers = False

    def tfidf_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute TF-IDF cosine similarity between two texts.
        
        Args:
            text_a: First text (original)
            text_b: Second text (compressed)
            
        Returns:
            Float in [0, 1], where 1.0 = identical
        """
        if not text_a.strip() or not text_b.strip():
            return 0.0

        words_a = re.findall(r"\b[a-zA-Z]{2,}\b", text_a.lower())
        words_b = re.findall(r"\b[a-zA-Z]{2,}\b", text_b.lower())

        # Build shared vocabulary
        vocab = sorted(set(words_a) | set(words_b))
        if not vocab:
            return 0.0

        # Compute IDF over both documents
        N = 2
        idf: Dict[str, float] = {}
        for term in vocab:
            df = (term in set(words_a)) + (term in set(words_b))
            idf[term] = math.log((N + 1) / (df + 1)) + 1.0

        vec_a = _tfidf_vector(text_a, vocab, idf)
        vec_b = _tfidf_vector(text_b, vocab, idf)

        return round(_cosine_sim(vec_a, vec_b), 4)

    def transformer_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute sentence-transformer embedding cosine similarity.
        
        Falls back to TF-IDF if transformers not available.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Float cosine similarity in [-1, 1]
        """
        if not self._has_transformers or not self._transformer_model:
            return self.tfidf_similarity(text_a, text_b)

        try:
            import numpy as np
            embeddings = self._transformer_model.encode([text_a, text_b])
            a, b = embeddings[0], embeddings[1]
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            return round(cos_sim, 4)
        except Exception:
            return self.tfidf_similarity(text_a, text_b)

    def evaluate(self, original: str, compressed: str) -> Dict[str, float]:
        """
        Full semantic evaluation between original and compressed prompts.
        
        Args:
            original: Original prompt text
            compressed: Compressed prompt text
            
        Returns:
            Dict with tfidf_similarity, semantic_similarity, method
        """
        tfidf_sim = self.tfidf_similarity(original, compressed)
        
        if self._has_transformers:
            semantic_sim = self.transformer_similarity(original, compressed)
            method = "sentence-transformers"
        else:
            semantic_sim = tfidf_sim
            method = "tfidf-cosine"

        return {
            "tfidf_similarity": tfidf_sim,
            "semantic_similarity": semantic_sim,
            "method": method,
            "meaning_preserved_pct": round(semantic_sim * 100, 2),
        }
