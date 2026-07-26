"""
TokenShrink - Scoring Module
Implements the Information Density Heuristic and TF-IDF scoring.

Score formula:
    score(s) = Σ tfidf(kw) for kw in unique_unused_keywords(s)
               ─────────────────────────────────────────────────
                        token_count(s)

If TF-IDF is unavailable, uniform keyword weight (1.0) is used.
"""

import math
import re
from typing import List, Dict, Set, Optional
from collections import Counter


class TFIDFScorer:
    """
    Lightweight TF-IDF implementation that works sentence-by-sentence.
    Uses scikit-learn when available; falls back to manual computation.
    """

    def __init__(self):
        self._use_sklearn = False
        self._vectorizer = None
        self._tfidf_matrix = None
        self._feature_names: List[str] = []
        self._word_scores: Dict[str, float] = {}
        self._try_sklearn()

    def _try_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._use_sklearn = True
        except ImportError:
            self._use_sklearn = False

    def fit(self, sentences: List[str]):
        """
        Fit TF-IDF on a corpus of sentences.
        
        Args:
            sentences: List of sentence strings (the full document)
        """
        if not sentences:
            return

        if self._use_sklearn:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 1),
                min_df=1,
                sublinear_tf=True,
            )
            try:
                self._tfidf_matrix = self._vectorizer.fit_transform(sentences)
                self._feature_names = self._vectorizer.get_feature_names_out().tolist()
                # Compute per-word global importance = mean TF-IDF across sentences
                arr = self._tfidf_matrix.toarray()
                mean_scores = arr.mean(axis=0)
                self._word_scores = {
                    word: float(mean_scores[i])
                    for i, word in enumerate(self._feature_names)
                }
            except Exception:
                self._word_scores = {}
        else:
            # Manual TF-IDF
            self._word_scores = self._manual_tfidf(sentences)

    def _manual_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Compute simplified TF-IDF without sklearn."""
        N = len(sentences)
        # Build document frequency
        doc_freq: Counter = Counter()
        tokenized = []
        for sent in sentences:
            words = set(re.findall(r"\b[a-zA-Z]{2,}\b", sent.lower()))
            doc_freq.update(words)
            tokenized.append(words)

        # Compute IDF
        idf: Dict[str, float] = {}
        for word, df in doc_freq.items():
            idf[word] = math.log((N + 1) / (df + 1)) + 1.0

        # Compute mean TF across all sentences
        all_words: Counter = Counter()
        for sent in sentences:
            words = re.findall(r"\b[a-zA-Z]{2,}\b", sent.lower())
            all_words.update(words)

        total = max(sum(all_words.values()), 1)
        scores: Dict[str, float] = {}
        for word, count in all_words.items():
            tf = count / total
            scores[word] = tf * idf.get(word, 1.0)

        return scores

    def word_score(self, word: str) -> float:
        """
        Get TF-IDF importance score for a word.
        
        Args:
            word: Lowercase word string
            
        Returns:
            Float importance score (0.0 if unknown)
        """
        return self._word_scores.get(word.lower(), 0.1)  # 0.1 = small default weight

    def is_fitted(self) -> bool:
        return bool(self._word_scores)


class InformationDensityScorer:
    """
    Computes Information Density scores for sentences.
    
    score(sentence) = Σ tfidf(kw) for kw in unique_unused_keywords
                     ─────────────────────────────────────────────
                                token_count(sentence)

    After sentence selection, already-used keywords are tracked and
    future scores are adjusted via the Dynamic Redundancy Filter.
    """

    def __init__(self, tfidf_scorer: Optional[TFIDFScorer] = None):
        self.tfidf = tfidf_scorer or TFIDFScorer()
        self.used_keywords: Set[str] = set()

    def reset(self):
        """Reset used-keyword tracking (call before each compression run)."""
        self.used_keywords = set()

    def score_sentence(
        self,
        keywords: List[str],
        token_count: int,
        redundancy_penalty: float = 0.5,
    ) -> float:
        """
        Compute information density score for a sentence.
        
        Args:
            keywords: List of keywords extracted from sentence
            token_count: Number of LLM tokens in the sentence
            redundancy_penalty: Weight multiplier for already-used keywords (default 0.5)
            
        Returns:
            Float score (higher = more informative per token)
        """
        if token_count == 0:
            return 0.0

        weighted_sum = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            base_score = self.tfidf.word_score(kw_lower)
            if kw_lower in self.used_keywords:
                # Penalize redundant keywords
                weighted_sum += base_score * redundancy_penalty
            else:
                weighted_sum += base_score

        return weighted_sum / token_count

    def score_all(
        self,
        sentences: List[str],
        keywords_per_sentence: List[List[str]],
        token_counts: List[int],
    ) -> List[float]:
        """
        Score all sentences (ignoring already-used keywords).
        
        Args:
            sentences: Raw sentence strings
            keywords_per_sentence: Keyword lists per sentence
            token_counts: Token counts per sentence
            
        Returns:
            List of float scores, same length as input
        """
        scores = []
        for kws, tc in zip(keywords_per_sentence, token_counts):
            scores.append(self.score_sentence(kws, tc))
        return scores

    def update_used_keywords(self, keywords: List[str]):
        """
        Mark keywords from a selected sentence as used.
        
        Args:
            keywords: Keywords to add to the used set
        """
        self.used_keywords.update(kw.lower() for kw in keywords)
