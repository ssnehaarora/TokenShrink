"""
TokenShrink - Main API Module
Public interface that orchestrates all compression modules.
"""

from typing import Optional, Dict, Any
from .algorithm import GreedyCompressor
from .similarity import SemanticSimilarityEvaluator


class TokenShrink:
    """
    Main TokenShrink API.
    
    Usage:
        ts = TokenShrink()
        result = ts.compress(
            text="Your long prompt here...",
            max_tokens=200,
        )
        print(result["compressed_text"])
        print(result["stats"])
    """

    def __init__(
        self,
        preserve_structure: bool = True,
        redundancy_penalty: float = 0.5,
    ):
        self.compressor = GreedyCompressor(
            preserve_structure=preserve_structure,
            redundancy_penalty=redundancy_penalty,
        )
        self.evaluator = SemanticSimilarityEvaluator()

    def compress(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        target_ratio: Optional[float] = None,
        max_sentences: Optional[int] = None,
        include_debug: bool = False,
        include_similarity: bool = True,
    ) -> Dict[str, Any]:
        """
        Compress a prompt using greedy information-density algorithm.
        
        Args:
            text: Input prompt (any length)
            max_tokens: Hard token budget for output
            target_ratio: Fraction of original tokens to keep (e.g. 0.5 = 50%)
            max_sentences: Maximum number of sentences to select
            include_debug: Include per-iteration debug log in output
            include_similarity: Compute semantic similarity metrics
            
        Returns:
            Dict containing:
                - compressed_text: Compressed prompt string
                - stats: Token counts and reduction metrics
                - selected_sentences: List of selected sentence objects
                - similarity: Semantic similarity scores (if include_similarity)
                - debug_log: Greedy iteration log (if include_debug)
        """
        if not text or not text.strip():
            return {
                "compressed_text": "",
                "stats": {},
                "selected_sentences": [],
                "error": "Empty input text",
            }

        # Run compression
        result = self.compressor.compress(
            text=text,
            max_tokens=max_tokens,
            target_ratio=target_ratio,
            max_sentences=max_sentences,
        )

        # Build response
        response: Dict[str, Any] = {
            "compressed_text": result.compressed_text,
            "original_text": text,
            "stats": result.stats,
            "selected_sentences": [
                {
                    "index": s.index,
                    "text": s.text,
                    "keywords": s.keywords,
                    "token_count": s.token_count,
                    "score": round(s.score, 6),
                    "selection_order": s.selection_order,
                    "is_anchor": s.is_anchor,
                }
                for s in result.selected_sentences
            ],
            "all_sentences": [
                {
                    "index": s.index,
                    "text": s.text,
                    "token_count": s.token_count,
                    "score": round(s.score, 6),
                    "selected": s.selected,
                    "is_anchor": s.is_anchor,
                }
                for s in result.all_sentences
            ],
        }

        # Semantic similarity evaluation
        if include_similarity and result.compressed_text:
            similarity = self.evaluator.evaluate(text, result.compressed_text)
            response["similarity"] = similarity

        # Debug log
        if include_debug:
            response["debug_log"] = result.debug_log

        return response
