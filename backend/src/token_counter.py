"""
TokenShrink - Token Counter Module
Uses tiktoken (GPT-compatible) for accurate token counting.
Falls back to whitespace tokenization if tiktoken fails.
"""

from typing import Union

class TokenCounter:
    """
    Accurate token counter compatible with GPT-style LLMs.
    
    Uses tiktoken's cl100k_base encoding (GPT-4 / GPT-3.5-turbo compatible).
    Falls back gracefully to word-count estimation.
    """

    TIKTOKEN_ENCODING = "cl100k_base"

    def __init__(self):
        self._encoder = None
        self._backend = "whitespace"
        self._init_encoder()

    def _init_encoder(self):
        """Initialize tiktoken encoder with fallback."""
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.TIKTOKEN_ENCODING)
            self._backend = "tiktoken"
        except Exception as e:
            self._backend = "whitespace"

    @property
    def backend(self) -> str:
        """Which tokenization backend is in use."""
        return self._backend

    def count(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Integer token count
        """
        if not text or not text.strip():
            return 0
        if self._backend == "tiktoken" and self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: words ≈ tokens (rough estimate, ~1.3x correction factor)
        words = len(text.split())
        return max(1, int(words * 1.3))

    def count_sentences(self, sentences: list) -> list:
        """
        Count tokens for each sentence in a list.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of integer token counts, same length as input
        """
        return [self.count(s) for s in sentences]

    def stats(self, original: str, compressed: str) -> dict:
        """
        Compute compression statistics.
        
        Args:
            original: Original text
            compressed: Compressed text
            
        Returns:
            Dict with original_tokens, compressed_tokens, reduction_pct, ratio
        """
        orig_count = self.count(original)
        comp_count = self.count(compressed)
        
        reduction_pct = 0.0
        if orig_count > 0:
            reduction_pct = round((1 - comp_count / orig_count) * 100, 2)
        
        ratio = 0.0
        if orig_count > 0:
            ratio = round(comp_count / orig_count, 4)

        return {
            "original_tokens": orig_count,
            "compressed_tokens": comp_count,
            "reduction_pct": reduction_pct,
            "compression_ratio": ratio,
            "tokens_saved": orig_count - comp_count,
            "backend": self._backend,
        }
