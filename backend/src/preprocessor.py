"""
TokenShrink - Preprocessing Module
Handles sentence splitting, tokenization, and stopword removal.
Falls back gracefully when NLTK data is unavailable.
"""

import re
import string
from typing import List, Set

# ── Built-in fallback stopwords (works even without NLTK data) ──────────────
BUILTIN_STOPWORDS: Set[str] = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","before","after",
    "above","below","between","each","few","more","most","other","some",
    "such","no","not","only","same","so","than","too","very","s","t",
    "can","will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn",
    "ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","i","me","my","myself","we","our","ours","ourselves","you",
    "your","yours","yourself","yourselves","he","him","his","himself","she",
    "her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that","these",
    "those","am","is","are","was","were","be","been","being","have","has",
    "had","having","do","does","did","doing","would","could","might","must",
    "shall","should","may","its","also","however","therefore","thus","hence",
    "whereas","while","although","though","because","since","unless","until",
    "whether","if","as","when","where","how","all","both","any","there",
    "here","then","than","that","its","been","over","under","again","further",
    "once","own","off","out","down","very","still","back","just","because",
}

def _try_nltk_stopwords() -> Set[str]:
    """Attempt to load NLTK stopwords; fall back to built-in set."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return BUILTIN_STOPWORDS

def _try_nltk_sent_tokenize(text: str) -> List[str]:
    """Attempt NLTK sentence tokenization; fall back to regex."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        return _regex_sent_tokenize(text)

def _regex_sent_tokenize(text: str) -> List[str]:
    """
    Robust regex-based sentence splitter.
    Handles abbreviations, decimals, and common edge cases.
    """
    # Protect common abbreviations
    abbrevs = r"(?<![A-Z])(?<![A-Z][a-z])(?<=\w)\.(?=\s+[a-z])"
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|U\.S|U\.K)\.", r"\1<DOT>", text, flags=re.IGNORECASE)
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])", text)
    # Restore protected dots
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences if sentences else [text.strip()]

def _word_tokenize(text: str) -> List[str]:
    """Simple word tokenizer that splits on non-alphanumeric chars."""
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text)


class TextPreprocessor:
    """
    Handles all text preprocessing for TokenShrink.
    
    Responsibilities:
    - Sentence segmentation
    - Word tokenization
    - Stopword & punctuation removal
    - Keyword extraction per sentence
    """

    def __init__(self):
        self.stopwords = _try_nltk_stopwords()
        self.punctuation = set(string.punctuation)

    def split_sentences(self, text: str) -> List[str]:
        """
        Split input text into individual sentences.
        
        Args:
            text: Raw input text
            
        Returns:
            List of sentence strings
        """
        text = text.strip()
        if not text:
            return []
        sentences = _try_nltk_sent_tokenize(text)
        # Filter empty sentences, normalize whitespace
        cleaned = [" ".join(s.split()) for s in sentences if s.strip()]
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words (lowercased).
        
        Args:
            text: Input string
            
        Returns:
            List of lowercase word tokens
        """
        tokens = _word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha() or t.isalnum()]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords and single-character tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list of meaningful tokens
        """
        return [
            t for t in tokens
            if t.lower() not in self.stopwords
            and len(t) > 1
            and t not in self.punctuation
        ]

    def extract_keywords(self, sentence: str) -> List[str]:
        """
        Extract meaningful keywords from a sentence.
        
        Pipeline: tokenize → lowercase → remove stopwords → deduplicate
        
        Args:
            sentence: Input sentence string
            
        Returns:
            List of unique keyword strings
        """
        tokens = self.tokenize(sentence)
        filtered = self.remove_stopwords(tokens)
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique = []
        for t in filtered:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def count_tokens_simple(self, text: str) -> int:
        """Quick whitespace-based token count (fallback)."""
        return len(text.split())

    def preprocess(self, text: str) -> dict:
        """
        Full preprocessing pipeline for an input text.
        
        Args:
            text: Raw prompt text
            
        Returns:
            Dict with keys: sentences, keywords_per_sentence, raw_token_counts
        """
        sentences = self.split_sentences(text)
        result = {
            "sentences": sentences,
            "keywords_per_sentence": [],
            "raw_token_counts": [],
        }
        for sent in sentences:
            kws = self.extract_keywords(sent)
            result["keywords_per_sentence"].append(kws)
            result["raw_token_counts"].append(self.count_tokens_simple(sent))
        return result
