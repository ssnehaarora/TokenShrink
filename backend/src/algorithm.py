"""
TokenShrink - Greedy Search Algorithm Module
Core compression engine using Information Density + Dynamic Redundancy Filtering.

Algorithm:
1. Score all sentences by information density
2. Greedily select highest-scoring sentence
3. Update used-keyword set (Dynamic Redundancy Filter)
4. Re-score remaining sentences
5. Repeat until token budget exhausted or target ratio achieved

Structure Preservation:
- First sentence is ALWAYS included (context anchor)
- Last sentence is ALWAYS included (instruction/conclusion)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .preprocessor import TextPreprocessor
from .scorer import TFIDFScorer, InformationDensityScorer
from .token_counter import TokenCounter


@dataclass
class SentenceRecord:
    """Tracks a sentence and its metadata during greedy selection."""
    index: int
    text: str
    keywords: List[str]
    token_count: int
    score: float = 0.0
    selected: bool = False
    selection_order: int = -1
    is_anchor: bool = False  # first or last sentence


@dataclass
class CompressionResult:
    """Full output of a compression run."""
    compressed_text: str
    original_text: str
    selected_sentences: List[SentenceRecord]
    all_sentences: List[SentenceRecord]
    stats: Dict[str, Any]
    debug_log: List[str] = field(default_factory=list)


class GreedyCompressor:
    """
    Greedy Search-based prompt compressor with Dynamic Redundancy Filtering.
    
    Implements:
    - TF-IDF informed Information Density scoring
    - Greedy selection with live re-scoring
    - Token-budget and compression-ratio stopping conditions
    - Anchor sentence preservation (first/last)
    - Detailed debug logging
    """

    def __init__(
        self,
        preserve_structure: bool = True,
        redundancy_penalty: float = 0.5,
    ):
        """
        Args:
            preserve_structure: If True, always keep first and last sentences
            redundancy_penalty: Penalty multiplier for already-used keywords [0, 1]
        """
        self.preprocessor = TextPreprocessor()
        self.token_counter = TokenCounter()
        self.preserve_structure = preserve_structure
        self.redundancy_penalty = redundancy_penalty

    def compress(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        target_ratio: Optional[float] = None,
        max_sentences: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress a prompt using greedy information-density selection.
        
        Args:
            text: Raw input prompt
            max_tokens: Maximum tokens in compressed output
            target_ratio: Target compression ratio, e.g. 0.5 = keep 50% of tokens
            max_sentences: Maximum sentences to select (alternative stopping condition)
            
        Returns:
            CompressionResult with compressed text and full metadata
        
        Note: If none of the stopping conditions are set, defaults to 0.6 ratio.
        """
        debug_log: List[str] = []

        # ── Default stopping condition ───────────────────────────────────
        if max_tokens is None and target_ratio is None and max_sentences is None:
            target_ratio = 0.6

        # ── Step 1: Preprocess ───────────────────────────────────────────
        preprocessed = self.preprocessor.preprocess(text)
        sentences_text = preprocessed["sentences"]
        keywords_per = preprocessed["keywords_per_sentence"]

        if not sentences_text:
            return self._empty_result(text)

        # ── Step 2: Count tokens ─────────────────────────────────────────
        token_counts = self.token_counter.count_sentences(sentences_text)
        total_tokens = sum(token_counts)
        debug_log.append(f"Original: {len(sentences_text)} sentences, {total_tokens} tokens")
        debug_log.append(f"Token backend: {self.token_counter.backend}")

        # ── Step 3: Determine token budget ───────────────────────────────
        if max_tokens is not None:
            budget = max_tokens
        elif target_ratio is not None:
            budget = max(1, int(total_tokens * target_ratio))
        else:
            budget = total_tokens  # no token limit, use sentence limit

        debug_log.append(f"Token budget: {budget} ({budget/max(total_tokens,1)*100:.1f}% of original)")

        # ── Step 4: Build TF-IDF model ───────────────────────────────────
        tfidf = TFIDFScorer()
        tfidf.fit(sentences_text)
        scorer = InformationDensityScorer(tfidf_scorer=tfidf)
        scorer.reset()

        # ── Step 5: Build sentence records ───────────────────────────────
        records: List[SentenceRecord] = []
        for i, (text_s, kws, tc) in enumerate(
            zip(sentences_text, keywords_per, token_counts)
        ):
            rec = SentenceRecord(
                index=i,
                text=text_s,
                keywords=kws,
                token_count=tc,
            )
            if self.preserve_structure and (i == 0 or i == len(sentences_text) - 1):
                rec.is_anchor = True
            records.append(rec)

        # ── Step 6: Pre-select anchor sentences ──────────────────────────
        selection_order = 0
        tokens_used = 0
        selected_indices = set()

        if self.preserve_structure and len(records) >= 1:
            anchors = [records[0]]
            if len(records) > 1:
                anchors.append(records[-1])
            for anchor in anchors:
                if tokens_used + anchor.token_count <= budget or tokens_used == 0:
                    anchor.selected = True
                    anchor.selection_order = selection_order
                    selection_order += 1
                    selected_indices.add(anchor.index)
                    tokens_used += anchor.token_count
                    scorer.update_used_keywords(anchor.keywords)
                    debug_log.append(
                        f"[ANCHOR] Sentence {anchor.index}: {anchor.text[:60]}... "
                        f"({anchor.token_count} tokens)"
                    )

        # ── Step 7: Greedy selection loop ─────────────────────────────────
        iteration = 0
        while True:
            iteration += 1

            # Check stopping conditions
            remaining_sentences = len([r for r in records if not r.selected])
            if remaining_sentences == 0:
                debug_log.append("Stopping: All sentences selected")
                break

            if max_sentences is not None and selection_order >= max_sentences:
                debug_log.append(f"Stopping: Max sentences ({max_sentences}) reached")
                break

            # Score un-selected sentences
            candidates = [r for r in records if not r.selected]
            for rec in candidates:
                rec.score = scorer.score_sentence(
                    rec.keywords,
                    max(rec.token_count, 1),
                    self.redundancy_penalty,
                )

            # Sort by score descending
            candidates.sort(key=lambda r: r.score, reverse=True)

            # Pick best candidate that fits budget
            chosen = None
            for candidate in candidates:
                if tokens_used + candidate.token_count <= budget:
                    chosen = candidate
                    break

            if chosen is None:
                debug_log.append(
                    f"Stopping: No remaining sentence fits within budget "
                    f"({budget - tokens_used} tokens remaining)"
                )
                break

            # Select the sentence
            chosen.selected = True
            chosen.selection_order = selection_order
            selection_order += 1
            selected_indices.add(chosen.index)
            tokens_used += chosen.token_count

            debug_log.append(
                f"[ITER {iteration}] Selected sentence {chosen.index} "
                f"(score={chosen.score:.4f}, tokens={chosen.token_count}, "
                f"total_used={tokens_used}): {chosen.text[:60]}..."
            )

            # Update Dynamic Redundancy Filter
            scorer.update_used_keywords(chosen.keywords)

        # ── Step 8: Reconstruct compressed text (preserve original order) ─
        selected_in_order = sorted(
            [r for r in records if r.selected],
            key=lambda r: r.index,
        )

        compressed_text = " ".join(r.text for r in selected_in_order)

        # ── Step 9: Compute final statistics ─────────────────────────────
        stats = self.token_counter.stats(text, compressed_text)
        stats["original_sentence_count"] = len(records)
        stats["compressed_sentence_count"] = len(selected_in_order)
        stats["sentence_retention_pct"] = round(
            len(selected_in_order) / max(len(records), 1) * 100, 2
        )
        stats["iterations"] = iteration
        stats["budget_used"] = tokens_used
        stats["budget_total"] = budget

        debug_log.append(
            f"Final: {stats['compressed_tokens']} tokens "
            f"({stats['reduction_pct']}% reduction), "
            f"{len(selected_in_order)}/{len(records)} sentences kept"
        )

        return CompressionResult(
            compressed_text=compressed_text,
            original_text=text,
            selected_sentences=selected_in_order,
            all_sentences=records,
            stats=stats,
            debug_log=debug_log,
        )

    def _empty_result(self, text: str) -> CompressionResult:
        stats = self.token_counter.stats(text, "")
        return CompressionResult(
            compressed_text="",
            original_text=text,
            selected_sentences=[],
            all_sentences=[],
            stats=stats,
            debug_log=["Empty input — nothing to compress"],
        )
