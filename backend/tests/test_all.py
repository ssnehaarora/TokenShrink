"""
TokenShrink - Test Suite
Tests all core modules: preprocessing, scoring, algorithm, token counting, similarity.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessor import TextPreprocessor
from src.token_counter import TokenCounter
from src.scorer import TFIDFScorer, InformationDensityScorer
from src.algorithm import GreedyCompressor
from src.similarity import SemanticSimilarityEvaluator
from src.tokenshrink import TokenShrink


# ── Sample texts ─────────────────────────────────────────────────────────────

SAMPLE_SHORT = """
Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
Neural networks are inspired by the human brain. They consist of layers of interconnected nodes.
Deep learning uses multiple layers to learn representations. It has achieved remarkable results in image recognition.
"""

SAMPLE_MEDIUM = """
Artificial intelligence has transformed numerous industries over the past decade. 
Companies across sectors from healthcare to finance are leveraging AI to automate tasks and gain insights.
Machine learning algorithms can identify patterns in data that humans might miss.
Natural language processing enables computers to understand and generate human language.
Large language models like GPT-4 and Claude can perform complex reasoning tasks.
These models are trained on vast amounts of text data using self-supervised learning.
The attention mechanism, introduced in the transformer architecture, revolutionized NLP.
Transfer learning allows models trained on one task to be fine-tuned for another.
Prompt engineering is the practice of crafting inputs to elicit desired outputs from language models.
The cost of running large models at scale is significant, making token efficiency crucial.
Reducing prompt length while preserving meaning can cut inference costs by 30-60%.
TokenShrink addresses this challenge using information-density-based greedy compression.
"""

SAMPLE_LONG = """
The rapid advancement of large language models has created both opportunities and challenges for developers and organizations.
These models, with their billions of parameters, have demonstrated remarkable capabilities in understanding and generating human language.
However, their computational requirements translate directly into operational costs that scale with the number of tokens processed.
Every API call to a service like OpenAI, Anthropic, or Google incurs costs proportional to input and output token counts.
For high-volume applications, even modest reductions in prompt length can result in substantial cost savings.
A 30% reduction in token usage across one million daily API calls can save thousands of dollars monthly.
Intelligent prompt compression aims to achieve this reduction without sacrificing the quality of model responses.
The key insight is that natural language contains significant redundancy and repetition.
Many sentences convey similar information, especially in longer documents and multi-paragraph prompts.
By identifying and removing the most redundant content, we can dramatically reduce token counts.
The challenge lies in determining which content is truly essential and which can be safely removed.
Simple heuristics like removing stopwords alone are insufficient for this task.
A more sophisticated approach considers the information density of each sentence relative to the whole.
TF-IDF (Term Frequency-Inverse Document Frequency) provides a principled way to measure word importance.
Words that appear frequently in a sentence but rarely across all sentences receive higher scores.
By aggregating these scores, we can estimate the information value of each sentence.
The greedy algorithm then iteratively selects the highest-value sentences within a token budget.
After each selection, the dynamic redundancy filter updates to penalize sentences with overlapping keywords.
This ensures the selected sentences collectively cover the maximum informational breadth.
The result is a compressed prompt that preserves the most important concepts from the original.
"""


def run_test(name: str, fn, *args, **kwargs):
    """Run a test and print result."""
    try:
        result = fn(*args, **kwargs)
        print(f"  ✓ {name}")
        return result
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None


def test_preprocessor():
    print("\n── Preprocessor Tests ──────────────────────────────────────────")
    pp = TextPreprocessor()

    sentences = run_test("Sentence splitting", pp.split_sentences, SAMPLE_SHORT)
    if sentences:
        assert len(sentences) >= 3, f"Expected ≥3 sentences, got {len(sentences)}"
        print(f"    Sentences found: {len(sentences)}")

    tokens = run_test("Tokenization", pp.tokenize, "Machine learning enables computers to learn.")
    if tokens:
        assert len(tokens) > 0
        print(f"    Tokens: {tokens}")

    kws = run_test("Keyword extraction", pp.extract_keywords, "Neural networks process information efficiently.")
    if kws:
        assert "networks" in kws or "neural" in kws
        print(f"    Keywords: {kws}")

    result = run_test("Full preprocess", pp.preprocess, SAMPLE_MEDIUM)
    if result:
        assert len(result["sentences"]) > 0
        print(f"    Preprocessed {len(result['sentences'])} sentences")


def test_token_counter():
    print("\n── Token Counter Tests ─────────────────────────────────────────")
    tc = TokenCounter()
    print(f"    Backend: {tc.backend}")

    count = run_test("Token count", tc.count, "Hello world, this is a test sentence.")
    if count:
        assert count > 0
        print(f"    Token count: {count}")

    stats = run_test("Compression stats", tc.stats, SAMPLE_MEDIUM, SAMPLE_SHORT)
    if stats:
        print(f"    Original: {stats['original_tokens']} → Compressed: {stats['compressed_tokens']} "
              f"(-{stats['reduction_pct']}%)")


def test_scorer():
    print("\n── Scorer Tests ────────────────────────────────────────────────")
    pp = TextPreprocessor()
    preprocessed = pp.preprocess(SAMPLE_MEDIUM)
    sentences = preprocessed["sentences"]
    keywords_per = preprocessed["keywords_per_sentence"]

    tfidf = TFIDFScorer()
    run_test("TF-IDF fit", tfidf.fit, sentences)
    
    score = run_test("Word score", tfidf.word_score, "learning")
    if score is not None:
        print(f"    Score for 'learning': {score:.4f}")

    scorer = InformationDensityScorer(tfidf_scorer=tfidf)
    tc = TokenCounter()
    token_counts = tc.count_sentences(sentences)

    scores = run_test("Score all sentences", scorer.score_all, sentences, keywords_per, token_counts)
    if scores:
        print(f"    Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        best_idx = scores.index(max(scores))
        print(f"    Highest scoring sentence: {sentences[best_idx][:70]}...")


def test_algorithm():
    print("\n── Algorithm Tests ─────────────────────────────────────────────")
    compressor = GreedyCompressor(preserve_structure=True)

    # Test 1: Ratio-based compression
    result = run_test("Compress with ratio=0.5", compressor.compress, SAMPLE_LONG, None, 0.5)
    if result:
        stats = result.stats
        print(f"    {stats['original_tokens']} → {stats['compressed_tokens']} tokens "
              f"(-{stats['reduction_pct']}%)")
        assert stats["compressed_tokens"] < stats["original_tokens"], "Should reduce tokens"
        assert result.compressed_text, "Should have compressed text"

    # Test 2: Token budget
    result2 = run_test("Compress with max_tokens=100", compressor.compress, SAMPLE_LONG, 100)
    if result2:
        print(f"    Tokens used: {result2.stats['compressed_tokens']} (budget: 100)")

    # Test 3: Short text
    result3 = run_test("Compress short text", compressor.compress, SAMPLE_SHORT)
    if result3:
        print(f"    Short text: {result3.stats['original_tokens']} → {result3.stats['compressed_tokens']}")

    # Test 4: Structure preservation
    if result:
        sentences = result.all_sentences
        anchors = [s for s in sentences if s.is_anchor and s.selected]
        print(f"    Anchor sentences preserved: {len(anchors)}/2")


def test_similarity():
    print("\n── Similarity Tests ────────────────────────────────────────────")
    evaluator = SemanticSimilarityEvaluator()

    sim = run_test("TF-IDF similarity", evaluator.tfidf_similarity, SAMPLE_MEDIUM, SAMPLE_SHORT)
    if sim is not None:
        print(f"    TF-IDF similarity (medium vs short): {sim:.4f}")

    sim_identical = run_test("Identical text similarity", evaluator.tfidf_similarity, SAMPLE_SHORT, SAMPLE_SHORT)
    if sim_identical is not None:
        assert sim_identical > 0.95, f"Identical texts should have similarity > 0.95, got {sim_identical}"
        print(f"    Identical text similarity: {sim_identical:.4f} ✓")

    eval_result = run_test("Full evaluation", evaluator.evaluate, SAMPLE_MEDIUM, SAMPLE_SHORT)
    if eval_result:
        print(f"    Method: {eval_result['method']}")
        print(f"    Meaning preserved: {eval_result['meaning_preserved_pct']}%")


def test_full_pipeline():
    print("\n── Full Pipeline Tests ─────────────────────────────────────────")
    ts = TokenShrink()

    # Test 1: Default compression
    result = run_test("Full pipeline default", ts.compress, SAMPLE_LONG)
    if result:
        stats = result["stats"]
        print(f"    {stats['original_tokens']} → {stats['compressed_tokens']} tokens "
              f"(-{stats['reduction_pct']}%)")
        if "similarity" in result:
            print(f"    Semantic similarity: {result['similarity']['semantic_similarity']:.4f}")

    # Test 2: With debug
    result2 = run_test("Full pipeline with debug", ts.compress, SAMPLE_MEDIUM, 
                        None, 0.5, None, True)
    if result2 and "debug_log" in result2:
        print(f"    Debug log entries: {len(result2['debug_log'])}")

    # Test 3: Token budget
    result3 = run_test("Full pipeline max_tokens=150", ts.compress, SAMPLE_LONG,
                        150, None, None, False)
    if result3:
        stats = result3["stats"]
        print(f"    Budget 150: used {stats['compressed_tokens']} tokens")


def test_edge_cases():
    print("\n── Edge Case Tests ─────────────────────────────────────────────")
    ts = TokenShrink()

    # Empty input
    r = run_test("Empty input", ts.compress, "")
    if r:
        assert "error" in r or r["compressed_text"] == ""

    # Single sentence
    r2 = run_test("Single sentence", ts.compress, "Machine learning is powerful.")
    if r2:
        print(f"    Single sentence: '{r2['compressed_text']}'")

    # Very tight budget
    r3 = run_test("Tight budget (5 tokens)", ts.compress, SAMPLE_MEDIUM, 5)
    if r3:
        print(f"    Tight budget result: {r3['stats']['compressed_tokens']} tokens")

    # Zero redundancy (fresh keywords dominate)
    r4 = run_test("No redundancy penalty", 
                  TokenShrink(redundancy_penalty=0.0).compress, SAMPLE_LONG, None, 0.4)
    if r4:
        print(f"    No penalty: -{r4['stats']['reduction_pct']}%")


def print_detailed_example():
    """Print a detailed human-readable compression example."""
    print("\n" + "═" * 70)
    print("  DETAILED COMPRESSION EXAMPLE")
    print("═" * 70)
    
    ts = TokenShrink()
    result = ts.compress(SAMPLE_LONG, target_ratio=0.4, include_debug=True)
    
    stats = result["stats"]
    print(f"\n  Input:  {stats['original_tokens']} tokens, "
          f"{result['stats']['original_sentence_count']} sentences")
    print(f"  Output: {stats['compressed_tokens']} tokens, "
          f"{result['stats']['compressed_sentence_count']} sentences")
    print(f"  Saved:  {stats['tokens_saved']} tokens ({stats['reduction_pct']}% reduction)")
    
    if "similarity" in result:
        sim = result["similarity"]
        print(f"  Semantic similarity: {sim['semantic_similarity']:.2%} "
              f"(method: {sim['method']})")
    
    print(f"\n  Compressed text:\n  {'─' * 60}")
    # Wrap at 65 chars
    words = result["compressed_text"].split()
    line = "  "
    for word in words:
        if len(line) + len(word) > 67:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    
    print(f"\n  Selected sentence scores:")
    for s in sorted(result["selected_sentences"], key=lambda x: x["selection_order"]):
        anchor = " [ANCHOR]" if s["is_anchor"] else ""
        print(f"  #{s['selection_order']+1} (sent {s['index']}) score={s['score']:.4f} "
              f"tokens={s['token_count']}{anchor}: {s['text'][:55]}...")
    
    print(f"\n  First 5 debug entries:")
    for entry in result.get("debug_log", [])[:5]:
        print(f"  · {entry}")


if __name__ == "__main__":
    print("TokenShrink Test Suite")
    print("=" * 70)
    
    test_preprocessor()
    test_token_counter()
    test_scorer()
    test_algorithm()
    test_similarity()
    test_full_pipeline()
    test_edge_cases()
    print_detailed_example()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
