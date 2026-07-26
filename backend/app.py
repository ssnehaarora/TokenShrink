"""
TokenShrink - Flask REST API Server
Exposes compression endpoints for the MERN frontend.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.tokenshrink import TokenShrink

app = Flask(__name__)
CORS(app)

# Initialize TokenShrink once at startup
ts = TokenShrink(preserve_structure=True, redundancy_penalty=0.5)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    from src.token_counter import TokenCounter
    tc = TokenCounter()
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "token_backend": tc.backend,
    })


@app.route("/api/compress", methods=["POST"])
def compress():
    """
    Main compression endpoint.
    
    Request body (JSON):
        text: str          - Input prompt (required)
        max_tokens: int    - Token budget (optional)
        target_ratio: float - Compression ratio e.g. 0.5 (optional)
        max_sentences: int  - Max sentences to keep (optional)
        include_debug: bool - Include debug log (optional, default false)
    
    Response (JSON):
        compressed_text, stats, selected_sentences, similarity, [debug_log]
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400

    max_tokens = data.get("max_tokens")
    target_ratio = data.get("target_ratio")
    max_sentences = data.get("max_sentences")
    include_debug = data.get("include_debug", False)

    # Validate numeric params
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens < 1:
                return jsonify({"error": "max_tokens must be >= 1"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "max_tokens must be an integer"}), 400

    if target_ratio is not None:
        try:
            target_ratio = float(target_ratio)
            if not (0.01 <= target_ratio <= 1.0):
                return jsonify({"error": "target_ratio must be between 0.01 and 1.0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "target_ratio must be a float"}), 400

    if max_sentences is not None:
        try:
            max_sentences = int(max_sentences)
            if max_sentences < 1:
                return jsonify({"error": "max_sentences must be >= 1"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "max_sentences must be an integer"}), 400

    try:
        result = ts.compress(
            text=text,
            max_tokens=max_tokens,
            target_ratio=target_ratio,
            max_sentences=max_sentences,
            include_debug=include_debug,
            include_similarity=True,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Compression failed: {str(e)}"}), 500


@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    """
    Count tokens in a text.
    
    Request body (JSON):
        text: str
    
    Response:
        token_count, word_count, char_count, backend
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "Field 'text' is required"}), 400

    from src.token_counter import TokenCounter
    tc = TokenCounter()
    text = data["text"]
    token_count = tc.count(text)

    return jsonify({
        "token_count": token_count,
        "word_count": len(text.split()),
        "char_count": len(text),
        "sentence_count": len([s for s in text.split(".") if s.strip()]),
        "backend": tc.backend,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"TokenShrink API running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
