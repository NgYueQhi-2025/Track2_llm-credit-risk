from flask import Flask, request, jsonify
import json

app = Flask(__name__)


def make_summary(prompt_text: str):
    # Very small heuristic to craft a JSON summary from prompt for testing
    summary = {
        "summary": "Mock summary: applicant describes small business activity and mixed payments.",
        "confidence": 0.75
    }
    return json.dumps(summary)


def make_risky(prompt_text: str):
    return json.dumps({"risky_phrases": ["late payments", "high credit usage"], "count": 2})


@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "ok", "info": "Mock Ollama-compatible server"})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    # Emulate an OpenAI-compatible chat.completions response
    try:
        body = request.get_json(force=True)
    except Exception:
        body = {}

    # Build a content string that contains a JSON object (as the real model might)
    content = make_summary(json.dumps(body))
    resp = {
        "choices": [
            {"message": {"content": content}}
        ]
    }
    return jsonify(resp)


@app.route('/generate', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
@app.route('/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def generate():
    # Emulate an Ollama-like response shape: {"result":[{"content":"..."}]}
    try:
        body = request.get_json(force=True)
    except Exception:
        body = {}

    # Choose content based on presence of keywords
    prompt = ''
    if isinstance(body, dict):
        prompt = body.get('prompt') or json.dumps(body.get('messages') or body)

    if 'risky' in prompt.lower():
        content = make_risky(prompt)
    else:
        content = make_summary(prompt)

    resp = {"result": [{"content": content}]}
    return jsonify(resp)


if __name__ == '__main__':
    # Run on the default port expected by the app
    # Bind to all interfaces to avoid potential loopback/IPv6 issues in some environments
    app.run(host='0.0.0.0', port=11434)
