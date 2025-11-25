import os
import json
import time
import logging
from typing import Any

_OPENAI_AVAILABLE = False
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None


def _safe_extract_json(text: str) -> Any:
    """Try to extract a JSON object from model output robustly.

    Returns parsed JSON on success or the original text on failure.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    # Find first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # fallback direct load
    try:
        return json.loads(s)
    except Exception:
        return s

# CORRECTION 1: Changed default mock=True to mock=False
def call_llm(prompt: str, mode: str = "summary", temperature: float = 0.0, use_cache: bool = True, mock: bool = False) -> str:
    """Call an LLM to perform a specific extraction `mode`.

    If `mock` is True, returns canned responses useful for offline testing.
    If an OpenAI API key is present and the openai package is installed, calls OpenAI ChatCompletion.
    The function returns the raw string output (usually JSON)."""
    
    # Mock responses for development and offline demos
    if mock:
        if mode == "summary":
            return json.dumps({"summary": "Applicant describes entrepreneurial experience; moderate financial detail.", "confidence": 0.82})
        if mode == "extract_risky":
            return json.dumps({"risky_phrases": ["late payments", "multiple loans"], "count": 2})
        if mode == "detect_contradictions":
            return json.dumps({"contradictions": ["claimed revenue inconsistent with bank statement"], "flag": 1})
        if mode == "sentiment":
            return json.dumps({"sentiment": "neutral", "score": 0.05})
        return json.dumps({"raw": "mocked"})

    # configure a module-level logger for debug tracing
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    # Default provider: prefer local Ollama-like gateways for developer flow,
    # but allow explicit override via LLM_PROVIDER (e.g. 'openai' or 'huggingface').
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    # If provider explicitly set to 'ollama' (local server), attempt HTTP calls
    if provider in ("ollama", "ollema", "local", "ollama_http"):
        # Try to use requests to call a local Ollama-like HTTP server.
        try:
            import requests
            _REQUESTS_AVAILABLE = True
        except Exception:
            _REQUESTS_AVAILABLE = False

        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("LLM provider 'ollama' selected but the 'requests' package is not installed.")

        base = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", os.getenv("OPENAI_MODEL", "llama2"))
        # Build simple prompts per mode
        if mode == "summary":
            user_prompt = (
                f"Analyze the applicant data and return JSON like: {json.dumps({'summary': '...', 'confidence': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "extract_risky":
            user_prompt = (
                f"Extract risky phrases. Return JSON like: {json.dumps({'risky_phrases': ['...'], 'count': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "detect_contradictions":
            user_prompt = (
                f"Detect contradictions. Return JSON like: {json.dumps({'contradictions': ['...'], 'flag': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "sentiment":
            user_prompt = (
                f"Return sentiment. JSON like: {json.dumps({'sentiment': 'neutral', 'score': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        else:
            user_prompt = f"Analyze input and return JSON. Input:\n{prompt}\n\nReturn only JSON."

        # Try a few common Ollama/OpenAI-compatible endpoints
        endpoints = ["/generate", "/api/generate", "/chat", "/api/chat", "/v1/chat/completions"]
        headers = {"Content-Type": "application/json"}
        last_exc = None
        for ep in endpoints:
            url = base.rstrip("/") + ep
            payloads = []
            # Different servers expect different payload shapes; try a few
            # Ollama-like: {"model": "<model>", "prompt": "..."}
            payloads.append({"model": model, "prompt": user_prompt})
            # OpenAI-compatible chat: {"model":"<model>", "messages":[{"role":"user","content":"..."}]}
            payloads.append({"model": model, "messages": [{"role": "user", "content": user_prompt}], "temperature": temperature})

            for payload in payloads:
                    # Debug: log the outgoing attempt
                    try:
                        logger.debug("LLM HTTP POST %s payload=%s", url, json.dumps(payload, ensure_ascii=False))
                    except Exception:
                        logger.debug("LLM HTTP POST %s (payload not JSON-serializable)", url)

                    # Use a slightly increased timeout to accommodate local dev boxes
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)

                    # Debug: log response status and truncated body
                    try:
                        body_preview = resp.text[:2000]
                        logger.debug("LLM HTTP RESPONSE %s status=%s body=%s", url, resp.status_code, body_preview)
                    except Exception:
                        logger.debug("LLM HTTP RESPONSE %s status=%s (body unavailable)", url, resp.status_code)
                    if resp.status_code != 200:
                        last_exc = RuntimeError(f"{url} returned {resp.status_code}: {resp.text}")
                        continue

                    # Try to parse common response shapes
                    text = None
                    try:
                        j = resp.json()
                        # Ollama sometimes returns {"result": [{"content":"..."}], ...}
                        if isinstance(j, dict) and "result" in j:
                            if isinstance(j["result"], list) and len(j["result"])>0:
                                # join content fields
                                parts = []
                                for item in j["result"]:
                                    if isinstance(item, dict) and "content" in item:
                                        parts.append(item["content"])
                                text = "\n".join(parts)
                        # OpenAI-like: {'choices':[{'message':{'content':'...'}}]}
                        if text is None and isinstance(j, dict) and "choices" in j:
                            try:
                                text = j["choices"][0]["message"]["content"]
                            except Exception:
                                try:
                                    text = j["choices"][0]["text"]
                                except Exception:
                                    text = None
                        # direct text field
                        if text is None and isinstance(j, dict) and "text" in j:
                            text = j["text"]
                        # last resort: raw text
                        if text is None:
                            text = resp.text

                        parsed = _safe_extract_json(text)
                        if isinstance(parsed, (dict, list)):
                            return json.dumps(parsed)
                        return text
                    except ValueError:
                        # not json, try raw text
                        text = resp.text
                        parsed = _safe_extract_json(text)
                        if isinstance(parsed, (dict, list)):
                            return json.dumps(parsed)
                        return text
                except Exception as e:
                    logger.exception("LLM HTTP attempt to %s failed: %s", url, e)
                    last_exc = e
                    continue
        raise RuntimeError(f"Ollama/local LLM call failed: {last_exc}")

    # If provider is Hugging Face (local TGI or HF Inference), attempt HTTP calls
    if provider in ("huggingface", "hf", "hugging_face"):
        try:
            import requests
            _REQUESTS_AVAILABLE = True
        except Exception:
            _REQUESTS_AVAILABLE = False

        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("LLM provider 'huggingface' selected but the 'requests' package is not installed.")

        base = os.getenv("LOCAL_LLM_URL", "http://localhost:8080")
        model = os.getenv("HF_MODEL", os.getenv("OPENAI_MODEL", "gpt2"))
        hf_key = os.getenv("HF_API_KEY")

        # Build simple prompts per mode (reuse same messages as other providers)
        if mode == "summary":
            user_prompt = (
                f"Analyze the applicant data and return JSON like: {json.dumps({'summary': '...', 'confidence': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "extract_risky":
            user_prompt = (
                f"Extract risky phrases. Return JSON like: {json.dumps({'risky_phrases': ['...'], 'count': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "detect_contradictions":
            user_prompt = (
                f"Detect contradictions. Return JSON like: {json.dumps({'contradictions': ['...'], 'flag': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        elif mode == "sentiment":
            user_prompt = (
                f"Return sentiment. JSON like: {json.dumps({'sentiment': 'neutral', 'score': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON.")
        else:
            user_prompt = f"Analyze input and return JSON. Input:\n{prompt}\n\nReturn only JSON."

        # Candidate endpoints for various HF-compatible servers (TGI, HF Inference, custom)
        endpoints = [
            # Router-compatible endpoints (Hugging Face now prefers router.huggingface.co)
            f"/api/models/{model}/infer",
            f"/api/models/{model}/generate",
            f"/api/models/{model}/predict",
            # Backwards-compatible HF shapes
            f"/v1/models/{model}/generate",
            f"/v1/models/{model}:predict",
            f"/v1/models/{model}/infer",
            f"/v1/models/{model}/predict",
            "/v1/generate",
            "/generate",
            "/v1/predict",
            "/predict",
            f"/models/{model}",
        ]

        headers = {"Content-Type": "application/json"}
        if hf_key:
            headers["Authorization"] = f"Bearer {hf_key}"

        last_exc = None
        for ep in endpoints:
            url = base.rstrip("/") + ep
            # Try a few payload shapes: HF inference-like and TGI-like
            payloads = []
            payloads.append({"inputs": user_prompt, "parameters": {"max_new_tokens": 256, "temperature": float(temperature)}})
            payloads.append({"inputs": user_prompt})
            payloads.append({"model": model, "inputs": user_prompt, "parameters": {"max_new_tokens": 256}})

            for payload in payloads:
                try:
                    try:
                        logger.debug("HF HTTP POST %s payload=%s", url, json.dumps(payload, ensure_ascii=False))
                    except Exception:
                        logger.debug("HF HTTP POST %s (payload not JSON-serializable)", url)

                    resp = requests.post(url, headers=headers, json=payload, timeout=15)

                    try:
                        body_preview = resp.text[:2000]
                        logger.debug("HF HTTP RESPONSE %s status=%s body=%s", url, resp.status_code, body_preview)
                    except Exception:
                        logger.debug("HF HTTP RESPONSE %s status=%s (body unavailable)", url, resp.status_code)

                    if resp.status_code != 200:
                        last_exc = RuntimeError(f"{url} returned {resp.status_code}: {resp.text}")
                        continue

                    # Try to interpret common HF response shapes
                    text = None
                    try:
                        j = resp.json()
                        # HF Inference API often returns {'generated_text': '...'} or a list of dicts
                        if isinstance(j, dict):
                            if 'generated_text' in j:
                                text = j['generated_text']
                            elif 'results' in j and isinstance(j['results'], list) and len(j['results'])>0:
                                # e.g., TGI returns {'results':[{'generated_text': '...'}]}
                                parts = []
                                for it in j['results']:
                                    if isinstance(it, dict) and 'generated_text' in it:
                                        parts.append(it['generated_text'])
                                text = '\n'.join(parts) if parts else None
                            elif 'outputs' in j:
                                # some servers return outputs array
                                try:
                                    if isinstance(j['outputs'], list):
                                        text = '\n'.join([o.get('generated_text') or str(o) for o in j['outputs']])
                                except Exception:
                                    text = None
                            elif 'result' in j:
                                # fallback to earlier pattern
                                if isinstance(j['result'], list) and len(j['result'])>0:
                                    parts = []
                                    for item in j['result']:
                                        if isinstance(item, dict) and 'content' in item:
                                            parts.append(item['content'])
                                    text = '\n'.join(parts)
                            elif 'error' in j:
                                last_exc = RuntimeError(f"HF server error: {j.get('error')}")
                                continue
                        elif isinstance(j, list):
                            # list of generations
                            parts = []
                            for it in j:
                                if isinstance(it, dict) and 'generated_text' in it:
                                    parts.append(it['generated_text'])
                            text = '\n'.join(parts) if parts else None

                        if text is None:
                            text = resp.text

                        parsed = _safe_extract_json(text)
                        if isinstance(parsed, (dict, list)):
                            return json.dumps(parsed)
                        return text
                    except ValueError:
                        text = resp.text
                        parsed = _safe_extract_json(text)
                        if isinstance(parsed, (dict, list)):
                            return json.dumps(parsed)
                        return text
                except Exception as e:
                    logger.exception("HF HTTP attempt to %s failed: %s", url, e)
                    last_exc = e
                    continue
        raise RuntimeError(f"HuggingFace/local LLM call failed: {last_exc}")

    # If OpenAI is available and API key set, call it
    api_key = os.getenv("OPENAI_API_KEY")
    if _OPENAI_AVAILABLE and api_key:
        openai.api_key = api_key
        system = (
            "You are a credit-risk feature extractor. Return only a single JSON object with the exact schema requested. "
            "Do not add any commentary or surrounding text."
        )
        if mode == "summary":
            user = (
                f"Analyze the applicant data and return JSON like: {json.dumps({'summary': '...', 'confidence': 0.0})}\n\n" +
                f"Input:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "extract_risky":
            user = (
                f"Extract risky phrases. Return JSON like: {json.dumps({'risky_phrases': ['...'], 'count': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "detect_contradictions":
            user = (
                f"Detect contradictions. Return JSON like: {json.dumps({'contradictions': ['...'], 'flag': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "sentiment":
            user = (
                f"Return sentiment. JSON like: {json.dumps({'sentiment': 'neutral', 'score': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        else:
            user = f"Analyze input and return JSON. Input:\n{prompt}\n\nReturn only JSON." 

        # Retry/backoff
        last_exc = None
        for attempt in range(3):
            try:
                # Support both pre-1.0 `openai.ChatCompletion.create` and the
                # new client-based API introduced in openai>=1.0.0.
                OpenAIClass = getattr(openai, "OpenAI", None)
                if OpenAIClass is not None:
                    # new client-based interface
                    client = OpenAIClass()
                    resp = client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=temperature,
                    )
                else:
                    # older interface
                    resp = openai.ChatCompletion.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=temperature,
                    )

                # Try a couple of response shapes (object with attributes or dict-like)
                out = None
                try:
                    out = resp.choices[0].message.content
                except Exception:
                    try:
                        out = resp["choices"][0]["message"]["content"]
                    except Exception:
                        out = str(resp)

                parsed = _safe_extract_json(out)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed)
                return out
            except Exception as e:
                last_exc = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"LLM provider call failed: {last_exc}")

    # If no provider available, raise with guidance
    raise RuntimeError("No LLM provider available. Set OPENAI_API_KEY and install openai, or set LLM_PROVIDER='ollama' and run a local Ollama-compatible server, or run in mock mode.")


if __name__ == '__main__':
    # quick local test
    prompt = "Name: Test; free_text: I run a small cafe and sometimes miss payments"
    # Added mock=False to test actual connection in main block
    print(call_llm(prompt, mode='summary', mock=False))
