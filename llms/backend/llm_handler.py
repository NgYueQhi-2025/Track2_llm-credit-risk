import os
import json
import time
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


def call_llm(prompt: str, mode: str = "summary", temperature: float = 0.0, use_cache: bool = True, mock: bool = True) -> str:
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
    raise RuntimeError("No LLM provider available. Set OPENAI_API_KEY and install openai, or run in mock mode.")


if __name__ == '__main__':
    # quick local test
    prompt = "Name: Test; free_text: I run a small cafe and sometimes miss payments"
    print(call_llm(prompt, mode='summary', mock=True))