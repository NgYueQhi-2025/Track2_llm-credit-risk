import os
import json
import hashlib
from typing import Dict, Any, Optional

# cache file next to this script
CACHE_FILE = os.path.join(os.path.dirname(__file__), ".llm_cache.json")


def _hash_input(s: str) -> str:
    """Return a short sha256 hex digest for cache keys."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    """Load cache dict from disk. Returns empty dict on error."""
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Atomically write cache dict to disk."""
    tmp = CACHE_FILE + ".tmp"
    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_FILE)


def call_llm(prompt: str, mode: str = 'summary', temperature: float = 0.0, use_cache: bool = True, mock: bool = True) -> str:
    """Call the LLM (mock or real). Returns raw string output.

    Caching: keyed by SHA256(prompt + '::' + mode + '::' + str(temperature))
    """
    # Mock responses for testing — keep inside function
    if mock:
        if mode == 'summary':
            return json.dumps({"summary": "Applicant describes entrepreneurial experience; moderate financial detail.", "confidence": 0.82})
        if mode == 'extract_risky':
            return json.dumps({"risky_phrases": ["late payments", "multiple loans"], "count": 2})
        if mode == 'detect_contradictions':
            return json.dumps({"contradictions": ["claimed revenue inconsistent with bank statement"], "flag": 1})
        if mode == 'sentiment':
            return json.dumps({"sentiment": "neutral", "score": 0.05})
        return json.dumps({"raw": "mocked"})

    key = _hash_input(prompt + '::' + mode + '::' + str(temperature))
    cache = _load_cache()
    if use_cache and key in cache:
        return cache[key]

    # === PLACEHOLDER: real LLM call ===
    # Replace the next line with your actual LLM invocation and set `result`.
    result = json.dumps({"raw": "<real-llm-response-placeholder>"})

    if use_cache:
        cache[key] = result
        try:
            _save_cache(cache)
        except Exception:
            pass

    return result


if __name__ == '__main__':
    # quick local test to verify cache behavior
    prompt = "Test prompt for caching"
    print('First call (mock) ->', call_llm(prompt, mode='summary', use_cache=True, mock=True))
    print('Second call (mock) ->', call_llm(prompt, mode='summary', use_cache=True, mock=True))
