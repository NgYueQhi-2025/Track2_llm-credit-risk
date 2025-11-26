import os
import json
import time
import logging
from typing import Any, Optional

# --- LLM Provider Imports ---
_OPENAI_AVAILABLE = False
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None

_GEMINI_AVAILABLE = False
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
    _GEMINI_AVAILABLE = True
    # Initialize client early. The SDK automatically reads the GEMINI_API_KEY env var.
    try:
        genai.configure()
        GEMINI_CLIENT = genai.Client()
    except Exception as e:
        # This will catch missing API key or other configuration errors
        logging.warning(f"Could not initialize Gemini Client: {e}")
        GEMINI_CLIENT = None
except Exception:
    GEMINI_CLIENT = None


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


def call_llm(
    prompt: str,
    mode: str = "summary",
    temperature: float = 0.0,
    use_cache: bool = True,
    mock: bool = False
) -> str:
    """Call an LLM to perform a specific extraction `mode`.

    If `mock` is True, returns canned responses useful for offline testing.
    The function returns the raw string output (usually JSON).
    """

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

    # Default provider: 'ollama' is the default fallback, which caused your error.
    # Set LLM_PROVIDER="gemini" in your secrets to force Gemini.
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    # --- GEMINI / GOOGLE AI PROVIDER ---
    if provider in ("gemini", "google", "google_ai") and GEMINI_CLIENT:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        system_instruction = (
            "You are a credit-risk feature extractor. Return only a single JSON object with the exact schema requested. "
            "Do not add any commentary or surrounding text. Your response must be valid, complete JSON."
        )

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

        # Retry/backoff
        last_exc = None
        for attempt in range(3):
            try:
                # Use the client initialized at the module level
                response = GEMINI_CLIENT.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(user_prompt)]),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=system_instruction
                    )
                )

                out = response.text
                parsed = _safe_extract_json(out)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed)
                return out

            except APIError as e:
                last_exc = e
                logger.error(f"Gemini API call failed (attempt {attempt+1}): {e}")
                time.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                last_exc = e
                logger.exception(f"Gemini call failed (attempt {attempt+1}): {e}")
                time.sleep(0.5 * (2 ** attempt))
        
        if last_exc:
            raise RuntimeError(f"Gemini LLM provider call failed after 3 attempts: {last_exc}")

    # --- OLLAMA / LOCAL HTTP PROVIDER ---
    if provider in ("ollama", "ollema", "local", "ollama_http"):
        # This block is what previously failed due to "Connection refused"
        # on Streamlit Cloud, as it relies on a local server (http://localhost:11434).
        try:
            import requests
            _REQUESTS_AVAILABLE = True
        except Exception:
            _REQUESTS_AVAILABLE = False

        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("LLM provider 'ollama' selected but the 'requests' package is not installed.")

        base = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", os.getenv("OPENAI_MODEL", "llama2"))
        
        # Build simple prompts per mode (omitted for brevity, assume original logic is here)
        # ... (Original prompt construction for Ollama/Local) ...

        # Retry logic (omitted for brevity, assume original logic is here)
        # ... (Original Ollama/Local retry and request logic) ...
        try:
            # Placeholder to prevent giant code block. If you enable this, 
            # the original complex request/parse logic must be here.
            raise RuntimeError("Ollama/local provider is selected and environment requires local server access.")
        except Exception as e:
             raise RuntimeError(f"Ollama/local LLM call failed: {e}")

    # --- HUGGING FACE PROVIDER ---
    if provider in ("huggingface", "hf", "hugging_face"):
        # ... (Original Hugging Face logic) ...
        try:
            # Placeholder to prevent giant code block. If you enable this, 
            # the original complex request/parse logic must be here.
            raise RuntimeError("HuggingFace provider is selected and requires key/URL configuration.")
        except Exception as e:
             raise RuntimeError(f"HuggingFace/local LLM call failed: {e}")

    # --- OPENAI PROVIDER ---
    api_key = os.getenv("OPENAI_API_KEY")
    if provider in ("openai", "gpt") and _OPENAI_AVAILABLE and api_key:
        # ... (Original OpenAI logic) ...
        # Placeholder to prevent giant code block.
        try:
            # The original OpenAI logic is complex and must be here.
            raise RuntimeError("OpenAI provider is selected and requires key/client setup.")
        except Exception as e:
             raise RuntimeError(f"OpenAI LLM provider call failed: {e}")

    # If no provider available, raise with guidance
    raise RuntimeError(
        "No LLM provider available. Set LLM_PROVIDER=\"gemini\" and "
        "GEMINI_API_KEY in your secrets. Alternatively, set OPENAI_API_KEY "
        "or run in mock mode."
    )


if __name__ == '__main__':
    # quick local test
    # NOTE: Set the environment variable LLM_PROVIDER="gemini" before running this locally
    prompt = "Name: Test; free_text: I run a small cafe and sometimes miss payments"
    # Added mock=False to test actual connection in main block
    print(call_llm(prompt, mode='summary', mock=False))
