# (Updated to include safe Gemini init and a per-word/per-chunk helper)
import os
import json
import time
import logging
from typing import Any, Optional, List, Dict

# --- LLM Provider Imports ---
_OPENAI_AVAILABLE = False
try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    openai = None

_GEMINI_AVAILABLE = False
GEMINI_CLIENT = None
try:
    from google import genai
    from google.genai.errors import APIError
    from google.genai import types
    _GEMINI_AVAILABLE = True
    # Initialize client using env var if present.
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            # the SDK supports configure(api_key=...) — this helps when running outside Cloud
            genai.configure(api_key=gemini_api_key)
        else:
            # genai.configure() will also attempt to read env var if available
            genai.configure()
        GEMINI_CLIENT = genai.Client()
    except Exception as e:
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

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    # real API key (if needed elsewhere)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

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
                f"Analyze the applicant data and return JSON like: {json.dumps({'summary': '...', 'confidence': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "extract_risky":
            user_prompt = (
                f"Extract risky phrases. Return JSON like: {json.dumps({'risky_phrases': ['...'], 'count': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "detect_contradictions":
            user_prompt = (
                f"Detect contradictions. Return JSON like: {json.dumps({'contradictions': ['...'], 'flag': 0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        elif mode == "sentiment":
            user_prompt = (
                f"Return sentiment. JSON like: {json.dumps({'sentiment': 'neutral', 'score': 0.0})}\n\nInput:\n{prompt}\n\nReturn only JSON."
            )
        else:
            user_prompt = f"Analyze input and return JSON. Input:\n{prompt}\n\nReturn only JSON."

        last_exc = None
        for attempt in range(3):
            try:
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

    # The rest of the providers (ollama, huggingface, openai) remain unchanged;
    # for brevity, raise if they are selected but not implemented here.
    if provider in ("ollama", "ollema", "local", "ollama_http"):
        raise RuntimeError("Ollama/local provider selected but not available in this environment.")
    if provider in ("huggingface", "hf", "hugging_face"):
        raise RuntimeError("HuggingFace provider selected but not implemented here.")
    api_key = os.getenv("OPENAI_API_KEY")
    if provider in ("openai", "gpt") and _OPENAI_AVAILABLE and api_key:
        raise RuntimeError("OpenAI provider selected but call logic not provided in this helper.")

    raise RuntimeError(
        "No LLM provider available. Set LLM_PROVIDER=\"gemini\" and "
        "GEMINI_API_KEY in your secrets. Alternatively, set OPENAI_API_KEY "
        "or run in mock mode."
    )


def process_text_word_by_word(
    text: str,
    mode: str = "extract_risky",
    chunk_size_words: int = 1,
    temperature: float = 0.0,
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """Process `text` in word-based (or chunk-based) units.

    Returns a list of dicts:
      [{'chunk': 'word_or_chunk', 'raw': raw_model_text, 'parsed': parsed_json_or_text, 'error': None}, ...]

    chunk_size_words: number of words per LLM call. 1 = per-word. Use larger values for speed and lower cost.
    """
    results: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip():
        return results

    if mock:
        # Return simple mocked responses for each chunk
        words = text.split()
        for i in range(0, len(words), chunk_size_words):
            chunk = " ".join(words[i:i + chunk_size_words])
            # a small canned structure to mimic the real extractor
            if mode == "extract_risky":
                parsed = {"risky_phrases": [], "count": 0}
            elif mode == "sentiment":
                parsed = {"sentiment": "neutral", "score": 0.0}
            else:
                parsed = {"raw": chunk}
            results.append({"chunk": chunk, "raw": json.dumps(parsed), "parsed": parsed, "error": None})
        return results

    # split into chunks of words
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size_words]) for i in range(0, len(words), chunk_size_words)]

    # For each chunk, call Gemini with a small focused prompt
    for chunk in chunks:
        # Construct a minimal prompt for the chunk depending on mode
        if mode == "extract_risky":
            prompt = f"Inspect this text fragment and return JSON: {{'risky_phrases': ['...'], 'count': 0}}. Fragment: \"{chunk}\". Return only JSON."
        elif mode == "sentiment":
            prompt = f"Return sentiment as JSON like: {{'sentiment': 'neutral','score': 0.0}} for the fragment: \"{chunk}\". Return only JSON."
        elif mode == "summary":
            prompt = f"Short summary JSON: {{'summary':'...','confidence':0.0}} of this fragment: \"{chunk}\". Return only JSON."
        else:
            prompt = f"Analyze and return JSON for fragment: \"{chunk}\". Return only JSON."

        try:
            parsed = None
            raw = None
            # Reuse call_llm flow for gemini — but we want to avoid duplication and keep retries consistent.
            # Directly use GEMINI_CLIENT here for lower overhead and more control:
            if GEMINI_CLIENT:
                try:
                    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                    resp = GEMINI_CLIENT.models.generate_content(
                        model=model,
                        contents=[
                            types.Content(role="user", parts=[types.Part.from_text(prompt)]),
                        ],
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            system_instruction="You are a lightweight extractor. Return only JSON."
                        )
                    )
                    raw = resp.text
                    parsed = _safe_extract_json(raw)
                except Exception as e:
                    # fallback to call_llm (which also does retries)
                    logging.exception("Direct Gemini call failed for chunk, falling back to call_llm.")
                    raw = call_llm(prompt, mode=mode, temperature=temperature, mock=False)
                    parsed = _safe_extract_json(raw)
            else:
                raise RuntimeError("Gemini client not initialized; ensure GEMINI_API_KEY is set and genai is available.")
            results.append({"chunk": chunk, "raw": raw, "parsed": parsed, "error": None})
        except Exception as e:
            logging.exception("Error calling LLM on chunk")
            results.append({"chunk": chunk, "raw": None, "parsed": None, "error": str(e)})

    return results


if __name__ == '__main__':
    prompt = "Name: Test; free_text: I run a small cafe and sometimes miss payments"
    # Example quick test. Ensure GEMINI_API_KEY and LLM_PROVIDER are configured if not using mock.
    print(call_llm(prompt, mode='summary', mock=True))
