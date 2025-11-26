# llms/backend/llm_handler.py

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
    # Initialize client using env var if present. Different genai SDK
    # releases expose different initialization patterns. Try the
    # most compatible approaches with graceful fallbacks.
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            # Prefer passing api_key directly to Client if supported.
            try:
                GEMINI_CLIENT = genai.Client(api_key=gemini_api_key)
            except TypeError:
                # If Client doesn't accept api_key, set a fallback env var
                # that the SDK may read and then instantiate the client.
                os.environ.setdefault("GOOGLE_API_KEY", gemini_api_key)
                GEMINI_CLIENT = genai.Client()
        else:
            # No explicit key provided; try default client creation.
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


## 1. CORE LLM CALL FUNCTION (Must be defined before process_text_word_by_word)
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

    # The rest of the providers (ollama, huggingface, openai)
    if provider in ("ollama", "ollema", "local", "ollama_http"):
        raise RuntimeError("Ollama/local provider selected but not available in this environment. Please use 'gemini'.")
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
#
# ---

## 2. CHUNKING HELPER FUNCTION (Now placed last)
def process_text_word_by_word(
    text: str,
    mode: str = "extract_risky",
    chunk_size_words: int = 1,
    temperature: float = 0.0,
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """Process `text` in word-based (or chunk-based) units.

    Returns a list of dicts:
      [{'chunk_text': 'word_or_chunk', 'raw': raw_model_text, 'parsed': parsed_json_or_text, 'error': None, 'word_start': 0, 'word_end': 0}, ...]

    chunk_size_words: number of words per LLM call. 1 = per-word. Use larger values for speed and lower cost.
    """
    results: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip():
        return results

    # split into chunks of words
    words = text.split()
    
    # For each chunk, call call_llm with a focused prompt
    for i in range(0, len(words), chunk_size_words):
        chunk_words = words[i:i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        
        start_index = i
        end_index = i + len(chunk_words) - 1

        # Construct a minimal prompt for the chunk depending on mode
        if mode == "extract_risky":
            prompt = f"Inspect this text fragment and return JSON: {{'risky_phrases': ['...'], 'count': 0}}. Fragment: \"{chunk_text}\". Return only JSON."
        elif mode == "sentiment":
            prompt = f"Return sentiment as JSON like: {{'sentiment': 'neutral','score': 0.0}} for the fragment: \"{chunk_text}\". Return only JSON."
        elif mode == "summary":
            prompt = f"Short summary JSON: {{'summary':'...','confidence':0.0}} of this fragment: \"{chunk_text}\". Return only JSON."
        else:
            prompt = f"Analyze and return JSON for fragment: \"{chunk_text}\". Return only JSON."

        try:
            # 1. Call the existing core LLM function
            raw_output = call_llm(
                prompt=prompt, # Pass the focused prompt, not just chunk_text
                mode=mode,
                temperature=temperature,
                use_cache=True,
                mock=mock
            )
            
            # 2. Safely parse the raw JSON output
            parsed_output = _safe_extract_json(raw_output)

            results.append({
                "chunk_text": chunk_text,
                "word_start": start_index,
                "word_end": end_index,
                "raw": raw_output,
                "parsed": parsed_output,
                "error": None
            })
            
        except Exception as e:
            # Handle API errors or parsing failures for this chunk gracefully
            results.append({
                "chunk_text": chunk_text,
                "word_start": start_index,
                "word_end": end_index,
                "raw": None,
                "parsed": None,
                "error": str(e)
            })
            # Log the error, but continue processing subsequent chunks
            logging.error(f"Error processing chunk {i}: {e}")

    return results

if __name__ == '__main__':
    prompt = "Name: Test; free_text: I run a small cafe and sometimes miss payments"
    # Example quick test. Ensure GEMINI_API_KEY and LLM_PROVIDER are configured if not using mock.
    print(call_llm(prompt, mode='summary', mock=True))
    
    # Test the chunking function using mock data
    long_text = "The applicant has a strong history of high revenue but noted several late credit card payments in the past year."
    print("\nChunking test (mock=True):")
    chunk_results = process_text_word_by_word(long_text, mode='extract_risky', chunk_size_words=5, mock=True)
    for result in chunk_results:
        print(f"Chunk: '{result.get('chunk_text')}' -> Parsed: {result.get('parsed')}")
