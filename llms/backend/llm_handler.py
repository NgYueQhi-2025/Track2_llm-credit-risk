# llms/backend/llm_handler.py

import os
import json
import time
import logging
import numpy as np
from typing import Any, Optional, List, Dict, Tuple

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

# Try to import gemini_adapter for RAG
_GEMINI_ADAPTER_AVAILABLE = False
try:
    from . import gemini_adapter
    _GEMINI_ADAPTER_AVAILABLE = True
except Exception as e:
    logging.warning(f"Could not import gemini_adapter: {e}")
    gemini_adapter = None


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


## 3. RAG HELPER FUNCTIONS

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_rag_index(index_dir: str) -> Optional[Tuple[List[Dict], np.ndarray]]:
    """
    Load RAG index artifacts from directory.
    
    Args:
        index_dir: Directory containing docs.jsonl and embeddings.npy
        
    Returns:
        Tuple of (documents, embeddings) or None if not found
    """
    docs_path = os.path.join(index_dir, 'docs.jsonl')
    embeddings_path = os.path.join(index_dir, 'embeddings.npy')
    
    if not os.path.exists(docs_path) or not os.path.exists(embeddings_path):
        logging.debug(f"RAG index not found in {index_dir}")
        return None
    
    try:
        # Load documents
        docs = []
        with open(docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        if len(docs) != len(embeddings):
            logging.warning(f"Mismatch: {len(docs)} docs but {len(embeddings)} embeddings")
            return None
        
        logging.info(f"Loaded RAG index: {len(docs)} documents from {index_dir}")
        return docs, embeddings
        
    except Exception as e:
        logging.error(f"Failed to load RAG index from {index_dir}: {e}")
        return None


def _retrieve_relevant_chunks(
    query_text: str,
    docs: List[Dict],
    doc_embeddings: np.ndarray,
    top_k: int = 3
) -> List[Dict]:
    """
    Retrieve top_k most relevant document chunks for query.
    
    Args:
        query_text: User query text
        docs: List of document dicts with 'text' field
        doc_embeddings: Numpy array of document embeddings
        top_k: Number of chunks to retrieve
        
    Returns:
        List of top_k document dicts with added 'score' field
    """
    if not _GEMINI_ADAPTER_AVAILABLE or gemini_adapter is None:
        logging.warning("gemini_adapter not available for retrieval")
        return []
    
    try:
        # Embed query
        query_embedding = gemini_adapter.embed_texts(query_text, provider='gemini')
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding[0]  # Get first embedding if batched
        
        # Compute similarities using vectorized operations for better performance
        # Normalize embeddings
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logging.warning("Query embedding has zero norm")
            return []
        
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        
        # Compute cosine similarities using vectorized dot product
        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build result with scores
        results = []
        for idx in top_indices:
            doc = docs[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        logging.debug(f"Retrieved {len(results)} chunks with scores: {[r['score'] for r in results]}")
        return results
        
    except Exception as e:
        logging.error(f"Failed to retrieve chunks: {e}")
        return []


def _build_rag_prompt(user_text: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build RAG prompt with retrieved context and negation-aware instructions.
    
    Args:
        user_text: Original user input text
        retrieved_chunks: List of retrieved document chunks
        
    Returns:
        Complete RAG prompt string
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        context_parts.append(f"[Context {i}, relevance={score:.3f}]: {text}")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    
    # Build negation-aware prompt with CRF schema
    prompt = f"""You are a credit risk assessment assistant. Analyze the applicant text using the provided context from credit risk research.

IMPORTANT INSTRUCTIONS:
1. Pay close attention to NEGATION words (no, never, not, without, etc.). If text says "no missed payments" that is POSITIVE.
2. Look for intensity modifiers (rarely, sometimes, often, always).
3. Consider temporal decay (recent issues are riskier than old ones).
4. Account for mitigating factors that reduce risk.

CONTEXT FROM RESEARCH:
{context}

APPLICANT TEXT TO ANALYZE:
{user_text}

Return a JSON object with this EXACT schema (CRF format):
{{
    "summary": "Brief assessment of credit risk (1-2 sentences)",
    "confidence": 0.0-1.0,
    "risky_phrases": ["phrase1", "phrase2", ...],
    "risk_score": 0.0-1.0,
    "risk_level": "low|medium|high",
    "mitigating_factors": ["factor1", "factor2", ...],
    "risk_factors": ["factor1", "factor2", ...],
    "negation_detected": true/false,
    "temporal_factors": ["recent event1", ...]
}}

Return ONLY the JSON object, no additional text."""
    
    return prompt


def call_llm_with_rag(
    text: str,
    index_dir: str = 'backend/llm_index',
    top_k: int = 3,
    use_cache: bool = True,
    mock: bool = True
) -> Dict[str, Any]:
    """
    Call LLM with RAG (Retrieval-Augmented Generation) using Gemini.
    
    This function:
    1. Loads RAG index artifacts (docs.jsonl, embeddings.npy)
    2. Computes embedding for input text
    3. Retrieves top_k relevant chunks via cosine similarity
    4. Builds RAG prompt with retrieved context
    5. Calls Gemini for generation
    6. Parses and returns structured JSON result
    
    Args:
        text: Input text to analyze
        index_dir: Directory containing RAG index artifacts
        top_k: Number of relevant chunks to retrieve
        use_cache: Whether to use caching (currently not implemented)
        mock: If True, use mock responses; if False, call real Gemini API
        
    Returns:
        Dict with parsed LLM response including summary, risky_phrases, etc.
        
    Example:
        >>> result = call_llm_with_rag("I sometimes miss credit card payments", mock=False)
        >>> print(result['risk_level'])
        'medium'
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    
    # Mock mode for testing
    if mock:
        logger.debug("Using mock mode for RAG call")
        return {
            "summary": "Mock RAG response: moderate risk applicant",
            "confidence": 0.75,
            "risky_phrases": ["mock phrase"],
            "risk_score": 0.5,
            "risk_level": "medium",
            "mitigating_factors": [],
            "risk_factors": ["mock risk"],
            "negation_detected": False,
            "temporal_factors": []
        }
    
    # Try to load RAG index
    rag_data = _load_rag_index(index_dir)
    
    if rag_data is None:
        # No index available, fall back to standard call_llm
        logger.info("No RAG index found, falling back to standard call_llm")
        try:
            raw_result = call_llm(text, mode='summary', mock=False, use_cache=use_cache)
            parsed = _safe_extract_json(raw_result)
            
            # Ensure we return a dict
            if isinstance(parsed, dict):
                return parsed
            return {"summary": str(parsed), "confidence": 0.5}
            
        except Exception as e:
            logger.error(f"Fallback call_llm failed: {e}")
            return {
                "summary": "Error: Could not process request",
                "confidence": 0.0,
                "error": str(e)
            }
    
    # Check if gemini_adapter is available
    if not _GEMINI_ADAPTER_AVAILABLE or gemini_adapter is None:
        logger.warning("gemini_adapter not available, falling back to call_llm")
        try:
            raw_result = call_llm(text, mode='summary', mock=False, use_cache=use_cache)
            parsed = _safe_extract_json(raw_result)
            if isinstance(parsed, dict):
                return parsed
            return {"summary": str(parsed), "confidence": 0.5}
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return {"summary": "Error", "confidence": 0.0, "error": str(e)}
    
    # RAG pipeline
    try:
        docs, doc_embeddings = rag_data
        
        # Retrieve relevant chunks
        logger.debug(f"Retrieving top {top_k} relevant chunks")
        retrieved = _retrieve_relevant_chunks(text, docs, doc_embeddings, top_k=top_k)
        
        if not retrieved:
            logger.warning("No chunks retrieved, using direct prompt")
            retrieved = []
        
        # Build RAG prompt
        rag_prompt = _build_rag_prompt(text, retrieved)
        logger.debug(f"Built RAG prompt with {len(retrieved)} chunks")
        
        # Call Gemini for generation
        logger.debug("Calling gemini_adapter.generate_from_prompt")
        raw_output = gemini_adapter.generate_from_prompt(
            prompt=rag_prompt,
            provider='gemini',
            temperature=0.0,
            max_output_tokens=1024
        )
        
        # Parse JSON result
        logger.debug(f"Raw output length: {len(raw_output)}")
        parsed = _safe_extract_json(raw_output)
        
        # Ensure we return a dict
        if isinstance(parsed, dict):
            logger.info(f"Successfully parsed RAG response: {list(parsed.keys())}")
            return parsed
        
        # If parsing failed, try to extract JSON substring
        logger.warning("Initial parse didn't return dict, attempting extraction")
        if isinstance(raw_output, str):
            # Try to find JSON in the output
            start = raw_output.find('{')
            end = raw_output.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw_output[start:end+1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    pass
        
        # Final fallback
        return {
            "summary": str(parsed),
            "confidence": 0.5,
            "raw_output": raw_output[:500]
        }
        
    except NotImplementedError as e:
        # Gemini adapter not fully implemented, fall back
        logger.info(f"RAG not fully available: {e}, falling back to call_llm")
        try:
            raw_result = call_llm(text, mode='summary', mock=False, use_cache=use_cache)
            parsed = _safe_extract_json(raw_result)
            if isinstance(parsed, dict):
                return parsed
            return {"summary": str(parsed), "confidence": 0.5}
        except Exception as e2:
            logger.error(f"Complete fallback failed: {e2}")
            return {"summary": "Error", "confidence": 0.0, "error": str(e2)}
            
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        # Try one more fallback to standard call
        try:
            raw_result = call_llm(text, mode='summary', mock=False, use_cache=use_cache)
            parsed = _safe_extract_json(raw_result)
            if isinstance(parsed, dict):
                return parsed
            return {"summary": str(parsed), "confidence": 0.5}
        except:
            return {
                "summary": "Error processing request",
                "confidence": 0.0,
                "error": str(e)
            }


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
