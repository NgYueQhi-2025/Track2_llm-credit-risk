"""
Gemini Adapter for LLM Credit Risk Application

This module provides integration with Google's Generative AI (Gemini) for:
1. Text embeddings generation
2. LLM text generation

Configuration:
  Set the following environment variables:
  - GOOGLE_API_KEY or GEMINI_API_KEY: Your Google API key (required)
  - GEMINI_MODEL: Model name for generation (default: 'gemini-2.0-flash-exp')
  - GEMINI_EMBED_MODEL: Model name for embeddings (default: 'models/text-embedding-004')
  - GOOGLE_PROJECT: Optional Google Cloud project ID
  - GOOGLE_REGION: Optional Google Cloud region

Fallback:
  If the Google SDK is not available or API keys are missing, the module will
  fall back to sentence-transformers for local embedding generation.

Example usage:
  ```python
  from llms.backend import gemini_adapter
  
  # Generate embeddings
  embeddings = gemini_adapter.embed_texts(['text1', 'text2'], provider='gemini')
  
  # Generate text
  response = gemini_adapter.generate_from_prompt(
      prompt="What is credit risk?",
      provider='gemini',
      temperature=0.0,
      max_output_tokens=1024
  )
  ```
"""

import os
import logging
import numpy as np
from typing import List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Try to import Google Generative AI SDK
_GOOGLE_GENAI_AVAILABLE = False
_GENAI_CLIENT = None
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    _GOOGLE_GENAI_AVAILABLE = True
    
    # Initialize client with API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            _GENAI_CLIENT = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")
        except TypeError:
            # Fallback if Client doesn't accept api_key directly
            os.environ.setdefault("GOOGLE_API_KEY", api_key)
            _GENAI_CLIENT = genai.Client()
            logger.info("Gemini client initialized with environment variable")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            _GENAI_CLIENT = None
    else:
        logger.info("No GOOGLE_API_KEY or GEMINI_API_KEY found, Gemini features disabled")
        
except ImportError as e:
    logger.warning(f"google-genai not available: {e}")
    _GOOGLE_GENAI_AVAILABLE = False

# Try to import sentence-transformers as fallback
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_LOCAL_EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("sentence-transformers available as fallback embedder")
except ImportError:
    logger.warning("sentence-transformers not available, local embedding fallback disabled")


def _get_local_embedder():
    """Lazy load the local embedding model."""
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is None and _SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Allow configuration via environment variable
            model_name = os.getenv('LOCAL_EMBED_MODEL', 'all-MiniLM-L6-v2')
            _LOCAL_EMBEDDER = SentenceTransformer(model_name)
            logger.info(f"Loaded local embedder: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load local embedder: {e}")
            _LOCAL_EMBEDDER = None
    return _LOCAL_EMBEDDER


def embed_texts(
    texts: Union[str, List[str]], 
    provider: str = 'gemini',
    model: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for one or more texts.
    
    Args:
        texts: Single text string or list of text strings to embed
        provider: Embedding provider ('gemini' or 'local')
        model: Optional model name override (default uses GEMINI_EMBED_MODEL env var)
    
    Returns:
        numpy array of embeddings with shape (n_texts, embedding_dim)
        
    Raises:
        RuntimeError: If neither Gemini nor local embedder is available
        NotImplementedError: If provider is not supported
    """
    # Normalize input to list
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        return np.array([])
    
    provider = provider.lower()
    
    # Try Gemini if requested
    if provider == 'gemini':
        if not _GOOGLE_GENAI_AVAILABLE or _GENAI_CLIENT is None:
            logger.warning("Gemini not available, falling back to local embedder")
            provider = 'local'
        else:
            try:
                embed_model = model or os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
                logger.debug(f"Generating embeddings for {len(texts)} texts using {embed_model}")
                
                embeddings = []
                # Batch process to avoid rate limits
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Call Gemini embedding API
                    response = _GENAI_CLIENT.models.embed_content(
                        model=embed_model,
                        contents=batch
                    )
                    
                    # Extract embeddings from response
                    if hasattr(response, 'embeddings'):
                        batch_embeddings = [emb.values for emb in response.embeddings]
                    elif hasattr(response, 'embedding'):
                        batch_embeddings = [response.embedding]
                    else:
                        # Try to access as dict
                        batch_embeddings = response.get('embeddings', response.get('embedding', []))
                    
                    embeddings.extend(batch_embeddings)
                
                result = np.array(embeddings, dtype=np.float32)
                logger.info(f"Generated {len(result)} embeddings with shape {result.shape}")
                return result
                
            except APIError as e:
                logger.error(f"Gemini API error during embedding: {e}")
                logger.warning("Falling back to local embedder")
                provider = 'local'
            except Exception as e:
                logger.error(f"Unexpected error with Gemini embeddings: {e}")
                logger.warning("Falling back to local embedder")
                provider = 'local'
    
    # Use local embedder
    if provider == 'local':
        embedder = _get_local_embedder()
        if embedder is None:
            raise RuntimeError(
                "No embedding provider available. Install sentence-transformers or "
                "set GOOGLE_API_KEY/GEMINI_API_KEY for Gemini embeddings."
            )
        
        try:
            logger.debug(f"Generating local embeddings for {len(texts)} texts")
            embeddings = embedder.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated {len(embeddings)} local embeddings")
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    raise NotImplementedError(f"Provider '{provider}' not supported. Use 'gemini' or 'local'.")


def generate_from_prompt(
    prompt: str,
    provider: str = 'gemini',
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None
) -> str:
    """
    Generate text from a prompt using Gemini.
    
    Args:
        prompt: Input prompt text
        provider: Generation provider ('gemini')
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_output_tokens: Maximum tokens to generate
        model: Optional model name override (default uses GEMINI_MODEL env var)
        system_instruction: Optional system instruction for the model
    
    Returns:
        Generated text string
        
    Raises:
        RuntimeError: If Gemini is not available
        NotImplementedError: If provider is not supported
    """
    provider = provider.lower()
    
    if provider != 'gemini':
        raise NotImplementedError(f"Only 'gemini' provider is supported, got '{provider}'")
    
    if not _GOOGLE_GENAI_AVAILABLE or _GENAI_CLIENT is None:
        raise RuntimeError(
            "Gemini not available. Install google-genai and set GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    
    gen_model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Use default system instruction if not provided
    if system_instruction is None:
        system_instruction = (
            "You are a helpful assistant for credit risk analysis. "
            "Provide accurate, structured responses in JSON format when requested."
        )
    
    try:
        logger.debug(f"Generating text with {gen_model}, temperature={temperature}")
        
        # Prepare generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction
        )
        
        # Call Gemini generation API
        response = _GENAI_CLIENT.models.generate_content(
            model=gen_model,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(prompt)])
            ],
            config=config
        )
        
        # Extract text from response
        if hasattr(response, 'text'):
            result = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content'):
                result = candidate.content.parts[0].text if candidate.content.parts else ""
            else:
                result = str(candidate)
        else:
            result = str(response)
        
        logger.info(f"Generated {len(result)} characters of text")
        return result
        
    except APIError as e:
        logger.error(f"Gemini API error during generation: {e}")
        raise RuntimeError(f"Gemini generation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        raise RuntimeError(f"Text generation failed: {e}")


# Module-level availability checks
def is_gemini_available() -> bool:
    """Check if Gemini is available and configured."""
    return _GOOGLE_GENAI_AVAILABLE and _GENAI_CLIENT is not None


def is_local_embedder_available() -> bool:
    """Check if local embedder is available."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE


if __name__ == '__main__':
    # Quick test
    print(f"Gemini available: {is_gemini_available()}")
    print(f"Local embedder available: {is_local_embedder_available()}")
    
    if is_gemini_available() or is_local_embedder_available():
        try:
            # Test embedding
            test_texts = ["This is a test", "Another test"]
            provider = 'gemini' if is_gemini_available() else 'local'
            embeddings = embed_texts(test_texts, provider=provider)
            print(f"\nEmbedding test ({provider}): Generated embeddings with shape {embeddings.shape}")
            
            # Test generation (Gemini only)
            if is_gemini_available():
                result = generate_from_prompt(
                    "What is 2+2? Respond in JSON: {\"answer\": N}",
                    provider='gemini',
                    temperature=0.0
                )
                print(f"\nGeneration test: {result[:100]}")
        except Exception as e:
            print(f"\nTest failed: {e}")
    else:
        print("\nNo providers available. Set GOOGLE_API_KEY or install sentence-transformers.")
