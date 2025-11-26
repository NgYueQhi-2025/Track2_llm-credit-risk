"""Backend app module for text processing with LLM.

This module provides helper functions for processing uploaded text using LLM
to extract risky phrases, sentiment, and other features.

Example usage:
    from backend.app import process_uploaded_text
    
    # Process text from an uploaded file
    results = process_uploaded_text(
        full_text="User text here...",
        use_mock=False,
        chunk_size=10
    )
    
    # Access extracted data
    risky_phrases = results['risky_phrases']
    total_count = results['risk_count']
"""

from llms.backend.llm_handler import process_text_word_by_word
import os
from typing import Dict, List, Any


def process_uploaded_text(
    full_text: str,
    use_mock: bool = None,
    chunk_size: int = 10,
    mode: str = 'extract_risky',
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Process uploaded text using LLM to extract features.
    
    Args:
        full_text: The text to process (extracted from PDF/JPG/PNG/CSV)
        use_mock: Whether to use mock LLM responses. If None, reads from USE_MOCK_LLM env var.
        chunk_size: Number of words per LLM call (default: 10, recommended: 5-50)
        mode: Processing mode ('extract_risky', 'sentiment', 'summary', etc.)
        temperature: LLM temperature parameter (default: 0.0)
    
    Returns:
        Dictionary with extracted features:
        - 'risky_phrases': List of detected risky phrases
        - 'risk_count': Total count of risky items
        - 'raw_results': Raw per-chunk results from LLM
    
    Example:
        >>> results = process_uploaded_text("Late payments on loans", use_mock=True)
        >>> print(results['risky_phrases'])
        ['late payments', 'loans']
    """
    if use_mock is None:
        use_mock = (os.environ.get('USE_MOCK_LLM', 'true').lower() == 'true')
    
    # Process text in chunks using LLM
    word_by_word_results = process_text_word_by_word(
        text=full_text,
        mode=mode,
        chunk_size_words=chunk_size,
        temperature=temperature,
        mock=use_mock
    )
    
    # Aggregate results: collect all risky phrases and total counts
    risky_phrases = []
    total_count = 0
    for item in word_by_word_results:
        if item.get("parsed") and isinstance(item["parsed"], dict):
            rp = item["parsed"].get("risky_phrases") or []
            cnt = item["parsed"].get("count") or 0
            risky_phrases.extend(rp)
            total_count += cnt
    
    return {
        'risky_phrases': risky_phrases,
        'risk_count': total_count,
        'raw_results': word_by_word_results
    }


# Example helper for sentiment analysis
def extract_sentiment(
    full_text: str,
    use_mock: bool = None,
    chunk_size: int = 10
) -> Dict[str, Any]:
    """Extract sentiment from text.
    
    Args:
        full_text: The text to analyze
        use_mock: Whether to use mock LLM responses
        chunk_size: Number of words per LLM call
    
    Returns:
        Dictionary with sentiment analysis results
    """
    if use_mock is None:
        use_mock = (os.environ.get('USE_MOCK_LLM', 'true').lower() == 'true')
    
    results = process_text_word_by_word(
        text=full_text,
        mode='sentiment',
        chunk_size_words=chunk_size,
        temperature=0.0,
        mock=use_mock
    )
    
    # Aggregate sentiment scores
    sentiments = []
    scores = []
    for item in results:
        if item.get("parsed") and isinstance(item["parsed"], dict):
            sent = item["parsed"].get("sentiment")
            score = item["parsed"].get("score")
            if sent:
                sentiments.append(sent)
            if score is not None:
                scores.append(score)
    
    # Calculate average sentiment score
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        'sentiments': sentiments,
        'average_score': avg_score,
        'raw_results': results
    }
