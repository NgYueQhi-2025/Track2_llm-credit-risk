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
import logging
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
    if scores:
        avg_score = sum(scores) / len(scores)
    else:
        avg_score = 0.0
        if full_text.strip():  # Only log if there was actual text to process
            logging.warning("No sentiment scores extracted from text. Check LLM responses.")
    
    return {
        'sentiments': sentiments,
        'average_score': avg_score,
        'raw_results': results
    }


# ============================================================================
# Streamlit App Interface
# ============================================================================
# This section runs when the file is executed with `streamlit run backend/app.py`

if __name__ == "__main__" or os.getenv("STREAMLIT_RUNTIME") is not None:
    # Only import streamlit if we're running as a Streamlit app
    try:
        import streamlit as st
        
        st.title("🔍 LLM-Based Risk Analysis Demo")
        st.markdown("""
        This demo shows how to use the LLM handler to extract risk features from text.
        
        **Features:**
        - Extract risky phrases from applicant text
        - Perform sentiment analysis
        - Process text in configurable chunks
        - Support for mock mode (no API key required)
        """)
        
        # Configuration sidebar
        st.sidebar.header("⚙️ Configuration")
        use_mock = st.sidebar.checkbox(
            "Use Mock Mode", 
            value=os.environ.get('USE_MOCK_LLM', 'true').lower() == 'true',
            help="Enable mock mode to test without API key"
        )
        chunk_size = st.sidebar.slider(
            "Chunk Size (words)", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="Number of words to process in each LLM call"
        )
        
        # Main input area
        st.header("📝 Input Text")
        sample_text = """I run a small business with annual revenue of $150,000.
I've had some late payments in the past due to cash flow issues.
Currently seeking a loan to expand operations."""
        
        input_text = st.text_area(
            "Enter text to analyze:",
            value=sample_text,
            height=150,
            help="Enter applicant text or business description"
        )
        
        # Analysis buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Extract Risky Phrases", type="primary"):
                with st.spinner("Analyzing text for risk indicators..."):
                    try:
                        results = process_uploaded_text(
                            full_text=input_text,
                            use_mock=use_mock,
                            chunk_size=chunk_size,
                            mode='extract_risky'
                        )
                        
                        st.success("✅ Analysis Complete!")
                        st.subheader("Results")
                        st.metric("Risk Count", results['risk_count'])
                        
                        if results['risky_phrases']:
                            st.write("**Detected Risky Phrases:**")
                            for phrase in results['risky_phrases']:
                                st.markdown(f"- {phrase}")
                        else:
                            st.info("No risky phrases detected")
                        
                        with st.expander("📊 View Raw Results"):
                            st.json(results['raw_results'])
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        
        with col2:
            if st.button("😊 Sentiment Analysis"):
                with st.spinner("Analyzing sentiment..."):
                    try:
                        results = extract_sentiment(
                            full_text=input_text,
                            use_mock=use_mock,
                            chunk_size=chunk_size
                        )
                        
                        st.success("✅ Analysis Complete!")
                        st.subheader("Results")
                        st.metric("Average Sentiment Score", f"{results['average_score']:.3f}")
                        
                        if results['sentiments']:
                            st.write("**Detected Sentiments:**")
                            for sent in set(results['sentiments']):
                                count = results['sentiments'].count(sent)
                                st.markdown(f"- {sent}: {count} occurrence(s)")
                        
                        with st.expander("📊 View Raw Results"):
                            st.json(results['raw_results'])
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **Environment Setup:**
        
        For real API calls:
        - Set `GEMINI_API_KEY` in environment
        - Set `USE_MOCK_LLM=false`
        
        For testing:
        - Leave mock mode enabled
        """)
        
    except ImportError:
        # Streamlit not available - this is fine when importing as a module
        pass
