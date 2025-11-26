# LLM Handler Import Fix

## Summary of Changes

This PR fixes import errors in the Streamlit app by addressing the following issues:

### 1. Missing Package Initialization Files
- **Added**: `llms/__init__.py` - Makes the `llms` package properly importable
- **Added**: `llms/backend/__init__.py` - Makes the `llms.backend` package properly importable

### 2. Enhanced Streamlit Secrets Support
- **Modified**: `llms/backend/llm_handler.py`
  - Added support for Streamlit secrets as a fallback for `GEMINI_API_KEY`
  - The code now tries to read from:
    1. Environment variable `GEMINI_API_KEY`
    2. Streamlit secrets (`st.secrets["GEMINI_API_KEY"]`) if env var not set
  - This allows the app to work both in local development and Streamlit Cloud deployment

### 3. Fixed Import-Time Execution Error
- **Modified**: `backend/app.py`
  - **Problem**: The original code tried to execute LLM calls at import time using an undefined variable `full_text`, causing `NameError`
  - **Solution**: Refactored the code into callable helper functions:
    - `process_uploaded_text()`: Main function for processing text with LLM
    - `extract_sentiment()`: Helper function for sentiment analysis
  - The module can now be safely imported without executing any LLM calls
  - Added comprehensive docstrings and usage examples

### 4. Added Validation Tests
- **Added**: `test_imports.py` - Comprehensive smoke tests that verify:
  - All modules can be imported successfully
  - Required functions are present
  - Mock mode works correctly
  - Data structures are valid

## Validation Commands

Run these commands to verify the changes:

```bash
# Test llm_handler import
python -c "from llms.backend import llm_handler; print('✓ ok', hasattr(llm_handler, 'process_text_word_by_word'))"

# Test backend.app import
python -c "from backend import app as demo; print('✓ ok', hasattr(demo, 'process_uploaded_text'))"

# Run comprehensive smoke tests
python test_imports.py
```

## Usage Example

```python
from backend.app import process_uploaded_text

# Process text extracted from uploaded file
text = "I have late payments on multiple loans"
results = process_uploaded_text(
    full_text=text,
    use_mock=False,  # Set to True for demo/testing without API key
    chunk_size=10
)

# Access results
print(f"Risky phrases: {results['risky_phrases']}")
print(f"Risk count: {results['risk_count']}")
```

## Environment Configuration

### For Local Development:
Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
export USE_MOCK_LLM="false"  # or "true" for demo mode
```

### For Streamlit Cloud:
Add to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
USE_MOCK_LLM = "false"
```

## Notes
- The code defaults to mock mode (`USE_MOCK_LLM=true`) if the environment variable is not set
- Per-word processing (chunk_size=1) is expensive; recommended chunk_size is 5-50 words
- The `google-genai` package must be installed (already in requirements.txt)
- Mock mode returns canned responses useful for testing without API calls
