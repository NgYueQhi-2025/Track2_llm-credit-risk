# Gemini RAG Hybrid Integration

This document describes how to set up and use the Gemini RAG (Retrieval-Augmented Generation) integration for enhanced credit risk assessment.

## Overview

The hybrid integration combines:
1. **RAG (Retrieval-Augmented Generation)**: Retrieves relevant credit risk research to provide context
2. **Gemini LLM**: Google's Generative AI for structured credit risk analysis
3. **Local Fallback**: Sentence-transformers for embedding when Gemini is unavailable

## Architecture

```
User Text → Embedding (Gemini/Local) → 
  Retrieve Top-K Chunks from Index → 
    Build RAG Prompt with Context → 
      Gemini Generation → 
        Structured JSON Output
```

## Quick Start

### 1. Install Dependencies

Ensure you have the required packages:

```bash
pip install google-genai>=0.14.0
pip install sentence-transformers  # Optional: for local fallback
pip install numpy
```

### 2. Set Environment Variables

#### Option A: Using Gemini (Recommended)

Set your Google API key:

```bash
# Linux/Mac
export GOOGLE_API_KEY="your-google-api-key-here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your-google-api-key-here"

# Windows CMD
set GOOGLE_API_KEY=your-google-api-key-here
```

Or use GEMINI_API_KEY:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**How to get a Google API Key:**
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and set it as an environment variable
5. **IMPORTANT**: Never commit API keys to git!

#### Option B: Using Local Embedder

If you don't have a Gemini API key, the system will automatically fall back to local embeddings using sentence-transformers. Just ensure the package is installed:

```bash
pip install sentence-transformers
```

#### Optional Configuration

```bash
# Customize model names
export GEMINI_MODEL="gemini-2.0-flash-exp"  # Default generation model
export GEMINI_EMBED_MODEL="models/text-embedding-004"  # Default embedding model

# Set LLM provider
export LLM_PROVIDER="gemini"

# For Google Cloud users (optional)
export GOOGLE_PROJECT="your-project-id"
export GOOGLE_REGION="us-central1"
```

### 3. Build the RAG Index

Before using RAG, you need to build the index with embedded research documents:

```bash
# Build index with sample research (uses Gemini if available, else local)
python tools/build_rag_index_gemini.py --generate-sample --output backend/llm_index

# Or build from custom research file
python tools/build_rag_index_gemini.py --input my_research.txt --output backend/llm_index

# Force local embedder
python tools/build_rag_index_gemini.py --generate-sample --provider local --output backend/llm_index
```

The build process will:
1. Load/generate credit risk research documents
2. Chunk the documents into manageable pieces
3. Generate embeddings using Gemini or local embedder
4. Save `docs.jsonl` and `embeddings.npy` to the output directory

Expected output:
```
✓ Successfully built RAG index in backend/llm_index
  - 25 documents
  - Embedding dimension: 768
  - Provider: gemini
```

### 4. Run the End-to-End Test

Verify everything works:

```bash
# Test with sample examples (auto-detects mock vs real mode)
python tools/e2e_test_gemini.py

# Force real Gemini mode (requires API key)
python tools/e2e_test_gemini.py --no-mock

# Force mock mode (no API calls)
python tools/e2e_test_gemini.py --mock

# Build index and test
python tools/e2e_test_gemini.py --build-index

# Use custom test examples
python tools/e2e_test_gemini.py --examples my_examples.json
```

Example output:
```
Gemini RAG End-to-End Test
================================================================================

Gemini available: True
Local embedder available: True
Mode: REAL GEMINI

✓ RAG index exists in backend/llm_index
✓ Loaded 10 examples from tools/sample_examples.json

[1/10] Testing: Example 1: Low Risk - Clean History
✓ Success: low risk

...

SUMMARY: 10/10 tests completed successfully
================================================================================
Risk distribution:
  Low risk:    4
  Medium risk: 3
  High risk:   3
```

## Usage in Code

### Basic RAG Call

```python
from llms.backend import llm_handler

# Analyze applicant text with RAG
result = llm_handler.call_llm_with_rag(
    text="I have no missed payments and stable income",
    index_dir='backend/llm_index',
    top_k=3,           # Retrieve top 3 relevant chunks
    use_cache=True,
    mock=False         # Set True for testing without API
)

print(f"Risk Level: {result['risk_level']}")
print(f"Summary: {result['summary']}")
print(f"Risky Phrases: {result['risky_phrases']}")
```

### Using Gemini Adapter Directly

```python
from llms.backend import gemini_adapter

# Generate embeddings
texts = ["text1", "text2", "text3"]
embeddings = gemini_adapter.embed_texts(texts, provider='gemini')
print(f"Embeddings shape: {embeddings.shape}")

# Generate text
response = gemini_adapter.generate_from_prompt(
    prompt="What factors indicate credit risk?",
    provider='gemini',
    temperature=0.0,
    max_output_tokens=1024
)
print(response)
```

### Check Provider Availability

```python
from llms.backend import gemini_adapter

if gemini_adapter.is_gemini_available():
    print("Gemini is configured and ready")
    provider = 'gemini'
elif gemini_adapter.is_local_embedder_available():
    print("Using local embedder fallback")
    provider = 'local'
else:
    print("No embedding provider available")
```

## Understanding the Output

The RAG pipeline returns a structured JSON object:

```json
{
    "summary": "Brief assessment of credit risk",
    "confidence": 0.85,
    "risky_phrases": ["missed payments", "high utilization"],
    "risk_score": 0.65,
    "risk_level": "medium",
    "mitigating_factors": ["stable employment", "improving trend"],
    "risk_factors": ["payment issues"],
    "negation_detected": false,
    "temporal_factors": ["recent issues"]
}
```

### Field Descriptions

- **summary**: 1-2 sentence assessment of creditworthiness
- **confidence**: 0.0-1.0, how confident the model is
- **risky_phrases**: Specific text that indicates risk
- **risk_score**: 0.0 (low) to 1.0 (high) numerical risk
- **risk_level**: "low", "medium", or "high" categorical risk
- **mitigating_factors**: Factors that reduce risk
- **risk_factors**: Factors that increase risk
- **negation_detected**: Whether negation was found (e.g., "no missed payments")
- **temporal_factors**: Time-related factors (recent vs historical)

## Advanced Configuration

### Custom RAG Index

Create your own research documents:

1. Write credit risk research in a text file (`research.txt`)
2. Build index:
   ```bash
   python tools/build_rag_index_gemini.py --input research.txt --output my_index
   ```
3. Use in code:
   ```python
   result = llm_handler.call_llm_with_rag(
       text="...",
       index_dir='my_index',
       top_k=5
   )
   ```

### Chunk Size Configuration

Adjust chunk size for different document types:

```bash
# Larger chunks (better context, fewer total chunks)
python tools/build_rag_index_gemini.py --generate-sample --chunk-size 1000

# Smaller chunks (more precise retrieval, more chunks)
python tools/build_rag_index_gemini.py --generate-sample --chunk-size 300
```

### Top-K Retrieval Tuning

Experiment with different top_k values:

```python
# More context (slower, potentially more accurate)
result = llm_handler.call_llm_with_rag(text, top_k=5)

# Less context (faster, more focused)
result = llm_handler.call_llm_with_rag(text, top_k=2)
```

## Troubleshooting

### "Gemini not available" Error

**Cause**: API key not set or invalid

**Solution**:
```bash
export GOOGLE_API_KEY="your-valid-key"
# Test
python -c "from llms.backend import gemini_adapter; print(gemini_adapter.is_gemini_available())"
```

### "No embedding provider available" Error

**Cause**: Neither Gemini nor sentence-transformers is available

**Solution**:
```bash
pip install sentence-transformers
```

### RAG Index Not Found

**Cause**: Index not built or wrong path

**Solution**:
```bash
# Build the index
python tools/build_rag_index_gemini.py --generate-sample --output backend/llm_index

# Verify files exist
ls backend/llm_index/
# Should see: docs.jsonl, embeddings.npy
```

### API Rate Limits

**Cause**: Too many requests to Gemini API

**Solution**:
- Use `mock=True` for testing
- Add delays between calls
- Consider caching results
- Use local embedder for development

### Import Errors

**Cause**: Running from wrong directory or missing `__init__.py`

**Solution**:
```bash
# Always run from repository root
cd /path/to/Track2_llm-credit-risk
python tools/e2e_test_gemini.py
```

## Security Best Practices

### Never Commit Secrets

```bash
# Add to .gitignore
echo "*.env" >> .gitignore
echo ".env.local" >> .gitignore
echo "secrets.txt" >> .gitignore
```

### Use Environment Variables

Store secrets in environment, not code:

```python
# ✓ Good
api_key = os.getenv("GOOGLE_API_KEY")

# ✗ Bad
api_key = "AIza..." # Never hardcode!
```

### Use Secrets Management

For production:
- Streamlit: Use `st.secrets`
- Cloud: Use Secret Manager (GCP, AWS, Azure)
- Local: Use `.env` files (not committed)

## Performance Tips

1. **Build index once**: Don't rebuild on every run
2. **Use appropriate chunk size**: 300-500 chars is usually good
3. **Tune top_k**: Start with 3, adjust based on results
4. **Cache responses**: Enable `use_cache=True` when possible
5. **Use mock mode for dev**: Faster and free

## Integration with Main App

To use RAG in the main Streamlit app:

```python
# In backend/integrations.py or backend/app.py
from llms.backend import llm_handler

# Replace call_llm with call_llm_with_rag
result = llm_handler.call_llm_with_rag(
    text=applicant_text,
    index_dir='backend/llm_index',
    top_k=3,
    mock=use_mock_mode
)

# Extract fields
summary = result.get('summary', '')
risk_level = result.get('risk_level', 'unknown')
risky_phrases = result.get('risky_phrases', [])
```

## Testing

### Unit Tests

Test individual components:

```bash
# Test gemini_adapter
python -c "from llms.backend import gemini_adapter; print('OK')"

# Test import
python -c "from llms.backend import llm_handler; print(llm_handler.call_llm_with_rag.__doc__)"
```

### Integration Tests

```bash
# Full e2e test
python tools/e2e_test_gemini.py

# Specific example
python tools/e2e_test_gemini.py --examples tools/sample_examples.json
```

### Mock Mode Testing

Always test in mock mode first:

```python
# No API calls, instant results
result = llm_handler.call_llm_with_rag(
    text="test text",
    mock=True
)
```

## Examples

### Example 1: Simple Analysis

```python
from llms.backend import llm_handler

text = "I have been employed for 5 years and never missed a payment"
result = llm_handler.call_llm_with_rag(text, mock=False)

print(f"Risk: {result['risk_level']}")  # "low"
print(f"Negation: {result['negation_detected']}")  # True
```

### Example 2: Batch Processing

```python
applicants = [
    "Clean credit history, stable income",
    "Recent bankruptcy, multiple late payments",
    "No collections, sometimes late with bills"
]

for i, text in enumerate(applicants):
    result = llm_handler.call_llm_with_rag(text, mock=False)
    print(f"Applicant {i+1}: {result['risk_level']} risk")
```

### Example 3: Custom Index

```python
# Build custom index
from tools import build_rag_index_gemini

build_rag_index_gemini.build_index(
    input_path='my_research.txt',
    output_dir='custom_index',
    provider='gemini'
)

# Use custom index
result = llm_handler.call_llm_with_rag(
    text="Analyze this",
    index_dir='custom_index',
    top_k=5
)
```

## Next Steps

1. ✓ Set up API key
2. ✓ Build RAG index
3. ✓ Run e2e test
4. → Integrate with main app
5. → Fine-tune prompts
6. → Expand research documents
7. → Deploy to production

## Support

- See [prompt_templates.md](prompt_templates.md) for prompt engineering
- Check [COSTS.md](COSTS.md) for API pricing
- Review [ARCHITECTURE.md](../ARCHITECTURE.md) for system design

## References

- [Gemini API Docs](https://ai.google.dev/docs)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/)
