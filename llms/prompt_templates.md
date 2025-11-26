# Prompt Templates for Credit Risk LLM

This document contains prompt templates used for various credit risk assessment tasks with Gemini and other LLM providers.

## Table of Contents
- [RAG Prompt Template](#rag-prompt-template)
- [CRF Structured Output Schema](#crf-structured-output-schema)
- [Negation-Aware Instructions](#negation-aware-instructions)
- [Standard Task Prompts](#standard-task-prompts)

---

## RAG Prompt Template

### Gemini RAG with Context Retrieval

This template is used when calling Gemini with Retrieval-Augmented Generation (RAG) for enhanced credit risk analysis.

```
You are a credit risk assessment assistant. Analyze the applicant text using the provided context from credit risk research.

IMPORTANT INSTRUCTIONS:
1. Pay close attention to NEGATION words (no, never, not, without, etc.). If text says "no missed payments" that is POSITIVE.
2. Look for intensity modifiers (rarely, sometimes, often, always).
3. Consider temporal decay (recent issues are riskier than old ones).
4. Account for mitigating factors that reduce risk.

CONTEXT FROM RESEARCH:
{retrieved_context}

APPLICANT TEXT TO ANALYZE:
{user_text}

Return a JSON object with this EXACT schema (CRF format):
{
    "summary": "Brief assessment of credit risk (1-2 sentences)",
    "confidence": 0.0-1.0,
    "risky_phrases": ["phrase1", "phrase2", ...],
    "risk_score": 0.0-1.0,
    "risk_level": "low|medium|high",
    "mitigating_factors": ["factor1", "factor2", ...],
    "risk_factors": ["factor1", "factor2", ...],
    "negation_detected": true/false,
    "temporal_factors": ["recent event1", ...]
}

Return ONLY the JSON object, no additional text.
```

**Usage in code:**
```python
from llms.backend import llm_handler

result = llm_handler.call_llm_with_rag(
    text="Applicant text here",
    index_dir='backend/llm_index',
    top_k=3,
    mock=False
)
```

---

## CRF Structured Output Schema

### Complete Response Format

The LLM should return a JSON object matching this schema for credit risk assessment:

```json
{
    "summary": "string - Brief 1-2 sentence assessment of applicant's creditworthiness",
    "confidence": "float - Confidence in assessment (0.0 to 1.0)",
    "risky_phrases": ["array of strings - Specific phrases indicating risk"],
    "risk_score": "float - Overall risk score (0.0 = low risk, 1.0 = high risk)",
    "risk_level": "string - 'low', 'medium', or 'high'",
    "mitigating_factors": ["array of strings - Factors that reduce risk"],
    "risk_factors": ["array of strings - Factors that increase risk"],
    "negation_detected": "boolean - Whether negation was found in text",
    "temporal_factors": ["array of strings - Time-related factors (recent vs old)"]
}
```

**Example Response:**
```json
{
    "summary": "Applicant shows excellent credit history with stable employment and no negative marks",
    "confidence": 0.92,
    "risky_phrases": [],
    "risk_score": 0.15,
    "risk_level": "low",
    "mitigating_factors": ["no missed payments", "stable employment", "low debt"],
    "risk_factors": [],
    "negation_detected": true,
    "temporal_factors": ["5 years employment"]
}
```

---

## Negation-Aware Instructions

### Critical Negation Patterns

When analyzing credit risk text, the LLM must correctly interpret negation:

**Positive Negations (Lower Risk):**
- "no missed payments" → GOOD
- "never defaulted" → GOOD
- "no late payments" → GOOD
- "never had collections" → GOOD
- "not delinquent" → GOOD
- "without bankruptcy" → GOOD

**Negative Statements (Higher Risk):**
- "missed payments" → BAD
- "defaulted" → BAD
- "late payments" → BAD
- "collections" → BAD
- "delinquent" → BAD
- "bankruptcy" → BAD

### Intensity Modifiers

The LLM should weight statements by frequency/intensity:

- **Always / Never** (strongest): "always pays on time", "never late"
- **Frequently / Rarely**: "frequently misses", "rarely late"
- **Often / Seldom**: "often delinquent", "seldom issues"
- **Sometimes / Occasionally**: "sometimes late", "occasionally misses"

### Temporal Considerations

Recent events carry more weight:
- "missed payment last month" → HIGH RISK
- "missed payment 5 years ago" → LOWER RISK (if clean since)
- "recent bankruptcy" → HIGH RISK
- "bankruptcy 10 years ago, rebuilt credit" → MEDIUM RISK

---

## Standard Task Prompts

### Summary Generation

```
You are a concise assistant. Read the applicant text below and return a JSON object with two keys:
- "summary": 1-2 sentence summary of creditworthiness
- "confidence": float between 0-1 indicating confidence

Applicant text:
{text}

Return only valid JSON, no additional commentary.
```

### Risky Phrase Extraction

```
Extract phrases from the applicant text that indicate financial risk or instability.

Consider:
- Payment issues (missed, late, delinquent)
- Defaults, bankruptcy, foreclosure
- Collections, judgments, liens
- Unstable income or employment
- High debt or utilization
- BUT: Pay attention to negation! "no missed payments" is NOT risky.

Return JSON: {"risky_phrases": [...], "count": N}

Text:
{text}

Return only JSON.
```

### Contradiction Detection

```
Analyze the applicant text for contradictions or inconsistent statements.

Examples of contradictions:
- Claims "high income" but also "struggling to pay bills"
- Says "never missed payment" but mentions "recent delinquency"
- Claims "stable job" but "unemployed last year"

Return JSON: {"contradictions": [...], "flag": 0 or 1}

Text:
{text}

Return only JSON.
```

### Sentiment Analysis

```
Analyze the sentiment of the applicant's financial description.

Sentiment should reflect overall financial health tone:
- Positive: confident, stable, strong financial position
- Neutral: mixed or unclear signals
- Negative: struggling, uncertain, financial stress

Return JSON: {"sentiment": "positive|neutral|negative", "score": -1.0 to 1.0}

Text:
{text}

Return only JSON.
```

---

## Configuration

### Environment Variables

Set these environment variables to configure the LLM:

```bash
# Required for Gemini
export GOOGLE_API_KEY="your-google-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"

# Optional configuration
export GEMINI_MODEL="gemini-2.0-flash-exp"
export GEMINI_EMBED_MODEL="models/text-embedding-004"
export LLM_PROVIDER="gemini"
```

### System Instructions

Default system instruction for Gemini:
```
You are a credit-risk feature extractor. Return only a single JSON object with the exact schema requested. Do not add any commentary or surrounding text. Your response must be valid, complete JSON.
```

---

## Best Practices

1. **Always validate negation**: Check for "no", "never", "not", "without", etc.
2. **Consider context**: A single risk factor doesn't determine creditworthiness
3. **Weight by time**: Recent events matter more than historical events
4. **Look for mitigating factors**: High income, stable employment, etc. can offset risks
5. **Use structured output**: Always return valid JSON matching the schema
6. **Be consistent**: Use the same risk_level categories (low/medium/high)
7. **Provide evidence**: List specific phrases from the text in risky_phrases

---

## Testing

Test prompts using the e2e test script:

```bash
# Test with sample examples
python tools/e2e_test_gemini.py

# Test with custom examples
python tools/e2e_test_gemini.py --examples my_examples.json

# Build index and test
python tools/e2e_test_gemini.py --build-index
```

---

## References

- [Gemini API Documentation](https://ai.google.dev/docs)
- Credit Risk Assessment Best Practices (see RAG index documents)
- CRF (Credit Risk Format) Schema specification
