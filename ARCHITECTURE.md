# System Architecture
This document explains the overall architecture of the **LLM Credit Risk** system.

# 🏗 High-Level Overview
CSV Upload → Preprocessing → LLM → Feature Extraction → ML Model → UI (Streamlit)
---

## 1. Data Layer
Users upload:
- CSV with applicant information  
- Or select built-in demo dataset  

Data is loaded into Pandas for preprocessing.
---

## 2. LLM Layer (OpenAI / Ollama / Mock Mode)
The LLM analyzes applicant free-text fields:

- Sentiment  
- Risk indicators  
- Contradictions  
- Behavioral cues  
- Credibility signals  

Outputs are cached to:
`backend/artifacts/llm_cache.json`

This layer supports multiple provider modes:

- `openai`: calls the OpenAI API (when `LLM_PROVIDER=openai`).
- `ollama`: calls a local Ollama-compatible HTTP gateway (when `LLM_PROVIDER=ollama` and `LOCAL_LLM_URL` is set). The handler will try common Ollama endpoints and payload shapes to be robust across gateway versions.
- `mock`: deterministic on-disk responses used for judged demos and offline testing.

This ensures:
- Determinism  
- No unnecessary API cost  
---

## 3. Feature Extraction Layer

`feature_extraction.py` converts LLM outputs → numeric ML features:

- `sentiment_score`  
- `risky_phrase_count`  
- `contradiction_flag` (0/1)  
- `credibility_score`  
- And domain-specific engineered features  

Stored in:
backend/artifacts/features.csv
---

## 4. ML Model Layer

`train_model.py` trains Logistic Regression / Random Forest model.

Artifacts:
- `model.pkl`
- `scaler.pkl`

Deterministic using `random_state=42`.
---

## 5. Backend Integration Layer

`integrations.py` connects:
UI → LLM → Feature Extractor → ML Model → Explanations

Handles:
- Exceptions
- Mock mode
- Caching
- Retry logic
---

## 6. Frontend Layer (Streamlit)

`app.py` provides:
- File upload  
- KPI summary  
- Applicant table  
- Local explanation panel  
- Story playback  

Uses:
- `st.cache_data`
- `st.spinner()`
- Clean, structured UI layout

---

## 7. Demo Mode

For reliability:
- All outputs can be loaded from cached artifacts  
- No real API calls required during judging  

---

# 🎯 Goals of Architecture
- Simple  
- Fast  
- Deterministic  
- Easy for all team members to contribute  
- Professional for hackathon judging  

---
