# Tech Stack (Slide content)

- LLM API: OpenAI (Chat Completions / Responses API). Mock mode available for offline demos.
- Backend: Python 3.10+, Flask for API endpoints (/health, /score, /train).
- LLM wrapper: backend/llm_handler.py (openai + mock, JSON prompts, file-based cache artifcats/llm_cache.json).
- Prompt templates: backend/prompts/*.txt (store in version control, keep short & consistent).
- Feature extraction: backend/feature_extraction.py -> artifacts/features.csv (numeric features: sentiment_score, risky_phrase_count, contradiction_flag, credibility_score).
- Models: scikit-learn (LogisticRegression, RandomForest). Training and persistence in backend/train_model.py (joblib.dump). Use random_state=42 for reproducibility.
- Reproducibility: prompt templates in VCS, cache stored locally, record COSTS.md, seed ML pipelines with random_state.
- Explainability: export feature importances (RandomForest) & use SHAP for deeper explainability in the dashboard/demo.
