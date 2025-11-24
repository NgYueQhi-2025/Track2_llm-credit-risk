# Track 2 — LLM-Based Risk Assessment for AI Credit Scoring

Tagline: Surface behavioral risk signals from application text + tabular data using an LLM + fusion classifier and an interpretable Streamlit dashboard.

---

## Summary

Traditional credit scoring relies heavily on structured numeric features and can miss behavioral and contextual signals contained in unstructured text (loan essays, transaction notes, emails). This prototype demonstrates how an LLM can extract summaries and risk indicators from free-text, convert them into numeric features, fuse those features with tabular data, and present interpretable risk scores in a Streamlit dashboard.

This repo contains a runnable Streamlit MVP that shows the end-to-end flow (upload → LLM extraction → feature engineering → classifier → explainability UI).

## Repo layout

- `backend/app.py` — Streamlit app (UI + demo pipeline).  
- `backend/ui_helpers.py` — small reusable UI components (KPI cards, table renderer, highlight helper).  
- `frontend/mock_index.html` — static mock page useful for slides.  
- `requirements.txt` — minimal dependencies for deployment.  
- `Procfile` — start command for Render.  

Work for UI/UX live on branch `feature/ui-layout`.

## MVP Features

- Upload CSV (tabular + a `text_notes` column) or pick a demo dataset.  
- Per-applicant LLM extraction: short summary, sentiment score, risky phrases/indicators.  
- Convert LLM outputs into numeric features (e.g., `sentiment_score`, `risky_phrase_count`).  
- Fuse text features with tabular features and run a classifier (heuristic / scikit-learn).  
- Dashboard: KPI cards, interactive applicant table, explanation and evidence panel, basic fairness snapshot.

## Quickstart — Run locally (Windows / PowerShell)

1. Clone and checkout UI branch:

```powershell
git clone https://github.com/NgYueQhi-2025/Track2_llm-credit-risk.git
cd Track2_llm-credit-risk
git checkout feature/ui-layout
```

2. Create & activate a Python virtual environment and install dependencies:

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the app:

```powershell
python -m streamlit run backend\app.py
# open http://localhost:8501 in your browser
```

Notes: if `streamlit` is not found, ensure `.venv` is activated. To avoid LLM API costs during demos, set `USE_MOCK_LLM=true` (see Environment Variables section).

## Data format

Expected CSV columns (minimum): `id`, `name`, `income`, `credit_score`, `text_notes`. Optional: `default_label` (supervised testing), `group` (for fairness checks).

Demo synthetic examples are loaded by default if you don't upload a CSV.

## Environment variables & secrets

Use host secret managers (Streamlit Cloud / Render) or local `.streamlit/secrets.toml` for development. Do NOT commit secrets to the repo.

Minimal (MVP) env vars:

- `USE_MOCK_LLM` — `true` / `false` (default `true` for demo).  
- `OPENAI_API_KEY` — OpenAI key (only if using OpenAI).  
- `HF_API_TOKEN` — Hugging Face token (optional).  
- `CACHE_TTL_SECONDS` — e.g. `3600`.  
- `RANDOM_SEED` — e.g. `42` for reproducible demos.

Access in code:

```python
import os
import streamlit as st

OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY') if hasattr(st, 'secrets') else os.environ.get('OPENAI_API_KEY')
USE_MOCK = (os.environ.get('USE_MOCK_LLM','true').lower() == 'true')
```

## Deployment

### Streamlit Community Cloud (recommended)

1. Push your branch to a repo accessible to your Streamlit account (fork if you are not the repo owner).  
2. Go to https://share.streamlit.io and sign in with GitHub.  
3. Click `New app`, select the repo and branch `feature/ui-layout`, and set the file to `backend/app.py`.  
4. Add secrets in App Settings → Secrets (e.g., `OPENAI_API_KEY`) or set `USE_MOCK_LLM=true`.  

Streamlit Cloud will install packages from `requirements.txt` and provide a persistent `https://...streamlitapp.com` URL.

### Render (alternative)

- Ensure `requirements.txt` and `Procfile` are present.  
- Build command: `pip install -r requirements.txt`  
- Start command (or use Procfile):

```
streamlit run backend/app.py --server.port $PORT --server.address 0.0.0.0
```

Add environment variables (OPENAI_API_KEY, USE_MOCK_LLM) in the Render dashboard. Render will provide a public URL.

### Temporary sharing

For quick demos you can use `cloudflared` or `ngrok` to expose `localhost:8501` with a temporary public URL — do not use this for production or to expose sensitive data.

## Architecture (overview)

```
[CSV + text] -> Preprocessing -> LLM API (summaries, risk indicators)
            -> Text -> numeric features -> Classifier (LogReg/RF/heuristic)
            -> Explainability (evidence sentences) -> Dashboard (Streamlit)
```

## Explainability & UI

For each applicant the UI shows: original text, LLM summary, extracted indicators (sentiment, risky phrases), numeric risk score, risk label, and a mock feature-contribution listing. The dashboard emphasizes evidence-first explanations and a recommended action.

## Testing & QA

- Smoke test: upload demo CSV → Run Pipeline → Verify table, KPIs, and explanation panel.  
- Use `USE_MOCK_LLM=true` for cost-free testing.  
- Add unit tests for feature engineering and any model-training code where appropriate.

## Slides & Screenshots

- Capture: KPI header + applicant table, explanation panel with highlighted evidence, and fairness chart.  
- Save screenshots to `frontend/` and include them in slides.  

## Contribution

- Branch from `feature/ui-layout` for UI work.  
- Keep PRs focused and include testing steps in PR descriptions.  

## License

Add your preferred license (MIT recommended for demos) and include a `LICENSE` file if desired.

## Team & Contact

- UI/UX Lead: Member B — Esther (GitHub: `25-ESTHERKONG`)  
- Backend / Model: (fill in team members)  

---

If you want, I can also add a short `docs/` slide text or export-ready screenshots after you deploy. Open an issue or request for any additions.
# Track2_llm-credit-risk