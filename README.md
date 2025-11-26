# LLM Credit Risk — Demo

This repository contains a Streamlit demo that uses an LLM (+ optional OCR) to extract behavioral risk signals from structured CSVs and unstructured uploads (PDF/PNG/JPG).

Quick overview
- UI entrypoint: `backend/app.py`
- Optional OCR libs: `pdfplumber`, `pytesseract`, `Pillow` (see `requirements-ocr.txt`)
- LLM integration: adapters live under `llms/backend/` (use `mock_mode` for offline testing)

How to publish to GitHub and let teammates view the app on Streamlit Community Cloud

1) Push this repo to GitHub (PowerShell commands)

   # create a new repository on GitHub first, then run locally:
   git init
   git add .
   git commit -m "Initial demo: LLM credit risk UI"
   git branch -M main
   git remote add origin https://github.com/<YOUR_ORG_OR_USERNAME>/<REPO_NAME>.git
   git push -u origin main

Replace `<YOUR_ORG_OR_USERNAME>` and `<REPO_NAME>` with your values. If you use SSH, use the SSH remote URL.

2) Connect the repo to Streamlit Community Cloud

- Open https://share.streamlit.io and sign in with GitHub.
- Click **New app** → choose your GitHub repo, branch `main`, and set `File path` to `backend/app.py`.
- Click **Deploy**.

Notes for teammates
- If you rely on OCR, the hosted Streamlit Cloud cannot install system packages (Tesseract) — use the Docker-based deploy or run locally with `requirements-ocr.txt` installed and Tesseract available on the host.
- If the app uses external LLM API keys, add them under Streamlit app Settings → Secrets (or in GitHub Actions if you add automated deploys).

Local run for testing

PowerShell (from repo root):
```
& ".\.venv\Scripts\Activate.ps1"  # if using venv
pip install -r requirements.txt
# optional OCR deps
pip install -r requirements-ocr.txt
streamlit run backend/app.py
```

CI / GitHub Actions
- This repo includes a simple CI that runs on pushes and PRs to `main` and checks Python syntax and installs requirements. See `.github/workflows/ci.yml`.

If you want, I can add an optional workflow to automatically deploy on push using an external action — tell me whether you'd like automatic deployment and whether you will store Streamlit credentials as GitHub secrets.

---
Project maintainer: edit the repository README and CI to match your org's process.
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

Minimal (MVP) env vars (new and existing):

- `USE_MOCK_LLM` — `true` / `false` (default `true` for demo).
- `LLM_PROVIDER` — `openai` / `ollama` / `mock` (default `openai` if `USE_MOCK_LLM=false`).
- `LOCAL_LLM_URL` — e.g. `http://127.0.0.1:11434` (when using a local Ollama-compatible HTTP gateway).
- `OPENAI_API_KEY` — OpenAI key (only if using OpenAI).
- `HF_API_TOKEN` — Hugging Face token (optional).
- `CACHE_TTL_SECONDS` — e.g. `3600`.
- `RANDOM_SEED` — e.g. `42` for reproducible demos.

Access in code (example):

```python
import os
import streamlit as st

USE_MOCK = (os.environ.get('USE_MOCK_LLM','true').lower() == 'true')
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai')
LOCAL_LLM_URL = os.environ.get('LOCAL_LLM_URL')
OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY') if hasattr(st, 'secrets') else os.environ.get('OPENAI_API_KEY')
```

Notes:
- `LLM_PROVIDER` controls which provider the `llms/backend/llm_handler.py` will attempt to use. The code supports `openai` (remote OpenAI API), `ollama` (local Ollama-compatible HTTP gateway), and `mock` (deterministic demo responses).
- When `LLM_PROVIDER=ollama`, set `LOCAL_LLM_URL` to the base URL of your Ollama HTTP gateway (for example `http://127.0.0.1:11434`). The handler will attempt common Ollama-compatible endpoints and payload shapes.
- For offline demos the repo includes a small mock Ollama server at `backend/artifacts/mock_ollama_server.py` which you can run locally to emulate a provider during development.

Run the mock server (PowerShell):

```powershell
# from repo root
# Ensure venv activated or use absolute python path
python backend\artifacts\mock_ollama_server.py
# Default: listens on http://127.0.0.1:11434
```

Use `LLM_PROVIDER=ollama` with `LOCAL_LLM_URL=http://127.0.0.1:11434` to point the app at the mock server for full-path testing without a paid provider.

## Mock vs Live Flows (safe demo & production switch)

This project supports two execution modes so you can demo reliably without external API calls (Mock mode) and switch to a Live LLM when you're ready.

- Mock mode (recommended for demos / judging):
    - The Streamlit UI includes a `Mock mode (no LLM/API)` checkbox in the sidebar. When enabled, the app will avoid real LLM API calls.
    - If precomputed artifacts exist, the app will use them for deterministic outputs:
        - `backend/artifacts/features.csv` — per-applicant numeric features (columns: `applicant_id`,`sentiment_score`,`risky_phrase_count`,`contradiction_flag`,`credibility_score`).
        - `backend/artifacts/model.pkl` — a small `sklearn` model used for `predict_proba`. If present, the UI will use it to generate risk scores.
    - A helper script `backend/artifacts/setup_demo.py` is provided to generate demo artifacts (toy `model.pkl`, `features.csv`, and `data/demo.csv`). Run it from the repo root:

```powershell
python backend\artifacts\setup_demo.py
```

    - After running the script, start the app and ensure the sidebar checkbox `Mock mode` is checked. The app will then: read `data/demo.csv` (or uploaded CSV), look up precomputed features in `backend/artifacts/features.csv` when possible, and use `backend/artifacts/model.pkl` to produce deterministic predictions.

- Live mode (production / real LLM):
    - Uncheck `Mock mode` in the UI or set `USE_MOCK_LLM=false` in your environment.
    - Implement a real provider call inside `llms/backend/llm_handler.py` — currently the file contains a `call_llm` stub with a `mock` parameter and a simple on-disk caching helper. Replace the `mock=False` branch of `call_llm` with your provider call (OpenAI, HF, etc.) and ensure you load API keys from env vars or `st.secrets`.
    - Example pattern inside `llms/backend/llm_handler.py`:

```python
# load API key securely
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def call_llm(prompt, mode='summary', mock=False):
        if mock:
                return ... # existing mock
        # call real provider here (openai.ChatCompletion.create or requests to an API)
        # return a JSON-serializable string matching the existing mock outputs
```

Notes & tips:
- The integrator layer (`backend/integrations.py`) implements retry logic (exponential backoff) for LLM extraction and will fall back to a deterministic heuristic if a model artifact is missing.
- For reliable demos (no cost, deterministic results), use `Mock mode` and run `backend/artifacts/setup_demo.py` once to populate demo artifacts.
- Keep secrets out of source control; use `st.secrets` or your host's secrets manager to provide `OPENAI_API_KEY` when running in Live mode.


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

---

## Auto-sync helper (keep workspace up-to-date)

This repository includes a small, intentionally conservative auto-sync helper to keep your local clone up-to-date with `origin/main`.

- `tools/auto_sync/auto_sync.ps1`: PowerShell script that fetches from the remote and attempts a `git pull --ff-only` on `origin/main` at a configurable interval (default 5 minutes). It will stop if a non-fast-forward update is required so you can resolve conflicts manually.
- `.vscode/tasks.json`: VS Code task `Auto Sync Repo (PowerShell)` — run this task to start the sync loop in a dedicated terminal.

Usage:

1. From VS Code: open the Command Palette → `Tasks: Run Task` → `Auto Sync Repo (PowerShell)`.
2. Or run directly in PowerShell from the repo root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\auto_sync\auto_sync.ps1 -IntervalMinutes 5
```

Notes:

- The script only performs fast-forward pulls to avoid creating automatic merge commits or conflicts.
- If your branch has diverged from the remote (non-fast-forward), the script prints instructions and stops — resolve manually before re-enabling auto-sync.
- Adjust `-IntervalMinutes` to a value that suits your workflow.
