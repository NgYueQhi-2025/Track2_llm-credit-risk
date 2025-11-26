Auto Deploy / Auto Push

This repository includes simple, opt-in tools to help you push local changes to the remote repository and keep a shared Streamlit app updated.

Files added:
- `tools/auto_sync/auto_push.ps1` — one-shot script: stages, commits (with timestamp), and pushes local changes to `origin/main`.
- `tools/auto_sync/auto_push_loop.ps1` — background loop: polls `git status` and calls `auto_push.ps1` when local changes are present.
- `.vscode/tasks.json` — added two VS Code tasks: `Auto Push Changes (one-shot)` and `Auto Push Loop (background)`.

How to use (recommended safe flow)
1. Review and test locally. Run a one-shot push to confirm behavior:

```powershell
Set-Location -LiteralPath 'C:\path\to\repo'
& '.\.venv\Scripts\Activate.ps1'
# Make changes in VS Code, then run:
python -c "import os; print('git status:'); os.system('git status --porcelain')"
# If you want to commit and push:
powershell -File .\tools\auto_sync\auto_push.ps1
```

2. Start a background auto-push (optional). This will check every 5 seconds and push when it detects local changes:

```powershell
powershell -File .\tools\auto_sync\auto_push_loop.ps1 -IntervalSeconds 5
```

3. Using VS Code tasks (UI): open the Command Palette (Ctrl+Shift+P) → `Tasks: Run Task` → pick `Auto Push Changes (one-shot)` or `Auto Push Loop (background)`.

Security & safety notes
- These scripts will auto-commit and push whatever is in your working directory. Use them only if you are comfortable with automatic commits. They are intended for small demos and personal projects, not for production teams.
- Do not store secrets in the repo. If you use provider API keys for deployments, configure them in your hosting provider (Streamlit Cloud, GitHub Secrets) instead of committing them.
- If you prefer manual control, run `auto_push.ps1` manually when ready.

How this helps Streamlit sharing
- If your Streamlit app is linked to the GitHub repository (Streamlit Cloud or other CI/CD integration), pushing to the repo will trigger redeploys automatically. No further action required on your side once the remote is updated.

If you want, I can:
- Add a safety prompt before auto-commit (require user confirmation in the loop), or
- Wire a GitHub Action to run tests and notify on deploy.

Tell me which option you prefer and I will implement it.