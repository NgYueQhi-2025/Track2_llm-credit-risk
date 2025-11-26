Running the Track2 LLM Credit Risk app

Use these helpers to ensure you always run the `Track2_llm-credit-risk` app (not any other copy in the workspace).

PowerShell (recommended on Windows)

From the repository root run:

    .\run_track2.ps1

This script will try to activate `.venv` if present and then run Streamlit for `Track2_llm-credit-risk\backend\app.py`.

Windows CMD

From the repository root run:

    run_track2.cmd

Bash / macOS / Linux

From the repository root run:

    ./run_track2.sh

Direct Streamlit command

If you prefer to run streamlit manually, run this from the repository root:

PowerShell / CMD:

    streamlit run Track2_llm-credit-risk\backend\app.py

Bash:

    streamlit run Track2_llm-credit-risk/backend/app.py

Notes
- Ensure your Python virtual environment is activated (scripts attempt to activate `.venv` automatically when present).
- If you use an alternate venv path, activate it before running the helpers.
- If you want VS Code launch configurations added, tell me and I will add a `launch.json` entry to run the Track2 app.
