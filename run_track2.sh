#!/usr/bin/env bash
# Bash helper to run the Track2 Streamlit app from repo root
VENV_ACT=".venv/bin/activate"
if [ -f "$VENV_ACT" ]; then
  # shellcheck disable=SC1091
  source "$VENV_ACT"
else
  echo "No .venv found at $VENV_ACT. Activate your Python environment manually if needed."
fi

APP_PATH="Track2_llm-credit-risk/backend/app.py"
if [ ! -f "$APP_PATH" ]; then
  echo "Error: $APP_PATH not found."
  exit 1
fi

echo "Starting Streamlit for Track2 app: $APP_PATH"
streamlit run "$APP_PATH"