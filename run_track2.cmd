@echo off
:: CMD helper to run the Track2 app from repository root
IF EXIST ".venv\Scripts\activate.bat" (
  call .venv\Scripts\activate.bat
) ELSE (
  echo No .venv activation script found. Activate your Python environment manually if needed.
)

set APP_PATH=Track2_llm-credit-risk\backend\app.py
if not exist "%APP_PATH%" (
  echo Error: %APP_PATH% not found.
  exit /b 1
)

echo Starting Streamlit for Track2 app: %APP_PATH%
streamlit run "%APP_PATH%"