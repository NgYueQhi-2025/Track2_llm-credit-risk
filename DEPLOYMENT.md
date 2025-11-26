Deployment notes — Tesseract & Streamlit Cloud

Summary
- Streamlit Community Cloud (app.streamlit.io) installs Python packages from `requirements.txt`, but it does NOT provide a way to install system-level packages (apt/apt-get) such as the Tesseract OCR binary.
- As a result, `pytesseract` (the Python wrapper) can be installed on Streamlit Cloud, but it will fail at runtime because the underlying `tesseract` executable is not present.

Options

1) Local development (recommended for OCR)
- Install Tesseract on your workstation (Windows / macOS / Linux). See the project's releases or the UB-Mannheim Windows builds.
- Install Python OCR packages: `pip install -r requirements.txt` (this repo now includes `pytesseract`, `pdfplumber`, `Pillow`).
- Run locally: `streamlit run backend/app.py`.

2) Deploy using a Docker container (recommended for hosted OCR)
- Build a Docker image that installs `tesseract-ocr` (system package) and Python deps. Example `Dockerfile` provided in this repo. This is suitable for Render, Railway (with Docker), Google Cloud Run, AWS ECS, etc.

3) Use an external OCR service (no system binary required)
- If you must deploy to Streamlit Cloud, host an OCR microservice elsewhere (e.g., Cloud Run or a small VM) that runs Tesseract, or use a managed OCR API (Google Vision, AWS Textract, etc.). The Streamlit app can call that service via HTTP.

4) Use a buildpack or platform that allows apt packages
- Some platforms provide custom buildpacks or Docker-based deploys that allow installation of apt packages; this is platform-specific and not supported on Streamlit Community Cloud.

Dockerfile (example)
- See `Dockerfile` in repo root for a minimal example that installs Tesseract and Python deps. Build & run locally to verify OCR works.

Troubleshooting
- If OCR fails locally with `pytesseract` errors, ensure `tesseract --version` works on the host and that the binary path is available (on Windows, often `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- For languages other than English, install additional tessdata files and set the `TESSDATA_PREFIX` environment variable.

If you want, I can:
- Add a UI notice that OCR was skipped because no Tesseract binary was found.
- Provide a tested Dockerfile variant for alpine/debian base images.

