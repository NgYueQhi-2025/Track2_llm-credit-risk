# Example Dockerfile to run this Streamlit app with Tesseract installed
# Base image
FROM python:3.11-slim

# Install system deps for OCR and PDF parsing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       libtesseract-dev \
       libleptonica-dev \
       pkg-config \
       poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit (use headless server mode)
CMD ["streamlit", "run", "backend/app.py", "--server.port=8501", "--server.headless=true"]
