"""Run a non-interactive demo: extract features via LLM handler and predict.
Prints JSON array of results to stdout.
"""
import os
import json
import argparse
import sys
# --- ADDED IMPORTS FOR FILE PROCESSING ---
from pypdf import PdfReader
from PIL import Image
import easyocr
import numpy as np
# -----------------------------------------

# Make running this script directly (python backend/artifacts/run_demo.py)
# work without requiring the user to set PYTHONPATH. Insert the repo root
# and the backend folder into sys.path so both package-style and bare
# imports used in the project resolve correctly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
backend_dir = os.path.join(repo_root, 'backend')
for _p in (repo_root, backend_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import app as demo_app
from backend import integrations

# Ensure we use the local mock provider unless explicitly overridden
os.environ.setdefault('LLM_PROVIDER', 'ollama')
os.environ.setdefault('LOCAL_LLM_URL', 'http://localhost:11434')

# --- ADDED TEXT EXTRACTION FUNCTION ---
def extract_text_from_file(file_path: str) -> str:
    """Extracts text from a PDF, PNG, or JPG file given its path."""
    text_content = ""
    file_type = os.path.splitext(file_path)[1].lower()
    
    # ⚠️ NOTE: This assumes the files exist on the file system where this script runs.
    
    if file_type == '.pdf':
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}", file=sys.stderr)

    elif file_type in ['.png', '.jpg', '.jpeg']:
        try:
            image = Image.open(file_path)
            # Use EasyOCR to read text from the image
            image_np = np.array(image)
            # Initialize Reader only once if possible, but for a demo this is fine
            reader = easyocr.Reader(['en'], verbose=False) 
            result = reader.readtext(image_np, detail=0)
            text_content = " ".join(result)
        except Exception as e:
            print(f"Error reading Image {file_path}: {e}", file=sys.stderr)
            
    return text_content
# --------------------------------------


def main(mock: bool = False):
    df = demo_app.load_demo_data('Demo A')
    results = []
    
    # Define the column containing the path to the document/file
    document_column = 'document_path' # **ADJUST THIS TO MATCH YOUR DATAFRAME**

    for _, r in df.iterrows():
        row = r.to_dict()
        
        # --- MODIFICATION: EXTRACT TEXT BEFORE RUNNING INTEGRATION ---
        document_path = row.get(document_column)
        if document_path and os.path.exists(document_path):
            extracted_text = extract_text_from_file(document_path)
            # Crucially, replace the original document path/reference with the extracted text
            # This is how the LLM handler gets the data to analyze.
            # Assuming 'text_to_analyze' is the key your LLM handler expects.
            row['text_to_analyze'] = extracted_text
        # -------------------------------------------------------------
        
        # Now, run feature extraction on the row containing the new text
        res = integrations.run_feature_extraction(row, mock=mock)
        
        feats = res.get('features', {})
        feats['_parsed'] = res.get('parsed', {})
        
        pred = integrations.predict(feats)
        
        merged = {**feats, **pred}
        merged['id'] = row.get('id')
        results.append(merged)
        
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run headless demo')
    parser.add_argument('--mock', action='store_true', help='Use local canned mock responses instead of calling an LLM')
    args = parser.parse_args()
    main(mock=args.mock)
