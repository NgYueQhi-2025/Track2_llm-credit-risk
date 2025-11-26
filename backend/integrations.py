# --- ASSUMING THIS IS INSIDE integrations.py ---
# Make sure you have the following imports at the top of your integrations.py
import os
import io
import time
from google import genai
from PIL import Image
from typing import Optional, Dict

# Initialize Gemini Client (ensure GEMINI_API_KEY is set in environment or loaded)
try:
    # Use GEMINI_API_KEY from environment or Streamlit secrets, as handled in main.py
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        MODEL_NAME = "gemini-2.5-flash"
    else:
        # Fallback for mock or local testing without key
        client = None
        MODEL_NAME = "mock-model"
except Exception as e:
    # Handle error during client initialization
    client = None
    MODEL_NAME = "error"


def analyze_file_with_gemini(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> str:
    """
    Uploads a file (PDF/Image) to Gemini Files API for processing and
    requests risk analysis on the content.

    Returns the extracted text content or a summary of the analysis.
    """
    if not client:
        return f"[Error: Gemini Client not initialized. Check API Key.]"

    # Define the multimodal prompt for text extraction and risk analysis
    RISK_ANALYSIS_PROMPT = (
        "You are an expert document analyst. Extract ALL readable text content "
        "from the provided document (PDF/Image). After extracting the full text, "
        "identify and list any high-risk financial or behavioral keywords (e.g., "
        "'fraud', 'illegal', 'debt', 'gambling', 'bankruptcy', 'lawsuit', 'eviction'). "
        "Format the output strictly as JSON with two keys: 'full_text' (string) "
        "and 'risky_keywords' (list of strings). If no text is found, set 'full_text' to 'None'."
    )
    
    uploaded_gemini_file = None
    
    # 1. Use the Files API for PDFs and large files
    if mime_type == 'application/pdf':
        try:
            # Upload the file bytes to the Gemini Files API
            file_to_upload = io.BytesIO(file_bytes)
            uploaded_gemini_file = client.files.upload(
                file=file_to_upload,
                mime_type=mime_type
            )
            
            # The prompt contents include the uploaded file object and the instruction text
            contents = [uploaded_gemini_file, RISK_ANALYSIS_PROMPT]
            
        except Exception as e:
            # Handle upload failure
            return f"[File API Upload Error for {file_name}: {e}]"
            
    # 2. Use Direct Part for Images (smaller files can be passed directly)
    elif mime_type.startswith('image/'):
        try:
            # Convert bytes to PIL Image object
            image = Image.open(io.BytesIO(file_bytes))
            contents = [image, RISK_ANALYSIS_PROMPT]
        except Exception as e:
            return f"[Image Processing Error for {file_name}: {e}]"

    else:
        return f"[Analysis Skipped: Unsupported MIME type {mime_type}]"


    # --- Call the Generative Model ---
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json") # Request JSON output
        )
        
        # Attempt to parse the JSON response text
        try:
            import json
            # The model should return text that is valid JSON
            parsed_json = json.loads(response.text)
            full_text = parsed_json.get("full_text", f"[JSON Error: Missing 'full_text' key]")
            
            # Keep only the first 2000 characters for the 'text_notes' field in the DataFrame
            return (full_text[:2000] + "...") if len(full_text) > 2000 else full_text
            
        except json.JSONDecodeError:
            return f"[JSON Parsing Error: Model did not return valid JSON. Raw output: {response.text[:500]}...]"

    except Exception as e:
        return f"[Gemini API Call Error: {e}]"
        
    finally:
        # CRITICAL: Clean up the file from the API service if it was uploaded
        if uploaded_gemini_file:
            client.files.delete(name=uploaded_gemini_file.name)
            # time.sleep(0.5) # Optional: short delay for cleanup
