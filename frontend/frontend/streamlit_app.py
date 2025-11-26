import streamlit as st
import os
from google import genai
from google.genai import types
from PIL import Image
import io
import mimetypes

# --- Configuration ---
# The API key must be set as an environment variable (GEMINI_API_KEY) or in Streamlit Secrets
# For local testing, ensure you have: export GEMINI_API_KEY='YOUR_KEY'
# Streamlit Cloud uses st.secrets
if "GEMINI_API_KEY" not in os.environ and "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

try:
    # Initialize the Gemini Client
    client = genai.Client()
    # gemini-2.5-flash is ideal for fast, multimodal tasks
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Error initializing Gemini client. Please check your GEMINI_API_KEY. Details: {e}")
    st.stop()

# --- Utility Function ---
def get_generative_parts(uploaded_file, user_prompt):
    """
    Prepares the multimodal parts for the Gemini API call.
    Converts Streamlit's UploadedFile into a types.Part object.
    """
    
    # 1. Get the raw bytes and MIME type from the Streamlit file
    file_bytes = uploaded_file.read()
    mime_type = uploaded_file.type

    # 2. Handle PDF files (which can be sent directly)
    if mime_type == 'application/pdf':
        media_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type
        )
    
    # 3. Handle image files (using PIL/Pillow for robust handling)
    elif mime_type in ['image/png', 'image/jpeg', 'image/jpg']:
        image = Image.open(io.BytesIO(file_bytes))
        media_part = image
    
    # 4. Handle unsupported files or as a fallback
    else:
        st.warning(f"Unsupported file type: {mime_type}. Treating as text, which may fail.")
        media_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type='text/plain' # Fallback, but likely to fail for complex binary files
        )

    # The contents array is the media part plus the text prompt
    contents = [media_part, user_prompt]
    return contents


# --- Streamlit UI and Logic ---
st.title("📄 Risk Keyword Analyzer with Gemini")
st.markdown("Upload a PDF, PNG, or JPG document for word-by-word risk analysis.")

# Define the risk-scoring system and prompt
RISK_PROMPT = """
You are an expert risk analyst. Your task is to analyze the provided document (PDF/Image) word-by-word.
Perform the following steps:
1. Extract all text content from the document.
2. Identify and list all words or phrases that represent a high risk (e.g., 'fraud', 'illegal', 'breach', 'embezzlement', 'money laundering', 'unauthorized access', 'confidentiality violation').
3. For each high-risk keyword/phrase, quote the full sentence it appears in.
4. Calculate a simple **Risk Score** from 0 to 100 based on the density and severity of the risky content.
5. Present the final output in a clear Markdown format.

**Format your final output STRICTLY as follows:**

## Document Analysis Report
**Total Word Count:** [Number of words in the document]
**Identified High-Risk Keywords:** [List of keywords separated by commas]
**Calculated Risk Score:** [Score]/100 (Explain the score in one sentence)

### Quotations of Risky Content
* Keyword: "..." | Sentence: "..."
* Keyword: "..." | Sentence: "..."
...
"""

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Document (PDF, PNG, JPG/JPEG)", 
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image("image_6661fc.png", caption="Uploaded Document Preview (for image/PDF files)", width=300) 
    
    if st.button("Start Risk Analysis"):
        with st.spinner("Analyzing document with Gemini..."):
            try:
                # 1. Prepare the multimodal prompt
                full_contents = get_generative_parts(uploaded_file, RISK_PROMPT)
                
                # 2. Call the Gemini API
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=full_contents
                )
                
                # 3. Display the result
                st.success("Analysis Complete!")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"An error occurred during the API call: {e}")

# Footer for API Key guidance
st.sidebar.info(
    "Set your GEMINI_API_KEY either as an environment variable or in Streamlit Secrets (`.streamlit/secrets.toml`)."
)
