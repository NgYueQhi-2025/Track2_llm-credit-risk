# --- Imports (Ensure these are at the top of your file) ---
import streamlit as st
import os
from google import genai
from google.genai import types
from PIL import Image
import io
import mimetypes

# ... (The rest of your initial setup for client and MODEL_NAME remains the same)

# --- Updated LLM Logic ---
if uploaded_file is not None:
    # Check the file type
    file_mime_type = uploaded_file.type
    st.image("image_6661fc.png", caption=f"Uploaded Document: {uploaded_file.name}", width=300)
    
    if st.button("Start Risk Analysis"):
        if file_mime_type == 'application/pdf':
            # --- PDF Handling: Use the Files API ---
            with st.spinner(f"Uploading and analyzing PDF: {uploaded_file.name}..."):
                try:
                    # 1. Upload the file using the Files API
                    # The file object must be passed as an io.BytesIO object
                    file_to_upload = io.BytesIO(uploaded_file.read())
                    uploaded_pdf = client.files.upload(
                        file=file_to_upload,
                        display_name=uploaded_file.name
                    )

                    # 2. Call the Gemini API with the file object and prompt
                    # The contents array contains the uploaded file object AND the text prompt
                    full_contents = [uploaded_pdf, RISK_PROMPT]
                    
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=full_contents
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"An error occurred during PDF analysis: {e}")
                finally:
                    # 3. CRITICAL: Delete the file from the API service after use
                    if 'uploaded_pdf' in locals():
                        client.files.delete(name=uploaded_pdf.name)
                        st.info(f"Cleaned up file {uploaded_pdf.name} from the API service.")

        elif file_mime_type in ['image/png', 'image/jpeg', 'image/jpg']:
            # --- Image Handling (simplified, similar to previous working code) ---
            with st.spinner(f"Analyzing Image: {uploaded_file.name}..."):
                try:
                    # Convert Streamlit UploadedFile to PIL Image object
                    image = Image.open(io.BytesIO(uploaded_file.read()))
                    
                    # The contents array is the image object plus the text prompt
                    full_contents = [image, RISK_PROMPT] 
                    
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=full_contents
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"An error occurred during image analysis: {e}")
        else:
            st.error(f"Unsupported file type: {file_mime_type}. Please upload a PDF, PNG, or JPG.")

# Footer for API Key guidance (Keep this)
st.sidebar.info(
    "Set your GEMINI_API_KEY either as an environment variable or in Streamlit Secrets (`.streamlit/secrets.toml`)."
)
