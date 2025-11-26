import time
from typing import Optional
import os
import sys
import importlib.util
import pandas as pd
import streamlit as st
# --- ADDED GEMINI IMPORTS ---
from google import genai
from google.genai.errors import APIError
# ----------------------------

# Ensure backend and project root are on sys.path so imports work when
# Streamlit runs the script directly (prevents ModuleNotFoundError for
# `ui_helpers` or `backend.ui_helpers`). This is robust across local
# and hosted environments.
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    import ui_helpers
except Exception:
    # fallback: try package import
    try:
        from backend import ui_helpers
    except Exception:
        # Last resort: load module from file path
        spec = importlib.util.spec_from_file_location("ui_helpers", os.path.join(_THIS_DIR, "ui_helpers.py"))
        ui_helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui_helpers)
import integrations
import shutil
from pathlib import Path

# --- ADDED: GEMINI CLIENT INITIALIZATION ---

@st.cache_resource
def get_gemini_client():
    """Initializes and caches the Gemini client securely using Streamlit secrets."""
    # Check if the API key is configured in the Streamlit secrets file
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ Google Gemini API Key not found!")
        st.markdown(
            "Please add your API key to `.streamlit/secrets.toml` with the key "
            "`GEMINI_API_KEY` or configure it in the Streamlit Cloud Secrets panel."
        )
        return None
    
    api_key = st.secrets["GEMINI_API_KEY"]
    
    try:
        # The client will be configured to use this API key.
        client = genai.Client(api_key=api_key)
        
        # Optional: Test connectivity to cache the check result
        # Note: This is a fast way to check if the key is valid once per session.
        # client.models.list()
        
        return client
    except Exception as e:
        st.error(f"❌ Gemini Client Initialization Failed: {e}")
        return None

# --- NEW: Check if the client is available and provide feedback ---

def check_llm_status(client: Optional[genai.Client]) -> bool:
    """Displays connection status in the sidebar."""
    if client is None:
        st.sidebar.error("LLM Status: Disconnected (Missing Key)")
        return False
        
    try:
        # Try a quick call to ensure the key is working and the service is reachable
        # Note: This will execute once per session due to st.cache_resource if placed there, 
        # but here it serves as a live check.
        _ = client.models.list_models()
        st.sidebar.success("LLM Status: Connected (Gemini API)")
        return True
    except APIError as e:
        if 'API_KEY_INVALID' in str(e):
            st.sidebar.error("LLM Status: Key Invalid")
        else:
            st.sidebar.error(f"LLM Status: Connection Error ({e})")
        return False
    except Exception as e:
        st.sidebar.error(f"LLM Status: Failed to connect ({e})")
        return False


# --- (Original code continues below) ---

st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")

# ... (show_onboarding_guide, load_demo_data, convert_df_to_csv functions are unchanged)

def extract_text_from_file(uploaded_file) -> str:
# ... (function body is unchanged, as the OCR logic is separate from the API call)
    
# ... (main function starts)

def main() -> None:
    # ... (session state initialization is unchanged)
    
    # ... (welcome dialog logic is unchanged)

    st.title("LLM Credit Risk — Demo UI")

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")
        
        # --- NEW: LLM Status check display ---
        gemini_client = get_gemini_client()
        llm_connected = check_llm_status(gemini_client)
        st.markdown("---")
        # -------------------------------------

        # ... (file uploader styling and logic is unchanged)
        
        # ... (uploaded_files logic is unchanged)
        
        st.markdown("---")
        st.subheader("Applicant scope")
        applicant_scope = st.selectbox("Who are you uploading?", ["Individuals", "Businesses"], index=0)

        demo = st.selectbox("Or choose a demo dataset", ["Demo A", "Demo B"])      
        
        # --- MODIFICATION: Mock mode now only controls the *prediction* part if LLM is connected ---
        # If the LLM connection fails, we should force mock mode
        mock_mode = st.checkbox("Mock mode (no LLM/API)", value=not llm_connected) 
        if not llm_connected and not mock_mode:
            st.warning("Forcing Mock Mode because the Gemini API is disconnected.")
            mock_mode = True 
        # -----------------------------------------------------------------------------------------
        
        run_button = st.button("Run Model", type="primary")
        st.markdown("---")
        st.caption("Tip: use the demo dataset for fastest demo flow.")
        
    # ... (data loading logic is unchanged)
    
    # ... (KPI cards are unchanged)
    
    # ... (Applicant Table and Explanations sections are unchanged)

    # ... (Story Playback Logic is unchanged)
    
    # Trigger model run
    if run_button:
        # Run combined pipeline: LLM feature extraction -> prediction
        if df is None or df.empty:
            st.error("No applicants to score. Upload a CSV or choose a demo dataset.")
        else:
            if not mock_mode and gemini_client is None:
                st.error("Cannot run in live mode: Gemini client failed to initialize. Check your API key.")
                return # Stop the run
                
            try:
                with st.spinner("Running model and LLM explainers..."):
                    # ... (rest of the run_button logic is unchanged)
                    # Note: The actual LLM call happens inside integrations.run_feature_extraction(row.to_dict(), mock=mock_mode)
                    # You must now update that function in the `integrations.py` file to accept the `gemini_client` 
                    # or configure the API key globally before it runs.
                    
                    progress = st.progress(0)
                    rows = []
                    features_list = []
                    total = len(df)
                    
                    for i, (_idx, row) in enumerate(df.iterrows(), start=1):
                        progress.progress(int((i - 1) / max(1, total) * 100))
                        
                        # The LLM call in `integrations.run_feature_extraction` needs to be updated 
                        # to use the Gemini client instead of Ollama/localhost when `mock=False`.
                        
                        # extract features (may call LLM)
                        # NOTE: The implementation of `run_feature_extraction` in `integrations.py` 
                        # must be changed to use `gemini_client` when `mock=False`.
                        res = integrations.run_feature_extraction(row.to_dict(), mock=mock_mode) 
                        
                        # ... (rest of the loop is unchanged)
                        
                        features = res.get("features", {})
                        # keep parsed for UI explanations
                        parsed = res.get("parsed", {})
                        features["_parsed"] = parsed
                        
                        # normalize parsed into flat fields for display and merging
                        try:
                            norm = integrations.expand_parsed_to_fields(parsed)
                        except Exception:
                            norm = {}
                        
                        # copy normalized known fields into features (do not overwrite existing numeric features)
                        for k, v in norm.items():
                            # use a slightly different key for phrases list
                            if k == 'risky_phrases':
                                features['risky_phrases_list'] = v
                            else:
                                if k not in features or features.get(k) is None:
                                    features[k] = v
                        features_list.append(features)
                        rows.append(row.to_dict())
                    progress.progress(90)

                    # ... (rest of the prediction and result merging logic is unchanged)

                    preds_rows = []
                    for i, feat in enumerate(features_list, start=1):
                        pred = integrations.predict(feat)
                        merged = {**feat, **pred}
                        preds_rows.append(merged)
                        
                    # convert to DataFrame
                    preds_df = pd.DataFrame(preds_rows)
                    
                    # join on applicant id where possible
                    if "id" in df.columns:
                        preds_df = preds_df.rename(columns={"applicant_id": "id"})
                        # try to align types so merge works (e.g. int vs str)
                        try:
                            preds_df["id"] = preds_df["id"].astype(df["id"].dtype)
                        except Exception:
                            pass
                        out_df = df.merge(preds_df, on="id", how="left")
                    else:
                        out_df = pd.concat([df.reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1)
                    
                    st.session_state["model_results"] = out_df
                    progress.progress(100)
                
                st.success("Completed model run")
                # Force rerun to update all components (tables, KPIs) immediately with the new session state data
                st.rerun()

            except Exception as e:
                st.error(f"Error during run: {e}")

if __name__ == "__main__":
    main()
