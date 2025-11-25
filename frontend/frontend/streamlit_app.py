import streamlit as st
import os
import json
import joblib
from backend.llm_handler import call_llm
from backend.feature_extraction import extract_features

# 1. Page Config
st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")
st.title("🏦 AI Credit Risk Analyzer")

# 2. Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    # This allows you to toggle the provider in the UI if you want
    provider = st.radio("LLM Provider", ["Ollama (Local)", "OpenAI", "Mock"], index=0)
    
    if provider == "Ollama (Local)":
        base_url = st.text_input("Ollama URL", value=os.getenv("LOCAL_LLM_URL", "http://localhost:11434"))
        # Dynamically update the environment variable for the session
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LOCAL_LLM_URL"] = base_url
    elif provider == "OpenAI":
        os.environ["LLM_PROVIDER"] = "openai"
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        # Use mock mode
        pass 

# 3. Main Input Area
st.subheader("Applicant Data")
applicant_text = st.text_area("Paste applicant notes/emails here:", height=150, 
    placeholder="e.g., Applicant owns a small cafe, revenue is consistent but missed one payment in 2022...")

if st.button("Analyze Risk"):
    if not applicant_text:
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Consulting the AI Model..."):
            try:
                # A. Determine if we should mock based on sidebar selection
                use_mock = (provider == "Mock")
                
                # B. Call LLM (using your backend function)
                # Note: We pass mock=use_mock to override the default if selected
                llm_response_str = call_llm(applicant_text, mode="summary", mock=use_mock)
                
                # C. Extract Features
                features = extract_features(llm_response_str, {})
                
                # D. Load Model & Predict
                # Ensure these paths exist in your repo structure
                if os.path.exists("artifacts/model.pkl") and os.path.exists("artifacts/scaler.pkl"):
                    model = joblib.load("artifacts/model.pkl")
                    scaler = joblib.load("artifacts/scaler.pkl")
                    
                    # Prepare vector [sentiment, risky_count, contra_flag, credibility]
                    # (Ensure this order matches your training code exactly)
                    vector = [
                        features["sentiment_score"],
                        features["risky_phrase_count"],
                        features["contradiction_flag"],
                        features["credibility_score"]
                    ]
                    
                    scaled_vector = scaler.transform([vector])
                    risk_score = model.predict_proba(scaled_vector)[0][1]
                    
                    # E. Display Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Risk Score", f"{risk_score:.2%}")
                        if risk_score > 0.5:
                            st.error("High Risk Detected")
                        else:
                            st.success("Low Risk")
                            
                    with col2:
                        st.json(features)
                        
                    with st.expander("View Raw LLM Output"):
                        st.code(llm_response_str)
                else:
                    st.error("Model artifacts not found. Please run training first.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
