import time
from typing import Optional
import os
import importlib.util
import pandas as pd
import streamlit as st

import ui_helpers
import integrations

st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")

# --- NEW: WELCOME DIALOG FUNCTION ---
@st.dialog("👋 Welcome to LLM Credit Risk")
def show_onboarding_guide():
    st.markdown("""
    **Transform raw application text into credit risk insights.**
    
    This tool uses AI to analyze loan applicants by combining their **financial data** (income, age) with **unstructured text** (interview notes, essays).
    
    ### 🚀 How to use this app:
    1.  **Upload Data:** Drag & drop your CSV file in the sidebar.
    2.  **Run Model:** Click the "Run Model" button to extract risk signals.
    3.  **Analyze:** Click on specific applicants to see why they were flagged.
    
    ### 📂 CSV Format Requirements:
    Your CSV file must have these columns:
    * `id` (unique number)
    * `name` (applicant name)
    * `income` (number)
    * `text_notes` (The text you want the AI to analyze)
    """)
    if st.button("I understand, let's start!"):
        st.session_state["first_visit"] = False
        st.rerun()

@st.cache_data
def load_demo_data(name: str) -> pd.DataFrame:
    """Return a tiny demo applicants dataframe based on a name."""
    if name == "Demo B":
        data = [
            {"id": 201, "name": "Alice B.", "age": 29, "income": 48000, "text_notes": "Steady job, but mentioned gambling debt in passing."},
            {"id": 202, "name": "Bob B.", "age": 46, "income": 62000, "text_notes": "Excellent references, no debt, very consistent saver."},
        ]
    else:
        data = [
            {"id": 101, "name": "Alice A.", "age": 31, "income": 52000, "text_notes": "Applicant is positive about repayment but has multiple small loans."},
            {"id": 102, "name": "Bob A.", "age": 45, "income": 72000, "text_notes": "Contradictory statements about employment history. High income but erratic spending."},
            {"id": 103, "name": "Carol A.", "age": 38, "income": 31000, "text_notes": "Low income but very stable job tenure. No risk flags detected."},
        ]
    return pd.DataFrame(data)

# --- NEW: TEMPLATE GENERATOR ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def main() -> None:
    st.title("LLM Credit Risk — Demo UI")
    
    # --- NEW: TRIGGER ONBOARDING ---
    if "first_visit" not in st.session_state:
        st.session_state["first_visit"] = True
    
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    # Show the dialog only if it's the first visit
    if st.session_state["first_visit"]:
        show_onboarding_guide()

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")
        
        # --- NEW: HELP & TEMPLATE SECTION ---
        with st.expander("ℹ️ Help & CSV Template"):
            st.write("Your CSV needs these columns:")
            st.code("id, name, income, text_notes")
            
            # Create a sample template for download
            sample_data = pd.DataFrame([
                {"id": 1, "name": "John Doe", "income": 50000, "age": 30, "text_notes": "Customer has good history."}
            ])
            csv_template = convert_df_to_csv(sample_data)
            
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_template,
                file_name="credit_risk_template.csv",
                mime="text/csv",
            )
        # ------------------------------------

        uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])
        demo = st.selectbox("Or choose a demo dataset", ["Demo A", "Demo B"])
        mock_mode = st.checkbox("Mock mode (no LLM/API)", value=True)
        run_button = st.button("Run Model", type="primary")
        st.markdown("---")
        st.caption("Tip: use the demo dataset for fastest demo flow.")

    # Load data
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = pd.DataFrame()
    else:
        df = load_demo_data(demo)

    # Top KPI cards
    k1, k2, k3 = st.columns([1, 1, 1])
    ui_helpers.kpi_card(k1, "Applicants", len(df))
    ui_helpers.kpi_card(k2, "Avg Income", f"${int(df['income'].mean()):,}" if not df.empty else "—")
    
    # Calculate High Risk % from saved results if available
    if st.session_state["model_results"] is not None:
         res_df = st.session_state["model_results"]
         if 'risk_label' in res_df.columns:
             high_pct = (res_df['risk_label'] == 'high').mean() * 100
             ui_helpers.kpi_card(k3, "High Risk (%)", f"{high_pct:.1f}%")
         else:
             ui_helpers.kpi_card(k3, "High Risk (%)", "—")
    else:
        ui_helpers.kpi_card(k3, "High Risk (%)", "—")


    # --- LAYOUT UPDATE: STACKED SECTIONS ---
    
    # 1. Applicant Table (Full Width)
    st.subheader("Applicant Table")
    display_df = st.session_state["model_results"] if st.session_state["model_results"] is not None else df
    ui_helpers.render_table(display_df)

    st.markdown("---")

    # 2. Explanations (Full Width below Table)
    st.subheader("Explanations & Story")

    # Put the ID selector in a smaller column so it doesn't stretch
    col_sel, col_space = st.columns([1, 3])
    with col_sel:
        # Ensure we don't crash if df is empty
        default_id = int(df['id'].iat[0]) if not df.empty else 0
        selected_id = st.number_input("Select applicant id", min_value=0, value=default_id)

    # Determine active dataframe (with or without scores)
    if st.session_state["model_results"] is not None:
        active_df = st.session_state["model_results"]
    else:
        active_df = df
        
    st.markdown("**Local Explanation**")
    
    # Find the selected row safely
    selected_row = pd.DataFrame()
    if not active_df.empty and 'id' in active_df.columns:
            selected_row = active_df[active_df['id'] == int(selected_id)]

    if not selected_row.empty:
        try:
            # Use iloc[0] on the filtered result
            r = selected_row.iloc[0]
            
            # Prefer any already-computed summary field
            if isinstance(r.get('summary'), str) and r.get('summary').strip():
                summary_text = r.get('summary')
                parsed = r.get('_parsed', {}) if isinstance(r.get('_parsed', {}), dict) else {}
            else:
                # If mock_mode is on, run a fast mock extraction for a better preview
                if mock_mode:
                    try:
                        ext = integrations.run_feature_extraction(r.to_dict(), mock=True)
                        parsed = ext.get('parsed', {}) if isinstance(ext.get('parsed', {}), dict) else {}
                        feats = ext.get('features', {}) if isinstance(ext.get('features', {}), dict) else {}
                        summary = parsed.get('summary', {}) if isinstance(parsed, dict) else {}
                        summary_text = summary.get('summary') if isinstance(summary, dict) else None
                    except Exception:
                        parsed = {}
                        summary_text = None
                else:
                    parsed = {}
                    summary_text = None

            # Build a local explanation from available signals
            sent_score = None
            if 'sentiment_score' in r:
                sent_score = r.get('sentiment_score')
            elif isinstance(parsed.get('sentiment'), dict):
                sent_score = parsed.get('sentiment', {}).get('score')

            risky_val = r.get('risky_phrases') or r.get('risky_phrases_list') or (parsed.get('extract_risky', {}) or {}).get('risky_phrases') or []
            if isinstance(risky_val, (list, tuple)):
                risky_text = ", ".join(map(str, risky_val))
            else:
                risky_text = str(risky_val) if risky_val else "None"

            # risk_score may not be present before run; try fallback fields
            risk_score = r.get('risk_score') or r.get('score') or (feats.get('risk_score') if 'feats' in locals() else None)
            
            # Simple recommendation heuristic
            try:
                rnum = float(risk_score) if risk_score is not None else None
            except Exception:
                rnum = None
            
            # --- DISPLAY DASHBOARD ---
            st.markdown("#### ➤ Applicant Profile Summary")
            st.info(summary_text or "No summary available.")

            st.markdown("#### ➤ Key Risk Signals")
            col1, col2, col3 = st.columns(3)
            with col1:
                if risk_score is not None:
                        st.metric("Risk Score", f"{float(risk_score):.2f}", delta="-High" if float(risk_score) > 0.5 else "Low", delta_color="inverse")
                else:
                        st.metric("Risk Score", "—")
            with col2:
                st.metric("Sentiment", f"{float(sent_score):.2f}" if sent_score is not None else "N/A")
            with col3:
                count = len(risky_val) if isinstance(risky_val, list) else 0
                st.metric("Risk Flags", count, delta="Flags" if count > 0 else "Clean", delta_color="inverse")

            if risky_text and risky_text != "None":
                st.caption("🚩 **Detected Risk Phrases:**")
                st.warning(risky_text)
            
            st.markdown("---")
            st.markdown("#### ➤ Recommendation")
            
            if rnum is None:
                st.warning("⚠️ **Model has not been run.** Click 'Run Model' to see scores.")
                recommendation = "Run model to see recommendation."
            elif rnum >= 0.7:
                st.error(
                    f"**🔴 DECLINE / MANUAL REVIEW**\n\n"
                    f"This applicant is flagged as **High Risk** ({int(rnum*100)}%). "
                    "Recommendation: Escalate to senior underwriter immediately."
                )
                recommendation = "Decline / Manual Review"
            elif rnum >= 0.4:
                st.warning(
                    f"**🟡 CONDITIONAL APPROVAL**\n\n"
                    f"This applicant is **Medium Risk** ({int(rnum*100)}%). "
                    "Recommendation: Request additional documentation (payslips/bank statements)."
                ) 
                recommendation = "Conditional Approval"
            else:
                st.success(
                    f"**🟢 APPROVE**\n\n"
                    f"This applicant is **Low Risk** ({int(rnum*100)}%). "
                    "Recommendation: Proceed with standard automated approval."
                )
                recommendation = "Approve"

        except Exception as e:
            st.error(f"Error displaying details: {e}")
            recommendation = "Error"
    else:
        st.info("Select a valid applicant ID to see details.")
        recommendation = "None"

    st.markdown("---")
    st.markdown("**Story Playback**")
    pcol1, pcol2, pcol3 = st.columns([1, 1, 2])
    prev_clicked = pcol1.button("◀ Prev")
    play_clicked = pcol2.button("Play")
    # placeholder progress widget for playback actions
    progress_placeholder = st.empty()
    progress_placeholder.progress(0)

    # Story Playback Logic
    if play_clicked and 'summary_text' in locals():
        steps = []
        steps.append(("Summary", summary_text))
        steps.append(("Local Explanation", recommendation))
        steps.append(("Risky Phrases", risky_text or "None"))
        sent_text = f"score={sent_score}" if sent_score is not None else "unknown"
        steps.append(("Sentiment", sent_text))

        for i, (title, body) in enumerate(steps, start=1):
            st.markdown(f"**Step {i}: {title}**")
            # Simple highlight
            st.info(body)
            progress_placeholder.progress(int(i / len(steps) * 100))
            time.sleep(0.4)
        progress_placeholder.progress(100)

    # Trigger model run
    if run_button:
        # Run combined pipeline: LLM feature extraction -> prediction
        if df is None or df.empty:
            st.error("No applicants to score. Upload a CSV or choose a demo dataset.")
        else:
            try:
                with st.spinner("Running model and LLM explainers..."):
                    progress = st.progress(0)
                    rows = []
                    features_list = []
                    total = len(df)
                    for i, (_idx, row) in enumerate(df.iterrows(), start=1):
                        progress.progress(int((i - 1) / max(1, total) * 100))
                        # extract features (may call LLM)
                        res = integrations.run_feature_extraction(row.to_dict(), mock=mock_mode)
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

                    # Predict for each extracted feature set
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
