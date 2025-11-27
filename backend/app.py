import time
from typing import Optional, List
import os
import sys
import importlib.util
import pandas as pd
import streamlit as st
import re # Added explicitly at top level for regex operations

# Ensure backend and project root are on sys.path so imports work when
# Streamlit runs the script directly.
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

st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")

# --- NEW: WELCOME DIALOG FUNCTION ---
def show_onboarding_guide():
        """Render the onboarding content into the current Streamlit container."""
        try:
                from streamlit.components.v1 import html as st_html

                boxed_html = """
                <div style="width:100%;max-width:1200px;margin:6px auto;padding:18px;border-radius:12px;background:#f7fbff;border:1px solid #d8ecff;box-sizing:border-box;font-family: 'Segoe UI', Roboto, Arial, sans-serif;">
                    <h2 style="margin:0 0 8px 0;font-weight:700;color:#0f1724;font-size:22px;">Welcome — LLM-Based Credit Risk Assessment Prototype</h2>
                    <p style="margin:6px 0 14px 0;color:#0f1724;line-height:1.6;font-size:15px;">This prototype demonstrates how Large Language Models (LLMs) can complement traditional credit scoring by combining structured financial fields with behavioral insights extracted from unstructured text.</p>

                    <div style="margin-bottom:10px;">
                        <h4 style="margin:6px 0 6px 0;font-size:16px;color:#0b2233;">What this system does</h4>
                        <ul style="margin:4px 0 10px 20px;color:#0b2233;line-height:1.5;font-size:14px;">
                            <li>Combine quantitative and narrative data into a unified applicant profile.</li>
                            <li>Extract behavioral signals from free-form text (tone, intent, repayment-related cues).</li>
                            <li>Produce an interpretable risk score with human-readable explanations and supporting evidence.</li>
                        </ul>
                    </div>

                    <div style="margin-bottom:10px;">
                        <h4 style="margin:6px 0 6px 0;font-size:16px;color:#0b2233;">How to use the app</h4>
                        <ol style="margin:4px 0 10px 20px;color:#0b2233;line-height:1.5;font-size:14px;">
                            <li><strong>Upload data</strong> — Use the sidebar to add CSV files or PDF/PNG/JPG documents.</li>
                            <li><strong>Run analysis</strong> — Click <em>Run Model</em> to process structured fields and analyze unstructured text with the LLM.</li>
                            <li><strong>Review results</strong> — Select an applicant to view the risk score and extracted behavioral indicators.</li>
                        </ol>
                    </div>
                </div>
                """
                st_html(boxed_html, height=380, scrolling=True)
        except Exception:
                st.markdown("## Welcome — LLM-Based Credit Risk Assessment Prototype")
                st.write("This prototype demonstrates how Large Language Models (LLMs) can complement traditional credit scoring.")

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

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


def extract_text_from_file(uploaded_file) -> str:
    """Try to extract text from uploaded files (PDF/Images/Text)."""
    name = getattr(uploaded_file, "name", "uploaded_file")
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Try PDF parsing first
    try:
        import pdfplumber
        try:
            uploaded_file.seek(0)
            with pdfplumber.open(uploaded_file) as pdf:
                texts = []
                for page in pdf.pages:
                    try:
                        t = page.extract_text() or ""
                        texts.append(t)
                    except Exception:
                        continue
                content = "\n\n".join([t for t in texts if t])
                if content:
                    return (content[:4000] + "...") if len(content) > 4000 else content
        except Exception:
            pass
    except Exception:
        pass

    # Try image OCR
    try:
        from PIL import Image
        import pytesseract

        def _find_tesseract_cmd() -> str | None:
            p = shutil.which("tesseract")
            if p: return p
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for c in candidates:
                if Path(c).exists(): return c
            return None

        try:
            tcmd = _find_tesseract_cmd()
            if tcmd:
                pytesseract.pytesseract.tesseract_cmd = tcmd

            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                return (text[:4000] + "...") if len(text) > 4000 else text
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: try to decode bytes
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            try:
                content = raw.decode("utf-8", errors="ignore")
            except Exception:
                content = None
        else:
            content = str(raw)
    except Exception:
        content = None

    if content:
        return (content[:4000] + "...") if len(content) > 4000 else content
    return f"[Extracted text unavailable for {name}]"


# --- UPDATED PARSING LOGIC ---
def parse_fields_from_text(text: str, filename: str = "") -> List[dict]:
    """
    Extract common applicant fields from document text using regex heuristics.
    Returns a LIST of dictionaries (one per applicant found in the text).
    """
    import re

    if not text:
        return []

    # 1. Split text into segments based on "Applicant Name:" or "Name:"
    # We use a positive lookahead (?=...) to split BUT keep the "Applicant Name" identifier in the chunk
    segments = re.split(r'(?=\b(?:Applicant )?Name:)', text, flags=re.IGNORECASE)

    parsed_applicants = []
    counter = 1

    for segment in segments:
        # Skip empty segments or segments that don't actually contain a name field
        if not re.search(r'\b(?:Applicant )?Name:', segment, flags=re.IGNORECASE):
            continue

        out = {
            "name": None,
            "age": None,
            "income": None,
            "requested_loan": None,
            "text_notes": segment.strip(), # We only store the text relevant to THIS applicant
        }

        # Normalize whitespace for easier regex matching
        t = "\n".join([line.strip() for line in segment.splitlines() if line.strip()])

        # --- EXTRACT FIELDS FROM THIS SPECIFIC SEGMENT ---

        # Name
        m = re.search(r"(?:Applicant )?Name:\s*(.+?)(?:\n|Applicant Age:|Applicant|$)", t, flags=re.IGNORECASE)
        if m:
            out["name"] = m.group(1).strip().rstrip(',')

        # Age
        m = re.search(r"Applicant Age:\s*(\d{1,3})", t, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(\d{1,3})\s+years?\s+old", t, flags=re.IGNORECASE)
        if m:
            try:
                out["age"] = int(m.group(1))
            except Exception:
                out["age"] = None

        # Income
        m = re.search(r"Annual Household Income:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"Income:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
        if m:
            s = m.group(1).replace(',', '')
            try:
                out["income"] = int(s)
            except Exception:
                out["income"] = None

        # Requested loan
        m = re.search(r"Requested Loan Amount:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"Requested Loan:\s*\$?([0-9,]+)", t, flags=re.IGNORECASE)
        if m:
            s = m.group(1).replace(',', '')
            try:
                out["requested_loan"] = int(s)
            except Exception:
                out["requested_loan"] = None

        # Fallback Name if missing
        if not out.get("name") and filename:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            out["name"] = f"{base_name}_{counter}"

        parsed_applicants.append(out)
        counter += 1

    return parsed_applicants

def main() -> None:
    # --- TRIGGER ONBOARDING ---
    if "first_visit" not in st.session_state:
        st.session_state["first_visit"] = True
    if "seen_welcome" not in st.session_state:
        st.session_state["seen_welcome"] = False
    if "dont_show_welcome" not in st.session_state:
        st.session_state["dont_show_welcome"] = False
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    if st.session_state["first_visit"] and not st.session_state.get("dont_show_welcome", False) and not st.session_state.get("seen_welcome", False):
        show_onboarding_guide()
        st.session_state['first_visit'] = False

    st.title("LLM Credit Risk — Demo UI")

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")
        st.markdown(
            """
            <style>
            div[data-testid="stFileUploader"] > div {
                border: 1px solid #e6eef8 !important;
                border-radius: 12px !important;
                padding: 12px 14px !important;
                background: #ffffff !important;
                box-shadow: 0 6px 18px rgba(15, 23, 36, 0.04) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Click to upload or drag and drop — Supported: CSV, PDF, PNG, JPG",
            type=["csv", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        st.subheader("Applicant scope")
        applicant_scope = st.selectbox("Who are you uploading?", ["Individuals", "Businesses"], index=0)

        mock_mode = False
        run_button = st.button("Run Model", type="primary")
        st.markdown("---")

    # --- UPDATED DATA LOADING LOGIC ---
    if 'uploaded_files' in locals() and uploaded_files:
        csvs = [f for f in uploaded_files if str(f.name).lower().endswith('.csv')]
        if csvs:
            try:
                csvs[0].seek(0)
                df = pd.read_csv(csvs[0])
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df = pd.DataFrame()
        else:
            rows = []
            next_id = 1
            # Iterate through every uploaded file
            for f in uploaded_files:
                # 1. Extract ALL text from the file (pages 1-3)
                text = extract_text_from_file(f)
                
                # 2. Parse text into a LIST of applicants
                applicants_found = parse_fields_from_text(text, getattr(f, 'name', ''))
                
                # 3. Loop through EACH applicant found in the single file
                for applicant in applicants_found:
                    full_text = applicant.get('text_notes')
                    
                    row = {
                        'id': next_id,
                        'name': applicant.get('name') or f"applicant_{next_id}",
                        'age': applicant.get('age'),
                        'income': applicant.get('income'),
                        'requested_loan': applicant.get('requested_loan'),
                        'credit_score': None,
                        'text_notes': full_text,
                        'text_preview': full_text[:200] + '...' if full_text and len(full_text) > 200 else (full_text or ""),
                    }
                    rows.append(row)
                    next_id += 1
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()
        st.info("Please upload applicant documents or a CSV to analyze.")

    # Top KPI cards
    k1, k2, k3 = st.columns([1, 1, 1])
    ui_helpers.kpi_card(k1, "Applicants", len(df))
    avg_income_display = "—"
    try:
        if 'income' in df.columns:
            incomes = df['income'].astype(float, errors='ignore') if hasattr(df['income'], 'astype') else df['income']
            incomes = pd.to_numeric(incomes, errors='coerce')
            if incomes.dropna().size > 0:
                avg_income_display = f"${int(incomes.mean()):,}"
    except Exception:
        avg_income_display = "—"
    ui_helpers.kpi_card(k2, "Avg Income", avg_income_display)
    
    if st.session_state["model_results"] is not None:
          res_df = st.session_state["model_results"]
          if 'risk_label' in res_df.columns:
              high_pct = (res_df['risk_label'] == 'high').mean() * 100
              ui_helpers.kpi_card(k3, "High Risk (%)", f"{high_pct:.1f}%")
          else:
              ui_helpers.kpi_card(k3, "High Risk (%)", "—")
    else:
        ui_helpers.kpi_card(k3, "High Risk (%)", "—")

    # 1. Applicant Table (Full Width)
    st.subheader("Applicant Table")
    display_df = st.session_state["model_results"] if st.session_state["model_results"] is not None else df
    ui_helpers.render_table(display_df)

    st.markdown("---")

    # 2. Explanations (Full Width below Table)
    st.subheader("Explanations & Story")

    col_sel, col_space = st.columns([1, 3])
    with col_sel:
        id_options = []
        try:
            if not df.empty and 'id' in df.columns:
                id_options = [str(x) for x in df['id'].tolist()]
        except Exception:
            id_options = []
        if not id_options:
            id_options = ["0"]
        selected_id_str = st.selectbox("Select applicant id", id_options, index=0)

    if st.session_state["model_results"] is not None:
        active_df = st.session_state["model_results"]
    else:
        active_df = df
        
    st.markdown("**Local Explanation**")
    
    selected_row = pd.DataFrame()
    try:
        if not active_df.empty and 'id' in active_df.columns:
            selected_row = active_df[active_df['id'].astype(str) == str(selected_id_str)]
    except Exception:
        selected_row = pd.DataFrame()

    if not selected_row.empty:
        try:
            r = selected_row.iloc[0]
            
            if isinstance(r.get('summary'), str) and r.get('summary').strip():
                summary_text = r.get('summary')
                parsed = r.get('_parsed', {}) if isinstance(r.get('_parsed', {}), dict) else {}
            else:
                try:
                    ext = integrations.run_feature_extraction(r.to_dict(), mock=mock_mode)
                    parsed = ext.get('parsed', {}) if isinstance(ext.get('parsed', {}), dict) else {}
                    feats = ext.get('features', {}) if isinstance(ext.get('features', {}), dict) else {}
                    summary = parsed.get('summary', {}) if isinstance(parsed, dict) else {}
                    summary_text = summary.get('summary') if isinstance(summary, dict) else None
                except Exception:
                    parsed = {}
                    summary_text = None

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

            risk_score = r.get('risk_score') or r.get('score') or (feats.get('risk_score') if 'feats' in locals() else None)
            
            try:
                rnum = float(risk_score) if risk_score is not None else None
            except Exception:
                rnum = None

            st.markdown("#### ➤ Applicant Profile Summary")
            st.info(summary_text or "No summary available.")

            st.markdown("#### ➤ Key Risk Signals")
            col1, col2, col3 = st.columns(3)
            with col1:
                if rnum is not None:
                    st.metric("Risk Score", f"{float(rnum):.2f}", delta=("High" if float(rnum) > 0.5 else "Low"), delta_color="inverse")
                else:
                    st.metric("Risk Score", "—")
            with col2:
                st.metric("Sentiment", f"{float(sent_score):+.2f}" if sent_score is not None else "N/A")
            with col3:
                count = len(risky_val) if isinstance(risky_val, list) else 0
                st.metric("Risk Flags", count, delta=("Flags" if count > 0 else "Clean"), delta_color="inverse")

            # Mini explanations
            try:
                risk_expl = "The model estimated a relative risk score based on detected behavioral signals."
                if rnum is not None:
                    risk_expl = f"📌 Risk Score Explanation: The model estimated a {float(rnum):.2f} relative risk based on stable income, liabilities, and repayment history."
                st.caption(risk_expl)
            except Exception:
                pass

            if risky_text and risky_text != "None":
                st.caption("🚩 **Detected Risk Phrases:**")
                st.warning(risky_text)

            st.markdown("---")
            st.markdown("#### ➤ Final Recommendation")

            if rnum is None:
                st.warning("⚠️ **Model has not been run.** Click 'Run Model' to see scores.")
                recommendation = "Run model to see recommendation."
            else:
                score_label = f"{float(rnum):.2f}"
                if rnum >= 0.7:
                    st.error(f"**🔴 DECLINE / MANUAL REVIEW — High Risk ({score_label})**\n\nThis applicant exhibits multiple high-risk signals.")
                    recommendation = "Decline / Manual Review"
                elif rnum >= 0.4:
                    st.warning(f"**🟡 CONDITIONAL APPROVAL — Moderate Risk ({score_label})**\n\nApplicant may qualify subject to additional verification.")
                    recommendation = "Conditional Approval"
                else:
                    st.success(f"**🟢 APPROVE — Low Risk ({score_label})**\n\nApplicant meets criteria for approval.")
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
    play_clicked = pcol2.button("Play")
    progress_placeholder = st.empty()
    progress_placeholder.progress(0)

    if play_clicked and 'summary_text' in locals():
        steps = []
        steps.append(("Summary", summary_text))
        steps.append(("Local Explanation", recommendation))
        steps.append(("Risky Phrases", risky_text or "None"))
        sent_text = f"score={sent_score}" if sent_score is not None else "unknown"
        steps.append(("Sentiment", sent_text))

        for i, (title, body) in enumerate(steps, start=1):
            st.markdown(f"**Step {i}: {title}**")
            st.info(body)
            progress_placeholder.progress(int(i / len(steps) * 100))
            time.sleep(0.4)
        progress_placeholder.progress(100)

    # Trigger model run
    if run_button:
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
                        try:
                            res = integrations.run_feature_extraction(row.to_dict(), mock=mock_mode)
                        except Exception:
                            res = {"features": {}, "parsed": {}}
                        features = res.get("features", {})
                        try:
                            if 'applicant_id' not in features:
                                if 'id' in row:
                                    features['applicant_id'] = row.get('id')
                                else:
                                    features['applicant_id'] = i
                        except Exception:
                            features['applicant_id'] = i
                        parsed = res.get("parsed", {})
                        features["_parsed"] = parsed
                        try:
                            norm = integrations.expand_parsed_to_fields(parsed)
                        except Exception:
                            norm = {}
                        for k, v in norm.items():
                            if k == 'risky_phrases':
                                features['risky_phrases_list'] = v
                            else:
                                if k not in features or features.get(k) is None:
                                    features[k] = v
                        features_list.append(features)
                        rows.append(row.to_dict())
                    progress.progress(90)

                    preds_rows = []
                    for i, feat in enumerate(features_list, start=1):
                        pred = integrations.predict(feat)
                        merged = {**feat, **pred}
                        preds_rows.append(merged)
                    
                    preds_df = pd.DataFrame(preds_rows)
                    
                    if "id" in df.columns:
                        preds_df = preds_df.rename(columns={"applicant_id": "id"})
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
                st.rerun()

            except Exception as e:
                st.error(f"Error during run: {e}")

if __name__ == "__main__":
    main()
