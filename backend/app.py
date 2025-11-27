import time
from typing import Optional
import os
import sys
import importlib.util
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")

# --- NEW: WELCOME DIALOG FUNCTION ---
def show_onboarding_guide():
        """Render the onboarding content into the current Streamlit container.

        This function is intentionally renderer-agnostic: it can be called inside
        a `st.modal()` context (preferred) or any container. Buttons update
        `st.session_state` to control visibility.
        """

        # Render a boxed welcome panel using HTML for a clear, framed layout
        try:
                from streamlit.components.v1 import html as st_html

                boxed_html = """
                <div style="width:100%;max-width:1200px;margin:6px auto;padding:18px;border-radius:12px;background:#f7fbff;border:1px solid #d8ecff;box-sizing:border-box;font-family: 'Segoe UI', Roboto, Arial, sans-serif;">
                    <h2 style="margin:0 0 8px 0;font-weight:700;color:#0f1724;font-size:22px;">Welcome — LLM-Based Credit Risk Assessment Prototype</h2>
                    <p style="margin:6px 0 14px 0;color:#0f1724;line-height:1.6;font-size:15px;">This prototype demonstrates how Large Language Models (LLMs) can complement traditional credit scoring by combining structured financial fields with behavioral insights extracted from unstructured text — for example, loan applications, customer messages, and transaction descriptions.</p>

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
                            <li><strong>Upload data</strong> — Use the sidebar to add CSV files (structured fields) or PDF/PNG/JPG documents (unstructured text; OCR applied when available).</li>
                            <li><strong>Run analysis</strong> — Click <em>Run Model</em> to process structured fields and analyze unstructured text with the LLM.</li>
                            <li><strong>Review results</strong> — Select an applicant to view the risk score, extracted behavioral indicators, transparent LLM explanations, and highlighted supporting text evidence.</li>
                        </ol>
                    </div>

                    <p style="margin:6px 0 8px 0;font-size:14px;color:#0b2233;"><strong>Supported file types:</strong> CSV, PDF, PNG, JPG</p>

                    <div style="margin-bottom:8px;">
                        <h4 style="margin:6px 0 6px 0;font-size:16px;color:#0b2233;">Purpose of this prototype</h4>
                        <p style="margin:6px 0 8px 0;color:#0f1724;line-height:1.5;font-size:14px;">Illustrate how LLMs can improve contextual understanding in credit risk evaluation, increase transparency by exposing model reasoning and textual evidence, and assist lenders in making fairer, more explainable decisions.</p>
                    </div>

                    <p style="margin-top:10px;font-size:13px;color:#374151;"><strong>Privacy note:</strong> This prototype is for demonstration and research. Avoid uploading sensitive personal data unless you control the dataset and environment.</p>
                </div>
                """

                # Render the boxed HTML with a more compact height and enable scrolling so content isn't cut off
                # Reduced height prevents large whitespace between the welcome and the main title.
                st_html(boxed_html, height=380, scrolling=True)
        except Exception:
                # Fallback to Streamlit native rendering if components aren't available
                st.markdown("## Welcome — LLM-Based Credit Risk Assessment Prototype")
                st.write(
                        "This prototype demonstrates how Large Language Models (LLMs) can complement "
                            "traditional credit scoring by combining structured financial fields with behavioral "
                            "insights extracted from unstructured text such as loan applications, customer messages, and "
                            "transaction descriptions."
                )

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


def extract_text_from_file(uploaded_file) -> str:
    """Try to extract text from uploaded files.

    - For PDFs: use `pdfplumber` when available.
    - For images: use `pytesseract` + `Pillow` when available.
    - Fallback: attempt to decode raw bytes or return a short placeholder.
    """
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
                    return (content[:2000] + "...") if len(content) > 2000 else content
        except Exception:
            # If pdfplumber fails on this file, continue to other methods
            pass
    except Exception:
        # pdfplumber not installed; skip PDF parsing
        pass

    # Try image OCR
    try:
        from PIL import Image
        import pytesseract

        # Auto-detect tesseract binary if available (useful on Windows)
        def _find_tesseract_cmd() -> str | None:
            # 1) In PATH
            p = shutil.which("tesseract")
            if p:
                return p
            # 2) Common Windows install locations
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            for c in candidates:
                if Path(c).exists():
                    return c
            return None

        try:
            tcmd = _find_tesseract_cmd()
            if tcmd:
                pytesseract.pytesseract.tesseract_cmd = tcmd

            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                return (text[:2000] + "...") if len(text) > 2000 else text
        except Exception:
            pass
    except Exception:
        # OCR deps not installed; skip image OCR
        pass

    # Fallback: try to decode bytes or return placeholder
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
        return (content[:300] + "...") if len(content) > 300 else content
    return f"[Extracted text unavailable for {name}]"


def parse_fields_from_text(text: str, filename: str = "") -> dict:
    """Extract common applicant fields from document text using regex heuristics.

    Looks for patterns like:
    - Applicant Name: <name>
    - Applicant Age: <digits>
    - Annual Household Income: $<number>
    - Requested Loan Amount: $<number>

    Returns a dict with keys: `name`, `age`, `income`, `requested_loan`, `text_notes`.
    """
    import re

    out = {
        "name": None,
        "age": None,
        "income": None,
        "requested_loan": None,
        "text_notes": text,
    }

    if not text:
        return out

    # Normalize whitespace
    t = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # Name: look for 'Applicant Name:' or 'Name:' prefixes
    m = re.search(r"Applicant Name:\s*(.+?)(?:\n|Applicant Age:|Applicant|$)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bName:\s*(.+?)(?:\n|$)", t, flags=re.IGNORECASE)
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

    # If name missing, fallback to filename as a human-friendly name
    if not out.get("name") and filename:
        out["name"] = os.path.splitext(os.path.basename(filename))[0]

    return out

def main() -> None:
    # --- NEW: TRIGGER ONBOARDING (show above the main title) ---
    if "first_visit" not in st.session_state:
        st.session_state["first_visit"] = True
    if "seen_welcome" not in st.session_state:
        st.session_state["seen_welcome"] = False
    if "dont_show_welcome" not in st.session_state:
        st.session_state["dont_show_welcome"] = False

    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    # If onboarding should show, render it inline at the top (above title)
    if st.session_state["first_visit"] and not st.session_state.get("dont_show_welcome", False) and not st.session_state.get("seen_welcome", False):
        show_onboarding_guide()
        # Mark that we've shown it once this session so uploads/actions won't re-open it
        st.session_state['first_visit'] = False

    st.title("LLM Credit Risk — Demo UI")

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")

        # --- File uploader (bigger visual area) ---
        # Streamlined professional uploader card
        st.markdown(
            """
            <style>
            /* Professional uploader card - broader selectors + !important overrides */
            div[data-testid="stFileUploader"], div[data-testid="stFileUploader"] > div, div[data-testid="stFileUploader"] * {
                box-sizing: border-box !important;
                font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
            }

            div[data-testid="stFileUploader"] > div {
                border: 1px solid #e6eef8 !important;
                border-radius: 12px !important;
                padding: 12px 14px !important;
                text-align: left !important;
                background: #ffffff !important;
                box-shadow: 0 6px 18px rgba(15, 23, 36, 0.04) !important;
                margin-bottom: 12px !important;
            }

            /* Title inside uploader */
            div[data-testid="stFileUploader"] h4, div[data-testid="stFileUploader"] label {
                margin: 0 0 6px 0 !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                color: #0b2233 !important;
            }

            /* Descriptive text */
            div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] .stMarkdown {
                margin: 0 0 8px 0 !important;
                color: #4b5563 !important;
                font-size: 13px !important;
                line-height: 1.45 !important;
            }

            /* Make the drop area visually distinct but subtle */
            div[data-testid="stFileUploader"] input[type="file"], div[data-testid="stFileUploader"] .css-1an6hbn {
                width: 100% !important;
                height: 110px !important;
                opacity: 0.999 !important; /* ensure clickable area */
                cursor: pointer !important;
                border-radius: 10px !important;
                background: transparent !important;
                border: none !important;
            }

            /* Tidy the internal button */
            div[data-testid="stFileUploader"] button, div[data-testid="stFileUploader"] .stButton button {
                margin-top: 8px !important;
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                box-shadow: none !important;
                color: #0b2233 !important;
            }

            /* Sidebar header polish */
            section[data-testid="stSidebar"] h2 {
                font-size: 18px !important;
                color: #0f1724 !important;
                margin-bottom: 6px !important;
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

        demo = st.selectbox("Or choose a demo dataset", ["Demo A", "Demo B"])      
        mock_mode = st.checkbox("Mock mode (no LLM/API)", value=True)
        run_button = st.button("Run Model", type="primary")
        st.markdown("---")
        st.caption("Tip: use the demo dataset for fastest demo flow.")

    # Load data: prefer CSV if provided, otherwise build from uploaded docs or demo
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
            # If user provided multiple files, give them incremental numeric ids
            for f in uploaded_files:
                text = extract_text_from_file(f)
                parsed = parse_fields_from_text(text, getattr(f, 'name', ''))
                # Store extracted text in 'text_notes' for feature extraction to analyze
                full_text = parsed.get('text_notes') or text
                row = {
                    'id': next_id,
                    'name': parsed.get('name') or getattr(f, 'name', '') or f"applicant_{next_id}",
                    'age': parsed.get('age'),
                    'income': parsed.get('income'),
                    'requested_loan': parsed.get('requested_loan'),
                    'credit_score': None,
                    'text_notes': full_text,  # Full text for LLM analysis
                    'text_preview': full_text[:200] + '...' if len(full_text) > 200 else full_text,  # Short preview for UI
                }
                rows.append(row)
                next_id += 1
            df = pd.DataFrame(rows)
    else:
        df = load_demo_data(demo)

    # Top KPI cards
    k1, k2, k3 = st.columns([1, 1, 1])
    ui_helpers.kpi_card(k1, "Applicants", len(df))
    # Safely compute average income: handle missing or non-numeric values
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
        # Use a selectbox that lists available applicant ids as strings to avoid
        # numeric casting issues when IDs are non-integer (uploaded CSVs).
        id_options = []
        try:
            if not df.empty and 'id' in df.columns:
                id_options = [str(x) for x in df['id'].tolist()]
        except Exception:
            id_options = []
        if not id_options:
            id_options = ["0"]
        selected_id_str = st.selectbox("Select applicant id", id_options, index=0)

    # Determine active dataframe (with or without scores)
    if st.session_state["model_results"] is not None:
        active_df = st.session_state["model_results"]
    else:
        active_df = df
        
    st.markdown("**Local Explanation**")
    
    # Find the selected row safely (match by string representation to be robust)
    selected_row = pd.DataFrame()
    try:
        if not active_df.empty and 'id' in active_df.columns:
            # match on stringified id values to avoid int/str mismatches
            selected_row = active_df[active_df['id'].astype(str) == str(selected_id_str)]
    except Exception:
        selected_row = pd.DataFrame()

    if not selected_row.empty:
        try:
            # Use iloc[0] on the filtered result
            r = selected_row.iloc[0]
            
            # Prefer any already-computed summary field
            if isinstance(r.get('summary'), str) and r.get('summary').strip():
                summary_text = r.get('summary')
                parsed = r.get('_parsed', {}) if isinstance(r.get('_parsed', {}), dict) else {}
            else:
                # If summary is missing, call run_feature_extraction with current mock_mode setting
                # to provide a preview (mock=True gives fast canned output, mock=False calls real LLM)
                try:
                    ext = integrations.run_feature_extraction(r.to_dict(), mock=mock_mode)
                    # Safely extract parsed dict and ensure it's actually a dict
                    parsed = ext.get('parsed', {}) if isinstance(ext.get('parsed', {}), dict) else {}
                    feats = ext.get('features', {}) if isinstance(ext.get('features', {}), dict) else {}
                    summary = parsed.get('summary', {}) if isinstance(parsed, dict) else {}
                    summary_text = summary.get('summary') if isinstance(summary, dict) else None
                except Exception:
                    # Defensive: if extraction fails for this applicant, use empty fallbacks
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
            # --- DISPLAY DASHBOARD (ENHANCED) ---
            st.markdown("#### ➤ Applicant Profile Summary")
            # If we have a computed summary, show it; otherwise fall back to brief text
            st.info(summary_text or "No summary available.")

            # Compose improved Key Risk Signals with mini-explanations
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

            # Mini explanations block (transparent interpretability)
            st.markdown("**Explanations**")
            # Risk score explanation
            try:
                risk_expl = "The model estimated a relative risk score based on detected behavioral signals, sentiment, and structured financial hints."
                if rnum is not None:
                    risk_expl = f"📌 Risk Score Explanation: The model estimated a {float(rnum):.2f} relative risk based on stable income, liabilities, and repayment history."
                st.caption(risk_expl)
            except Exception:
                pass

            # Sentiment explanation
            try:
                sent_expl = "📌 Sentiment Explanation: Narrative sentiment indicates borrower tone and repayment intent."
                if sent_score is not None:
                    sent_expl = f"📌 Sentiment Explanation: Sentiment is {float(sent_score):+.2f}, indicating {'positive' if sent_score>0 else 'neutral' if sent_score==0 else 'negative'} tone and repayment intent."
                st.caption(sent_expl)
            except Exception:
                pass

            # Risk flags explanation
            try:
                if count == 0:
                    flags_expl = "📌 Risk Flags Explanation: No behavioural red flags detected."
                else:
                    flags_expl = f"📌 Risk Flags Explanation: Detected {count} flagged behaviour(s) — {risky_text}."
                st.caption(flags_expl)
            except Exception:
                pass

            if risky_text and risky_text != "None":
                st.caption("🚩 **Detected Risk Phrases:**")
                st.warning(risky_text)

            st.markdown("---")
            st.markdown("#### ➤ Final Recommendation")

            # Render a more professional recommendation block with consistent phrasing
            if rnum is None:
                st.warning("⚠️ **Model has not been run.** Click 'Run Model' to see scores.")
                recommendation = "Run model to see recommendation."
            else:
                score_label = f"{float(rnum):.2f}"
                if rnum >= 0.7:
                    st.error(f"**🔴 DECLINE / MANUAL REVIEW — High Risk ({score_label})**\n\nThis applicant exhibits multiple high-risk signals. Recommendation: escalate to senior underwriter and request comprehensive documentation.")
                    recommendation = "Decline / Manual Review"
                elif rnum >= 0.4:
                    st.warning(f"**🟡 CONDITIONAL APPROVAL — Moderate Risk ({score_label})**\n\nApplicant may qualify subject to additional verification (income, bank statements). Recommendation: request documents and re-assess before funding.")
                    recommendation = "Conditional Approval"
                else:
                    st.success(f"**🟢 APPROVE — Low Risk ({score_label})**\n\nApplicant meets criteria for approval. Income stability, low liabilities, and strong repayment history indicate high reliability. Proceed with automated approval under standard terms.")
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
                        # Extract features: pass mock_mode to control LLM usage (mock=True uses canned outputs)
                        try:
                            res = integrations.run_feature_extraction(row.to_dict(), mock=mock_mode)
                        except Exception:
                            # Defensive: if extraction fails, use fallback empty features
                            res = {"features": {}, "parsed": {}}
                        features = res.get("features", {})
                        # Ensure applicant_id exists so downstream dataframe merge is robust
                        try:
                            if 'applicant_id' not in features:
                                # prefer explicit 'id' from the source row
                                if 'id' in row:
                                    features['applicant_id'] = row.get('id')
                                else:
                                    features['applicant_id'] = i
                        except Exception:
                            features['applicant_id'] = i
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
