# app.py — Updated full version with customer-message processing
# Based on user's uploaded file. :contentReference[oaicite:1]{index=1}
#
# New features:
# - Auto-detect customer service messages (e.g., "missed payment", "waive fee")
# - analyze_customer_message() provides summary, sentiment, risky phrases, risk score, and recommendation
# - Integrates results into the existing explanation UI
# - Falls back to integrations.run_customer_message_extraction() if provided in integrations module

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
import re

st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")

# ---------------------------
# New helper functions
# ---------------------------

def is_customer_message(text: str) -> bool:
    """Heuristic: return True if text looks like a customer service message (missed payment, request to waive fee, hospitalization, salary delay, etc.)."""
    if not text or not isinstance(text, str):
        return False
    txt = text.lower()
    customer_indicators = [
        "late fee", "late payment", "missed payment", "waive", "waive the late", "hospital", "hospitalized",
        "salary", "paycheck", "paycheck was delayed", "i can make the full payment", "can you please waive",
        "sorry", "apolog", "payment this friday", "unable to pay", "missed work", "paycheck was delayed"
    ]
    hits = sum(1 for phrase in customer_indicators if phrase in txt)
    # also if text is short and written like a message or email (has "Hi," or "Regards")
    casual_message = bool(re.search(r"\bhi[, ]|\bhello[, ]|\bregards\b|\bthanks\b|\bsincerely\b", txt))
    # treat as customer message if enough indicators or casual message + single late-payment mention
    return (hits >= 1) or (casual_message and ("late" in txt or "pay" in txt))

def simple_sentiment_score(text: str) -> float:
    """Very small heuristic sentiment: counts positive and negative words and returns normalized score [-1,1]."""
    if not text or not isinstance(text, str):
        return 0.0
    text = text.lower()
    positive = {"thank", "thanks", "appreciate", "grateful", "able", "will", "can make", "confident", "responsible", "on time", "paid"}
    negative = {"late", "delay", "delayed", "missed", "sorry", "problem", "unable", "issue", "hospital", "hospitalized", "sick"}
    pos_count = sum(text.count(w) for w in positive)
    neg_count = sum(text.count(w) for w in negative)
    if pos_count + neg_count == 0:
        return 0.0
    score = (pos_count - neg_count) / (pos_count + neg_count)
    # clamp
    if score > 1: score = 1.0
    if score < -1: score = -1.0
    return float(score)

def extract_risky_phrases_from_message(text: str):
    """Return a small list of 'risky phrases' indicating flags (e.g., 'two late payments', 'hospitalized', 'salary delayed')."""
    phrases = []
    txt = text.lower()
    patterns = [
        (r"\b\d+\s+late payments?\b", "recent late payments"),
        (r"\bone late payment\b", "one late payment"),
        (r"\btwo late payments\b", "two late payments"),
        (r"hospitaliz", "medical hospitalization"),
        (r"paycheck (?:was )?delayed", "delayed paycheck"),
        (r"salary (?:was )?delayed", "salary delayed"),
        (r"missed work", "missed work"),
        (r"unable to pay", "unable to pay"),
        (r"waive (?:the )?late fee", "request to waive late fee"),
        (r"request to waive", "request to waive"),
        (r"late fee", "late fee mentioned"),
        (r"temporary income disruption", "temporary income disruption"),
    ]
    for pat, label in patterns:
        if re.search(pat, txt):
            phrases.append(label)
    # dedupe
    return list(dict.fromkeys(phrases))

def analyze_customer_message(text: str, mock: bool = True) -> dict:
    """
    Analyze a customer message and produce:
    - parsed: dict with summary etc.
    - features: dict with risk_score, sentiment_score, risky_phrases, recommendation
    If integrations provides an specialized extractor `run_customer_message_extraction`, prefer it.
    """
    # Prefer integrations' dedicated function if present
    try:
        if hasattr(integrations, "run_customer_message_extraction"):
            return integrations.run_customer_message_extraction(text, mock=mock)
    except Exception:
        # If the integrations function exists but errors, fall back to local
        pass

    # Local fallback analysis (lightweight heuristics)
    sentiment = simple_sentiment_score(text)  # -1..1
    risky_phrases = extract_risky_phrases_from_message(text)
    # Determine base risk score heuristically:
    # Start with low baseline and bump for risky phrases and negative sentiment
    risk = 0.2  # baseline (low)
    # each risky phrase adds up to 0.12
    risk += min(0.12 * len(risky_phrases), 0.4)
    # penalize negative sentiment slightly
    if sentiment < -0.2:
        risk += 0.15
    elif sentiment < 0:
        risk += 0.06
    # cap
    if risk > 0.95:
        risk = 0.95

    # Determine recommendation logic similar to app UI
    if risk >= 0.7:
        label = "high"
        recommendation = "Decline / Manual Review - High Risk."
    elif risk >= 0.4:
        label = "moderate"
        # For customer message about a one-off late payment, suggest conditional action
        recommendation = "Conditional — follow-up and request supporting doc (e.g., hospital note), consider temporary waiver."
    else:
        label = "low"
        recommendation = "Approve late fee waiver and monitor next payment cycle."

    # Build summary text (human readable)
    # Detect if medical-related
    med_related = bool(re.search(r"hospitaliz|hospital|medical", text.lower()))
    # detect one-off wording patterns
    one_off = any(re.search(pat, text.lower()) for pat in [r"one late payment", r"two late payments", r"missed work", r"temporary income disruption", r"paycheck (?:was )?delayed", r"salary (?:was )?delayed"])
    prior_good = bool(re.search(r"always paid|paid on time|consistently paid|good standing|no previous", text.lower()))
    summary_lines = []
    summary_lines.append("The customer has a generally positive repayment history prior to this incident." if prior_good else "Limited explicit prior repayment history in message.")
    if med_related:
        summary_lines.append("The late payment appears tied to a short-term, documented medical event causing temporary income delay.")
    elif one_off:
        summary_lines.append("The late payment appears tied to a short-term income disruption (one-off).")
    else:
        summary_lines.append("The message indicates a short-term cashflow issue.")

    tone_label = "neutral"
    if sentiment > 0.2:
        tone_label = "positive (cooperative, apologetic)"
    elif sentiment < -0.2:
        tone_label = "negative (angry or evasive)"
    else:
        tone_label = "neutral / factual"

    summary_lines.append(f"The tone indicates {tone_label}. The borrower expresses willingness to pay promptly.")
    summary_text = " ".join(summary_lines)

    parsed = {
        "summary": {"summary": summary_text},
        "sentiment": {"score": sentiment, "label": tone_label},
        "extract_risky": {"risky_phrases": risky_phrases},
    }

    features = {
        "risk_score": round(risk, 2),
        "risk_label": label,
        "sentiment_score": round(sentiment, 2),
        "risky_phrases": risky_phrases,
        "recommendation": recommendation,
    }

    return {"parsed": parsed, "features": features}

# ---------------------------
# Keep existing functions (PDF OCR, parse_fields_from_text) — unchanged except minor compatibility edits
# ---------------------------

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

# ---------------------------
# Main app code (keeps your UI and logic)
# ---------------------------

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
        # `show_onboarding_guide()` exists in original file (kept above via original code)
        try:
            show_onboarding_guide()
        except Exception:
            pass
        # Mark that we've shown it once this session so uploads/actions won't re-open it
        st.session_state['first_visit'] = False

    st.title("LLM Credit Risk — Demo UI")

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")

        # --- File uploader (bigger visual area) ---
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
                # Identify if this file contains a customer message
                is_msg = is_customer_message(full_text)
                row = {
                    'id': next_id,
                    'name': parsed.get('name') or (f"message_{next_id}" if is_msg else getattr(f, 'name', '') or f"applicant_{next_id}"),
                    'age': parsed.get('age'),
                    'income': parsed.get('income'),
                    'requested_loan': parsed.get('requested_loan'),
                    'credit_score': None,
                    'text_notes': full_text,  # Full text for LLM analysis
                    'text_preview': full_text[:200] + '...' if len(full_text) > 200 else full_text,  # Short preview for UI
                    'is_customer_message': is_msg,
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
    if st.session_state.get("model_results") is not None:
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
    display_df = st.session_state["model_results"] if st.session_state.get("model_results") is not None else df
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
    if st.session_state.get("model_results") is not None:
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

            # Determine whether this is a customer message (if present in row)
            is_msg_row = bool(r.get('is_customer_message')) if 'is_customer_message' in r else is_customer_message(r.get('text_notes', ''))

            # Prefer any already-computed summary field
            if isinstance(r.get('summary'), str) and r.get('summary').strip():
                summary_text = r.get('summary')
                parsed = r.get('_parsed', {}) if isinstance(r.get('_parsed', {}), dict) else {}
            else:
                # If summary is missing, branch:
                # - For customer messages: call analyze_customer_message
                # - Else: call integrations.run_feature_extraction as before
                parsed = {}
                feats = {}
                summary_text = None
                try:
                    if is_msg_row:
                        # Prefer integrations.run_customer_message_extraction if available
                        try:
                            if hasattr(integrations, "run_customer_message_extraction"):
                                res = integrations.run_customer_message_extraction(r.get('text_notes', ''), mock=mock_mode)
                            else:
                                res = analyze_customer_message(r.get('text_notes', ''), mock=mock_mode)
                        except Exception:
                            # Fallback to local analyze
                            res = analyze_customer_message(r.get('text_notes', ''), mock=mock_mode)
                    else:
                        # Existing flow for normal loan application parsing (structured + LLM)
                        try:
                            res = integrations.run_feature_extraction(r.to_dict(), mock=mock_mode)
                        except Exception:
                            res = {"features": {}, "parsed": {}}
                    parsed = res.get('parsed', {}) if isinstance(res.get('parsed', {}), dict) else {}
                    feats = res.get('features', {}) if isinstance(res.get('features', {}), dict) else {}
                    # prefer summary under parsed.summary.summary (matches earlier structure)
                    summary = parsed.get('summary', {}) if isinstance(parsed, dict) else {}
                    summary_text = summary.get('summary') if isinstance(summary, dict) else None
                except Exception:
                    parsed = {}
                    feats = {}
                    summary_text = None

            # Build a local explanation from available signals
            sent_score = None
            if 'sentiment_score' in r:
                sent_score = r.get('sentiment_score')
            elif isinstance(parsed.get('sentiment'), dict):
                sent_score = parsed.get('sentiment', {}).get('score')
            elif isinstance(feats.get('sentiment_score'), (int, float)):
                sent_score = feats.get('sentiment_score')

            risky_val = r.get('risky_phrases') or r.get('risky_phrases_list') or (parsed.get('extract_risky', {}) or {}).get('risky_phrases') or feats.get('risky_phrases') or []
            if isinstance(risky_val, (list, tuple)):
                risky_text = ", ".join(map(str, risky_val))
            else:
                risky_text = str(risky_val) if risky_val else "None"

            # risk_score may not be present before run; try fallback fields
            risk_score = r.get('risk_score') or r.get('score') or feats.get('risk_score') or feats.get('score')
            
            # Simple recommendation heuristic
            try:
                rnum = float(risk_score) if risk_score is not None else None
            except Exception:
                rnum = None

            # --- DISPLAY DASHBOARD (ENHANCED) ---
            st.markdown("#### 🔎 Applicant Profile Summary")
            # If we have a computed summary, show it; otherwise fall back to brief text
            # For customer messages ensure we show the "summary" returned by analyze_customer_message
            if summary_text:
                st.info(summary_text)
            else:
                # Fallback message when no LLM summary is available
                st.info("No summary available. The document will be processed when you run the model.")

            # Compose improved Key Risk Signals with mini-explanations
            st.markdown("#### ⚠ Key Risk Signals")
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

            # Show structured fields if any
            st.markdown("#### ➤ Structured Fields (extracted / available)")
            sf1, sf2, sf3, sf4 = st.columns(4)
            with sf1:
                st.caption("Employment")
                st.write(r.get('employment_status') or "Unknown")
            with sf2:
                st.caption("Credit Score")
                st.write(r.get('credit_score') or "Unknown")
            with sf3:
                st.caption("Requested Loan")
                st.write(r.get('requested_loan') or "Unknown")
            with sf4:
                st.caption("Loan Purpose")
                st.write(r.get('loan_purpose') or "Unknown")

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
            st.markdown("#### 🟢 Recommendation")

            # Show recommendation if available in features or fallback to heuristic
            rec_text = None
            if isinstance(feats.get('recommendation'), str):
                rec_text = feats.get('recommendation')
            elif isinstance(r.get('recommendation'), str):
                rec_text = r.get('recommendation')
            else:
                if rnum is None:
                    st.warning("⚠️ **Model has not been run.** Click 'Run Model' to see scores.")
                    rec_text = "Run model to see recommendation."
                else:
                    score_label = f"{float(rnum):.2f}"
                    if rnum >= 0.7:
                        rec_text = f"🔴 DECLINE / MANUAL REVIEW — High Risk ({score_label})"
                    elif rnum >= 0.4:
                        rec_text = f"🟡 CONDITIONAL — Moderate Risk ({score_label})"
                    else:
                        rec_text = f"🟢 APPROVE — Low Risk ({score_label})"

            if rec_text:
                # For the customer-message use-case we want a clear action recommendation
                if is_msg_row and "waive" in (r.get('text_notes') or "").lower():
                    # If message explicitly asks to waive, show targeted recommendation
                    if "approve" in rec_text.lower() or "approve" in (feats.get('recommendation') or "").lower():
                        st.success(f"Recommendation: {rec_text}\n\nAction: Approve late fee waiver and monitor next payment cycle.")
                    else:
                        st.info(f"Recommendation: {rec_text}\n\nSuggested action: Consider waiver if supporting documentation provided.")
                else:
                    # general display
                    if "approve" in rec_text.lower() or rec_text.startswith("🟢"):
                        st.success(rec_text)
                    elif rec_text.startswith("🟡"):
                        st.warning(rec_text)
                    else:
                        st.error(rec_text)

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

    # Story Playback Logic (unchanged)
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

    # Trigger model run (unchanged except we ensure customer-message features are preserved)
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
                        # If this row was marked as customer message, run the customer message extractor/pipeline
                        try:
                            if row.get('is_customer_message'):
                                try:
                                    if hasattr(integrations, "run_customer_message_extraction"):
                                        res = integrations.run_customer_message_extraction(row.get('text_notes', ''), mock=mock_mode)
                                    else:
                                        res = analyze_customer_message(row.get('text_notes', ''), mock=mock_mode)
                                except Exception:
                                    res = analyze_customer_message(row.get('text_notes', ''), mock=mock_mode)
                            else:
                                res = integrations.run_feature_extraction(row.to_dict(), mock=mock_mode)
                        except Exception:
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
                        # Also include structured OCR fields from the original df row (if present)
                        try:
                            # e.g., credit_score, employment_status, requested_loan, loan_purpose
                            for fld in ['credit_score', 'employment_status', 'requested_loan', 'loan_purpose', 'income', 'age', 'name']:
                                if fld in row and (fld not in features or features.get(fld) is None):
                                    features[fld] = row.get(fld)
                        except Exception:
                            pass

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
