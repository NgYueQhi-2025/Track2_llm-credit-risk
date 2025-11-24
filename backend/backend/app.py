import time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from backend import ui_helpers


st.set_page_config(page_title="Track2 — LLM Credit Risk (Demo)", layout="wide")


@st.cache_data
def load_demo_data(name: str) -> pd.DataFrame:
    """Return small demo applicants dataframe with a text column."""
    if name == "Synthetic Loan Batch":
        data = [
            {"id": 1001, "name": "Ana Lopez", "age": 33, "income": 54000, "credit_score": 680,
             "text_notes": "Looking to refinance; steady income but recent late payments."},
            {"id": 1002, "name": "Bashir Khan", "age": 45, "income": 42000, "credit_score": 590,
             "text_notes": "Business expansion loan; cashflow tight last 3 months; optimistic tone."},
            {"id": 1003, "name": "C. Zhang", "age": 29, "income": 76000, "credit_score": 730,
             "text_notes": "Applying for small personal loan; mentions ADA medical expenses."},
        ]
    else:
        data = [
            {"id": 2001, "name": "Dana S.", "age": 38, "income": 31000, "credit_score": 610,
             "text_notes": "Short, vague essay; multiple contradictions about employment."},
            {"id": 2002, "name": "Evan P.", "age": 52, "income": 120000, "credit_score": 780,
             "text_notes": "Examples of stable income and conservative spending; apologetic tone for late payments."},
        ]
    return pd.DataFrame(data)


def fake_llm_extract(texts: List[str]) -> List[Dict]:
    """Return mocked LLM outputs for a list of texts.

    Each entry contains: summary, sentiment_score (-1..1), risky_phrases list.
    """
    outs = []
    for t in texts:
        low = "vague" in t or "contradict" in t or "late" in t
        risky = []
        if "late" in t or "cashflow" in t or "tight" in t:
            risky.append("payment_history")
        if "contradict" in t or "vague" in t:
            risky.append("contradiction")
        # sentiment proxy
        s = -0.3 if low and len(risky) > 0 else 0.2
        summary = (t[:140] + "...") if len(t) > 140 else t
        outs.append({"summary": summary, "sentiment_score": s, "risky_phrases": risky})
    return outs


def engineer_text_features(llm_outputs: List[Dict]) -> pd.DataFrame:
    """Create numeric features from LLM outputs."""
    rows = []
    for out in llm_outputs:
        rows.append({
            "summary": out["summary"],
            "sentiment_score": out["sentiment_score"],
            "risky_phrase_count": len(out["risky_phrases"]),
            "risky_phrases": ", ".join(out["risky_phrases"]) if out["risky_phrases"] else "",
        })
    return pd.DataFrame(rows)


def heuristic_score(row: pd.Series) -> float:
    """Simple deterministic scoring function combining tabular + text features.

    Score in [0,1] where higher is more risky.
    """
    # base from credit_score
    cs = row.get("credit_score", 650)
    cs_term = max(0, (700 - cs) / 300)  # lower credit_score -> higher
    inc = row.get("income", 40000)
    inc_term = max(0, (60000 - inc) / 60000)
    sent = row.get("sentiment_score", 0.0)
    risk_phr = row.get("risky_phrase_count", 0)
    score = 0.5 * cs_term + 0.3 * inc_term + 0.2 * max(0, -sent) + 0.05 * risk_phr
    return min(max(score, 0.0), 1.0)


def run_pipeline(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Run the mock pipeline: LLM extract -> text features -> scoring."""
    if df is None or df.empty:
        return df

    texts = df.get("text_notes", pd.Series([""] * len(df))).fillna("").tolist()
    # In a real app cache keys should be sha of text + model params
    llm_outs = fake_llm_extract(texts)
    text_feats = engineer_text_features(llm_outs)
    df2 = pd.concat([df.reset_index(drop=True), text_feats.reset_index(drop=True)], axis=1)
    # compute score
    df2["risk_score"] = df2.apply(heuristic_score, axis=1)
    df2["risk_label"] = df2["risk_score"].apply(lambda s: "high" if s >= 0.5 else "low")
    return df2


def main() -> None:
    st.title("Track 2 — LLM-based Credit Risk (Demo)")
    st.write("Surface behavioral risk signals from application text and tabular features.")

    # Sidebar controls
    with st.sidebar:
        st.header("Input & Settings")
        uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])
        demo = st.selectbox("Or choose a demo dataset", ["Synthetic Loan Batch", "Small Bank Batch"])
        st.markdown("---")
        st.subheader("Column mapping")
        # preview columns if uploaded
        sample_df = None
        if uploaded is not None:
            try:
                sample_df = pd.read_csv(uploaded, nrows=5)
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
        cols = list(sample_df.columns) if sample_df is not None else ["id", "name", "income", "credit_score", "text_notes"]
        id_col = st.selectbox("ID column", options=cols, index=0)
        text_col = st.selectbox("Text column", options=cols, index=min(4, len(cols) - 1))
        label_col = st.selectbox("Label column (optional)", options=[None] + cols, index=0)
        st.markdown("---")
        st.subheader("Model & LLM")
        llm_provider = st.selectbox("LLM provider (demo)", ["Mock LLM", "OpenAI (not configured)"])
        classifier = st.selectbox("Tabular model", ["Heuristic (demo)", "LogisticRegression (optional)"])
        use_cache = st.checkbox("Use cached LLM outputs", value=True)
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.success("Cache cleared")
        st.markdown("---")
        run_button = st.button("Run Pipeline")

    # Load data
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = pd.DataFrame()
    else:
        df = load_demo_data(demo)

    # Top KPIs
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    ui_helpers.kpi_card(k1, "Applicants", len(df))
    avg_risk = "—"
    if "risk_score" in df.columns:
        avg_risk = f"{df['risk_score'].mean():.2f}"
    ui_helpers.kpi_card(k2, "Avg Risk", avg_risk)
    ui_helpers.kpi_card(k3, "High Risk (%)", "—")
    ui_helpers.kpi_card(k4, "Model", classifier)

    # Middle layout: table + explanation
    table_col, explain_col = st.columns([2, 1])

    with table_col:
        st.subheader("Applicants")
        # filtering controls
        filt1, filt2 = st.columns([2, 1])
        search = filt1.text_input("Search name/id")
        min_risk = filt2.slider("Min risk score", 0.0, 1.0, 0.0, 0.01)

        # run pipeline if requested
        if run_button:
            with st.spinner("Running pipeline — extracting text signals and scoring..."):
                progress = st.progress(0)
                time.sleep(0.2)
                progress.progress(10)
                df = run_pipeline(df, use_cache=use_cache)
                progress.progress(80)
                time.sleep(0.2)
                progress.progress(100)
            st.success("Pipeline complete")

        display_df = df.copy()
        if "risk_score" in display_df.columns:
            display_df = display_df[display_df["risk_score"] >= min_risk]
        if search:
            display_df = display_df[display_df.apply(lambda r: search.lower() in str(r.get("name", "")).lower() or search in str(r.get("id", "")), axis=1)]

        # render table (basic)
        ui_helpers.render_table(display_df)

        # choose an applicant to inspect
        ids = display_df["id"].tolist() if not display_df.empty else []
        selected = st.selectbox("Select applicant id to inspect", options=[None] + ids)

    with explain_col:
        st.subheader("Explanation & Evidence")
        if selected is None:
            st.info("Select an applicant to see LLM summary and evidence")
        else:
            row = df[df["id"] == selected].reset_index(drop=True)
            if row.empty:
                st.warning("Selected applicant not found in data")
            else:
                row = row.iloc[0]
                st.markdown("**Original Text**")
                st.write(row.get("text_notes", ""))
                st.markdown("**LLM Summary**")
                st.write(row.get("summary", "(not available)"))
                st.markdown("**Extracted Indicators**")
                st.write(f"Sentiment score: {row.get('sentiment_score', '—')}")
                st.write(f"Risk phrases: {row.get('risky_phrases', '')}")
                st.markdown("**Recommended Action**")
                rs = row.get("risk_score", None)
                if rs is not None:
                    if rs >= 0.7:
                        st.error(f"High risk — score {rs:.2f}")
                    elif rs >= 0.4:
                        st.warning(f"Medium risk — score {rs:.2f}")
                    else:
                        st.success(f"Low risk — score {rs:.2f}")
                st.markdown("---")
                st.markdown("**Feature contributions (mock)**")
                st.write("Credit score (−), Income (−), Sentiment (−)")

    # Bottom: charts & diagnostics
    st.markdown("---")
    st.subheader("Diagnostics & Charts")
    chart_col1, chart_col2 = st.columns([1, 1])
    with chart_col1:
        st.markdown("**Risk Distribution**")
        if "risk_score" in df.columns:
            st.bar_chart(df["risk_score"].value_counts().sort_index())
        else:
            st.info("No scores yet — run the pipeline")
    with chart_col2:
        st.markdown("**Top Risk Phrases (mock)**")
        # simple aggregation
        if "risky_phrases" in df.columns:
            phrases = df["risky_phrases"].str.get(0) if isinstance(df["risky_phrases"].dtype, object) else None
            st.write(df["risky_phrases"].value_counts().head(10))
        else:
            st.info("No extracted phrases yet")

    # Footer
    st.markdown("---")
    st.caption("Architecture: CSV + text → LLM extraction → feature engineering → classifier → explainability → dashboard")


if __name__ == "__main__":
    main()
