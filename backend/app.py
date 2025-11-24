import time
from typing import Optional

import pandas as pd
import streamlit as st

from backend import ui_helpers


st.set_page_config(page_title="LLM Credit Risk — Demo", layout="wide")


@st.cache_data
def load_demo_data(name: str) -> pd.DataFrame:
    """Return a tiny demo applicants dataframe based on a name."""
    if name == "Demo B":
        data = [
            {"id": 201, "name": "Alice B.", "age": 29, "income": 48000},
            {"id": 202, "name": "Bob B.", "age": 46, "income": 62000},
        ]
    else:
        data = [
            {"id": 101, "name": "Alice A.", "age": 31, "income": 52000},
            {"id": 102, "name": "Bob A.", "age": 45, "income": 72000},
            {"id": 103, "name": "Carol A.", "age": 38, "income": 31000},
        ]
    return pd.DataFrame(data)


@st.cache_data
def fake_model_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate model predictions (cached for demo speed)."""
    import numpy as np

    df = df.copy()
    # deterministic pseudo-random score based on id
    df["score"] = (df["id"] % 100) / 100.0
    df["risk_label"] = df["score"].apply(lambda s: "low" if s < 0.5 else "high")
    # small delay to make progress bar visible
    time.sleep(0.6)
    return df


def main() -> None:
    st.title("LLM Credit Risk — Demo UI")

    # Sidebar: upload, demo selector, run
    with st.sidebar:
        st.header("Inputs")
        uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])
        demo = st.selectbox("Or choose a demo dataset", ["Demo A", "Demo B"])
        run_button = st.button("Run Model")
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
    ui_helpers.kpi_card(k3, "High Risk (%)", "—")

    # Main layout: table and explanation
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Applicant Table")
        ui_helpers.render_table(df)

    with right:
        st.subheader("Explanations & Story")
        selected_id = st.number_input("Select applicant id", min_value=0, value=int(df['id'].iat[0]) if not df.empty else 0)
        st.markdown("**Local Explanation**")
        st.markdown(ui_helpers.highlight_snippet("This is a placeholder explanation for the selected applicant. Replace with LLM output."), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Story Playback**")
        pcol1, pcol2, pcol3 = st.columns([1, 1, 2])
        if pcol1.button("◀ Prev"):
            st.info("Step backward (placeholder)")
        if pcol2.button("Play"):
            st.info("Play story (placeholder)")
        st.progress(0)

    # Trigger model run
    if run_button:
        with st.spinner("Running model and LLM explainers..."):
            progress = st.progress(0)
            for i in range(1, 5):
                time.sleep(0.15)
                progress.progress(i * 20)
            preds = fake_model_predict(df)
        st.success("Completed model run")
        # update KPI
        high_pct = (preds['risk_label'] == 'high').mean() * 100
        k3.metric(label="High Risk (%)", value=f"{high_pct:.1f}%")
        st.subheader("Predictions")
        ui_helpers.render_table(preds)


if __name__ == "__main__":
    main()
