import time
from typing import Optional

import os
import importlib.util
import pandas as pd
import streamlit as st


from backend import ui_helpers
from backend import integrations


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
        mock_mode = st.checkbox("Mock mode (no LLM/API)", value=True)
        # removed debug parsed-JSON toggle; we'll show a local explanation instead
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
        # Show a live preview explanation for the selected applicant even before running the full pipeline.
        preview_explanation = None
        if selected_id is not None and not df.empty:
            try:
                preview_row = df[df['id'] == int(selected_id)].reset_index(drop=True)
                if not preview_row.empty:
                    r = preview_row.iloc[0]
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

                    risky_val = r.get('risky_phrases') or (parsed.get('extract_risky', {}) or {}).get('risky_phrases') or []
                    if isinstance(risky_val, (list, tuple)):
                        risky_text = ", ".join(map(str, risky_val))
                    else:
                        risky_text = str(risky_val) if risky_val else "None"

                    # risk_score may not be present before run; try fallback fields
                    risk_score = r.get('risk_score') or r.get('score') or (feats.get('risk_score') if 'feats' in locals() else None)
                    risk_label = r.get('risk_label', '—')

                    # Simple recommendation heuristic
                    try:
                        rnum = float(risk_score) if risk_score is not None else None
                    except Exception:
                        rnum = None
                    if rnum is None:
                        recommendation = "No score available — run the model for final recommendation."
                    elif rnum >= 0.7:
                        recommendation = "Recommend escalation / manual review (high risk)."
                    elif rnum >= 0.4:
                        recommendation = "Recommend enhanced due diligence (medium risk)."
                    else:
                        recommendation = "Low risk — standard processing recommended."

                    preview_explanation = (
                        f"Summary: {summary_text or 'No summary available'}\n\n"
                        f"Signals:\n- Sentiment score: {sent_score}\n- Risk phrases: {risky_text}\n- Risk score: {risk_score} ({risk_label})\n\n"
                        f"Recommendation: {recommendation}"
                    )
            except Exception:
                preview_explanation = None

        if preview_explanation:
            st.markdown(ui_helpers.highlight_snippet(preview_explanation), unsafe_allow_html=True)
        else:
            st.markdown(ui_helpers.highlight_snippet("No local explanation available yet. Select an applicant and click 'Run Model' (or enable Mock mode for a quick preview)."), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Story Playback**")
        pcol1, pcol2, pcol3 = st.columns([1, 1, 2])
        prev_clicked = pcol1.button("◀ Prev")
        play_clicked = pcol2.button("Play")
        # placeholder progress widget for playback actions
        progress_placeholder = st.empty()
        progress_placeholder.progress(0)

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
                    import pandas as pd

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
                    progress.progress(100)
                st.success("Completed model run")
                # update KPI
                high_pct = (out_df['risk_label'] == 'high').mean() * 100
                k3.metric(label="High Risk (%)", value=f"{high_pct:.1f}%")
                st.subheader("Predictions")
                ui_helpers.render_table(out_df)

                # Show explanation for selected applicant if available
                if not out_df.empty:
                    selected_row = out_df[out_df['id'] == int(selected_id)] if 'id' in out_df.columns else out_df.iloc[[0]]
                    if not selected_row.empty:
                        sel = selected_row.iloc[0]
                        # Try to get a short summary from available fields
                        summary_text = None
                        if isinstance(sel.get('summary'), str) and sel.get('summary').strip():
                            summary_text = sel.get('summary')
                        else:
                            parsed = sel.get('_parsed', {}) if isinstance(sel.get('_parsed', {}), dict) else {}
                            summary = parsed.get('summary', {}) if isinstance(parsed, dict) else {}
                            summary_text = summary.get('summary', 'No summary available') if isinstance(summary, dict) else str(summary)

                        # Local explanation constructed from available numeric/text signals
                        sent_score = sel.get('sentiment_score') if sel.get('sentiment_score') is not None else (
                            (sel.get('sentiment') or {}).get('score') if isinstance(sel.get('sentiment'), dict) else None
                        )
                        risky_val = sel.get('risky_phrases') or sel.get('risky_phrases_list') or []
                        if isinstance(risky_val, str):
                            risky_text = risky_val
                        elif isinstance(risky_val, (list, tuple)):
                            risky_text = ", ".join(map(str, risky_val))
                        else:
                            risky_text = str(risky_val)
                        risk_score = sel.get('risk_score') if sel.get('risk_score') is not None else sel.get('score')
                        risk_label = sel.get('risk_label', '—')

                        # Recommendation text based on risk_score (simple heuristic)
                        try:
                            rnum = float(risk_score) if risk_score is not None else None
                        except Exception:
                            rnum = None
                        if rnum is None:
                            recommendation = "No score available — run model to compute recommendation."
                        elif rnum >= 0.7:
                            recommendation = "Recommend escalation / manual review (high risk)."
                        elif rnum >= 0.4:
                            recommendation = "Recommend enhanced due diligence (medium risk)."
                        else:
                            recommendation = "Low risk — standard processing recommended."

                        local_explanation = (
                            f"Summary: {summary_text}\n\n"
                            f"Signals:\n- Sentiment score: {sent_score}\n- Risk phrases: {risky_text or 'None'}\n- Risk score: {risk_score} ({risk_label})\n\n"
                            f"Recommendation: {recommendation}"
                        )

                        st.markdown(ui_helpers.highlight_snippet(local_explanation), unsafe_allow_html=True)

                        # Story Playback: use the local explanation and signals
                        if play_clicked:
                            steps = []
                            steps.append(("Summary", summary_text))
                            steps.append(("Local Explanation", recommendation))
                            steps.append(("Risky Phrases", risky_text or "None"))
                            sent_text = f"score={sent_score}" if sent_score is not None else "unknown"
                            steps.append(("Sentiment", sent_text))

                            for i, (title, body) in enumerate(steps, start=1):
                                st.markdown(f"**Step {i}: {title}**")
                                st.markdown(ui_helpers.highlight_snippet(body), unsafe_allow_html=True)
                                progress_placeholder.progress(int(i / len(steps) * 100))
                                time.sleep(0.4)
                            progress_placeholder.progress(100)

            except Exception as e:
                st.error(f"Error during run: {e}")


if __name__ == "__main__":
    main()
