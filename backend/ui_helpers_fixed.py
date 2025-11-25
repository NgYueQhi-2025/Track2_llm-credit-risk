import streamlit as st
import pandas as pd


def kpi_card(col: st.delta_generator.DeltaGenerator, title: str, value, help_text: str | None = None) -> None:
    """Render a small KPI card inside the provided column.

    Args:
        col: a Streamlit column to render into (e.g. one of st.columns())
        title: card title
        value: main value to show (string or number)
        help_text: optional caption
    """
    with col:
        try:
            if isinstance(value, (int, float)):
                st.metric(label=title, value=value)
            else:
                st.write(f"**{title}**")
                st.write(value)
            if help_text:
                st.caption(help_text)
        except Exception:
            st.write(f"**{title}**: {value}")


def render_table(df: pd.DataFrame) -> None:
    """Render a dataframe with basic options."""
    if df is None or df.empty:
        st.info("No data to show")
        return
    st.dataframe(df.reset_index(drop=True), width='stretch')


def highlight_snippet(text: str) -> str:
    """Return HTML-wrapped snippet for display in Streamlit.

    Use `unsafe_allow_html=True` when rendering the returned HTML with `st.markdown`.
    """
    if text is None:
        return ""
    escaped = str(text).replace("\n", "<br>")
    html = f"<div style='padding:10px;background:#f8f9fb;border-radius:6px'>{escaped}</div>"
    return html
