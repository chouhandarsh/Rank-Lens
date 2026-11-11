import sys
from pathlib import Path

import streamlit as st

source_directory = Path(__file__).resolve().parent / "src"
if not source_directory.is_dir():
    raise RuntimeError(f"Chatbot source package was not found: {source_directory}")
# The project root holds the shared `nirf` package used for the map charts.
project_root = Path(__file__).resolve().parents[1]
for import_path in (project_root, source_directory):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from nirf.theme import apply_theme
from nirf_chatbot.streamlit_ui import render_chat_page


st.set_page_config(page_title="NIRF Insights", layout="wide")
apply_theme()

with st.container(border=True):
    logo_column, title_column = st.columns([1, 6], vertical_alignment="center")
    with logo_column:
        logo_path = Path(__file__).resolve().parent / "assets" / "nirf_logo.png"
        if logo_path.is_file():
            st.image(str(logo_path), width=112)
        else:
            st.warning(f"NIRF logo image was not found: {logo_path}")
    with title_column:
        st.title("NIRF Insights")
        st.caption("Ask a question about NIRF 2025 institutes and explore the results.")

render_chat_page(title="Chat with NIRF data", subtitle="Ask a question to get an answer, result table, chart, and optional plotting code.")
