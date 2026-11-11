"""Apple-style dark theme: pure black pages, a grey sidebar, and chart colors.

Base colors live in ``.streamlit/config.toml``. This module holds the chart
colors chosen to stay legible on black, and the styling Streamlit's theme
settings cannot express.
"""
from __future__ import annotations

import streamlit as st

# Apple's dark-mode system colors.
BLUE = "#0A84FF"
GREEN = "#30D158"
RED = "#FF453A"
GREY = "#98989D"
WHITE = "#F5F5F7"

CARD = "#1C1C1E"
NO_DATA = "#2C2C2E"

# Low values stay dim against the dark card and high values stand out.
SEQUENTIAL_SCALE = [[0.0, "#123A66"], [0.5, BLUE], [1.0, "#BDE6FF"]]
SEQUENTIAL_SCALE_REVERSED = [[0.0, "#BDE6FF"], [0.5, BLUE], [1.0, "#123A66"]]
WARM_SCALE = [[0.0, "#4A2308"], [0.5, "#FF9F0A"], [1.0, "#FFE3A6"]]
# Zero sits on neutral grey rather than the white of a light-theme scale.
DIVERGING_SCALE = [[0.0, RED], [0.5, "#3A3A3C"], [1.0, BLUE]]
QUARTILE_COLORS = ["#2F5F8F", BLUE, "#64D2FF", "#D6F1FF"]

_CSS = """
<style>
:root {
    --nirf-black: #000000;
    --nirf-card: #1C1C1E;
    --nirf-card-raised: #2C2C2E;
    --nirf-sidebar: #2C2C2E;
    --nirf-control: #3A3A3C;
    --nirf-hairline: rgba(255, 255, 255, 0.10);
    --nirf-text: #F5F5F7;
    --nirf-text-muted: #98989D;
    --nirf-blue: #0A84FF;
}

/* Page */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--nirf-black);
    -webkit-font-smoothing: antialiased;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0.72);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
}
.block-container {
    padding-top: 3.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Type */
h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.015em;
}
.page-subtitle {
    color: var(--nirf-text-muted);
    font-size: 1.05rem;
    margin-top: -0.4rem;
}
.section-heading {
    font-weight: 600;
    margin-bottom: 0.25rem;
}
hr {
    border-color: var(--nirf-hairline) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--nirf-sidebar);
    border-right: 1px solid var(--nirf-hairline);
}
[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    letter-spacing: -0.015em;
    padding-bottom: 0;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 1rem !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #AEAEB2;
}

/* Sidebar navigation: a source list rather than radio buttons */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stRadio"]),
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] > div,
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 2px;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    width: 100% !important;
    box-sizing: border-box;
    margin: 0;
    padding: 0.45rem 0.7rem;
    border-radius: 9px;
    cursor: pointer;
    outline: none;
    transition: background-color 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div {
    width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background: rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: var(--nirf-blue);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
    color: #FFFFFF;
    font-weight: 500;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-focus-visible] {
    outline: 2px solid #FFFFFF;
    outline-offset: 1px;
}

/* Cards */
[data-testid="stMetric"] {
    background: var(--nirf-card);
    padding: 1rem 1.15rem;
    border-radius: 16px;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--nirf-control);
}
[data-testid="stMetricLabel"] p {
    color: var(--nirf-text-muted);
    font-size: 0.82rem;
}
[data-testid="stMetricValue"] {
    font-weight: 600;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
[data-testid="stPlotlyChart"] {
    background: var(--nirf-card);
    border-radius: 18px;
    overflow: hidden;
}
/* Streamlit paints plots with the page's black; let the card show through. */
[data-testid="stPlotlyChart"] .main-svg {
    background: transparent !important;
}
[data-testid="stPlotlyChart"] .legend .bg {
    fill: transparent !important;
}
[data-testid="stPlotlyChart"] .modebar-group {
    background: transparent !important;
}
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stExpander"] details {
    background: var(--nirf-card);
    border: none;
    border-radius: 16px;
}
[data-testid="stAlert"] > div {
    border-radius: 14px;
}
[data-testid="stCode"], .stCode pre {
    border-radius: 14px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border: none;
    border-radius: 980px;
    background: var(--nirf-card-raised);
    color: var(--nirf-text);
    font-weight: 500;
    transition: background-color 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #48484A;
    color: var(--nirf-text);
}
[data-testid="stSidebar"] .stButton > button {
    background: var(--nirf-control);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #48484A;
}

/* Tabs: a segmented control */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    width: fit-content;
    max-width: 100%;
    padding: 2px;
    background: var(--nirf-card);
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 0.35rem 0.95rem;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: var(--nirf-control);
}
.stTabs [aria-selected="true"] p {
    color: var(--nirf-text);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--nirf-card);
    border-radius: 18px;
    padding: 1rem 1.15rem;
}
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: var(--nirf-black);
}
[data-testid="stChatInput"] {
    border-radius: 22px;
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
    }
}
</style>
"""


def apply_theme() -> None:
    """Inject the theme's CSS; call once after ``st.set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)
