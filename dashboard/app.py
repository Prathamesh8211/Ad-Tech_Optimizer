"""
Ad-Tech Optimizer Dashboard
Unified Single-Workspace | Light Mode | 3 Tabs | Path-Safe v11 (Clean Sidebar)
"""

import os
import sys

# ============================================================
# PATH LOCK — makes the app work no matter where you start it from
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import importlib

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Ad-Tech Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_loader import load_sample_data
from utils.model_loader import load_models

import pages._01_dashboard_home as dashboard_home
import pages._02_predictions as predictions
import pages._03_performance_insights as insights
import pages._04_recommendations as recommendations

for module in (dashboard_home, predictions, insights, recommendations):
    importlib.reload(module)


# ============================================================
# SESSION STATE
# ============================================================
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

if "analytics_section" not in st.session_state:
    st.session_state.analytics_section = "Strategic Predictions"

if "global_category_filter" not in st.session_state:
    st.session_state.global_category_filter = None

if "global_device_filter" not in st.session_state:
    st.session_state.global_device_filter = "All Devices"

# Single unified perspective
st.session_state.workspace = "Unified"
st.session_state.user_type = "Business Owner"
st.session_state.theme_mode = "☀️ Light Mode"


# ============================================================
# LIGHT THEME + READABILITY OVERRIDES
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

    .stApp,
    [data-testid="stAppViewContainer"],
    header[data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    html, body, [class*="css"], .stMarkdown, p, span, label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700 !important;
        color: #0f172a !important;
        user-select: none !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label { color: #0f172a !important; }

    /* Dropdowns / inputs: white with dark text */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input, select, textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] * { color: #0f172a !important; }

    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Buttons: blue with white text */
    div.stButton > button, button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
    }

    div.stButton > button p, div.stButton > button span,
    button p, button span { color: #ffffff !important; }

    div.stButton > button:hover, button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    div.stButton > button:disabled, button:disabled {
        background-color: #cbd5e1 !important;
        color: #475569 !important;
    }

    /* Metric cards: light */
    div[data-testid="metric-container"], .custom-metric-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 16px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01) !important;
    }

    div[data-testid="stMetricValue"], .custom-metric-value,
    .custom-metric-card, .custom-metric-card * { color: #0f172a !important; }

    div[data-testid="stMetricLabel"], .custom-metric-label { color: #64748b !important; }

    /* Convert any dark inline-styled box to light */
    .main [style*="1e293b"], .main [style*="0f172a"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px;
    }

    .main [style*="1e293b"] *, .main [style*="0f172a"] * { color: #0f172a !important; }

    div.stAlert { color: #0f172a !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING WITH FALLBACKS (silent, no UI controls)
# ============================================================
def get_sample_data():
    # 1) Project's own loader
    try:
        df = load_sample_data()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # 2) CSV inside the dashboard folder
    try:
        df = pd.read_csv(os.path.join(APP_DIR, "sample_ad_performance.csv"))
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) Synthetic data one level up
    try:
        df = pd.read_csv(
            os.path.join(APP_DIR, "..", "Synthetic_Data", "raw_synthetic_ad_click_data.csv")
        )
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()


df_gold = get_sample_data()

if df_gold is None or not hasattr(df_gold, "copy"):
    df_gold = pd.DataFrame()

st.session_state.df_gold = df_gold

try:
    models = load_models()
    st.session_state.models = models if models is not None else {}
except Exception:
    st.session_state.models = {}


# ============================================================
# SIDEBAR (clean — no data source, no debug info)
# ============================================================
with st.sidebar:

    st.markdown("# Ad-Tech Optimizer")
    st.markdown("---")

    df_filtered = st.session_state.df_gold.copy()

    if not df_filtered.empty and "ad_category" in df_filtered.columns:
        category_filter = st.session_state.global_category_filter

        if category_filter:
            if isinstance(category_filter, list):
                all_categories_count = df_filtered["ad_category"].nunique()
                if 0 < len(category_filter) < all_categories_count:
                    df_filtered = df_filtered[
                        df_filtered["ad_category"].isin(category_filter)
                    ]
            elif category_filter != "All Categories":
                df_filtered = df_filtered[df_filtered["ad_category"] == category_filter]

    st.markdown("### Export Dataset")

    export_format = st.selectbox(
        "Format",
        ["Select Format...", "CSV Document (.csv)", "JSON Document (.json)"],
        index=0,
        label_visibility="collapsed",
        help="Download the active dataset in your chosen format.",
    )

    if export_format == "CSV Document (.csv)" and not df_filtered.empty:
        st.download_button(
            label="Download CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="ad_performance_report_unified.csv",
            mime="text/csv",
            use_container_width=True,
        )
    elif export_format == "JSON Document (.json)" and not df_filtered.empty:
        st.download_button(
            label="Download JSON",
            data=df_filtered.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="ad_performance_report_unified.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# ACTIVE FILTER DISPLAY
# ============================================================
def display_active_filters():
    active_filters = []

    category_filter = st.session_state.get("global_category_filter")

    if category_filter:
        if isinstance(category_filter, list):
            all_categories_count = 0
            if (
                "df_gold" in st.session_state
                and not st.session_state.df_gold.empty
                and "ad_category" in st.session_state.df_gold.columns
            ):
                all_categories_count = len(st.session_state.df_gold["ad_category"].unique())

            if 0 < len(category_filter) < all_categories_count:
                active_filters.append(f"Categories: <strong>{', '.join(category_filter)}</strong>")
        elif category_filter != "All Categories":
            active_filters.append(f"Category: <strong>{category_filter}</strong>")

    if active_filters:
        filter_text = " & ".join(active_filters)
        st.markdown(
            f"""
            <div style="background-color: #FEF9C3; border: 1px solid #FDE047; border-radius: 8px; padding: 12px 18px; margin-top: 10px; margin-bottom: 25px; display: flex; align-items: center; gap: 10px; justify-content: center;">
                <span style="font-size: 15.5px; color: #713F12; font-weight: 600; margin: 0; font-family: 'Plus Jakarta Sans', sans-serif;">
                    Active Global Filters &mdash; {filter_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HEADER + 3 TABS
# ============================================================
st.markdown(
    """
    <h1 style='text-align: center; text-transform: uppercase; font-weight: 900; color: #0f172a; margin-top: 0px; margin-bottom: 20px; font-size: 2.2rem; letter-spacing: -0.5px;'>
        Ad-Tech Performance Optimizer
    </h1>
    """,
    unsafe_allow_html=True,
)

# If NO data could be loaded at all, show a clear message (never a blank page)
if st.session_state.df_gold.empty:
    st.error(
        "⚠️ No data could be loaded. Check that `sample_ad_performance.csv` exists "
        "inside the `dashboard` folder, then rerun the app."
    )
    st.stop()

nav_options = ["Overview", "Analytics", "Budget Optimizer"]

st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > label { display: none; }

    div[data-testid="stRadio"] > div {
        gap: 0px !important;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 24px;
    }

    div[data-testid="stRadio"] > div > label {
        display: inline-flex !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        border-bottom: 3px solid transparent !important;
        margin-bottom: -2px !important;
        cursor: pointer !important;
        border-radius: 0 !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stRadio"] > div > label:hover { color: #0f172a !important; }

    div[data-testid="stRadio"] > div > label[data-selected="true"],
    div[data-testid="stRadio"] > div > label[aria-checked="true"] {
        color: #E53E3E !important;
        border-bottom: 3px solid #E53E3E !important;
        font-weight: 700 !important;
    }

    div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

selected_tab = st.radio(
    "Navigation",
    nav_options,
    index=st.session_state.active_tab if st.session_state.active_tab < len(nav_options) else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="tab_radio",
)

st.session_state.active_tab = nav_options.index(selected_tab)
active = st.session_state.active_tab

# ============================================================
# TAB CONTENT
# ============================================================
if active == 0:
    dashboard_home.show(st.session_state.df_gold, st.session_state.user_type)

elif active == 1:
    display_active_filters()

    st.markdown("### Analytics")

    analytics_section = st.selectbox(
        "Analytics Section",
        ["Strategic Predictions", "Performance Insights"],
        index=0 if st.session_state.analytics_section == "Strategic Predictions" else 1,
        label_visibility="collapsed",
        key="analytics_section_selector",
    )

    st.session_state.analytics_section = analytics_section

    if df_filtered.empty:
        st.info("No data available for the selected filters.")
    else:
        if analytics_section == "Strategic Predictions":
            predictions.show(df_filtered, st.session_state.models)
        else:
            insights.show(df_filtered, st.session_state.user_type)

elif active == 2:
    display_active_filters()
    recommendations.show(st.session_state.df_gold, st.session_state.user_type)


# ============================================================
# FOOTER
# ============================================================
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.caption("Ad-Tech Optimizer | Unified Dashboard | Powered by Databricks")


# ============================================================
# CURSOR FIX
# ============================================================
st.markdown(
    """
    <style>
    * { cursor: default !important; }

    button,
    div.stButton > button,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] input,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] input,
    div[data-testid="stSlider"] [role="slider"],
    div[data-testid="stSlider"] [role="track"],
    div.stTabs [role="tab"],
    section[data-testid="stSidebar"] a,
    section[data-testid="stSidebar"] label,
    a, a * { cursor: pointer !important; }

    button:disabled,
    div[aria-disabled="true"],
    div[data-baseweb="select"][aria-disabled="true"] > div { cursor: not-allowed !important; }
    </style>
    """,
    unsafe_allow_html=True,
)