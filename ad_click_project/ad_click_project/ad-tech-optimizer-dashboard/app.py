"""
Ad-Tech Optimizer Dashboard
Main Application Entry Point - Reload Trigger v8
"""

import streamlit as st
import numpy as np
import importlib
from utils.data_loader import load_gold_data, load_sample_data
from utils.model_loader import load_models

# Import page modules
import pages._01_dashboard_home as dashboard_home
import pages._02_predictions as predictions
import pages._03_performance_insights as insights
import pages._04_recommendations as recommendations
import pages._05_ai_copilot as ai_copilot
from llm.copilot import render_copilot_sidebar


# Force reload modules on every rerun to pick up edits
importlib.reload(dashboard_home)
importlib.reload(predictions)
importlib.reload(insights)
importlib.reload(recommendations)
importlib.reload(ai_copilot)


# Page Configuration
st.set_page_config(
    page_title="Ad-Tech Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Premium Styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main page container padding spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /*  Global cursor reset: no I-beam anywhere on static content  */
    * {
        cursor: default !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700 !important;
        color: #0f172a !important;
        user-select: none !important;
    }
    
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 16px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        border-color: #cbd5e1;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b !important;
    }
    
    /* Info Box / Alert styling */
    div.stAlert {
        border-radius: 16px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.01);
        padding: 16px 20px;
    }
    
    /* Button formatting */
    div.stButton > button {
        border-radius: 12px;
        transition: all 0.2s ease-in-out;
        cursor: pointer !important;
    }
    
    /* Scrollbar aesthetics */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Horizontal Page Tabs styling */
    div.stTabs [role="tablist"] {
        gap: 4px !important;
    }
    div.stTabs [role="tab"] {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        padding-left: 6px !important;
        padding-right: 6px !important;
        cursor: pointer !important;
    }
    
    /* Enlarge the text inside the tab buttons */
    div.stTabs [role="tab"] p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /*  Cursor Rules  */

    /* Selectbox / dropdown trigger */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] > div {
        cursor: pointer !important;
    }

    /* Metric cards — they have hover lift so should show pointer */
    div[data-testid="metric-container"] {
        cursor: default !important;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] input {
        cursor: pointer !important;
    }

    /* Checkboxes */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] input {
        cursor: pointer !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] [role="slider"],
    div[data-testid="stSlider"] [role="track"] {
        cursor: pointer !important;
    }

    /* Sidebar nav items */
    section[data-testid="stSidebar"] a,
    section[data-testid="stSidebar"] label {
        cursor: pointer !important;
    }

    /* Links and anchor elements */
    a, a * {
        cursor: pointer !important;
    }

    /* Disabled elements — show not-allowed so user knows */
    button:disabled,
    div[aria-disabled="true"],
    div[data-baseweb="select"][aria-disabled="true"] > div {
        cursor: not-allowed !important;
    }

    </style>
""",unsafe_allow_html=True)

# Initialize session state for workspace and page_idx
if "workspace" not in st.session_state:
    st.session_state.workspace = None
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "data_source" not in st.session_state:
    st.session_state.data_source = "Live Production Data (S3)"

# Check if workspace is selected (Landing Page view)
if st.session_state.workspace is None:
    # Inject CSS to hide sidebar on landing page
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; margin-top: 0px;'>Ad-Tech Performance Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.15rem; color: #64748b; margin-bottom: 50px;'>Select your workspace to load custom analytics profiles and predictions:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #0f172a; margin-bottom: 12px; font-weight: 700;'>Business Owner</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748b; font-size: 14.5px; line-height: 1.6; min-height: 75px;'>Strategic KPIs, high-level ROAS, profitability tracking, and ROAS-weighted budget allocation models.</p>", unsafe_allow_html=True)
            st.write("")
            if st.button("Enter Business Workspace", use_container_width=True, key="btn_business"):
                st.session_state.workspace = "Business"
                st.session_state.page_idx = 0
                st.session_state.should_scroll_top = True
                st.rerun()
            
    with col2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #0f172a; margin-bottom: 12px; font-weight: 700;'>Agency & Campaign</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #64748b; font-size: 14.5px; line-height: 1.6; min-height: 75px;'>Granular CTR/CPC indicators, ML performance forecasts, and ad format recommenders.</p>", unsafe_allow_html=True)
            st.write("")
            if st.button("Enter Agency Workspace", use_container_width=True, key="btn_agency"):
                st.session_state.workspace = "Agency"
                st.session_state.page_idx = 0
                st.session_state.should_scroll_top = True
                st.rerun()

else:
    # Inject JavaScript to scroll back to top if entering a workspace
    if st.session_state.get("should_scroll_top", False):
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                var mainContainer = window.parent.document.querySelector('.main');
                if (mainContainer) {
                    mainContainer.scrollTop = 0;
                }
            </script>
            """,
            height=0
        )
        st.session_state.should_scroll_top = False
    # ============================================================
    # DATA LOADING (Done first so sidebar export has immediate access)
    # ============================================================
    selected_source = st.session_state.get("data_source", "Live Production Data (S3)")
    
    if selected_source == "Live Production Data (S3)":
        st.session_state.df_gold = load_gold_data()
        if st.session_state.df_gold.empty:
            st.warning("Could not load data from S3. Falling back to local Demo Synthetic Data.")
            st.session_state.df_gold = load_sample_data()
    else:
        st.session_state.df_gold = load_sample_data()
    
    # Load models
    st.session_state.models = load_models()

    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        # Clean header at the top
        st.markdown("# Ad-Tech Optimizer")
        st.markdown("---")
        
        st.session_state.user_type = "Business Owner" if st.session_state.workspace == "Business" else "Agency"
        
        # Data Source Selection Dropdown
        st.selectbox(
            "Data Source",
            ["Live Production Data (S3)", "Demo Synthetic Data (Local)"],
            key="data_source",
            help="Choose between Live S3 tables and local Synthetic Demo records"
        )
        
        # Initialize global filter session states if not present
        if "global_category_filter" not in st.session_state:
            st.session_state.global_category_filter = None  # will be set after first render
        if "global_device_filter" not in st.session_state:
            st.session_state.global_device_filter = "All Devices"
            
        # Filter dataset globally based on session choices set by the home page
        df_filtered = st.session_state.df_gold.copy()
        if not df_filtered.empty:
            cat_filter = st.session_state.global_category_filter
            if cat_filter:
                if isinstance(cat_filter, list):
                    all_cats_cnt = df_filtered['ad_category'].nunique()
                    if 0 < len(cat_filter) < all_cats_cnt:
                        df_filtered = df_filtered[df_filtered['ad_category'].isin(cat_filter)]
                elif cat_filter != "All Categories":
                    df_filtered = df_filtered[df_filtered['ad_category'] == cat_filter]
            if st.session_state.workspace == "Agency" and st.session_state.global_device_filter != "All Devices":
                df_filtered = df_filtered[df_filtered['ad_device'] == st.session_state.global_device_filter]
        st.markdown("### Export Dataset")
        
        export_format = st.selectbox(
            "Format",
            ["Select Format...", "CSV Document (.csv)", "JSON Document (.json)"],
            index=0,
            label_visibility="collapsed",
            help="Download the active dataset in your chosen format"
        )
        
        if export_format == "CSV Document (.csv)" and not df_filtered.empty:
            csv_data = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"ad_performance_report_{st.session_state.workspace.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        elif export_format == "JSON Document (.json)" and not df_filtered.empty:
            json_data = df_filtered.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"ad_performance_report_{st.session_state.workspace.lower()}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Switch Workspace Back Button
        if st.button("Switch Workspace", use_container_width=True):
            st.session_state.workspace = None
            st.rerun()




    # ============================================================
    # MAIN CONTENT
    # ============================================================
    
    def display_active_filters():
        active_filters = []
        cat_filter = st.session_state.get("global_category_filter")
        if cat_filter:
            if isinstance(cat_filter, list):
                all_cats_cnt = len(st.session_state.df_gold['ad_category'].unique()) if hasattr(st.session_state, 'df_gold') and not st.session_state.df_gold.empty else 0
                if 0 < len(cat_filter) < all_cats_cnt:
                    active_filters.append(f"Categories: <strong>{', '.join(cat_filter)}</strong>")
            elif cat_filter != "All Categories":
                active_filters.append(f"Category: <strong>{cat_filter}</strong>")
        if st.session_state.get("workspace") == "Agency" and st.session_state.get("global_device_filter", "All Devices") != "All Devices":
            active_filters.append(f"Device: <strong>{st.session_state.global_device_filter}</strong>")
            
        if active_filters:
            filter_text = " & ".join(active_filters)
            st.markdown(f"""
                <div style="background-color: #FEF9C3; border: 1px solid #FDE047; border-radius: 8px; padding: 12px 18px; margin-top: 10px; margin-bottom: 25px; display: flex; align-items: center; gap: 10px; justify-content: center;">
                    <span style="font-size: 15.5px; color: #713F12; font-weight: 600; margin: 0; font-family: 'Plus Jakarta Sans', sans-serif;">
                        Active Global Filters &mdash; {filter_text}
                    </span>
                </div>
            """, unsafe_allow_html=True)

    #  Session-state tab index (persists across reruns — fixes redirect bug) 
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Dynamic Navigation labels based on Workspace
    if st.session_state.workspace == "Business":
        st.markdown("<h1 style='text-align: center; text-transform: uppercase; font-weight: 900; color: #0f172a; margin-top: 0px; margin-bottom: 20px; font-size: 2.2rem; letter-spacing: -0.5px;'>Business Owner Dashboard</h1>", unsafe_allow_html=True)
        nav_options = [
            "Financial Overview",
            "Strategic Predictions",
            "Performance Insights",
            "Budget Optimizer",
            "AI Analytics Copilot"
        ]
    else:
        st.markdown("<h1 style='text-align: center; text-transform: uppercase; font-weight: 900; color: #0f172a; margin-top: 0px; margin-bottom: 20px; font-size: 2.2rem; letter-spacing: -0.5px;'>Agency &amp; Campaign Workspace</h1>", unsafe_allow_html=True)
        nav_options = [
            "Campaign Analytics",
            "ML Performance Forecaster",
            "Granular Performance Insights",
            "Creative Recommender",
            "AI Analytics Copilot"
        ]

    #  Tab bar using st.radio (horizontal) — no CSS leaking 
    st.markdown("""
        <style>
        /* Style the radio tab bar to look like tabs */
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
        div[data-testid="stRadio"] > div > label:hover {
            color: #0f172a !important;
        }
        div[data-testid="stRadio"] > div > label[data-selected="true"],
        div[data-testid="stRadio"] > div > label[aria-checked="true"] {
            color: #E53E3E !important;
            border-bottom: 3px solid #E53E3E !important;
            font-weight: 700 !important;
        }
        /* Hide the radio circle dot */
        div[data-testid="stRadio"] > div > label > div:first-child {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    selected_tab = st.radio(
        "Navigation",
        nav_options,
        index=st.session_state.active_tab if st.session_state.active_tab < len(nav_options) else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="tab_radio"
    )
    st.session_state.active_tab = nav_options.index(selected_tab)
    active = st.session_state.active_tab

    #  Render active tab content 
    if active == 0:
        dashboard_home.show(st.session_state.df_gold, st.session_state.user_type)
    elif active == 1:
        display_active_filters()
        predictions.show(df_filtered, st.session_state.models)
    elif active == 2:
        display_active_filters()
        insights.show(df_filtered, st.session_state.user_type)
    elif active == 3:
        display_active_filters()
        recommendations.show(st.session_state.df_gold, st.session_state.user_type)
    elif active == 4:
        display_active_filters()
        ai_copilot.show(df_filtered, st.session_state.user_type)



# ============================================================
# FOOTER
# ============================================================

st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("Ad-Tech Optimizer v2.0 | Data from Gold Layer | Powered by Databricks")

# ============================================================
# CURSOR FIX — injected LAST on every render so it always wins
# over Streamlit's own internal styles
# ============================================================
st.markdown("""
    <style>
    /* Global: no I-beam on any static content */
    * { cursor: default !important; }

    /* Restore pointer on every interactive element */
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
    a, a * {
        cursor: pointer !important;
    }

    /* Disabled elements */
    button:disabled,
    div[aria-disabled="true"],
    div[data-baseweb="select"][aria-disabled="true"] > div {
        cursor: not-allowed !important;
    }
    </style>
""", unsafe_allow_html=True)
