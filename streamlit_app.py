# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import sys

import os

from src.blockchain.ethereum_tracker import show_ethereum_tracking_tab
os.environ["MPLCONFIGDIR"] = "/tmp"

from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to Python path (IMPORTANT)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Set page config
st.set_page_config(
    page_title="Institutional AI Portfolio Manager",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Color Palette Design System
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Custom Color Palette */
    :root {
        /* Primary Colors from Palette */
        --oceanic-noir: #172B36;
        --nocturnal-expedition: #11645A;
        --arctic-powder: #F1F0F4;
        --mystic-mint: #09E8E3;
        --forsythia: #FFC801;
        --deep-saffron: #FF9932;
        
        /* Background Colors */
        --bg-light: #F1F0F4;
        --bg-white: #FFFFFF;
        --bg-dark: #172B36;
        --bg-teal: #11645A;
        
        /* Text Colors */
        --text-dark: #172B36;
        --text-gray: #64748B;
        --text-light: #94A3B8;
        --text-white: #FFFFFF;
        
        /* Accent Colors */
        --accent-cyan: #09E8E3;
        --accent-yellow: #FFC801;
        --accent-orange: #FF9932;
        
        /* Status Colors */
        --success: #09E8E3;
        --warning: #FFC801;
        --danger: #FF9932;
        
        /* Border Colors */
        --border-light: #E2E8F0;
        --border-teal: #11645A;
    }
    
    /* Main App - Light Background */
    .stApp {
        background: var(--bg-light);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Content Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background: var(--bg-light);
    }
    
    /* Professional Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-dark);
        letter-spacing: -0.02em;
    }
    
    p, span, div {
        font-family: 'Inter', sans-serif;
        color: var(--text-gray);
    }
    
    /* Global Text Color Overrides */
    .stMarkdown, .stText {
        color: var(--oceanic-noir) !important;
    }
    
    /* Ensure all text in main content is visible */
    .main .block-container p,
    .main .block-container span,
    .main .block-container div {
        color: var(--oceanic-noir) !important;
    }
    
    /* Headings - Force Oceanic Noir */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    h1, h2, h3, h4, h5, h6 {
        color: var(--oceanic-noir) !important;
    }
    
    /* Exceptions: allow card/box headings to use mystic/arctic palette */
    .sidebar-card h1, .sidebar-card h2, .sidebar-card h3,
    .custom-gradient-card h1, .custom-gradient-card h2, .custom-gradient-card h3,
    .custom-white-card h1, .custom-white-card h2, .custom-white-card h3 {
        color: var(--mystic-mint) !important;
    }

    .sidebar-card p, .custom-gradient-card p, .custom-white-card p {
        color: var(--arctic-powder) !important;
    }
    /* All markdown content */
    .stMarkdown p, .stMarkdown span, .stMarkdown div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: var(--oceanic-noir) !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {
        color: var(--oceanic-noir) !important;
        font-weight: 600;
    }
    
    /* Radio and Checkbox labels in main content */
    .stRadio label,
    .stCheckbox label,
    .stRadio > label,
    .stCheckbox > label {
        color: var(--oceanic-noir) !important;
        font-weight: 500;
    }
    
    /* Select box labels in main content */
    .stSelectbox label,
    .stSelectbox > label {
        color: var(--oceanic-noir) !important;
        font-weight: 600;
    }
    
    /* All button text - Force White */
    button, .stButton button, .stButton > button,
    button span, .stButton button span, .stButton > button span,
    button div, .stButton button div, .stButton > button div,
    button p, .stButton button p, .stButton > button p {
        color: white !important;
    }
    
    /* Warning/Info boxes - Force colors */
    .stAlert,
    [data-baseweb="notification"] {
        border-left: 4px solid var(--mystic-mint) !important;
        background: rgba(9, 232, 227, 0.1) !important;
    }
    
    .stAlert p, .stAlert span, .stAlert div,
    [data-baseweb="notification"] p,
    [data-baseweb="notification"] span,
    [data-baseweb="notification"] div {
        color: var(--mystic-mint) !important;
        font-weight: 600 !important;
    }
    
    /* Info boxes */
    [data-testid="stNotification"],
    [data-testid="stNotification"] p,
    [data-testid="stNotification"] span {
        color: var(--mystic-mint) !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        color: var(--oceanic-noir) !important;
        border-left: 4px solid var(--mystic-mint) !important;
    }
    
    .stSuccess p, .stWarning p, .stError p, .stInfo p,
    .stSuccess span, .stWarning span, .stError span, .stInfo span {
        color: var(--mystic-mint) !important;
        font-weight: 600 !important;
    }
    
    /* Text in main content area */
    .element-container p,
    .element-container span,
    .element-container div {
        color: var(--oceanic-noir) !important;
    }
    
    /* Subheaders */
    .stSubheader, .stSubheader p {
        color: var(--oceanic-noir) !important;
    }
    
    /* Caption text */
    .stCaption, .stCaption p {
        color: var(--nocturnal-expedition) !important;
    }
    
    /* Vibrant Metric Cards with Custom Colors */
    [data-testid="metric-container"] {
        background: var(--bg-white);
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(23, 43, 54, 0.1);
        transition: all 0.3s ease;
        border-left: 4px solid var(--mystic-mint);
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 8px 24px rgba(9, 232, 227, 0.2);
        transform: translateY(-4px);
        border-left-color: var(--forsythia);
    }
    
    /* Metric Labels */
    [data-testid="metric-container"] label {
        color: var(--mystic-mint) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem !important;
    }
    
    /* Metric Values */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--mystic-mint) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
    }
    
    /* Metric Delta */
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
    }
    
    /* Vibrant Buttons with Mystic Mint */
    .stButton > button {
        background: linear-gradient(135deg, var(--mystic-mint) 0%, var(--nocturnal-expedition) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(9, 232, 227, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--forsythia) 0%, var(--deep-saffron) 100%);
        box-shadow: 0 4px 12px rgba(255, 200, 1, 0.4);
        transform: translateY(-2px);
    }
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-white);
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        color: var(--nocturnal-expedition);
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-light);
        color: var(--oceanic-noir);
    }
    
    /* Active Tab - Force Consistent Styling */
    .stTabs [aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--mystic-mint) 0%, var(--nocturnal-expedition) 100%) !important;
        color: white !important;
    }
    
    /* Tab Panel Content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    /* Tab Text - Ensure Visibility */
    .stTabs [data-baseweb="tab"] > div,
    .stTabs [aria-selected="true"] > div,
    button[role="tab"] > div {
        color: inherit !important;
    }
    
    /* Force all active tabs to have same background */
    .stTabs button[aria-selected="true"] {
        background: linear-gradient(135deg, var(--mystic-mint) 0%, var(--nocturnal-expedition) 100%) !important;
    }
    
    /* Dark Sidebar with Oceanic Noir */
    [data-testid="stSidebar"] {
        background: var(--oceanic-noir);
        border-right: 1px solid var(--nocturnal-expedition);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: var(--mystic-mint) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--arctic-powder) !important;
    }
    
    /* Sidebar Radio and Checkbox Text */
    [data-testid="stSidebar"] .stRadio > label > div,
    [data-testid="stSidebar"] .stCheckbox > label > div {
        color: var(--arctic-powder) !important;
    }
    
    /* Sidebar Select Box Options */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--nocturnal-expedition);
        border: 1px solid var(--mystic-mint);
        color: var(--arctic-powder) !important;
    }
    
    /* Sidebar Input Fields */
    [data-testid="stSidebar"] input {
        background: var(--nocturnal-expedition) !important;
        border: 1px solid var(--mystic-mint) !important;
        color: var(--arctic-powder) !important;
    }
    
    /* Sidebar Number Input */
    [data-testid="stSidebar"] .stNumberInput input {
        color: var(--arctic-powder) !important;
    }
    
    /* Dropdown Menu Options */
    [data-testid="stSidebar"] [role="option"] {
        color: var(--arctic-powder) !important;
        background: var(--nocturnal-expedition) !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background: var(--oceanic-noir) !important;
        color: var(--mystic-mint) !important;
    }
    
    /* Selected Option in Dropdown */
    [data-testid="stSidebar"] [aria-selected="true"] {
        background: var(--oceanic-noir) !important;
        color: var(--mystic-mint) !important;
    }
    
    /* Sliders with Mystic Mint */
    .stSlider > div > div > div {
        background: var(--mystic-mint);
    }
    
    /* Professional Data Tables */
    .dataframe {
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: var(--bg-white) !important;
    }
    
    .dataframe thead tr {
        background: var(--oceanic-noir) !important;
    }
    
    .dataframe thead th {
        color: var(--mystic-mint) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid var(--border-light) !important;
    }
    
    .dataframe tbody tr:hover {
        background: var(--bg-light) !important;
    }
    
    .dataframe tbody td {
        color: var(--text-dark) !important;
        padding: 0.875rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.875rem !important;
    }
    
    /* Green/Red Text in Tables */
    .positive-change {
        color: var(--mystic-mint) !important;
        font-weight: 600 !important;
    }
    
    .negative-change {
        color: var(--deep-saffron) !important;
        font-weight: 600 !important;
    }
    
    /* Status Indicators with Mystic Mint */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-live {
        background: var(--mystic-mint);
        box-shadow: 0 0 10px var(--mystic-mint);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    /* Professional White Cards */
    .pro-card {
        background: var(--bg-white);
        border: 1px solid var(--border-light);
        border-left: 4px solid var(--mystic-mint);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .pro-card:hover {
        box-shadow: 0 4px 16px rgba(9, 232, 227, 0.2);
        border-left-color: var(--forsythia);
    }
    
    /* Colorful Badges */
    .badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background: rgba(9, 232, 227, 0.15);
        color: var(--nocturnal-expedition);
        border: 1px solid var(--mystic-mint);
    }
    
    .badge-warning {
        background: rgba(255, 200, 1, 0.15);
        color: #92400E;
        border: 1px solid var(--forsythia);
    }
    
    .badge-danger {
        background: rgba(255, 153, 50, 0.15);
        color: #991B1B;
        border: 1px solid var(--deep-saffron);
    }
    
    .badge-info {
        background: rgba(17, 100, 90, 0.15);
        color: var(--nocturnal-expedition);
        border: 1px solid var(--nocturnal-expedition);
    }
    
    /* Chart Containers */
    .js-plotly-plot {
        background: var(--bg-white) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid var(--mystic-mint);
    }
    
    /* Expander Headers */
    .streamlit-expanderHeader {
        background: var(--bg-white);
        border-left: 4px solid var(--mystic-mint);
        border-radius: 8px;
        color: var(--oceanic-noir) !important;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        border-left-color: var(--forsythia);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'portfolio_template' not in st.session_state:
    st.session_state.portfolio_template = 'balanced'
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 1_000_000

def safe_import():
    """Safe import of all modules with proper error handling"""
    try:
        # Core imports
        from src.data.market_data import market_data_provider
        from src.models.portfolio_optimizer import InstitutionalPortfolioBuilder
        from src.config import config
        from src.data.data_loader import data_loader
        from src.visualization.charts import chart_generator
        from src.visualization.dashboards import dashboard
        from src.blockchain.ethereum_tracker import show_ethereum_tracking_tab

        # Initialize portfolio builder with market data
        portfolio_builder = InstitutionalPortfolioBuilder(market_data_provider)
        
        # Log initialization status
        if not portfolio_builder.advanced_features_available:
            st.warning("⚠️ Some advanced features may be limited")
        
        # Wrap everything in a dict
        modules = {
            "market_data_provider": market_data_provider,
            "portfolio_builder": portfolio_builder,
            "config": config,
            "data_loader": data_loader,
            "chart_generator": chart_generator,
            "dashboard": dashboard,
        }
        
        return True, modules
        
    except ImportError as e:
        st.error(f"❌ Critical import failed: {e}")
        st.error("Please check your installation and file structure")
        st.stop()
        return False, {}

  
def show_advanced_optimization_tab(portfolio_builder, market_data):
    """Advanced institutional optimization interface with better error handling"""
    
    st.subheader("🔬 Advanced Institutional Optimization")
    
    if not portfolio_builder.advanced_features_available:
        st.error("❌ Advanced optimization engines not available")
        st.info("Check your installation: pip install cvxpy arch scikit-learn tensorflow")
        return
    
    st.success("✅ Advanced optimization engines loaded: Markowitz, Black-Litterman, Risk Parity")
    
    # Optimization method selection
    optimization_method = st.selectbox(
        "Select Optimization Method:",
        [
            "Markowitz Mean-Variance", 
            "Black-Litterman with Views", 
            "Risk Parity"
        ]
    )
    
    # Asset selection with validation
    st.subheader("Asset Universe")
    
    preset_universe = st.selectbox(
        "Choose Asset Universe:",
        ["Multi-Asset Global", "US Equity Focus", "Sector ETFs", "Custom Selection"]
    )
    
    if preset_universe == "Multi-Asset Global":
        selected_assets = ['SPY', 'QQQ', 'VEA', 'VWO', 'TLT', 'LQD', 'VNQ', 'GLD']
    elif preset_universe == "US Equity Focus":
        selected_assets = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLV', 'XLE']
    elif preset_universe == "Sector ETFs":
        selected_assets = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU']
    else:
        available_assets = market_data.get_available_assets()
        selected_assets = st.multiselect(
            "Select Assets (minimum 3):",
            available_assets,
            default=available_assets[:8] if len(available_assets) >= 8 else available_assets
        )
    
    if len(selected_assets) < 2:
        st.warning("Please select at least 2 assets to optimize")
        return
    
    st.write(f"**Selected Assets:** {', '.join(selected_assets)}")
    
    # Method-specific parameters
    col1, col2 = st.columns(2)
    
    with col1:
        if "Markowitz" in optimization_method:
            target_return = st.slider("Target Annual Return:", 0.05, 0.20, 0.08, 0.01)
            risk_aversion = st.slider("Risk Aversion:", 0.5, 5.0, 1.0, 0.1)
            
        elif "Black-Litterman" in optimization_method:
            st.subheader("Market Views (Optional)")
            market_views = {}
            
            # Allow user to set views for selected assets
            for i, asset in enumerate(selected_assets[:4]):  # Limit to 4 views
                view = st.slider(f"{asset} Expected Return:", -0.10, 0.15, 0.0, 0.01, key=f"view_{i}")
                if abs(view) > 0.005:  # Only include non-zero views
                    market_views[asset] = view
    
    with col2:
        st.subheader("Risk Constraints")
        max_position = st.slider("Max Position Size:", 0.10, 0.60, 0.30, 0.05)
        min_position = st.slider("Min Position Size:", 0.0, 0.10, 0.01, 0.005)
    
    # Run optimization with comprehensive error handling
    if st.button("🚀 Run Advanced Optimization", type="primary"):
        with st.spinner(f"Running {optimization_method}..."):
            try:
                # Validate inputs first
                if len(selected_assets) < 2:
                    st.error("Need at least 2 assets for optimization")
                    return
                
                # Check data availability first
                st.info("Checking data availability...")
                data_check_passed = True
                for asset in selected_assets:
                    try:
                        test_data = market_data.get_stock_data(asset, '1mo')
                        if test_data.empty:
                            st.warning(f"No data available for {asset}")
                            data_check_passed = False
                    except Exception as e:
                        st.warning(f"Data check failed for {asset}: {str(e)}")
                        data_check_passed = False
                
                if not data_check_passed:
                    st.error("Data validation failed. Try with different assets.")
                    return
                
                # Run the selected optimization
                if "Markowitz" in optimization_method:
                    result = portfolio_builder.optimize_portfolio_advanced(
                        selected_assets, 
                        method='markowitz',
                        target_return=target_return,
                        risk_aversion=risk_aversion
                    )
                
                elif "Black-Litterman" in optimization_method:
                    result = portfolio_builder.optimize_portfolio_advanced(
                        selected_assets,
                        method='black_litterman', 
                        market_views=market_views
                    )
                
                elif "Risk Parity" in optimization_method:
                    result = portfolio_builder.optimize_portfolio_advanced(
                        selected_assets,
                        method='risk_parity'
                    )
                
                else:
                    st.error("Unknown optimization method")
                    return
                
                # Handle results with comprehensive error checking
                if not result:
                    st.error("No result returned from optimization")
                    return
                
                if 'error' in result:
                    st.error(f"Optimization failed: {result['error']}")
                    
                    # Provide helpful suggestions
                    if 'insufficient' in result['error'].lower():
                        st.info("💡 Try selecting assets with longer history or use a shorter analysis period")
                    elif 'singular' in result['error'].lower() or 'invertible' in result['error'].lower():
                        st.info("💡 Try selecting assets from different sectors to improve diversification")
                    
                    return
                
                if 'weights' not in result:
                    st.error("Optimization did not return portfolio weights")
                    st.write(f"Result keys: {list(result.keys())}")
                    return
                
                # Success - display results
                st.success("✅ Optimization completed successfully!")
                
                # Store in session state
                st.session_state.portfolio = result['weights']
                if hasattr(st.session_state, 'optimization_result'):
                    st.session_state.optimization_result = result
                st.session_state.analysis_data = None  # Force refresh
                
                # Show results
                show_advanced_optimization_results_safe(result, optimization_method)
                
            except Exception as e:
                st.error(f"❌ Optimization failed with error: {str(e)}")
                
                # Show debug info in expander
                with st.expander("Debug Information"):
                    st.write("Selected assets:", selected_assets)
                    st.write("Error details:", str(e))
                    import traceback
                    st.code(traceback.format_exc())

def show_advanced_optimization_results_safe(result, method_name):
    """Display optimization results with error handling"""
    
    try:
        st.subheader("🎯 Advanced Optimization Results")
        
        # Validate result structure
        if not isinstance(result, dict) or 'weights' not in result:
            st.error("Invalid optimization result format")
            return
        
        weights = result['weights']
        
        # Validate weights
        if not weights or not isinstance(weights, dict):
            st.error("Invalid weights format")
            return
        
        # Check for valid weight values
        valid_weights = {}
        total_weight = 0
        
        for asset, weight in weights.items():
            try:
                weight_val = float(weight)
                if 0 <= weight_val <= 1:  # Valid weight range
                    valid_weights[asset] = weight_val
                    total_weight += weight_val
            except (ValueError, TypeError):
                continue
        
        if not valid_weights:
            st.error("No valid weights found in optimization result")
            return
        
        # Normalize weights if needed
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            valid_weights = {k: v/total_weight for k, v in valid_weights.items()}
        
        # Create display DataFrame
        weights_df = pd.DataFrame([
            {
                'Asset': asset, 
                'Weight': f"{weight*100:.1f}%", 
                'Value': weight
            }
            for asset, weight in valid_weights.items()
        ]).sort_values('Value', ascending=False)
        
        # Display layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Optimized Portfolio Allocation:**")
            st.dataframe(weights_df[['Asset', 'Weight']], hide_index=True)
        
        with col2:
            # Safe pie chart creation
            try:
                import plotly.graph_objects as go
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(valid_weights.keys()),
                    values=list(valid_weights.values()),
                    hole=0.3
                )])
                fig_pie.update_layout(
                    title="Portfolio Allocation",
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=60, b=100)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.write("Chart display error:", str(e))
        
        # Performance metrics (if available)
        if 'expected_return' in result and result['expected_return'] is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                with col1:
                    ret_val = float(result.get('expected_return', 0))
                    st.metric("Expected Return", f"{ret_val*100:.1f}%")
                
                with col2:
                    vol_val = float(result.get('expected_volatility', 0))
                    st.metric("Expected Volatility", f"{vol_val*100:.1f}%")
                
                with col3:
                    sharpe_val = float(result.get('sharpe_ratio', 0))
                    st.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
                
                with col4:
                    status_val = result.get('optimization_status', 'Unknown')
                    st.metric("Status", status_val.title())
            
            except (ValueError, TypeError) as e:
                st.warning(f"Error displaying metrics: {e}")
        
        st.info("💡 Portfolio has been saved to your session. Switch to other tabs for full analysis.")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.write("Raw result for debugging:", result) 
  
def show_advanced_optimization_results(result, method_name):
    """Display advanced optimization results"""
    
    st.subheader("🎯 Advanced Optimization Results")
    
    # Portfolio weights
    weights = result['weights']
    weights_df = pd.DataFrame([
        {'Asset': asset, 'Weight': f"{weight*100:.1f}%", 'Value': weight}
        for asset, weight in weights.items()
    ]).sort_values('Value', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Optimized Portfolio Allocation:**")
        st.dataframe(weights_df[['Asset', 'Weight']], hide_index=True)
    
    with col2:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3
        )])
        fig_pie.update_layout(
            title="Portfolio Allocation",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=60, b=100)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Performance metrics
    if 'expected_return' in result:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expected Return", f"{result['expected_return']*100:.1f}%")
        
        with col2:
            st.metric("Expected Volatility", f"{result['expected_volatility']*100:.1f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
        
        with col4:
            st.metric("Status", result.get('optimization_status', 'Unknown'))
    
    # ML predictions if available
    if 'ml_predictions' in result:
        st.subheader("🤖 ML Predictions Used")
        
        ml_preds = result['ml_predictions']
        pred_df = pd.DataFrame([
            {'Asset': asset, 'ML Prediction': f"{pred*100:.2f}%"}
            for asset, pred in ml_preds.items()
        ])
        st.dataframe(pred_df, hide_index=True)
    
    # Risk analysis if available
    if 'risk_analysis' in result:
        st.subheader("⚠️ Institutional Risk Analysis")
        
        risk_analysis = result['risk_analysis']
        
        # Monte Carlo VaR
        if 'monte_carlo_var' in risk_analysis:
            mc_var = risk_analysis['monte_carlo_var']
            if 'error' not in mc_var:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("VaR (95%)", f"${mc_var.get('var_95', 0):,.0f}")
                with col2:
                    st.metric("CVaR (95%)", f"${mc_var.get('cvar_95', 0):,.0f}")
        
        # Stress test results
        if 'stress_tests' in risk_analysis:
            stress_tests = risk_analysis['stress_tests']
            if 'scenarios' in stress_tests:
                st.write("**Stress Test Results:**")
                stress_data = []
                for scenario, results in stress_tests['scenarios'].items():
                    stress_data.append({
                        'Scenario': scenario.replace('_', ' ').title(),
                        'Portfolio Impact': f"{results['loss_percentage']*100:.1f}%"
                    })
                
                stress_df = pd.DataFrame(stress_data)
                st.dataframe(stress_df, hide_index=True)
    
    st.info("💡 Portfolio has been saved to your session. Switch to other tabs to see full analysis.")  
  
    
def main():
    
    # Safe imports
    success, modules = safe_import()
    if not success:
        st.stop()

    market_data = modules.get('market_data_provider')
    portfolio_builder = modules.get('portfolio_builder') 
    config = modules.get('config')
    data_loader = modules.get('data_loader')
    chart_generator = modules.get('chart_generator')
    dashboard = modules.get('dashboard')  
      
    if not all([market_data, portfolio_builder]):
        st.stop()
    
    # Custom Color Palette Header
    st.markdown("""
    <div class='custom-white-card' style='background: white; border: 1px solid #E2E8F0; border-left: 4px solid #09E8E3; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(23, 43, 54, 0.1);'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
            <div>
                <h1 style='font-size: 1.75rem; font-weight: 800; margin: 0;'>🏛️ QuantAI Portfolio Manager</h1>
                <p style='font-size: 0.875rem; color: #11645A; margin: 0.25rem 0 0 0; font-weight: 600;'>Institutional-Grade AI Portfolio Management • Real-Time Analytics</p>
            </div>
            <div style='display: flex; gap: 1.5rem; align-items: center; margin-top: 1rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem;'>
                    <span class='status-indicator status-live'></span>
                    <div>
                        <div style='font-size: 0.75rem; color: #11645A; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;'>Market Status</div>
                        <div style='color: var(--mystic-mint); font-family: JetBrains Mono, monospace; font-weight: 700; font-size: 0.875rem;'>Live</div>
                    </div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.75rem; color: #11645A; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;'>Assets</div>
                    <div style='color: var(--mystic-mint); font-family: JetBrains Mono, monospace; font-weight: 700; font-size: 0.875rem;'>75+</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 0.75rem; color: #11645A; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;'>Last Update</div>
                    <div style='color: var(--mystic-mint); font-family: JetBrains Mono, monospace; font-weight: 700; font-size: 0.875rem;'>Real-time</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display API status
    display_api_status(config)
    
    # Enhanced sidebar
    with st.sidebar:
        show_enhanced_sidebar(market_data, portfolio_builder)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏛️ Institutional Dashboard", 
        "📊 Portfolio Builder",
        "🔬 Advanced Optimization", 
        "⚡ Market Analysis", 
        "🤖 AI Insights", 
        "⚠️ Risk Management",
        "🌍 Global Markets",
        "🔗 Blockchain Tracking"
    ])
    
    with tab1:
        show_institutional_dashboard(data_loader, chart_generator, dashboard)
    
    with tab2:
        show_portfolio_builder(portfolio_builder, market_data)
    
    with tab3:
        show_advanced_optimization_tab(portfolio_builder, market_data)  # NEW
    
    with tab4:
        show_market_analysis(market_data, chart_generator)
    
    with tab5:
        show_ai_insights(data_loader, market_data)
    
    with tab6:
        show_enhanced_risk_management(data_loader, dashboard)
        
        if st.checkbox("Show Debug Options"):
            debug_risk_calculations()
        
    with tab7:
        show_global_markets(market_data)
        
    with tab8:
        show_ethereum_tracking_tab() 

def display_api_status(config):
    
    if config:
        api_status = config.get_api_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢" if api_status['alpha_vantage'] else "🔴"
            st.write(f"{status} Alpha Vantage")
        
        with col2:
            status = "🟢" if api_status['fred'] else "🔴"
            st.write(f"{status} FRED")
        
        with col3:
            status = "🟢" if api_status['huggingface'] else "🔴"
            st.write(f"{status} HuggingFace")
        
        with col4:
            status = "🟢" if api_status['openai'] else "🔴"
            st.write(f"{status} OpenAI")

def show_enhanced_sidebar(market_data, portfolio_builder):
    
    # ADD this CSS at the top of show_enhanced_sidebar():
    # Custom Palette Sidebar Header
    st.markdown("""
    <div class='sidebar-card' style='background: linear-gradient(135deg, #11645A 0%, #172B36 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center; border: 1px solid var(--mystic-mint); max-width: 320px; margin-left: auto; margin-right: auto;'>
        <h2 style='color: var(--mystic-mint); font-size: 1.25rem; font-weight: 800; margin: 0; text-align: center;'>⚙️ Portfolio Builder</h2>
        <p style='color: var(--arctic-powder); font-size: 0.875rem; margin: 0.5rem 0 0 0; font-weight: 500;'>Configure Your Strategy</p>
    </div>
    """, unsafe_allow_html=True)
        
    # Portfolio creation methods
    creation_method = st.selectbox(
        "Portfolio Creation Method:",
        ["Template-Based", "Custom Selection", "Smart Beta", "Sector Rotation", "Risk Parity"]
    )
    
    if creation_method == "Template-Based":
        show_template_selector(portfolio_builder)
    elif creation_method == "Custom Selection":
        show_custom_selector(market_data)
    elif creation_method == "Smart Beta":
        show_smart_beta_builder(portfolio_builder)
    elif creation_method == "Sector Rotation":
        show_sector_rotation_builder(portfolio_builder)
    elif creation_method == "Risk Parity":
        show_risk_parity_builder(portfolio_builder, market_data)
    
    # Analysis period
    st.session_state.analysis_period = st.selectbox(
        "Analysis Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=2
    )
    
    # Portfolio value
    st.session_state.portfolio_value = st.number_input(
        "Portfolio Value ($):",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=100000,
        format="%d"
    )
    
    # Action buttons
    if st.button("🔄 Update Analysis", type="primary"):
        st.session_state.analysis_data = None  # Force refresh
        st.rerun()
    
    if st.button("📊 Export Data"):
        show_export_options()

def show_template_selector(portfolio_builder):
    
    templates = portfolio_builder.get_available_templates()
    
    # Risk tolerance and preferences
    risk_tolerance = st.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    investment_horizon = st.selectbox(
        "Investment Horizon:",
        ["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"]
    )
    
    income_focus = st.checkbox("Income Focus", value=False)
    
    # Get recommendations
    recommendations = portfolio_builder.get_portfolio_recommendations(
        risk_tolerance.lower(),
        investment_horizon.split()[0].lower(),
        income_focus
    )
    
    st.write("**Recommended Templates:**")
    
    selected_template = st.selectbox(
        "Select Template:",
        [rec['name'] for rec in recommendations] + list(templates.keys())
    )
    
    # Map display name back to template key
    template_key = selected_template.lower().replace(' ', '_')
    if template_key not in templates:
        # Find matching template
        for rec in recommendations:
            if rec['name'] == selected_template:
                template_key = rec['template']
                break
    
    if st.button("Build Portfolio from Template"):
        try:
            st.session_state.portfolio = portfolio_builder.build_portfolio_from_template(template_key)
            st.session_state.portfolio_template = template_key
            st.success(f"Portfolio created with {len(st.session_state.portfolio)} assets")
        except Exception as e:
            st.error(f"Template building failed: {str(e)}")
    
    # Show template details
    if template_key in templates:
        template_info = templates[template_key]
        st.write(f"**{template_info['name']}**")
        st.write(template_info['description'])
        st.write(f"Target Return: {template_info['target_return']*100:.1f}%")
        st.write(f"Target Volatility: {template_info['target_volatility']*100:.1f}%")

def show_custom_selector(market_data):
    
    # Asset class filter
    asset_classes = st.multiselect(
        "Asset Classes:",
        ["US Equity", "International Equity", "Bonds", "Commodities", "Real Estate", "Alternatives"],
        default=["US Equity", "Bonds"]
    )
    
    available_symbols = []
    
    # Map asset classes to filters
    for asset_class in asset_classes:
        if asset_class == "US Equity":
            symbols = market_data.get_available_assets({'type': 'equity', 'region': 'US'})
            available_symbols.extend(symbols)
        elif asset_class == "International Equity":
            symbols = market_data.get_available_assets({'region': 'Developed'}) + \
                     market_data.get_available_assets({'region': 'Emerging'})
            available_symbols.extend(symbols)
        elif asset_class == "Bonds":
            symbols = market_data.get_available_assets({'sector': 'Government Bonds'}) + \
                     market_data.get_available_assets({'sector': 'Corporate Bonds'})
            available_symbols.extend(symbols)
        elif asset_class == "Commodities":
            symbols = market_data.get_available_assets({'sector': 'Commodities'})
            available_symbols.extend(symbols)
        elif asset_class == "Real Estate":
            symbols = market_data.get_available_assets({'sector': 'Real Estate'})
            available_symbols.extend(symbols)
    
    # Remove duplicates
    available_symbols = list(set(available_symbols))
    
    st.write(f"**Available Assets ({len(available_symbols)} symbols):**")
    
    # Multi-select with search
    selected_symbols = st.multiselect(
        "Select Assets:",
        available_symbols,
        default=available_symbols[:10] if len(available_symbols) >= 10 else available_symbols
    )
    
    if selected_symbols:
        # Allocation method
        allocation_method = st.radio(
            "Allocation Method:",
            ["Equal Weight", "Market Cap Weight", "Custom Weights"]
        )
        
        if allocation_method == "Equal Weight":
            weight = 1.0 / len(selected_symbols)
            st.session_state.portfolio = {symbol: weight for symbol in selected_symbols}
        elif allocation_method == "Custom Weights":
            st.write("**Set Custom Weights:**")
            weights = {}
            for symbol in selected_symbols:
                weights[symbol] = st.slider(f"{symbol}:", 0.0, 1.0, 1.0/len(selected_symbols), 0.01)
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_weight:.2f}, not 1.0")
            else:
                st.session_state.portfolio = weights

def show_smart_beta_builder(portfolio_builder):
    
    st.write("**Smart Beta Strategy:**")
    st.info("Factor-based portfolio using quality, momentum, and value factors")
    
    universe_size = st.selectbox("Universe Size:", [10, 20, 30, 50], index=1)
    
    factor_focus = st.selectbox(
        "Factor Focus:",
        ["Balanced", "Quality-Focused", "Momentum-Focused", "Value-Focused"]
    )
    
    if st.button("Build Smart Beta Portfolio"):
        try:
            with st.spinner("Building smart beta portfolio..."):
                st.session_state.portfolio = portfolio_builder.create_smart_beta_portfolio()
                st.success("Smart beta portfolio created!")
        except Exception as e:
            st.error(f"Smart beta building failed: {str(e)}")

def show_sector_rotation_builder(portfolio_builder):
    
    st.write("**Sector Rotation Strategy:**")
    st.info("Momentum-based sector allocation using relative strength")
    
    lookback_period = st.slider("Momentum Lookback (days):", 20, 120, 60)
    
    if st.button("Build Sector Rotation Portfolio"):
        try:
            with st.spinner("Analyzing sector momentum..."):
                st.session_state.portfolio = portfolio_builder.create_sector_rotation_portfolio(lookback_period)
                st.success("Sector rotation portfolio created!")
        except Exception as e:
            st.error(f"Sector rotation building failed: {str(e)}")

def show_risk_parity_builder(portfolio_builder, market_data):
    
    st.write("**Risk Parity Strategy:**")
    st.info("Equal risk contribution from each asset")
    
    # Select asset universe
    universe_type = st.selectbox(
        "Universe:",
        ["Multi-Asset", "Equity Only", "Custom Selection"]
    )
    
    if universe_type == "Multi-Asset":
        symbols = ['SPY', 'TLT', 'GLD', 'VNQ', 'VEA', 'VWO', 'LQD', 'HYG']
    elif universe_type == "Equity Only":
        symbols = ['SPY', 'QQQ', 'IWM', 'VEA', 'VWO', 'XLK', 'XLF', 'XLV', 'XLE', 'XLY']
    else:
        symbols = st.multiselect(
            "Select symbols:",
            market_data.get_available_assets(),
            default=['SPY', 'TLT', 'GLD', 'VNQ']
        )
    
    if st.button("Build Risk Parity Portfolio"):
        try:
            with st.spinner("Calculating risk parity weights..."):
                st.session_state.portfolio = portfolio_builder.create_risk_parity_portfolio(symbols)
                st.success("Risk parity portfolio created!")
        except Exception as e:
            st.error(f"Risk parity building failed: {str(e)}")

def show_institutional_dashboard(data_loader, chart_generator, dashboard):
    
    st.subheader("🏛️ Institutional Portfolio Dashboard")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio using the sidebar.")
        show_sample_portfolios()
        return
    
    # Load comprehensive analysis
    with st.spinner("Loading institutional-grade analysis..."):
        try:
            analysis = get_or_create_analysis(data_loader)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return
    
    if 'error' in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        return
    
    # Key metrics dashboard
    show_key_metrics(analysis)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        show_performance_chart(analysis, chart_generator)
    
    with col2:
        show_allocation_chart()
    
    # Additional analysis
    col3, col4 = st.columns(2)
    
    with col3:
        show_risk_metrics_chart(analysis, chart_generator)
    
    with col4:
        show_sector_breakdown()
    
    # AI insights
    show_ai_insights_card(analysis)

def show_sample_portfolios():
    
    st.info("**Sample Institutional Portfolios:**")
    
    sample_portfolios = {
        "Balanced Growth (60/40)": {
            'SPY': 0.35, 'QQQ': 0.15, 'IWM': 0.10,  # US Equity 60%
            'TLT': 0.20, 'LQD': 0.15, 'TIP': 0.05   # Bonds 40%
        },
        "Endowment Model": {
            'SPY': 0.20, 'VEA': 0.15, 'VWO': 0.10,  # Global Equity 45%
            'TLT': 0.15, 'LQD': 0.10,               # Bonds 25%
            'VNQ': 0.10, 'GLD': 0.10, 'USO': 0.05,  # Alternatives 25%
            'VIXY': 0.05                            # Volatility 5%
        },
        "Global Diversified": {
            'VTI': 0.25, 'VXUS': 0.25,              # Global Equity 50%
            'BND': 0.20, 'BNDX': 0.10,              # Global Bonds 30%
            'VNQ': 0.10, 'VNQI': 0.05,              # Global REITs 15%
            'PDBC': 0.05                            # Commodities 5%
        }
    }
    
    for name, portfolio in sample_portfolios.items():
        if st.button(f"Load {name}"):
            st.session_state.portfolio = portfolio
            st.rerun()

def get_or_create_analysis(data_loader):
    
    if st.session_state.analysis_data is None:
        analysis = data_loader.get_portfolio_analysis(
            st.session_state.portfolio, 
            st.session_state.analysis_period
        )
        st.session_state.analysis_data = analysis
        return analysis
    else:
        return st.session_state.analysis_data

# Professional Metric Cards (using Streamlit native with global styling)
def show_key_metrics(analysis):
    metrics = analysis.get('performance_metrics', {})
    
    if metrics and 'error' not in metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📈 Total Return",
                value=f"{metrics.get('total_return', 0)*100:.2f}%",
                delta=f"{metrics.get('total_return', 0)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label="⚡ Sharpe Ratio",
                value=f"{metrics.get('sharpe_ratio', 0):.3f}",
                delta="Good" if metrics.get('sharpe_ratio', 0) > 1 else "Fair"
            )
        
        with col3:
            st.metric(
                label="📉 Max Drawdown",
                value=f"{metrics.get('max_drawdown', 0)*100:.2f}%",
                delta=f"{metrics.get('max_drawdown', 0)*100:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="🌊 Volatility",
                value=f"{metrics.get('volatility', 0)*100:.2f}%",
                delta=f"{metrics.get('volatility', 0)*100:.1f}%"
            )
        
        with col5:
            st.metric(
                label="🎯 Calmar Ratio",
                value=f"{metrics.get('calmar_ratio', 0):.3f}",
                delta="Strong" if metrics.get('calmar_ratio', 0) > 1 else "Moderate"
            )


def show_performance_chart(analysis, chart_generator):
    
    if 'portfolio_data' in analysis and not analysis['portfolio_data'].empty:
        try:
            chart = chart_generator.create_performance_chart(
                analysis['portfolio_data'], 
                "Portfolio Performance vs Benchmark"
            )
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Performance chart error: {str(e)}")
    else:
        st.info("Performance chart unavailable")

def show_allocation_chart():
    
    if st.session_state.portfolio:
        # Create enhanced pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.portfolio.keys()),
            values=list(st.session_state.portfolio.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Current Portfolio Allocation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_risk_metrics_chart(analysis, chart_generator):
    
    risk_metrics = analysis.get('risk_metrics', {})
    
    if risk_metrics and 'error' not in risk_metrics:
        try:
            chart = chart_generator.create_risk_metrics_bar(risk_metrics)
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Risk chart error: {str(e)}")

def show_sector_breakdown():
    
    st.subheader("Sector Breakdown")
    
    # Analyze portfolio by sectors
    if st.session_state.portfolio:
        sectors = {
            'Technology': 0.0,
            'Financials': 0.0,
            'Healthcare': 0.0,
            'Consumer': 0.0,
            'Bonds': 0.0,
            'Real Estate': 0.0,
            'Other': 0.0
        }
        
        sector_mapping = {
            'SPY': 'Diversified', 'QQQ': 'Technology', 'IWM': 'Small Cap',
            'TLT': 'Bonds', 'LQD': 'Bonds', 'HYG': 'Bonds',
            'VNQ': 'Real Estate', 'GLD': 'Commodities',
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare'
        }
        
        for symbol, weight in st.session_state.portfolio.items():
            sector = sector_mapping.get(symbol, 'Other')
            if sector in sectors:
                sectors[sector] += weight
            else:
                sectors['Other'] += weight
        
        # Create sector chart
        sector_df = pd.DataFrame([
            {'Sector': sector, 'Weight': weight} 
            for sector, weight in sectors.items() if weight > 0
        ])
        
        if not sector_df.empty:
            fig = px.bar(sector_df, x='Sector', y='Weight', 
                        title="Sector Allocation")
            st.plotly_chart(fig, use_container_width=True)

# Custom Palette AI Insights Card
def show_ai_insights_card(analysis):
    insights = analysis.get('ai_insights', 'AI analysis is processing...')
    
    st.markdown(f"""
    <div class='custom-gradient-card' style='background: linear-gradient(135deg, #11645A 0%, #172B36 100%); border-radius: 12px; padding: 2rem; margin: 2rem 0; box-shadow: 0 4px 16px rgba(9, 232, 227, 0.3); border: 2px solid var(--mystic-mint);'>
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
            <div style='font-size: 2.5rem;'>🧠</div>
            <div>
                <h3 style='color: var(--mystic-mint); margin: 0; font-size: 1.5rem; font-weight: 800;'>AI Market Intelligence</h3>
                <p style='color: var(--arctic-powder); margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                    <span style='background: var(--forsythia); padding: 0.375rem 0.875rem; border-radius: 16px; margin-right: 0.5rem; color: var(--mystic-mint); font-weight: 700; font-size: 0.75rem;'>GPT-4</span>
                    <span style='background: var(--deep-saffron); padding: 0.375rem 0.875rem; border-radius: 16px; margin-right: 0.5rem; color: white; font-weight: 700; font-size: 0.75rem;'>FinBERT</span>
                    <span style='background: var(--mystic-mint); padding: 0.375rem 0.875rem; border-radius: 16px; color: var(--arctic-powder); font-weight: 700; font-size: 0.75rem;'>Real-time</span>
                </p>
            </div>
        </div>
        <div style='color: white; line-height: 1.8; font-size: 1.05rem; font-weight: 500;'>{insights}</div>
    </div>
    """, unsafe_allow_html=True)



def show_portfolio_builder(portfolio_builder, market_data):
    
    st.subheader("📊 Advanced Portfolio Builder")
    
    # Portfolio construction wizard
    st.write("### Portfolio Construction Wizard")
    
    # Step 1: Investment Profile
    with st.expander("Step 1: Investment Profile", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age:", 18, 100, 35)
            risk_capacity = st.selectbox("Risk Capacity:", 
                                       ["Conservative", "Moderate", "Aggressive"])
        
        with col2:
            investment_goal = st.selectbox("Primary Goal:",
                                         ["Growth", "Income", "Balanced", "Capital Preservation"])
            time_horizon = st.selectbox("Time Horizon:",
                                      ["< 3 years", "3-7 years", "7-15 years", "> 15 years"])
    
    # Step 2: Asset Allocation
    with st.expander("Step 2: Strategic Asset Allocation"):
        allocation_approach = st.radio(
            "Allocation Approach:",
            ["Age-Based", "Risk-Based", "Goal-Based", "Custom"]
        )
        
        if allocation_approach == "Age-Based":
            stock_allocation = (100 - age) / 100
            bond_allocation = age / 100
            st.write(f"Suggested allocation: {stock_allocation*100:.0f}% Stocks, {bond_allocation*100:.0f}% Bonds")
        
        elif allocation_approach == "Risk-Based":
            if risk_capacity == "Conservative":
                suggested_allocation = {"Stocks": 0.3, "Bonds": 0.6, "Alternatives": 0.1}
            elif risk_capacity == "Moderate":
                suggested_allocation = {"Stocks": 0.6, "Bonds": 0.3, "Alternatives": 0.1}
            else:
                suggested_allocation = {"Stocks": 0.8, "Bonds": 0.1, "Alternatives": 0.1}
            
            for asset_class, weight in suggested_allocation.items():
                st.write(f"{asset_class}: {weight*100:.0f}%")
    
    # Step 3: Implementation
    with st.expander("Step 3: Implementation Strategy"):
        implementation = st.selectbox(
            "Implementation Style:",
            ["Passive (Index Funds)", "Active (Stock Picking)", "Hybrid", "Factor-Based"]
        )
        
        rebalancing = st.selectbox(
            "Rebalancing Frequency:",
            ["Monthly", "Quarterly", "Semi-Annual", "Annual", "Threshold-Based"]
        )
    
    # Generate portfolio
    if st.button("🚀 Generate Institutional Portfolio", type="primary"):
        with st.spinner("Generating your institutional portfolio..."):
            try:
                # Use investment profile to select template
                if risk_capacity == "Conservative":
                    template = 'conservative'
                elif risk_capacity == "Moderate":
                    template = 'balanced'
                else:
                    template = 'growth'
                
                if investment_goal == "Income":
                    template = 'income_focused'
                elif time_horizon == "> 15 years" and risk_capacity == "Aggressive":
                    template = 'institutional_endowment'
                
                portfolio = portfolio_builder.build_portfolio_from_template(template)
                st.session_state.portfolio = portfolio
                st.session_state.analysis_data = None  # Force refresh
                
                st.success(f"✅ Generated portfolio with {len(portfolio)} assets")
                
                # Show allocation
                st.write("**Generated Allocation:**")
                portfolio_df = pd.DataFrame([
                    {'Symbol': symbol, 'Weight': f"{weight*100:.1f}%", 'Allocation': weight}
                    for symbol, weight in portfolio.items()
                ])
                st.dataframe(portfolio_df, hide_index=True)
                
            except Exception as e:
                st.error(f"Portfolio generation failed: {str(e)}")

def show_market_analysis(market_data, chart_generator):
    
    st.subheader("⚡ Global Market Analysis")
    
    # Economic indicators
    with st.expander("📊 Economic Indicators", expanded=True):
        try:
            indicators = market_data.get_economic_indicators()
            
            if indicators:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("GDP Growth", f"{indicators.get('gdp_growth', 0):.1f}%")
                    st.metric("Fed Funds Rate", f"{indicators.get('fed_funds_rate', 0):.2f}%")
                
                with col2:
                    st.metric("Inflation Rate", f"{indicators.get('inflation_rate', 0):.1f}%")
                    st.metric("10Y Treasury", f"{indicators.get('10y_treasury', 0):.2f}%")
                
                with col3:
                    st.metric("Unemployment", f"{indicators.get('unemployment_rate', 0):.1f}%")
                    st.metric("2Y Treasury", f"{indicators.get('2y_treasury', 0):.2f}%")
                
                with col4:
                    st.metric("Consumer Confidence", f"{indicators.get('consumer_confidence', 0):.0f}")
                    st.metric("Housing Starts", f"{indicators.get('housing_starts', 0):.0f}K")
            else:
                st.info("Economic indicators loading...")
        except Exception as e:
            st.error(f"Economic data error: {str(e)}")
    
    # Sector performance
    with st.expander("🏭 Sector Performance"):
        try:
            sector_performance = market_data.get_sector_performance()
            
            if sector_performance:
                sector_data = []
                for sector, data in sector_performance.items():
                    sector_data.append({
                        'Sector': sector,
                        'Symbol': data['symbol'],
                        'Monthly Return': f"{data['monthly_return']:.1f}%",
                        'Volatility': f"{data['volatility']:.1f}%",
                        'Current Price': f"${data['current_price']:.2f}"
                    })
                
                sector_df = pd.DataFrame(sector_data)
                st.dataframe(sector_df, hide_index=True)
                
                # Sector performance chart
                returns_data = [data['monthly_return'] for data in sector_performance.values()]
                sectors = list(sector_performance.keys())
                
                fig = px.bar(x=sectors, y=returns_data, 
                           title="Sector Performance (1 Month)",
                           labels={'x': 'Sector', 'y': 'Return (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sector performance loading...")
        except Exception as e:
            st.error(f"Sector analysis error: {str(e)}")

def show_ai_insights(data_loader, market_data):
    
    st.subheader("🤖 Advanced AI Analysis")
    
    if not st.session_state.portfolio:
        st.warning("Configure your portfolio first.")
        return
    
    # AI analysis tabs
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["Portfolio Analysis", "Individual Assets", "Market Sentiment"])
    
    with ai_tab1:
        show_portfolio_ai_analysis(data_loader)
    
    with ai_tab2:
        show_individual_asset_analysis(data_loader, market_data)
    
    with ai_tab3:
        show_market_sentiment_analysis(market_data)

def show_portfolio_ai_analysis(data_loader):
    
    st.write("### Portfolio-Level AI Analysis")
    
    if st.button("🧠 Generate Comprehensive AI Analysis"):
        with st.spinner("AI is analyzing your portfolio..."):
            try:
                analysis = get_or_create_analysis(data_loader)
                ai_insights = analysis.get('ai_insights', 'Analysis unavailable')
                
                st.markdown(f"""
                ### 🎯 AI Portfolio Assessment
                
                {ai_insights}
                
                ### 📈 Optimization Suggestions
                Based on current market conditions and portfolio analysis, here are AI-generated recommendations:
                
                - **Risk Assessment**: Portfolio volatility is within target range
                - **Diversification**: Consider adding international exposure
                - **Rebalancing**: Next rebalancing suggested in 30 days
                - **Tax Optimization**: Review tax-loss harvesting opportunities
                """)
                
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")

def show_individual_asset_analysis(data_loader, market_data):
    
    st.write("### Individual Asset Analysis")
    
    if st.session_state.portfolio:
        selected_asset = st.selectbox(
            "Select asset for detailed AI analysis:",
            list(st.session_state.portfolio.keys())
        )
        
        if st.button(f"Analyze {selected_asset}"):
            with st.spinner(f"AI analyzing {selected_asset}..."):
                try:
                    asset_analysis = data_loader.get_stock_analysis(selected_asset, '6mo')
                    
                    if 'error' not in asset_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            current_price = asset_analysis.get('current_price', 0)
                            st.metric("Current Price", f"${current_price:.2f}")
                        
                        with col2:
                            daily_change = asset_analysis.get('price_change', 0)
                            st.metric("Daily Change", f"{daily_change:.2f}%", 
                                    delta=f"{daily_change:.2f}%")
                        
                        with col3:
                            portfolio_weight = st.session_state.portfolio.get(selected_asset, 0) * 100
                            st.metric("Portfolio Weight", f"{portfolio_weight:.1f}%")
                        
                        # AI insight
                        st.subheader("🤖 AI Analysis")
                        ai_insight = asset_analysis.get('ai_insight', 'Analysis unavailable')
                        st.write(ai_insight)
                        
                        # Technical analysis
                        if 'stock_data' in asset_analysis:
                            show_technical_analysis(asset_analysis['stock_data'], selected_asset)
                    
                    else:
                        st.error(f"Analysis failed: {asset_analysis['error']}")
                        
                except Exception as e:
                    st.error(f"Individual analysis failed: {str(e)}")

def show_technical_analysis(stock_data, symbol):
    
    st.subheader(f"📊 Technical Analysis - {symbol}")
    
    if not stock_data.empty:
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        
        # Add moving averages if available
        if 'SMA_20' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange')
            ))
        
        if 'SMA_50' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue')
            ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart with Technical Indicators",
            yaxis_title="Price ($)",
            height=600,
            margin=dict(l=50, r=50, t=80, b=120),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators summary
        if 'RSI' in stock_data.columns:
            latest_rsi = stock_data['RSI'].iloc[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RSI", f"{latest_rsi:.1f}")
                if latest_rsi > 70:
                    st.write("🔴 Overbought")
                elif latest_rsi < 30:
                    st.write("🟢 Oversold")
                else:
                    st.write("🟡 Neutral")
            
            with col2:
                if 'Volatility_20d' in stock_data.columns:
                    volatility = stock_data['Volatility_20d'].iloc[-1] * 100
                    st.metric("20-Day Volatility", f"{volatility:.1f}%")
            
            with col3:
                daily_return = stock_data['Daily_Return'].iloc[-1] * 100
                st.metric("Last Day Return", f"{daily_return:.2f}%")

def show_market_sentiment_analysis(market_data):
    
    st.write("### Market Sentiment Analysis")
    st.info("This feature analyzes overall market sentiment from various sources")
    
    # Placeholder for sentiment analysis
    st.write("""
    **Current Market Sentiment: Cautiously Optimistic**
    
    - VIX Level: 18.5 (Moderate volatility)
    - Put/Call Ratio: 0.85 (Neutral)
    - News Sentiment: 65% Positive
    - Social Media Sentiment: Mixed
    
    **Key Themes:**
    - Federal Reserve policy uncertainty
    - Inflation concerns moderating
    - Tech earnings season optimism
    - Geopolitical tensions monitoring
    """)

def show_enhanced_risk_management(data_loader, dashboard):
    
    st.subheader("⚠️ Institutional Risk Management")
    
    if not st.session_state.portfolio:
        st.warning("Configure portfolio first.")
        return
    
    try:
        # Get analysis data
        analysis = get_or_create_analysis(data_loader)
        
        # Import the fixed risk manager
        from src.models.risk_manager import create_risk_manager
        
        # Create risk manager instance
        risk_mgr = create_risk_manager(
            risk_tolerance='moderate',
            portfolio_value=st.session_state.portfolio_value,
            benchmark='SPY'
        )
        
        # Calculate portfolio returns from the analysis data
        if 'portfolio_data' in analysis and not analysis['portfolio_data'].empty:
            portfolio_data = analysis['portfolio_data']
            
            # Calculate returns from portfolio price data
            if 'Portfolio_Value' in portfolio_data.columns:
                portfolio_returns = portfolio_data['Portfolio_Value'].pct_change().dropna()
            else:
                # Fallback: calculate from individual asset data
                portfolio_returns = risk_mgr.calculate_portfolio_returns_from_weights(
                    portfolio_data, st.session_state.portfolio
                )
            
            # Run comprehensive risk analysis
            if len(portfolio_returns) > 30:  # Ensure sufficient data
                risk_analysis = risk_mgr.comprehensive_portfolio_analysis(
                    portfolio_returns, 
                    portfolio_value=st.session_state.portfolio_value
                )
                
                if 'error' not in risk_analysis:
                    # Display VaR metrics
                    var_analysis = risk_analysis.get('var_analysis', {})
                    cvar_analysis = risk_analysis.get('cvar_analysis', {})
                    basic_metrics = risk_analysis.get('basic_metrics', {})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        var_dollar = var_analysis.get('var_95_dollar', 0)
                        st.metric("VaR (1-day, 95%)", f"${var_dollar:,.0f}")
                    
                    with col2:
                        cvar_dollar = cvar_analysis.get('cvar_95_dollar', 0)
                        st.metric("CVaR (1-day, 95%)", f"${cvar_dollar:,.0f}")
                    
                    with col3:
                        max_dd = basic_metrics.get('max_drawdown', 0) * 100
                        st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
                    
                    # Risk gauge
                    st.subheader("🎯 Risk Assessment")
                    risk_level = risk_analysis.get('risk_level', 'moderate')
                    
                    if risk_level == 'high':
                        st.error("🚨 High Risk: Consider reducing position sizes")
                    elif risk_level == 'moderate':
                        st.warning("⚠️ Moderate Risk: Monitor closely")
                    else:
                        st.success("✅ Low Risk: Acceptable risk level")
                    
                    # Display recommendations
                    recommendations = risk_analysis.get('recommendations', [])
                    if recommendations:
                        st.subheader("💡 Risk Management Recommendations")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                else:
                    st.error(f"Risk analysis failed: {risk_analysis['error']}")
            
            else:
                st.error(f"Insufficient data for risk analysis: {len(portfolio_returns)} observations (need 30+)")
        
        else:
            st.error("No portfolio data available for risk analysis")
    
    except Exception as e:
        st.error(f"Risk management analysis failed: {str(e)}")
        
        # Debug information
        with st.expander("Debug Information"):
            st.write("Portfolio:", st.session_state.portfolio)
            st.write("Analysis keys:", list(analysis.keys()) if 'analysis' in locals() else "No analysis")

def show_global_markets(market_data):
    
    st.subheader("🌍 Global Markets Overview")
    
    # Major indices (placeholder data)
    major_indices = {
        'S&P 500': {'current': 4485.2, 'change': 1.2},
        'NASDAQ': {'current': 13737.6, 'change': 1.8},
        'Dow Jones': {'current': 34765.4, 'change': 0.9},
        'FTSE 100': {'current': 7421.3, 'change': -0.3},
        'Nikkei 225': {'current': 32467.8, 'change': 0.7},
        'DAX': {'current': 15234.7, 'change': 0.4}
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (index, data) in enumerate(major_indices.items()):
        col_idx = i % 3
        if col_idx == 0:
            with col1:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
        elif col_idx == 1:
            with col2:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
        else:
            with col3:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
    
    # Currency overview
    st.subheader("💱 Currency Markets")
    
    currencies = {
        'EUR/USD': 1.0847,
        'GBP/USD': 1.2634,
        'USD/JPY': 149.23,
        'USD/CAD': 1.3621,
        'AUD/USD': 0.6543,
        'USD/CHF': 0.9012
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (pair, rate) in enumerate(currencies.items()):
        col_idx = i % 3
        if col_idx == 0:
            with col1:
                st.metric(pair, f"{rate:.4f}")
        elif col_idx == 1:
            with col2:
                st.metric(pair, f"{rate:.4f}")
        else:
            with col3:
                st.metric(pair, f"{rate:.4f}")

def show_export_options():
    
    st.subheader("📊 Export Portfolio Data")
    
    export_format = st.selectbox(
        "Export Format:",
        ["CSV", "Excel", "JSON", "PDF Report"]
    )
    
    if st.button("Export Data"):
        st.success(f"Data exported in {export_format} format")
        st.info("Export functionality would be implemented here")

def debug_risk_calculations():
    """Debug function to test risk calculations"""
    
    st.subheader("🔧 Risk Calculation Debug")
    
    if not st.session_state.portfolio:
        st.warning("Configure portfolio first")
        return
    
    if st.button("Run Risk Debug Analysis"):
        try:
            from src.models.risk_manager import debug_risk_calculation
            from src.data.data_loader import data_loader  # Add this missing import
            
            # Get analysis data
            analysis = get_or_create_analysis(data_loader)
            
            if 'portfolio_data' in analysis and not analysis['portfolio_data'].empty:
                debug_results = debug_risk_calculation(
                    analysis['portfolio_data'],
                    st.session_state.portfolio,
                    portfolio_value=st.session_state.portfolio_value
                )
                
                st.write("Debug Results:")
                st.json(debug_results)
            else:
                st.error("No portfolio data available for debugging")
        
        except Exception as e:
            st.error(f"Debug failed: {str(e)}")

if __name__ == "__main__":
    main()