import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="JEE Rank Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — PREMIUM DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(18, 18, 30, 0.7);
        --bg-card-hover: rgba(25, 25, 40, 0.85);
        --border-color: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #f0f0f5;
        --text-secondary: #8b8b9e;
        --text-muted: #5a5a6e;
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --gradient-primary: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        --gradient-cyan: linear-gradient(135deg, #06b6d4, #0891b2);
        --gradient-emerald: linear-gradient(135deg, #10b981, #059669);
        --gradient-rose: linear-gradient(135deg, #f43f5e, #e11d48);
        --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
        --radius-lg: 16px;
        --radius-xl: 24px;
    }

    /* ── Global Styles ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    .stApp > header {
        background: transparent !important;
    }

    /* ── Main Content Area ── */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] li {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }

    /* ── Typography ── */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
    }

    h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }

    p, li, span {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Slider Styles ── */
    .stSlider > div > div {
        background: transparent !important;
    }

    .stSlider [data-testid="stTickBar"] {
        background: rgba(99, 102, 241, 0.1) !important;
    }

    div[data-testid="stSlider"] label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    .stSlider > div > div > div[role="slider"] {
        background: var(--accent-indigo) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.5) !important;
    }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-lg) !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: var(--border-glow) !important;
        box-shadow: var(--shadow-glow) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        background: var(--gradient-primary) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border-color) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-indigo) !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-lg) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--border-color) !important;
        margin: 2rem 0 !important;
    }

    /* ── Select Box / Number Input ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* ── Custom Animated Background Orbs ── */
    .hero-section {
        position: relative;
        padding: 2rem 0 1rem 0;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 999px;
        padding: 6px 16px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #a5b4fc;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
        animation: fadeInUp 0.6s ease;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #f0f0f5 0%, #a5b4fc 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeInUp 0.6s ease 0.1s both;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
        animation: fadeInUp 0.6s ease 0.2s both;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl);
        padding: 2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease;
    }

    .glass-card:hover {
        border-color: var(--border-glow);
        box-shadow: var(--shadow-glow);
    }

    /* ── Prediction Cards ── */
    .pred-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-xl);
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: var(--shadow-lg);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .pred-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }

    .pred-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.5);
    }

    .pred-card.indigo::before { background: var(--gradient-primary); }
    .pred-card.cyan::before { background: var(--gradient-cyan); }
    .pred-card.emerald::before { background: var(--gradient-emerald); }
    .pred-card.rose::before { background: var(--gradient-rose); }

    .pred-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }

    .pred-label.indigo { color: #a5b4fc; }
    .pred-label.cyan { color: #67e8f9; }
    .pred-label.emerald { color: #6ee7b7; }
    .pred-label.rose { color: #fda4af; }

    .pred-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .pred-sub {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 400;
    }

    /* ── Metric Bar ── */
    .metric-bar-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }

    .metric-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    .metric-bar-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 500;
    }

    .metric-bar-value {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-bar-track {
        background: rgba(255, 255, 255, 0.05);
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }

    .metric-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-bar-fill.indigo { background: var(--gradient-primary); }
    .metric-bar-fill.emerald { background: var(--gradient-emerald); }
    .metric-bar-fill.cyan { background: var(--gradient-cyan); }
    .metric-bar-fill.rose { background: var(--gradient-rose); }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        padding: 3rem 0 1rem 0;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }

    .footer a {
        color: var(--accent-indigo);
        text-decoration: none;
    }

    /* ── Plotly Chart Container ── */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }

    /* ── Info Box ── */
    .info-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #c7d2fe;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Radio buttons ── */
    .stRadio > div {
        gap: 0.5rem !important;
    }

    .stRadio label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .pred-value { font-size: 1.6rem; }
        .main .block-container { padding: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("jee_marks_percentile_rank_2009_2026.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(data):
    clean = data.copy()
    clean = clean[clean['rank'] > 0]

    clean['log_rank'] = np.log1p(clean['rank'])

    if 'percentile' not in clean.columns:
        clean['percentile'] = 100 * (1 - clean['rank'] / clean['rank'].max())

    feature_cols = ['marks', 'year']
    if 'total_candidates' in clean.columns:
        feature_cols.append('total_candidates')

    X = clean[feature_cols]
    y_log = clean['log_rank']
    y_pct = clean['percentile']
    y_raw = clean['rank']

    X_train, X_test, y_log_train, y_log_test, y_pct_train, y_pct_test, y_raw_train, y_raw_test = train_test_split(
        X, y_log, y_pct, y_raw, test_size=0.2, random_state=42
    )

    # Linear (log-rank)
    lin = LinearRegression()
    lin.fit(X_train, y_log_train)
    y_pred_lin = np.expm1(lin.predict(X_test))

    # Polynomial (log-rank)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    pol = LinearRegression()
    pol.fit(X_poly_train, y_log_train)
    y_pred_pol = np.expm1(pol.predict(X_poly_test))

    # Percentile model
    pct_model = LinearRegression()
    pct_model.fit(X_train, y_pct_train)
    y_pred_pct = pct_model.predict(X_test)
    y_pred_pct = np.clip(y_pred_pct, 0, 100)

    metrics = {
        "lin_mae": mean_absolute_error(y_raw_test, y_pred_lin),
        "lin_r2": r2_score(y_raw_test, y_pred_lin),
        "pol_mae": mean_absolute_error(y_raw_test, y_pred_pol),
        "pol_r2": r2_score(y_raw_test, y_pred_pol),
        "pct_mae": mean_absolute_error(y_pct_test, y_pred_pct),
        "pct_r2": r2_score(y_pct_test, y_pred_pct),
    }

    return lin, pol, poly, pct_model, metrics, feature_cols, clean

lin_model, poly_model, poly_feat, pct_model, metrics, feature_cols, clean_data = train_models(df)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #f0f0f5; letter-spacing: -0.02em;">JEE Rank Predictor</div>
        <div style="font-size: 0.75rem; color: #5a5a6e; margin-top: 4px;">ML-Powered Predictions</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #5a5a6e; margin-bottom: 0.8rem;">
        📝 Input Parameters
    </div>
    """, unsafe_allow_html=True)

    marks = st.slider(
        "🎯 Your Marks",
        min_value=0,
        max_value=300,
        value=150,
        step=1,
        help="Enter your JEE Main marks (0–300)"
    )

    year = st.slider(
        "📅 Exam Year",
        min_value=2009,
        max_value=2026,
        value=2025,
        step=1,
        help="Select the JEE Main exam year"
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #5a5a6e; margin-bottom: 0.8rem;">
        ⚙️ Model Info
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>3 Models Active</strong><br>
        <span style="color: #8b8b9e;">Linear Regression, Polynomial (deg 3), and Percentile-based models trained on <strong>{len(clean_data):,}</strong> data points from <strong>2009–2026</strong>.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.75rem; color: #5a5a6e; line-height: 1.7;">
        <strong style="color: #8b8b9e;">How it works:</strong><br>
        ① Enter your marks & year<br>
        ② Models predict your rank<br>
        ③ Compare across methods<br>
        ④ Explore visual analytics
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">✨ ML-Powered · 3 Models · Live Predictions</div>
    <div class="hero-title">JEE Rank Predictor</div>
    <div class="hero-subtitle">
        Predict your JEE Main rank using advanced machine learning models 
        trained on 17 years of historical data (2009–2026).
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
inp_dict = {'marks': marks, 'year': year}
if 'total_candidates' in feature_cols:
    inp_dict['total_candidates'] = 1400000

inp = pd.DataFrame([inp_dict])[feature_cols]

lin_pred = max(1, int(np.expm1(lin_model.predict(inp))[0]))
poly_pred = max(1, int(np.expm1(poly_model.predict(poly_feat.transform(inp)))[0]))
pct_pred = float(np.clip(pct_model.predict(inp)[0], 0, 100))
rank_from_pct = max(1, int((100 - pct_pred) / 100 * 1400000))

avg_rank = int(np.mean([lin_pred, poly_pred, rank_from_pct]))


# ─────────────────────────────────────────────
# PREDICTION CARDS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="pred-card indigo">
        <div class="pred-label indigo">Linear Model</div>
        <div class="pred-value">{lin_pred:,}</div>
        <div class="pred-sub">Estimated Rank</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="pred-card cyan">
        <div class="pred-label cyan">Polynomial Model</div>
        <div class="pred-value">{poly_pred:,}</div>
        <div class="pred-sub">Degree-3 Prediction</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="pred-card emerald">
        <div class="pred-label emerald">Percentile</div>
        <div class="pred-value">{pct_pred:.2f}%</div>
        <div class="pred-sub">Rank ≈ {rank_from_pct:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="pred-card rose">
        <div class="pred-label rose">Ensemble Average</div>
        <div class="pred-value">{avg_rank:,}</div>
        <div class="pred-sub">Mean of All Models</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS: Analytics, Model Perf, Data Explorer
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Visual Analytics", "🧮 Model Performance", "🔍 Data Explorer"])


# ──── TAB 1: Visual Analytics ────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Marks vs Rank scatter (filtered by year range)
        year_data = clean_data[clean_data['year'].between(year - 2, year + 2)]
        if len(year_data) == 0:
            year_data = clean_data

        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=year_data['marks'],
            y=year_data['rank'],
            mode='markers',
            marker=dict(
                size=6,
                color=year_data['percentile'],
                colorscale=[[0, '#f43f5e'], [0.5, '#8b5cf6'], [1, '#06b6d4']],
                opacity=0.6,
                line=dict(width=0),
                colorbar=dict(
                    title=dict(text="Pctl", font=dict(color='#8b8b9e', size=11)),
                    tickfont=dict(color='#8b8b9e', size=10),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                )
            ),
            name='Historical Data',
            hovertemplate='<b>Marks:</b> %{x}<br><b>Rank:</b> %{y:,}<extra></extra>'
        ))

        # Highlight user's prediction
        fig_scatter.add_trace(go.Scatter(
            x=[marks],
            y=[avg_rank],
            mode='markers+text',
            marker=dict(size=16, color='#f59e0b', symbol='diamond',
                        line=dict(width=2, color='#fbbf24')),
            text=['You'],
            textposition='top center',
            textfont=dict(color='#f59e0b', size=12, family='Inter'),
            name='Your Prediction',
            hovertemplate='<b>Your Marks:</b> %{x}<br><b>Predicted Rank:</b> %{y:,}<extra></extra>'
        ))

        fig_scatter.update_layout(
            title=dict(text='Marks vs Rank Distribution', font=dict(size=16, color='#f0f0f5', family='Inter'), x=0),
            xaxis=dict(title='Marks', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)',
                       zerolinecolor='rgba(255,255,255,0.06)'),
            yaxis=dict(title='Rank', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)',
                       zerolinecolor='rgba(255,255,255,0.06)', autorange='reversed'),
            plot_bgcolor='rgba(12,12,20,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#8b8b9e'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(size=11, color='#8b8b9e'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=50, r=30, t=60, b=50),
            height=420,
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    with viz_col2:
        # Model comparison gauge-like chart
        models_df = pd.DataFrame({
            'Model': ['Linear', 'Polynomial', 'Percentile', 'Ensemble'],
            'Predicted Rank': [lin_pred, poly_pred, rank_from_pct, avg_rank],
            'Color': ['#6366f1', '#06b6d4', '#10b981', '#f43f5e']
        })

        fig_bar = go.Figure()

        for _, row in models_df.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row['Predicted Rank']],
                y=[row['Model']],
                orientation='h',
                marker=dict(
                    color=row['Color'],
                    line=dict(width=0),
                    opacity=0.85
                ),
                text=f"  {row['Predicted Rank']:,}",
                textposition='outside',
                textfont=dict(color=row['Color'], size=13, family='JetBrains Mono'),
                name=row['Model'],
                showlegend=False,
                hovertemplate=f"<b>{row['Model']}</b><br>Rank: {row['Predicted Rank']:,}<extra></extra>"
            ))

        fig_bar.update_layout(
            title=dict(text='Model Predictions Comparison', font=dict(size=16, color='#f0f0f5', family='Inter'), x=0),
            xaxis=dict(title='Predicted Rank', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)',
                       zerolinecolor='rgba(255,255,255,0.06)'),
            yaxis=dict(color='#8b8b9e', categoryorder='array',
                       categoryarray=['Ensemble', 'Percentile', 'Polynomial', 'Linear']),
            plot_bgcolor='rgba(12,12,20,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#8b8b9e'),
            margin=dict(l=100, r=80, t=60, b=50),
            height=420,
            bargap=0.35,
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Second row of charts
    st.markdown("<br>", unsafe_allow_html=True)
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        # Percentile Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct_pred,
            number=dict(
                font=dict(size=42, color='#f0f0f5', family='JetBrains Mono'),
                suffix='%'
            ),
            delta=dict(reference=90, increasing=dict(color='#10b981'), decreasing=dict(color='#f43f5e')),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#5a5a6e',
                          tickfont=dict(color='#5a5a6e', size=10)),
                bar=dict(color='#6366f1', thickness=0.3),
                bgcolor='rgba(255,255,255,0.03)',
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color='rgba(244,63,94,0.1)'),
                    dict(range=[50, 80], color='rgba(245,158,11,0.1)'),
                    dict(range=[80, 95], color='rgba(6,182,212,0.1)'),
                    dict(range=[95, 100], color='rgba(16,185,129,0.1)'),
                ],
                threshold=dict(
                    line=dict(color='#f59e0b', width=3),
                    thickness=0.8,
                    value=pct_pred
                ),
            ),
            title=dict(text='Your Percentile Score', font=dict(size=14, color='#8b8b9e')),
        ))

        fig_gauge.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#8b8b9e'),
            margin=dict(l=30, r=30, t=80, b=30),
            height=350,
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

    with viz_col4:
        # Year-wise trend for the given marks
        marks_range = clean_data[(clean_data['marks'].between(marks - 5, marks + 5))]
        trend = marks_range.groupby('year').agg(
            avg_rank=('rank', 'mean'),
            avg_pct=('percentile', 'mean')
        ).reset_index()

        if len(trend) > 0:
            fig_trend = go.Figure()

            fig_trend.add_trace(go.Scatter(
                x=trend['year'],
                y=trend['avg_rank'],
                mode='lines+markers',
                line=dict(color='#8b5cf6', width=3, shape='spline'),
                marker=dict(size=8, color='#8b5cf6', line=dict(width=2, color='#a78bfa')),
                fill='tozeroy',
                fillcolor='rgba(139,92,246,0.08)',
                name='Avg Rank',
                hovertemplate='<b>Year:</b> %{x}<br><b>Avg Rank:</b> %{y:,.0f}<extra></extra>'
            ))

            fig_trend.update_layout(
                title=dict(text=f'Rank Trend for ~{marks} Marks (±5)', font=dict(size=16, color='#f0f0f5', family='Inter'), x=0),
                xaxis=dict(title='Year', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)',
                           dtick=1, zerolinecolor='rgba(255,255,255,0.06)'),
                yaxis=dict(title='Average Rank', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)',
                           zerolinecolor='rgba(255,255,255,0.06)', autorange='reversed'),
                plot_bgcolor='rgba(12,12,20,0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#8b8b9e'),
                margin=dict(l=50, r=30, t=60, b=50),
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data for this marks range to show trend.")


# ──── TAB 2: Model Performance ────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    # Performance Metric Bars
    max_mae = max(metrics['lin_mae'], metrics['pol_mae']) * 1.1

    st.markdown(f"""
    <div class="metric-bar-container">
        <div class="metric-bar-header">
            <span class="metric-bar-label">🔵 Linear Regression — MAE</span>
            <span class="metric-bar-value">{metrics['lin_mae']:,.0f}</span>
        </div>
        <div class="metric-bar-track">
            <div class="metric-bar-fill indigo" style="width: {min(metrics['lin_mae']/max_mae*100, 100):.1f}%;"></div>
        </div>
    </div>

    <div class="metric-bar-container">
        <div class="metric-bar-header">
            <span class="metric-bar-label">🟣 Polynomial Regression — MAE</span>
            <span class="metric-bar-value">{metrics['pol_mae']:,.0f}</span>
        </div>
        <div class="metric-bar-track">
            <div class="metric-bar-fill cyan" style="width: {min(metrics['pol_mae']/max_mae*100, 100):.1f}%;"></div>
        </div>
    </div>

    <div class="metric-bar-container">
        <div class="metric-bar-header">
            <span class="metric-bar-label">🟢 Percentile Model — MAE</span>
            <span class="metric-bar-value">{metrics['pct_mae']:.2f}</span>
        </div>
        <div class="metric-bar-track">
            <div class="metric-bar-fill emerald" style="width: {min(metrics['pct_mae']/10*100, 100):.1f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # R² Score cards
    r2_col1, r2_col2, r2_col3 = st.columns(3)

    with r2_col1:
        r2_pct_lin = max(0, metrics['lin_r2'] * 100)
        st.markdown(f"""
        <div class="pred-card indigo" style="text-align: center;">
            <div class="pred-label indigo">Linear R² Score</div>
            <div class="pred-value">{metrics['lin_r2']:.4f}</div>
            <div class="pred-sub">Variance Explained: {r2_pct_lin:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with r2_col2:
        r2_pct_pol = max(0, metrics['pol_r2'] * 100)
        st.markdown(f"""
        <div class="pred-card cyan" style="text-align: center;">
            <div class="pred-label cyan">Polynomial R² Score</div>
            <div class="pred-value">{metrics['pol_r2']:.4f}</div>
            <div class="pred-sub">Variance Explained: {r2_pct_pol:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with r2_col3:
        r2_pct_pct = max(0, metrics['pct_r2'] * 100)
        st.markdown(f"""
        <div class="pred-card emerald" style="text-align: center;">
            <div class="pred-label emerald">Percentile R² Score</div>
            <div class="pred-value">{metrics['pct_r2']:.4f}</div>
            <div class="pred-sub">Variance Explained: {r2_pct_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Residuals visualization
    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache_data
    def get_residuals(_data, _feature_cols):
        clean = _data.copy()
        clean = clean[clean['rank'] > 0]
        clean['log_rank'] = np.log1p(clean['rank'])

        if 'percentile' not in clean.columns:
            clean['percentile'] = 100 * (1 - clean['rank'] / clean['rank'].max())

        X = clean[_feature_cols]
        y_raw = clean['rank']

        X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)

        y_log_train = np.log1p(y_train)

        lin = LinearRegression().fit(X_train, y_log_train)
        lin_pred = np.expm1(lin.predict(X_test))
        lin_residuals = y_test - lin_pred

        return y_test, lin_pred, lin_residuals

    y_test, y_pred, residuals = get_residuals(df, feature_cols)

    fig_residuals = go.Figure()

    fig_residuals.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=5,
            color=residuals,
            colorscale=[[0, '#f43f5e'], [0.5, '#5a5a6e'], [1, '#06b6d4']],
            opacity=0.5,
            showscale=False,
        ),
        hovertemplate='<b>Predicted:</b> %{x:,.0f}<br><b>Residual:</b> %{y:,.0f}<extra></extra>'
    ))

    fig_residuals.add_hline(y=0, line_dash="dash", line_color="rgba(99, 102, 241, 0.5)", line_width=1)

    fig_residuals.update_layout(
        title=dict(text='Residuals Plot (Linear Model)', font=dict(size=16, color='#f0f0f5', family='Inter'), x=0),
        xaxis=dict(title='Predicted Rank', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(title='Residual (Actual − Predicted)', color='#8b8b9e', gridcolor='rgba(255,255,255,0.04)'),
        plot_bgcolor='rgba(12,12,20,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#8b8b9e'),
        margin=dict(l=60, r=30, t=60, b=50),
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig_residuals, use_container_width=True)


# ──── TAB 3: Data Explorer ────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1, 2])

    with exp_col1:
        selected_year = st.selectbox("Select Year", sorted(clean_data['year'].unique(), reverse=True), index=0)
        marks_range = st.slider("Marks Range", 0, 300, (50, 250))

    with exp_col2:
        filtered = clean_data[
            (clean_data['year'] == selected_year) &
            (clean_data['marks'].between(marks_range[0], marks_range[1]))
        ].sort_values('marks', ascending=False)

        st.markdown(f"""
        <div style="color: #8b8b9e; font-size: 0.85rem; margin-bottom: 0.5rem;">
            Showing <strong style="color: #a5b4fc;">{len(filtered)}</strong> records for year <strong style="color: #a5b4fc;">{selected_year}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            filtered[['marks', 'percentile', 'rank', 'total_candidates']].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

    # Heatmap: average rank by marks bucket and year
    st.markdown("<br>", unsafe_allow_html=True)

    heat_data = clean_data.copy()
    heat_data['marks_bucket'] = pd.cut(heat_data['marks'], bins=range(0, 310, 30),
                                        labels=[f"{i}-{i+29}" for i in range(0, 300, 30)])
    heatmap_df = heat_data.groupby(['marks_bucket', 'year'])['rank'].mean().unstack(fill_value=0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=np.log1p(heatmap_df.values),
        x=heatmap_df.columns.astype(str),
        y=[str(b) for b in heatmap_df.index],
        colorscale=[[0, '#0a0a0f'], [0.2, '#1e1b4b'], [0.4, '#4338ca'], [0.6, '#6366f1'],
                     [0.8, '#a78bfa'], [1.0, '#e0e7ff']],
        hovertemplate='<b>Marks:</b> %{y}<br><b>Year:</b> %{x}<br><b>Log Avg Rank:</b> %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Log(Rank)', font=dict(color='#8b8b9e')),
                      tickfont=dict(color='#8b8b9e'), bgcolor='rgba(0,0,0,0)', borderwidth=0),
    ))

    fig_heat.update_layout(
        title=dict(text='Average Rank Heatmap (Log Scale)', font=dict(size=16, color='#f0f0f5', family='Inter'), x=0),
        xaxis=dict(title='Year', color='#8b8b9e'),
        yaxis=dict(title='Marks Range', color='#8b8b9e', autorange='reversed'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#8b8b9e'),
        margin=dict(l=80, r=30, t=60, b=50),
        height=450,
    )

    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">
        Built with ❤️ using <strong>Streamlit</strong> & <strong>Scikit-Learn</strong>
    </div>
    <div>
        Data: JEE Main Results (2009–2026) · Models: Linear, Polynomial, Percentile Regression
    </div>
</div>
""", unsafe_allow_html=True)
