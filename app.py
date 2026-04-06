import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

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

    .stApp { background: var(--bg-primary) !important; font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; }
    .stApp > header { background: transparent !important; }
    .main .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

    section[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border-color) !important; }
    section[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] li { color: var(--text-secondary) !important; font-size: 0.9rem !important; }

    h1 { font-family: 'Inter', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.03em !important; color: var(--text-primary) !important; }
    h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; color: var(--text-primary) !important; }
    p, li, span { font-family: 'Inter', sans-serif !important; }

    .stSlider > div > div { background: transparent !important; }
    .stSlider [data-testid="stTickBar"] { background: rgba(99, 102, 241, 0.1) !important; }
    div[data-testid="stSlider"] label { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.95rem !important; }
    .stSlider > div > div > div[role="slider"] { background: var(--accent-indigo) !important; border: 2px solid rgba(255,255,255,0.2) !important; box-shadow: 0 0 16px rgba(99,102,241,0.5) !important; }

    div[data-testid="stMetric"] { background: var(--bg-card) !important; border: 1px solid var(--border-color) !important; border-radius: var(--radius-lg) !important; padding: 1.5rem !important; backdrop-filter: blur(20px) !important; transition: all 0.3s ease !important; box-shadow: var(--shadow-lg) !important; }
    div[data-testid="stMetric"]:hover { border-color: var(--border-glow) !important; box-shadow: var(--shadow-glow) !important; transform: translateY(-2px) !important; }
    div[data-testid="stMetric"] label { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.85rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important; font-size: 2rem !important; background: var(--gradient-primary) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; }

    .stTabs [data-baseweb="tab-list"] { background: var(--bg-secondary) !important; border-radius: 12px !important; padding: 4px !important; gap: 4px !important; border: 1px solid var(--border-color) !important; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-secondary) !important; border-radius: 8px !important; font-weight: 500 !important; font-size: 0.9rem !important; padding: 0.6rem 1.2rem !important; border: none !important; transition: all 0.2s ease !important; }
    .stTabs [aria-selected="true"] { background: var(--accent-indigo) !important; color: white !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

    .stSelectbox > div > div, .stNumberInput > div > div > input { background: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; border-radius: 10px !important; color: var(--text-primary) !important; }

    .hero-section { position: relative; padding: 2rem 0 1rem 0; overflow: hidden; }
    .hero-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25); border-radius: 999px; padding: 6px 16px; font-size: 0.8rem; font-weight: 600; color: #a5b4fc; letter-spacing: 0.04em; margin-bottom: 1rem; animation: fadeInUp 0.6s ease; }
    .hero-title { font-size: 3rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1.1; margin-bottom: 0.5rem; background: linear-gradient(135deg, #f0f0f5 0%, #a5b4fc 50%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: fadeInUp 0.6s ease 0.1s both; }
    .hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); font-weight: 400; line-height: 1.6; max-width: 600px; animation: fadeInUp 0.6s ease 0.2s both; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    .glass-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-xl); padding: 2rem; backdrop-filter: blur(20px); box-shadow: var(--shadow-lg); transition: all 0.3s ease; animation: fadeInUp 0.5s ease; }
    .glass-card:hover { border-color: var(--border-glow); box-shadow: var(--shadow-glow); }

    .pred-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-xl); padding: 1.8rem; backdrop-filter: blur(20px); box-shadow: var(--shadow-lg); transition: all 0.4s ease; position: relative; overflow: hidden; }
    .pred-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: var(--radius-xl) var(--radius-xl) 0 0; }
    .pred-card:hover { transform: translateY(-4px); box-shadow: 0 24px 48px rgba(0,0,0,0.5); }
    .pred-card.indigo::before { background: var(--gradient-primary); }
    .pred-card.cyan::before { background: var(--gradient-cyan); }
    .pred-card.emerald::before { background: var(--gradient-emerald); }
    .pred-card.rose::before { background: var(--gradient-rose); }
    .pred-card.amber::before { background: linear-gradient(135deg, #f59e0b, #d97706); }

    .pred-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
    .pred-label.indigo { color: #a5b4fc; } .pred-label.cyan { color: #67e8f9; } .pred-label.emerald { color: #6ee7b7; } .pred-label.rose { color: #fda4af; } .pred-label.amber { color: #fcd34d; }
    .pred-value { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; color: var(--text-primary); line-height: 1; margin-bottom: 0.3rem; }
    .pred-sub { font-size: 0.85rem; color: var(--text-muted); font-weight: 400; }

    .category-badge { display: inline-block; padding: 6px 18px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.03em; }
    .cat-elite { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
    .cat-top { background: rgba(6,182,212,0.15); color: #67e8f9; border: 1px solid rgba(6,182,212,0.3); }
    .cat-high { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
    .cat-comp { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
    .cat-not { background: rgba(244,63,94,0.15); color: #fda4af; border: 1px solid rgba(244,63,94,0.3); }

    .metric-bar-container { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 1.2rem 1.5rem; margin-bottom: 0.8rem; }
    .metric-bar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; }
    .metric-bar-label { color: var(--text-secondary); font-size: 0.85rem; font-weight: 500; }
    .metric-bar-value { font-family: 'JetBrains Mono', monospace; color: var(--text-primary); font-size: 0.85rem; font-weight: 600; }
    .metric-bar-track { background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; overflow: hidden; }
    .metric-bar-fill { height: 100%; border-radius: 3px; transition: width 1s ease; }
    .metric-bar-fill.indigo { background: var(--gradient-primary); }
    .metric-bar-fill.emerald { background: var(--gradient-emerald); }
    .metric-bar-fill.cyan { background: var(--gradient-cyan); }
    .metric-bar-fill.rose { background: var(--gradient-rose); }

    .info-box { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 1rem 1.2rem; color: #c7d2fe; font-size: 0.9rem; line-height: 1.6; }
    .footer { text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 3rem 0 1rem 0; border-top: 1px solid var(--border-color); margin-top: 3rem; }
    .footer a { color: var(--accent-indigo); text-decoration: none; }

    hr { border-color: var(--border-color) !important; margin: 2rem 0 !important; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-primary); } ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    @media (max-width: 768px) { .hero-title { font-size: 2rem; } .pred-value { font-size: 1.6rem; } .main .block-container { padding: 1rem !important; } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("jee_marks_percentile_rank_2009_2026.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Build year → total_candidates lookup for auto-fill
year_candidates_map = df.groupby('Year')['Total_Candidates'].first().to_dict()


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (from notebook)
# ─────────────────────────────────────────────
def get_category(ratio):
    """Notebook insight: Category based on RankRatio = Rank / Total_Candidates"""
    if ratio <= 0.005:
        return 'Elite (Top 0.5%)'
    elif ratio <= 0.02:
        return 'Top Tier (0.5% - 2%)'
    elif ratio <= 0.05:
        return 'Highly Competitive (2% - 5%)'
    elif ratio <= 0.10:
        return 'Competitive (5% - 10%)'
    else:
        return 'Not Prepared (>10%)'

CATEGORY_STYLES = {
    'Elite (Top 0.5%)': ('cat-elite', '🏆'),
    'Top Tier (0.5% - 2%)': ('cat-top', '⭐'),
    'Highly Competitive (2% - 5%)': ('cat-high', '🔥'),
    'Competitive (5% - 10%)': ('cat-comp', '💪'),
    'Not Prepared (>10%)': ('cat-not', '📚'),
}


# ─────────────────────────────────────────────
# TRAIN ALL MODELS (notebook-faithful)
# ─────────────────────────────────────────────
@st.cache_resource
def train_all_models(_data):
    data = _data.copy()
    data = data[data['Rank'] > 0]

    # ── Feature Engineering ──
    data['RankRatio'] = data['Rank'] / data['Total_Candidates']
    data['Category'] = data['RankRatio'].apply(get_category)

    sc = StandardScaler()
    encoder = LabelEncoder()

    # ═══════════════════════════════════════════
    # REGRESSION: Random Forest on log(Rank)
    # Notebook insight: RF handles skewed rank + outliers better than linear
    # ═══════════════════════════════════════════
    X_reg = data[['Year', 'Marks', 'Total_Candidates']]
    Y_reg = np.log1p(data['Rank'])

    X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(
        X_reg, Y_reg, test_size=0.2, random_state=42
    )

    X_train_r_sc = sc.fit_transform(X_train_r)
    X_test_r_sc = sc.transform(X_test_r)

    # Polynomial LR (notebook baseline for comparison)
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_r_sc)
    X_test_poly = poly.transform(X_test_r_sc)
    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train_r)
    y_lr_pred = np.expm1(lr.predict(X_test_poly))

    # Random Forest Regressor (notebook's best model)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_split=3,
        min_samples_leaf=5, random_state=42
    )
    rf.fit(X_train_r_sc, Y_train_r)
    y_rf_pred = np.expm1(rf.predict(X_test_r_sc))

    y_test_actual = np.expm1(Y_test_r)

    # Confidence interval — 10th-90th percentile across trees
    # Notebook insight: "tried 5-95 but too large due to outlier trees, 10-90 balances coverage vs usability"
    all_tree_preds_test = np.array([tree.predict(X_test_r_sc) for tree in rf.estimators_])
    lower_test = np.percentile(all_tree_preds_test, 10, axis=0)
    upper_test = np.percentile(all_tree_preds_test, 90, axis=0)
    lower_rank_test = np.expm1(lower_test)
    upper_rank_test = np.expm1(upper_test)
    coverage = np.mean((y_test_actual >= lower_rank_test) & (y_test_actual <= upper_rank_test))
    avg_range_width = (upper_rank_test - lower_rank_test).mean()

    reg_metrics = {
        'lr_r2': r2_score(y_test_actual, y_lr_pred),
        'lr_mae': mean_absolute_error(y_test_actual, y_lr_pred),
        'lr_rmse': np.sqrt(mean_squared_error(y_test_actual, y_lr_pred)),
        'rf_r2': r2_score(y_test_actual, y_rf_pred),
        'rf_mae': mean_absolute_error(y_test_actual, y_rf_pred),
        'rf_rmse': np.sqrt(mean_squared_error(y_test_actual, y_rf_pred)),
        'rf_r2_train': rf.score(X_train_r_sc, Y_train_r),
        'rf_r2_test': rf.score(X_test_r_sc, Y_test_r),
        'coverage': coverage,
        'avg_range_width': avg_range_width,
    }

    # Adjusted R²
    n = len(y_test_actual)
    p_lr = X_test_poly.shape[1]
    p_rf = X_test_r_sc.shape[1]
    reg_metrics['lr_adj_r2'] = 1 - (1 - reg_metrics['lr_r2']) * (n - 1) / (n - p_lr - 1)
    reg_metrics['rf_adj_r2'] = 1 - (1 - reg_metrics['rf_r2']) * (n - 1) / (n - p_rf - 1)

    # ═══════════════════════════════════════════
    # CLASSIFICATION: XGBoost + SMOTE
    # Notebook insight: same marks → different rank across years due to candidate count
    # Rank is density-driven, non-linear → tree models handle this locally
    # ═══════════════════════════════════════════
    sc_cls = StandardScaler()
    X_cls = data[['Year', 'Marks', 'Rank']]
    Y_cls = encoder.fit_transform(data['Category'])

    X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(
        X_cls, Y_cls, test_size=0.2, random_state=42
    )
    X_train_c_sc = sc_cls.fit_transform(X_train_c)
    X_test_c_sc = sc_cls.transform(X_test_c)

    # SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_c_sm, Y_train_c_sm = smote.fit_resample(X_train_c_sc, Y_train_c)

    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.02, max_depth=3,
        subsample=0.8, random_state=42, use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train_c_sm, Y_train_c_sm)

    y_xgb_pred = xgb.predict(X_test_c_sc)
    xgb_acc = accuracy_score(Y_test_c, y_xgb_pred)
    xgb_report = classification_report(Y_test_c, y_xgb_pred, output_dict=True,
                                       target_names=encoder.classes_)
    cv_scores = cross_val_score(xgb, X_train_c_sm, Y_train_c_sm, cv=5)

    cls_metrics = {
        'xgb_acc': xgb_acc,
        'xgb_train_acc': xgb.score(X_train_c_sm, Y_train_c_sm),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'report': xgb_report,
    }

    return (rf, lr, poly, xgb, sc, sc_cls, encoder,
            reg_metrics, cls_metrics, data)


(rf_model, lr_model, poly_feat, xgb_model, scaler_reg, scaler_cls, label_encoder,
 reg_metrics, cls_metrics, clean_data) = train_all_models(df)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #f0f0f5;">JEE Rank Predictor</div>
        <div style="font-size: 0.75rem; color: #5a5a6e; margin-top: 4px;">ML-Powered · RF + XGBoost</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#5a5a6e;margin-bottom:0.8rem;">📝 Input Parameters</div>', unsafe_allow_html=True)

    marks = st.slider("🎯 Your Marks", 0, 300, 150, 1, help="Enter your JEE Main marks (0–300)")
    year = st.slider("📅 Exam Year", 2009, 2026, 2025, 1, help="Select exam year")

    # Auto-fill total candidates from historical data
    default_candidates = year_candidates_map.get(year, 1400000)
    total_candidates = st.number_input(
        "👥 Total Candidates",
        min_value=500000, max_value=2000000,
        value=int(default_candidates), step=50000,
        help="Auto-filled from historical data. Adjust if needed."
    )

    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
        <strong>5 Models Active</strong><br>
        <span style="color: #8b8b9e;">RF Regressor, Poly LR (rank) + XGBoost, SVC, RF (category) trained on <strong>{len(clean_data):,}</strong> data points from <strong>2009–2026</strong>.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.75rem; color: #5a5a6e; line-height: 1.7;">
        <strong style="color: #8b8b9e;">How it works:</strong><br>
        ① Enter marks, year & candidates<br>
        ② RF predicts rank + confidence range<br>
        ③ XGBoost classifies your category<br>
        ④ Explore analytics & model metrics
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">✨ RF + XGBoost · SMOTE · Confidence Intervals</div>
    <div class="hero-title">JEE Rank Predictor</div>
    <div class="hero-subtitle">
        Predict your JEE Main rank using Random Forest Regression with tree-based confidence intervals,
        and get your competitive category via XGBoost classification — trained on 17 years of data.
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
inp_raw = np.array([[year, marks, total_candidates]])
inp_scaled = scaler_reg.transform(inp_raw)

# RF rank prediction
rf_log_pred = rf_model.predict(inp_scaled)[0]
rf_rank = max(1, int(np.expm1(rf_log_pred)))

# Confidence interval from individual trees (notebook's 10-90 percentile approach)
all_tree_preds = np.array([tree.predict(inp_scaled)[0] for tree in rf_model.estimators_])
rf_lower = max(1, int(np.expm1(np.percentile(all_tree_preds, 10))))
rf_upper = max(1, int(np.expm1(np.percentile(all_tree_preds, 90))))

# Polynomial LR prediction
inp_poly = poly_feat.transform(inp_scaled)
lr_rank = max(1, int(np.expm1(lr_model.predict(inp_poly)[0])))

# Percentile from rank
pct_pred = max(0.0, min(100.0, 100 * (1 - rf_rank / total_candidates)))

# XGBoost category prediction
inp_cls_raw = np.array([[year, marks, rf_rank]])
inp_cls_scaled = scaler_cls.transform(inp_cls_raw)
cat_encoded = xgb_model.predict(inp_cls_scaled)[0]
cat_label = label_encoder.inverse_transform([cat_encoded])[0]
cat_style, cat_icon = CATEGORY_STYLES.get(cat_label, ('cat-comp', '📊'))


# ─────────────────────────────────────────────
# PREDICTION CARDS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="pred-card indigo">
        <div class="pred-label indigo">RF Predicted Rank</div>
        <div class="pred-value">{rf_rank:,}</div>
        <div class="pred-sub">Random Forest (300 trees)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="pred-card cyan">
        <div class="pred-label cyan">Rank Range (80% CI)</div>
        <div class="pred-value" style="font-size:1.6rem;">{rf_lower:,} – {rf_upper:,}</div>
        <div class="pred-sub">10th–90th percentile</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="pred-card emerald">
        <div class="pred-label emerald">Percentile</div>
        <div class="pred-value">{pct_pred:.2f}%</div>
        <div class="pred-sub">Based on RF rank</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="pred-card rose">
        <div class="pred-label rose">Poly LR Rank</div>
        <div class="pred-value">{lr_rank:,}</div>
        <div class="pred-sub">Degree-3 Polynomial</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="pred-card amber">
        <div class="pred-label amber">Category</div>
        <div style="margin: 0.5rem 0;">
            <span class="category-badge {cat_style}">{cat_icon} {cat_label}</span>
        </div>
        <div class="pred-sub">XGBoost + SMOTE</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 Visual Analytics", "🧮 Model Performance", "🔍 Data Explorer"])

PLOT_BG = 'rgba(12,12,20,0.8)'
PAPER_BG = 'rgba(0,0,0,0)'
GRID_COLOR = 'rgba(255,255,255,0.04)'
FONT_CFG = dict(family='Inter', color='#8b8b9e')

# ──── TAB 1: Visual Analytics ────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        year_data = clean_data[clean_data['Year'].between(year - 2, year + 2)]
        if len(year_data) == 0:
            year_data = clean_data

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=year_data['Marks'], y=year_data['Rank'], mode='markers',
            marker=dict(size=6, color=year_data['Percentile'],
                        colorscale=[[0,'#f43f5e'],[0.5,'#8b5cf6'],[1,'#06b6d4']],
                        opacity=0.6, line=dict(width=0),
                        colorbar=dict(title=dict(text="Pctl", font=dict(color='#8b8b9e',size=11)),
                                      tickfont=dict(color='#8b8b9e',size=10), bgcolor='rgba(0,0,0,0)', borderwidth=0)),
            name='Historical', hovertemplate='<b>Marks:</b> %{x}<br><b>Rank:</b> %{y:,}<extra></extra>'
        ))
        # User's prediction diamond
        fig_scatter.add_trace(go.Scatter(
            x=[marks], y=[rf_rank], mode='markers+text',
            marker=dict(size=16, color='#f59e0b', symbol='diamond', line=dict(width=2, color='#fbbf24')),
            text=['You'], textposition='top center', textfont=dict(color='#f59e0b', size=12),
            name='Your Prediction', hovertemplate='<b>Marks:</b> %{x}<br><b>Rank:</b> %{y:,}<extra></extra>'
        ))
        fig_scatter.update_layout(
            title=dict(text='Marks vs Rank Distribution', font=dict(size=16, color='#f0f0f5'), x=0),
            xaxis=dict(title='Marks', color='#8b8b9e', gridcolor=GRID_COLOR),
            yaxis=dict(title='Rank', color='#8b8b9e', gridcolor=GRID_COLOR, autorange='reversed'),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=FONT_CFG,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                        font=dict(size=11, color='#8b8b9e'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=50, r=30, t=60, b=50), height=420,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with viz_col2:
        models_df = pd.DataFrame({
            'Model': ['RF Regressor', 'Poly LR', 'RF Lower', 'RF Upper'],
            'Predicted Rank': [rf_rank, lr_rank, rf_lower, rf_upper],
            'Color': ['#6366f1', '#f43f5e', '#10b981', '#06b6d4']
        })
        fig_bar = go.Figure()
        for _, row in models_df.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row['Predicted Rank']], y=[row['Model']], orientation='h',
                marker=dict(color=row['Color'], opacity=0.85),
                text=f"  {row['Predicted Rank']:,}", textposition='outside',
                textfont=dict(color=row['Color'], size=13, family='JetBrains Mono'),
                name=row['Model'], showlegend=False,
                hovertemplate=f"<b>{row['Model']}</b><br>Rank: {row['Predicted Rank']:,}<extra></extra>"
            ))
        fig_bar.update_layout(
            title=dict(text='Model Predictions Comparison', font=dict(size=16, color='#f0f0f5'), x=0),
            xaxis=dict(title='Predicted Rank', color='#8b8b9e', gridcolor=GRID_COLOR),
            yaxis=dict(color='#8b8b9e', categoryorder='array',
                       categoryarray=['RF Upper', 'RF Lower', 'Poly LR', 'RF Regressor']),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=FONT_CFG,
            margin=dict(l=110, r=80, t=60, b=50), height=420, bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Second row
    st.markdown("<br>", unsafe_allow_html=True)
    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pct_pred,
            number=dict(font=dict(size=42, color='#f0f0f5', family='JetBrains Mono'), suffix='%'),
            delta=dict(reference=90, increasing=dict(color='#10b981'), decreasing=dict(color='#f43f5e')),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor='#5a5a6e', tickfont=dict(color='#5a5a6e', size=10)),
                bar=dict(color='#6366f1', thickness=0.3), bgcolor='rgba(255,255,255,0.03)', borderwidth=0,
                steps=[dict(range=[0,50], color='rgba(244,63,94,0.1)'), dict(range=[50,80], color='rgba(245,158,11,0.1)'),
                       dict(range=[80,95], color='rgba(6,182,212,0.1)'), dict(range=[95,100], color='rgba(16,185,129,0.1)')],
                threshold=dict(line=dict(color='#f59e0b', width=3), thickness=0.8, value=pct_pred),
            ),
            title=dict(text='Your Percentile Score', font=dict(size=14, color='#8b8b9e')),
        ))
        fig_gauge.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=FONT_CFG, margin=dict(l=30,r=30,t=80,b=30), height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with viz_col4:
        marks_range_data = clean_data[clean_data['Marks'].between(marks - 5, marks + 5)]
        trend = marks_range_data.groupby('Year').agg(avg_rank=('Rank', 'mean')).reset_index()
        if len(trend) > 0:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend['Year'], y=trend['avg_rank'], mode='lines+markers',
                line=dict(color='#8b5cf6', width=3, shape='spline'),
                marker=dict(size=8, color='#8b5cf6', line=dict(width=2, color='#a78bfa')),
                fill='tozeroy', fillcolor='rgba(139,92,246,0.08)', name='Avg Rank',
                hovertemplate='<b>Year:</b> %{x}<br><b>Avg Rank:</b> %{y:,.0f}<extra></extra>'
            ))
            fig_trend.update_layout(
                title=dict(text=f'Rank Trend for ~{marks} Marks (±5)', font=dict(size=16, color='#f0f0f5'), x=0),
                xaxis=dict(title='Year', color='#8b8b9e', gridcolor=GRID_COLOR, dtick=1),
                yaxis=dict(title='Average Rank', color='#8b8b9e', gridcolor=GRID_COLOR, autorange='reversed'),
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=FONT_CFG,
                margin=dict(l=50,r=30,t=60,b=50), height=350, showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough data for this marks range to show trend.")


# ──── TAB 2: Model Performance ────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Regression Model Comparison")

    reg_comp = pd.DataFrame({
        'Model': ['Polynomial Linear Regression', 'Random Forest Regressor'],
        'R² Score': [f"{reg_metrics['lr_r2']:.4f}", f"{reg_metrics['rf_r2']:.4f}"],
        'Adjusted R²': [f"{reg_metrics['lr_adj_r2']:.4f}", f"{reg_metrics['rf_adj_r2']:.4f}"],
        'MAE': [f"{reg_metrics['lr_mae']:,.0f}", f"{reg_metrics['rf_mae']:,.0f}"],
        'RMSE': [f"{reg_metrics['lr_rmse']:,.0f}", f"{reg_metrics['rf_rmse']:,.0f}"],
    })
    st.dataframe(reg_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # RF specific metrics
    rf_c1, rf_c2, rf_c3, rf_c4 = st.columns(4)
    with rf_c1:
        st.markdown(f"""
        <div class="pred-card indigo" style="text-align:center;">
            <div class="pred-label indigo">RF R² (Train)</div>
            <div class="pred-value" style="font-size:1.8rem;">{reg_metrics['rf_r2_train']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with rf_c2:
        st.markdown(f"""
        <div class="pred-card cyan" style="text-align:center;">
            <div class="pred-label cyan">RF R² (Test)</div>
            <div class="pred-value" style="font-size:1.8rem;">{reg_metrics['rf_r2_test']:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with rf_c3:
        st.markdown(f"""
        <div class="pred-card emerald" style="text-align:center;">
            <div class="pred-label emerald">CI Coverage</div>
            <div class="pred-value" style="font-size:1.8rem;">{reg_metrics['coverage']:.1%}</div>
            <div class="pred-sub">10th–90th percentile</div>
        </div>""", unsafe_allow_html=True)
    with rf_c4:
        st.markdown(f"""
        <div class="pred-card rose" style="text-align:center;">
            <div class="pred-label rose">Avg Range Width</div>
            <div class="pred-value" style="font-size:1.8rem;">{reg_metrics['avg_range_width']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏷️ Classification Model Performance (XGBoost + SMOTE)")

    cls_c1, cls_c2, cls_c3 = st.columns(3)
    with cls_c1:
        st.markdown(f"""
        <div class="pred-card indigo" style="text-align:center;">
            <div class="pred-label indigo">Test Accuracy</div>
            <div class="pred-value" style="font-size:1.8rem;">{cls_metrics['xgb_acc']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with cls_c2:
        st.markdown(f"""
        <div class="pred-card cyan" style="text-align:center;">
            <div class="pred-label cyan">Train Accuracy</div>
            <div class="pred-value" style="font-size:1.8rem;">{cls_metrics['xgb_train_acc']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with cls_c3:
        st.markdown(f"""
        <div class="pred-card emerald" style="text-align:center;">
            <div class="pred-label emerald">Cross-Val (5-fold)</div>
            <div class="pred-value" style="font-size:1.8rem;">{cls_metrics['cv_mean']:.1%}</div>
            <div class="pred-sub">± {cls_metrics['cv_std']:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # Classification report table
    st.markdown("<br>", unsafe_allow_html=True)
    report = cls_metrics['report']
    report_rows = []
    for cat in label_encoder.classes_:
        if cat in report:
            r = report[cat]
            report_rows.append({
                'Category': cat,
                'Precision': f"{r['precision']:.3f}",
                'Recall': f"{r['recall']:.3f}",
                'F1-Score': f"{r['f1-score']:.3f}",
                'Support': int(r['support']),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


# ──── TAB 3: Data Explorer ────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([1, 2])

    with exp_col1:
        selected_year = st.selectbox("Select Year", sorted(clean_data['Year'].unique(), reverse=True), index=0)
        marks_filter = st.slider("Marks Range", 0, 300, (50, 250))

    with exp_col2:
        filtered = clean_data[
            (clean_data['Year'] == selected_year) &
            (clean_data['Marks'].between(marks_filter[0], marks_filter[1]))
        ].sort_values('Marks', ascending=False)

        st.markdown(f"""
        <div style="color:#8b8b9e;font-size:0.85rem;margin-bottom:0.5rem;">
            Showing <strong style="color:#a5b4fc;">{len(filtered)}</strong> records for year <strong style="color:#a5b4fc;">{selected_year}</strong>
        </div>""", unsafe_allow_html=True)

        st.dataframe(
            filtered[['Marks','Percentile','Rank','Total_Candidates','RankRatio','Category']].reset_index(drop=True),
            use_container_width=True, height=400,
        )

    # Heatmap
    st.markdown("<br>", unsafe_allow_html=True)
    heat_data = clean_data.copy()
    heat_data['marks_bucket'] = pd.cut(heat_data['Marks'], bins=range(0, 310, 30),
                                       labels=[f"{i}-{i+29}" for i in range(0, 300, 30)])
    heatmap_df = heat_data.groupby(['marks_bucket', 'Year'])['Rank'].mean().unstack(fill_value=0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=np.log1p(heatmap_df.values), x=heatmap_df.columns.astype(str),
        y=[str(b) for b in heatmap_df.index],
        colorscale=[[0,'#0a0a0f'],[0.2,'#1e1b4b'],[0.4,'#4338ca'],[0.6,'#6366f1'],[0.8,'#a78bfa'],[1.0,'#e0e7ff']],
        hovertemplate='<b>Marks:</b> %{y}<br><b>Year:</b> %{x}<br><b>Log Avg Rank:</b> %{z:.1f}<extra></extra>',
        colorbar=dict(title=dict(text='Log(Rank)', font=dict(color='#8b8b9e')),
                      tickfont=dict(color='#8b8b9e'), bgcolor='rgba(0,0,0,0)', borderwidth=0),
    ))
    fig_heat.update_layout(
        title=dict(text='Average Rank Heatmap (Log Scale)', font=dict(size=16, color='#f0f0f5'), x=0),
        xaxis=dict(title='Year', color='#8b8b9e'), yaxis=dict(title='Marks Range', color='#8b8b9e', autorange='reversed'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=FONT_CFG,
        margin=dict(l=80,r=30,t=60,b=50), height=450,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">
        Built with ❤️ using <strong>Streamlit</strong>, <strong>Scikit-Learn</strong>, <strong>XGBoost</strong> & <strong>SMOTE</strong>
    </div>
    <div>
        Data: JEE Main Results (2009–2026) · Models: Random Forest, Polynomial LR, XGBoost Classifier
    </div>
</div>
""", unsafe_allow_html=True)
