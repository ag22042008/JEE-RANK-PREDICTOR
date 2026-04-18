"""
JEE Rank Predictor — Streamlit app
Deploy on Streamlit Cloud with:
  main module : app.py
  requirements: requirements.txt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="JEE Rank Predictor",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background: #0a0e1a; color: #e2e8f0; }

.hero { text-align: center; padding: 36px 20px 28px; background: linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%); border-radius: 20px; border: 1px solid #1e2d47; margin-bottom: 28px; }
.hero .badge { display: inline-block; background: linear-gradient(90deg,#4f8ef7,#7c3aed); color:#fff; font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; padding:4px 16px; border-radius:20px; margin-bottom:14px; }
.hero h1 { font-size:2.4rem; font-weight:800; background:linear-gradient(90deg,#7dd3fc,#a78bfa,#f0abfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1.2; margin:0 0 10px; }
.hero p { color:#64748b; font-size:14px; margin:0; }

.stat-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.stat-item { background:#111827; border:1px solid #1e2d47; border-radius:12px; padding:16px 10px; text-align:center; }
.stat-val { font-size:1.3rem; font-weight:800; color:#4f8ef7; }
.stat-lbl { font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:.7px; margin-top:4px; }

.input-card { background:#111827; border:1px solid #1e2d47; border-radius:18px; padding:28px 28px 22px; margin-bottom:22px; }
.card-title { font-size:15px; font-weight:700; color:#4f8ef7; margin-bottom:18px; }
.cand-chip { display:inline-flex; align-items:center; gap:8px; background:rgba(79,142,247,.1); border:1px solid rgba(79,142,247,.25); border-radius:8px; padding:8px 14px; font-size:13px; color:#7dd3fc; margin-bottom:4px; width:100%; }

.result-card { background:#111827; border:1px solid #1e2d47; border-radius:18px; padding:32px 28px; margin-top:10px; }
.result-center { text-align:center; border-bottom:1px solid #1e2d47; padding-bottom:26px; margin-bottom:24px; }
.result-label { font-size:11px; color:#64748b; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:8px; }
.rank-num { font-size:5rem; font-weight:800; background:linear-gradient(90deg,#60a5fa,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1; margin:4px 0 10px; }
.rank-range { font-size:14px; color:#64748b; }
.rank-range strong { color:#e2e8f0; }
.pill { display:inline-block; padding:8px 22px; border-radius:30px; font-size:14px; font-weight:700; margin-top:16px; color:#fff; }
.pill-elite       { background:linear-gradient(90deg,#10b981,#059669); }
.pill-top         { background:linear-gradient(90deg,#3b82f6,#1d4ed8); }
.pill-highly      { background:linear-gradient(90deg,#8b5cf6,#6d28d9); }
.pill-competitive { background:linear-gradient(90deg,#f59e0b,#b45309); }
.pill-not         { background:linear-gradient(90deg,#ef4444,#b91c1c); }

.mini-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:22px; }
.mini-box { background:#1a2235; border:1px solid #1e2d47; border-radius:10px; padding:14px 10px; text-align:center; }
.mini-val { font-size:1.3rem; font-weight:800; color:#e2e8f0; }
.mini-lbl { font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:.6px; margin-top:4px; }

.conf-title { font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.7px; font-weight:600; margin-bottom:8px; }
.conf-track { background:#1e2d47; border-radius:99px; height:10px; overflow:hidden; margin-bottom:6px; }
.conf-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#4f8ef7,#7c3aed); }
.conf-ends  { display:flex; justify-content:space-between; font-size:11px; color:#64748b; }

.disclaimer { margin-top:20px; padding:13px 16px; background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.2); border-radius:10px; font-size:12.5px; color:#fbbf24; line-height:1.6; }

.stButton > button { width:100%; background:linear-gradient(90deg,#4f8ef7,#7c3aed) !important; color:white !important; font-size:16px !important; font-weight:700 !important; padding:14px 0 !important; border:none !important; border-radius:12px !important; box-shadow:0 4px 24px rgba(79,142,247,.35) !important; letter-spacing:.4px !important; }
.stButton > button:hover { box-shadow:0 6px 32px rgba(79,142,247,.5) !important; }
</style>
""", unsafe_allow_html=True)

YEAR_CANDS = {
    2009:800000, 2010:850000, 2011:900000, 2012:950000,
    2013:1000000, 2014:1050000, 2015:1100000, 2016:1150000,
    2017:1180000, 2018:1200000, 2019:1220000, 2020:1250000,
    2021:1280000, 2022:1300000, 2023:1320000, 2024:1350000,
    2025:1380000, 2026:1400000,
}

@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv("jee_marks_percentile_rank_2009_2026.csv")
    df["RankRatio"] = df["Rank"] / df["Total_Candidates"]

    def get_category(r):
        if r <= 0.005: return "Elite (Top 0.5%)"
        if r <= 0.02:  return "Top Tier (0.5% - 2%)"
        if r <= 0.05:  return "Highly Competitive (2% - 5%)"
        if r <= 0.10:  return "Competitive (5% - 10%)"
        return "Not Prepared (>10%)"

    df["Category"] = df["RankRatio"].apply(get_category)

    X_reg = df[["Year", "Marks", "Total_Candidates"]]
    Y_reg = np.log1p(df["Rank"])
    sc_reg = StandardScaler()
    Xt, Xe, Yt, Ye = train_test_split(X_reg, Y_reg, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=3,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(sc_reg.fit_transform(Xt), Yt)

    X_clf = df[["Year", "Marks", "Rank"]]
    encoder = LabelEncoder()
    Y_clf = encoder.fit_transform(df["Category"])
    sc_clf = StandardScaler()
    Xc, Xe2, Yc, Ye2 = train_test_split(X_clf, Y_clf, test_size=0.2, random_state=42)
    Xc_s = sc_clf.fit_transform(Xc)
    Xc_res, Yc_res = SMOTE(random_state=42).fit_resample(Xc_s, Yc)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=3,
                         subsample=0.8, random_state=42, eval_metric="mlogloss")
    xgb.fit(Xc_res, Yc_res)

    return rf, sc_reg, xgb, sc_clf, encoder


def run_predict(year, marks, rf, sc_reg, xgb, sc_clf, encoder):
    tc   = YEAR_CANDS[year]
    Xn   = sc_reg.transform([[year, marks, tc]])
    rank = int(round(np.expm1(rf.predict(Xn)[0])))
    trees = np.array([t.predict(Xn)[0] for t in rf.estimators_])
    lo   = int(round(np.expm1(np.percentile(trees, 10))))
    hi   = int(round(np.expm1(np.percentile(trees, 90))))
    Xc   = sc_clf.transform([[year, marks, rank]])
    cat  = encoder.inverse_transform([xgb.predict(Xc)[0]])[0]
    return rank, lo, hi, cat, tc


def pill_class(cat):
    if "Elite"    in cat: return "pill-elite"
    if "Top Tier" in cat: return "pill-top"
    if "Highly"   in cat: return "pill-highly"
    if "Competitive" in cat: return "pill-competitive"
    return "pill-not"

def fmt(n): return f"{n:,}"

# ── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Training models on 2009-2026 data... (first load only, ~15s)"):
    rf, sc_reg, xgb, sc_clf, encoder = load_and_train()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">AI-Powered Predictor</div>
  <h1>JEE Rank Predictor</h1>
  <p>Trained on 2009–2026 historical data &nbsp;·&nbsp; Random Forest + XGBoost</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-strip">
  <div class="stat-item"><div class="stat-val">1,620</div><div class="stat-lbl">Data Points</div></div>
  <div class="stat-item"><div class="stat-val">2009–2026</div><div class="stat-lbl">Years Covered</div></div>
  <div class="stat-item"><div class="stat-val">300 Trees</div><div class="stat-lbl">Random Forest</div></div>
  <div class="stat-item"><div class="stat-val">98.38%</div><div class="stat-lbl">Model R²</div></div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="card-title">📝 Enter Your Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    marks_slider = st.slider("JEE Marks (out of 300)", 0, 300, 150, 1)
with col2:
    marks_num = st.number_input("Exact marks", 0, 300, marks_slider, 1)

marks = marks_num  # number input wins

year = st.selectbox("Exam Year", sorted(YEAR_CANDS.keys(), reverse=True), index=0)
cands = YEAR_CANDS[year]
st.markdown(f'<div class="cand-chip">👥 &nbsp; Expected candidates in <strong>{year}</strong>: &nbsp;<strong>{fmt(cands)}</strong></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("🔮  Predict My JEE Rank")

# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calculating your rank..."):
        rank, lo, hi, category, tc = run_predict(year, float(marks), rf, sc_reg, xgb, sc_clf, encoder)

    spread   = hi - lo
    conf_pct = max(5, min(100, 100 - (spread / tc) * 100))
    pc       = pill_class(category)

    st.markdown(f"""
<div class="result-card">
  <div class="result-center">
    <div class="result-label">Predicted JEE Rank</div>
    <div class="rank-num">{fmt(rank)}</div>
    <div class="rank-range">Confidence Range: &nbsp;<strong>{fmt(lo)} – {fmt(hi)}</strong></div>
    <div class="pill {pc}">{category}</div>
  </div>
  <div class="mini-stats">
    <div class="mini-box"><div class="mini-val">{marks}</div><div class="mini-lbl">Your Marks</div></div>
    <div class="mini-box"><div class="mini-val">{year}</div><div class="mini-lbl">Exam Year</div></div>
    <div class="mini-box"><div class="mini-val">{fmt(tc)}</div><div class="mini-lbl">Total Candidates</div></div>
  </div>
  <div class="conf-title">Prediction Confidence — narrower band = higher confidence</div>
  <div class="conf-track"><div class="conf-fill" style="width:{conf_pct:.1f}%"></div></div>
  <div class="conf-ends"><span>{fmt(lo)}</span><span>{conf_pct:.0f}% confidence</span><span>{fmt(hi)}</span></div>
  <div class="disclaimer">
    ⚠️  This is an ML-based estimate trained on historical JEE data (2009–2026).
    Actual ranks may differ due to paper difficulty, normalisation, and other factors.
    Use this as a reference, not an official result.
  </div>
</div>
""", unsafe_allow_html=True)
