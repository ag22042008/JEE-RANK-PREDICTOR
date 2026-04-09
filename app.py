# ============================================================
#  JEE Rank Predictor — Production Streamlit App
#  Dataset : jee_marks_percentile_rank_2009_2026.csv
#  Models  : RandomForest (Regression) + XGBoost (Classification)
#  Author  : Auto-generated from Jupyter Notebook
# ============================================================

# ── SECTION 1: IMPORTS ──────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ── SECTION 2: PAGE CONFIGURATION ───────────────────────────
st.set_page_config(
    page_title="JEE Rank Predictor · 2009–2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── SECTION 3: CUSTOM CSS — Glassmorphism Dark Theme ────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Global background ── */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 40%, #0a1628 70%, #080d18 100%);
            color: #e2e8f0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: rgba(10,14,26,0.92) !important;
            border-right: 1px solid rgba(99,179,237,0.15);
            backdrop-filter: blur(20px);
        }

        /* ── Glass card ── */
        .glass-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(99,179,237,0.18);
            border-radius: 16px;
            padding: 24px 28px;
            margin: 10px 0;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.45),
                        inset 0 1px 0 rgba(255,255,255,0.06);
            transition: border-color .3s ease, box-shadow .3s ease;
        }
        .glass-card:hover {
            border-color: rgba(99,179,237,0.40);
            box-shadow: 0 8px 40px rgba(66,153,225,0.18);
        }

        /* ── Metric / result cards ── */
        .metric-card {
            background: linear-gradient(135deg,
                rgba(66,153,225,0.12) 0%,
                rgba(49,130,206,0.06) 100%);
            border: 1px solid rgba(99,179,237,0.28);
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 36px rgba(66,153,225,0.22);
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #63b3ed, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #90cdf4;
            text-transform: uppercase;
            letter-spacing: .09em;
            margin-top: 6px;
            font-weight: 600;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #718096;
            margin-top: 4px;
        }

        /* ── Result hero card ── */
        .result-hero {
            background: linear-gradient(135deg,
                rgba(128,90,213,0.20) 0%,
                rgba(66,153,225,0.15) 50%,
                rgba(49,151,149,0.12) 100%);
            border: 1px solid rgba(159,122,234,0.40);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
            box-shadow: 0 0 60px rgba(128,90,213,0.15);
        }
        .result-rank {
            font-size: 4rem;
            font-weight: 900;
            background: linear-gradient(135deg, #f6e05e, #f6ad55, #fc8181);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }
        .result-label {
            font-size: 1rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: .12em;
            margin-top: 8px;
        }

        /* ── Category badge ── */
        .category-badge {
            display: inline-block;
            padding: 8px 22px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
            letter-spacing: .04em;
            margin-top: 12px;
        }
        .badge-tit     { background: linear-gradient(135deg,#f6e05e,#f6ad55); color:#1a202c; }
        .badge-iit     { background: linear-gradient(135deg,#68d391,#38a169); color:#1a202c; }
        .badge-nit-top { background: linear-gradient(135deg,#63b3ed,#3182ce); color:#fff; }
        .badge-nit     { background: linear-gradient(135deg,#a78bfa,#6b46c1); color:#fff; }
        .badge-gfti    { background: linear-gradient(135deg,#fc8181,#c53030); color:#fff; }
        .badge-lower   { background: linear-gradient(135deg,#a0aec0,#4a5568); color:#fff; }

        /* ── Section headings ── */
        .section-title {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -.02em;
            background: linear-gradient(90deg, #63b3ed 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 4px;
        }
        .section-sub {
            color: #718096;
            font-size: 0.92rem;
            margin-bottom: 18px;
        }

        /* ── Insight callout ── */
        .insight-box {
            background: rgba(99,179,237,0.07);
            border-left: 3px solid #63b3ed;
            border-radius: 0 10px 10px 0;
            padding: 12px 18px;
            margin: 8px 0;
            font-size: 0.9rem;
            color: #cbd5e0;
        }
        .insight-box b { color: #90cdf4; }

        /* ── Animated predict button ── */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #4299e1 0%, #6b46c1 100%);
            color: #fff;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            border-radius: 12px;
            padding: 14px 36px;
            width: 100%;
            cursor: pointer;
            transition: all .3s ease;
            box-shadow: 0 4px 20px rgba(66,153,225,0.35);
            letter-spacing: .04em;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(135deg, #63b3ed 0%, #805ad5 100%);
            box-shadow: 0 6px 28px rgba(128,90,213,0.5);
            transform: translateY(-2px);
        }
        div[data-testid="stButton"] > button:active {
            transform: translateY(0px);
        }

        /* ── Sliders & widgets ── */
        .stSlider > div > div > div { background: #4299e1 !important; }
        .stSelectbox > div > div { background: rgba(255,255,255,0.04) !important; }
        div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.05) !important;
            border-color: rgba(99,179,237,0.30) !important;
        }
        div[data-testid="stNumberInput"] > div > div > input {
            background: rgba(255,255,255,0.05) !important;
            border-color: rgba(99,179,237,0.30) !important;
            color: #e2e8f0 !important;
        }

        /* ── Tabs ── */
        button[data-baseweb="tab"] {
            color: #718096 !important;
            font-weight: 600 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #63b3ed !important;
            border-bottom-color: #63b3ed !important;
        }

        /* ── Divider ── */
        hr { border-color: rgba(99,179,237,0.12) !important; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }

        /* ── Hero banner ── */
        .hero-banner {
            text-align: center;
            padding: 48px 24px 36px;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #f6ad55 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            margin-bottom: 16px;
        }
        .hero-sub {
            font-size: 1.1rem;
            color: #718096;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.7;
        }

        /* ── Warning / info bars ── */
        .info-bar {
            background: rgba(66,153,225,0.10);
            border: 1px solid rgba(66,153,225,0.25);
            border-radius: 10px;
            padding: 10px 18px;
            font-size: 0.88rem;
            color: #90cdf4;
            margin: 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── SECTION 4: CONSTANTS & MAPPINGS ─────────────────────────
CATEGORY_BINS   = [0, 2500, 10000, 35000, 150000, 500000, 1_500_000]
CATEGORY_LABELS = ["Top IIT", "IIT Zone", "NIT Top", "NIT/IIIT", "GFTI/State", "Lower Tier"]
LABEL_MAP       = {l: i for i, l in enumerate(CATEGORY_LABELS)}
INV_LABEL_MAP   = {i: l for i, l in enumerate(CATEGORY_LABELS)}

BADGE_CLASS = {
    "Top IIT"    : "badge-tit",
    "IIT Zone"   : "badge-iit",
    "NIT Top"    : "badge-nit-top",
    "NIT/IIIT"   : "badge-nit",
    "GFTI/State" : "badge-gfti",
    "Lower Tier" : "badge-lower",
}

# Total candidates trend (actual 2009-2026, linear extrapolation beyond)
_YEAR_CANDIDATES = {yr: int(800_000 + (yr - 2009) * 32_000)
                    for yr in range(2009, 2060)}
# Override with exact known values
_KNOWN = {
    2009:800000,2010:850000,2011:900000,2012:950000,2013:1000000,
    2014:1050000,2015:1100000,2016:1150000,2017:1180000,2018:1200000,
    2019:1220000,2020:1250000,2021:1280000,2022:1300000,2023:1320000,
    2024:1350000,2025:1380000,2026:1400000,
}
_YEAR_CANDIDATES.update(_KNOWN)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#cbd5e0", family="Inter, Segoe UI, sans-serif"),
    xaxis=dict(gridcolor="rgba(99,179,237,0.10)", zerolinecolor="rgba(99,179,237,0.10)"),
    yaxis=dict(gridcolor="rgba(99,179,237,0.10)", zerolinecolor="rgba(99,179,237,0.10)"),
    colorway=["#63b3ed","#a78bfa","#68d391","#f6ad55","#fc8181","#76e4f7"],
)

CATEGORY_COLOR_MAP = {
    "Top IIT"   : "#f6e05e",
    "IIT Zone"  : "#68d391",
    "NIT Top"   : "#63b3ed",
    "NIT/IIIT"  : "#a78bfa",
    "GFTI/State": "#fc8181",
    "Lower Tier": "#a0aec0",
}


# ── SECTION 5: DATA LOADING & PREPROCESSING ─────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load CSV, perform all feature engineering, return clean DataFrame."""
    df = pd.read_csv(path)

    # ── Feature Engineering ──────────────────────────────────
    # RankRatio: normalised rank (0 = best, 1 = worst)
    df["RankRatio"] = df["Rank"] / df["Total_Candidates"]

    # Category: rank-based college admission tier label
    df["Category"] = pd.cut(
        df["Rank"],
        bins=CATEGORY_BINS,
        labels=CATEGORY_LABELS,
        right=True,
    ).astype(str)

    # Numeric category label for ML
    df["CategoryCode"] = df["Category"].map(LABEL_MAP)

    # Marks bucket (useful for groupby insights)
    df["MarksBucket"] = pd.cut(
        df["Marks"],
        bins=[0, 50, 100, 150, 200, 250, 300],
        labels=["0–50", "51–100", "101–150", "151–200", "201–250", "251–300"],
    ).astype(str)

    # Percentile band
    df["PctBand"] = pd.cut(
        df["Percentile"],
        bins=[0, 50, 80, 90, 95, 99, 100],
        labels=["<50", "50–80", "80–90", "90–95", "95–99", "99–100"],
    ).astype(str)

    return df


# ── SECTION 6: MODEL TRAINING ────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_all_models(df: pd.DataFrame):
    """
    Train RandomForest regressor (log1p target) and XGBoost classifier.
    Returns dict with models, scalers, metrics, feature importance arrays.
    """
    # ── 6A. REGRESSION ── predict log1p(Rank) from Marks, Year, Total_Candidates
    REG_FEATURES = ["Marks", "Year", "Total_Candidates"]
    X_reg = df[REG_FEATURES].values
    y_reg = np.log1p(df["Rank"].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    scaler_reg = StandardScaler()
    X_tr_s = scaler_reg.fit_transform(X_tr)
    X_te_s = scaler_reg.transform(X_te)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)

    # Metrics on test set
    y_pred_log = rf.predict(X_te_s)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_te)
    reg_r2  = r2_score(y_true, y_pred)
    reg_mae = mean_absolute_error(y_true, y_pred)
    reg_rmse= np.sqrt(mean_squared_error(y_true, y_pred))

    # Feature importance (regression)
    rf_fi = dict(zip(REG_FEATURES, rf.feature_importances_))

    # ── 6B. CLASSIFICATION ── predict Category from Marks, Year,
    #         Total_Candidates, RankRatio
    CLS_FEATURES = ["Marks", "Year", "Total_Candidates", "RankRatio"]
    X_cls = df[CLS_FEATURES].values
    y_cls = df["CategoryCode"].values

    scaler_cls = StandardScaler()
    X_cls_s    = scaler_cls.fit_transform(X_cls)

    # SMOTE to balance classes (k_neighbors=3 because smallest class has ~18 rows)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_cls_s, y_cls)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_res, y_res)

    # Classification metrics on full (unbalanced) set
    cls_preds = xgb.predict(scaler_cls.transform(X_cls))
    cls_acc   = accuracy_score(y_cls, cls_preds)
    cls_report = classification_report(
        y_cls, cls_preds,
        target_names=CATEGORY_LABELS,
        output_dict=True,
    )
    xgb_fi = dict(zip(CLS_FEATURES, xgb.feature_importances_))

    return dict(
        rf=rf, scaler_reg=scaler_reg, reg_features=REG_FEATURES,
        reg_r2=reg_r2, reg_mae=reg_mae, reg_rmse=reg_rmse, rf_fi=rf_fi,
        xgb=xgb, scaler_cls=scaler_cls, cls_features=CLS_FEATURES,
        cls_acc=cls_acc, cls_report=cls_report, xgb_fi=xgb_fi,
    )


# ── SECTION 7: PREDICTION HELPERS ───────────────────────────
def predict_rank(models: dict, marks: int, year: int) -> dict:
    """
    Predict JEE rank, uncertainty band (P10–P90) from individual trees,
    then predict the college tier category using the predicted rank.
    """
    total_candidates = _YEAR_CANDIDATES.get(year, 1_400_000)

    # ── Regression ──
    X_in = np.array([[marks, year, total_candidates]])
    X_sc = models["scaler_reg"].transform(X_in)

    # Collect individual tree predictions for uncertainty
    tree_preds = np.array([
        np.expm1(t.predict(X_sc)[0])
        for t in models["rf"].estimators_
    ])
    pred_rank = int(np.expm1(models["rf"].predict(X_sc)[0]))
    rank_p10  = int(np.percentile(tree_preds, 10))
    rank_p90  = int(np.percentile(tree_preds, 90))

    # Clip to [1, total_candidates]
    pred_rank = max(1, min(pred_rank, total_candidates))
    rank_p10  = max(1, min(rank_p10,  total_candidates))
    rank_p90  = max(1, min(rank_p90,  total_candidates))

    # ── Classification ──
    rank_ratio = pred_rank / total_candidates
    X_cls_in   = np.array([[marks, year, total_candidates, rank_ratio]])
    X_cls_sc   = models["scaler_cls"].transform(X_cls_in)
    cat_code   = models["xgb"].predict(X_cls_sc)[0]
    cat_proba  = models["xgb"].predict_proba(X_cls_sc)[0]
    category   = INV_LABEL_MAP[int(cat_code)]

    # Predicted percentile from rank
    pred_pct = round(100 * (1 - pred_rank / total_candidates), 4)
    pred_pct = max(0.0, min(100.0, pred_pct))

    return dict(
        rank=pred_rank,
        rank_p10=rank_p10,
        rank_p90=rank_p90,
        percentile=pred_pct,
        category=category,
        cat_proba=cat_proba,
        total_candidates=total_candidates,
        rank_ratio=rank_ratio,
    )


# ── SECTION 8: EDA CHARTS ───────────────────────────────────
def chart_marks_distribution(df: pd.DataFrame) -> go.Figure:
    """Marks distribution: histogram + KDE overlay"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Marks"],
        nbinsx=60,
        name="Frequency",
        marker=dict(color="#63b3ed", opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
        hovertemplate="Marks: %{x}<br>Count: %{y}<extra></extra>",
    ))
    # KDE approximation via density histogram
    counts, bins = np.histogram(df["Marks"], bins=80)
    bin_mid = (bins[:-1] + bins[1:]) / 2
    kde_y   = counts / counts.max() * df["Marks"].max() * 0.35
    fig.add_trace(go.Scatter(
        x=bin_mid, y=kde_y,
        mode="lines", name="Density",
        line=dict(color="#a78bfa", width=2.5),
        hovertemplate="Marks: %{x:.0f}<br>Density: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="📊 Marks Distribution (All Years)", font=dict(size=15)),
        xaxis_title="Marks", yaxis_title="Count",
        **PLOTLY_TEMPLATE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


def chart_rank_distribution(df: pd.DataFrame) -> go.Figure:
    """Log-scale rank distribution by category."""
    fig = px.histogram(
        df, x="Rank", color="Category",
        nbins=80,
        log_x=True,
        color_discrete_map=CATEGORY_COLOR_MAP,
        title="📈 Rank Distribution by College Tier (Log Scale)",
        labels={"Rank": "AIR Rank (log scale)", "count": "Count"},
    )
    fig.update_traces(opacity=0.80)
    fig.update_layout(**PLOTLY_TEMPLATE, height=380,
                      legend_title_text="College Tier")
    return fig


def chart_marks_vs_rank(df: pd.DataFrame) -> go.Figure:
    """Scatter: Marks vs Rank, coloured by year."""
    fig = px.scatter(
        df, x="Marks", y="Rank",
        color="Year",
        color_continuous_scale="Blues",
        opacity=0.55,
        title="🔵 Marks vs AIR Rank (All Years)",
        labels={"Marks": "JEE Marks", "Rank": "AIR Rank"},
        hover_data={"Percentile": ":.2f", "Total_Candidates": ":,"},
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**PLOTLY_TEMPLATE, height=400,
                      coloraxis_colorbar=dict(title="Year", thickness=12))
    return fig


def chart_marks_vs_percentile(df: pd.DataFrame) -> go.Figure:
    """Scatter: Marks vs Percentile with regression trend line."""
    fig = px.scatter(
        df, x="Marks", y="Percentile",
        color="Category",
        color_discrete_map=CATEGORY_COLOR_MAP,
        trendline="lowess",
        trendline_color_override="#f6e05e",
        opacity=0.50,
        title="📉 Marks vs Percentile (LOWESS Trend)",
        labels={"Marks": "JEE Marks", "Percentile": "Percentile Score"},
        hover_data={"Year": True, "Rank": ":,"},
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
    fig.update_layout(**PLOTLY_TEMPLATE, height=400, legend_title_text="Tier")
    return fig


def chart_total_candidates_trend(df: pd.DataFrame) -> go.Figure:
    """Year-wise total candidates growth (area chart)."""
    tc = df.groupby("Year")["Total_Candidates"].first().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tc["Year"], y=tc["Total_Candidates"],
        mode="lines+markers",
        fill="tozeroy",
        name="Total Candidates",
        line=dict(color="#63b3ed", width=3),
        fillcolor="rgba(99,179,237,0.12)",
        marker=dict(size=8, color="#63b3ed",
                    line=dict(color="#fff", width=1.5)),
        hovertemplate="Year: %{x}<br>Candidates: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="📅 Total JEE Candidates by Year (2009–2026)",
                   font=dict(size=15)),
        xaxis_title="Year", yaxis_title="Total Candidates",
        **PLOTLY_TEMPLATE, height=360,
    )
    return fig


def chart_median_marks_per_year(df: pd.DataFrame) -> go.Figure:
    """Box plot of Marks distribution per year."""
    fig = px.box(
        df, x="Year", y="Marks",
        color_discrete_sequence=["#a78bfa"],
        title="📦 Marks Distribution per Year (Box Plot)",
        labels={"Marks": "JEE Marks", "Year": "Year"},
        points=False,
    )
    fig.update_traces(fillcolor="rgba(167,139,250,0.20)",
                      line=dict(color="#a78bfa", width=1.5))
    fig.update_layout(**PLOTLY_TEMPLATE, height=400)
    return fig


def chart_percentile_by_category(df: pd.DataFrame) -> go.Figure:
    """Violin: Percentile distribution per category tier."""
    fig = px.violin(
        df, x="Category", y="Percentile",
        color="Category",
        color_discrete_map=CATEGORY_COLOR_MAP,
        box=True,
        points=False,
        title="🎻 Percentile Distribution per College Tier",
        labels={"Percentile": "Percentile", "Category": "College Tier"},
        category_orders={"Category": CATEGORY_LABELS},
    )
    fig.update_layout(**PLOTLY_TEMPLATE, height=420, showlegend=False)
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap of numeric features."""
    num_cols   = ["Marks", "Percentile", "Rank", "Total_Candidates",
                  "RankRatio", "Year"]
    corr_df    = df[num_cols].corr()
    text_vals  = [[f"{v:.2f}" for v in row] for row in corr_df.values]

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=num_cols, y=num_cols,
        colorscale=[
            [0.0,  "#1a365d"],
            [0.25, "#2c5282"],
            [0.5,  "#1a202c"],
            [0.75, "#553c9a"],
            [1.0,  "#6b46c1"],
        ],
        zmid=0,
        text=text_vals,
        texttemplate="%{text}",
        hovertemplate="%{x} ↔ %{y} : %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🔥 Feature Correlation Matrix", font=dict(size=15)),
        **PLOTLY_TEMPLATE, height=420,
        xaxis=dict(tickangle=-30),
    )
    return fig


def chart_rf_feature_importance(rf_fi: dict) -> go.Figure:
    """Horizontal bar chart for RandomForest feature importance."""
    items = sorted(rf_fi.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h",
        marker=dict(
            color=[v for _, v in items],
            colorscale="Blues",
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🌲 RF Regressor — Feature Importance",
                   font=dict(size=15)),
        xaxis_title="Importance", yaxis_title="Feature",
        **PLOTLY_TEMPLATE, height=300,
    )
    return fig


def chart_xgb_feature_importance(xgb_fi: dict) -> go.Figure:
    """Horizontal bar chart for XGBoost feature importance."""
    items = sorted(xgb_fi.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h",
        marker=dict(
            color=[v for _, v in items],
            colorscale="Purples",
            showscale=False,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🚀 XGBoost Classifier — Feature Importance",
                   font=dict(size=15)),
        xaxis_title="Importance", yaxis_title="Feature",
        **PLOTLY_TEMPLATE, height=300,
    )
    return fig


def chart_rank_vs_rankratio(df: pd.DataFrame) -> go.Figure:
    """RankRatio vs Marks to show normalised competition over years."""
    fig = px.scatter(
        df, x="Marks", y="RankRatio",
        color="Year",
        color_continuous_scale="Viridis",
        opacity=0.55,
        title="⚖️ Normalised Rank Ratio vs Marks (Year-wise)",
        labels={"RankRatio": "Rank Ratio (Rank / Total Candidates)",
                "Marks": "JEE Marks"},
        hover_data={"Rank": ":,", "Total_Candidates": ":,"},
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**PLOTLY_TEMPLATE, height=400,
                      coloraxis_colorbar=dict(title="Year", thickness=12))
    return fig


def chart_category_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of category distribution."""
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.pie(
        counts, names="Category", values="Count",
        hole=0.48,
        color="Category",
        color_discrete_map=CATEGORY_COLOR_MAP,
        title="🍩 Dataset Composition by College Tier",
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=12,
        hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(**PLOTLY_TEMPLATE, height=380,
                      legend_title_text="College Tier")
    return fig


def chart_marks_bucket_avg_rank(df: pd.DataFrame) -> go.Figure:
    """Bar chart: avg rank per marks bucket."""
    grp = (df.groupby("MarksBucket", observed=True)["Rank"]
             .mean().reset_index()
             .rename(columns={"Rank": "Avg Rank"}))
    order = ["0–50","51–100","101–150","151–200","201–250","251–300"]
    grp["MarksBucket"] = pd.Categorical(grp["MarksBucket"],
                                        categories=order, ordered=True)
    grp = grp.sort_values("MarksBucket")
    fig = px.bar(
        grp, x="MarksBucket", y="Avg Rank",
        color="Avg Rank",
        color_continuous_scale="RdYlGn_r",
        title="🎯 Average AIR Rank by Marks Bucket",
        labels={"MarksBucket": "Marks Range", "Avg Rank": "Average AIR Rank"},
        text_auto=".0f",
    )
    fig.update_traces(textfont_size=11,
                      hovertemplate="Marks: %{x}<br>Avg Rank: %{y:,.0f}<extra></extra>")
    fig.update_layout(**PLOTLY_TEMPLATE, height=380,
                      coloraxis_showscale=False)
    return fig


def chart_year_median_rank(df: pd.DataFrame) -> go.Figure:
    """Median rank by year — shows competition trend."""
    med = df.groupby("Year")["Rank"].median().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=med["Year"], y=med["Rank"],
        mode="lines+markers",
        name="Median Rank",
        line=dict(color="#f6ad55", width=3),
        marker=dict(size=9, color="#f6ad55",
                    line=dict(color="#fff", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(246,173,85,0.08)",
        hovertemplate="Year: %{x}<br>Median Rank: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="📆 Median AIR Rank Trend (2009–2026)",
                   font=dict(size=15)),
        xaxis_title="Year", yaxis_title="Median AIR Rank",
        **PLOTLY_TEMPLATE, height=360,
    )
    return fig


# ── SECTION 9: PAGE RENDERS ──────────────────────────────────
def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
          <div class="hero-title">🎓 JEE Rank Intelligence</div>
          <div class="hero-sub">
            ML-powered AIR Rank predictor trained on 17 years of JEE data (2009–2026).<br>
            Enter your marks to predict rank, percentile, and college admission tier.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_kpis(models: dict, df: pd.DataFrame):
    """Top-level KPI strip — dataset & model quality metrics."""
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, f"{len(df):,}",       "Total Records",      "2009–2026"),
        (c2, "18",                  "Years Covered",      "2009 to 2026"),
        (c3, f"{models['reg_r2']:.4f}",  "RF Regressor R²",  "on 20 % hold-out"),
        (c4, f"{models['reg_mae']:,.0f}", "RF Mean Abs Error", "predicted rank units"),
        (c5, f"{models['cls_acc']*100:.1f}%", "XGB Accuracy",   "6-class tier"),
    ]
    for col, val, label, sub in kpis:
        with col:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="metric-value">{val}</div>
                      <div class="metric-label">{label}</div>
                      <div class="metric-sub">{sub}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


def render_eda_section(df: pd.DataFrame, models: dict):
    """Full EDA / Insights section with all charts."""
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Deep dive into JEE trends, distributions and correlations (2009–2026)</div>',
                unsafe_allow_html=True)

    # ── Row 1: Marks + Rank distributions ──
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_marks_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_rank_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Insight callouts ──
    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 1 — Uniform Marks Sampling:</b>
          The dataset contains a perfectly uniform spread of marks from 2 to 297 across all
          18 years (90 rows per year), making it an ideal benchmark dataset for rank modelling.
        </div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 2 — Heavy-tailed Rank Distribution:</b>
          Rank follows a right-skewed distribution (majority of candidates concentrated in
          the 50K–500K range), confirming the competitive nature of JEE at the top ranks.
        </div>""", unsafe_allow_html=True)

    # ── Row 2: Marks vs Rank + Marks vs Percentile ──
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_marks_vs_rank(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c4:
        st.plotly_chart(chart_marks_vs_percentile(df),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 3 — High Marks–Rank Correlation (r = −0.898):</b>
          Marks and Rank are strongly negatively correlated. Above ~200 marks, the rank
          curve is exponentially steep — a 10-mark gain above 200 can improve rank
          by 20,000+ positions.
        </div>""", unsafe_allow_html=True)

    # ── Row 3: Total candidates trend + median rank trend ──
    st.markdown("---")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(chart_total_candidates_trend(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c6:
        st.plotly_chart(chart_year_median_rank(df),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 4 — Competition Is Growing:</b>
          Total JEE candidates grew from 800K (2009) to 1.4M (2026) — a <b>75 % increase</b>.
          Correspondingly, the median rank shifts upward, meaning the same absolute marks
          yields a lower percentile today than a decade ago.
        </div>""", unsafe_allow_html=True)

    # ── Row 4: Box marks per year + RankRatio scatter ──
    st.markdown("---")
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(chart_median_marks_per_year(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c8:
        st.plotly_chart(chart_rank_vs_rankratio(df),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 5 — Stable Mark Distribution, Growing Pool:</b>
          The mark distribution is consistent across all years (same range, similar spread),
          but the RankRatio plot confirms that identical marks command a relatively worse
          normalised rank as the candidate pool expands year on year.
        </div>""", unsafe_allow_html=True)

    # ── Row 5: Percentile violin + Category pie ──
    st.markdown("---")
    c9, c10 = st.columns(2)
    with c9:
        st.plotly_chart(chart_percentile_by_category(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c10:
        st.plotly_chart(chart_category_pie(df),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Row 6: Avg rank per marks bucket ──
    st.markdown("---")
    c11, c12 = st.columns(2)
    with c11:
        st.plotly_chart(chart_marks_bucket_avg_rank(df),
                        use_container_width=True, config={"displayModeBar": False})
    with c12:
        st.plotly_chart(chart_correlation_heatmap(df),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """<div class="insight-box">
          💡 <b>Insight 6 — Percentile & Rank Are Almost Perfectly Anticorrelated (r = −0.977):</b>
          As expected. The slight deviation from −1.0 comes from year-to-year variation in
          total candidates — the same percentile maps to slightly different ranks in different years.
        </div>""", unsafe_allow_html=True)

    # ── Feature importances ──
    st.markdown("---")
    st.markdown('<div class="section-title">🤖 Model Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What each model learned about the features</div>',
                unsafe_allow_html=True)

    ci1, ci2 = st.columns(2)
    with ci1:
        st.plotly_chart(chart_rf_feature_importance(models["rf_fi"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            """<div class="insight-box">
              💡 <b>RF Regressor:</b> <b>Marks</b> dominates rank prediction.
              Year and Total_Candidates contribute but cannot override a candidate's score.
            </div>""", unsafe_allow_html=True)
    with ci2:
        st.plotly_chart(chart_xgb_feature_importance(models["xgb_fi"]),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            """<div class="insight-box">
              💡 <b>XGBoost Classifier:</b> <b>RankRatio</b> (normalised rank) is the
              dominant signal for tier classification — the absolute rank relative to total
              candidates perfectly separates college tiers.
            </div>""", unsafe_allow_html=True)

    # ── Classification report as styled table ──
    st.markdown("---")
    st.markdown('<div class="section-title">📋 XGBoost Classification Report</div>',
                unsafe_allow_html=True)
    cr = models["cls_report"]
    rows = []
    for cls in CATEGORY_LABELS:
        if cls in cr:
            rows.append({
                "College Tier"  : cls,
                "Precision"     : f"{cr[cls]['precision']:.3f}",
                "Recall"        : f"{cr[cls]['recall']:.3f}",
                "F1-Score"      : f"{cr[cls]['f1-score']:.3f}",
                "Support"       : int(cr[cls]["support"]),
            })
    report_df = pd.DataFrame(rows)
    st.dataframe(
        report_df.style.set_properties(**{"color": "#e2e8f0"}),
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_section(models: dict, df: pd.DataFrame):
    """Prediction UI — input form + result display."""
    st.markdown('<div class="section-title">🎯 Predict Your JEE Rank</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter your details below to get AI-powered rank prediction with uncertainty range</div>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        col_inp, col_out = st.columns([1, 1.3], gap="large")

        with col_inp:
            st.markdown("#### ✏️ Your Details")

            marks = st.slider(
                "📝 JEE Marks (out of 300)",
                min_value=0, max_value=300, value=150, step=1,
                help="Enter your actual or expected JEE Mains marks",
            )
            year = st.selectbox(
                "📅 Exam Year",
                options=list(range(2009, 2030)),
                index=list(range(2009, 2030)).index(2026),
                help="Select the year of your JEE Mains exam",
            )

            # Live preview of total candidates
            tc = _YEAR_CANDIDATES.get(year, 1_400_000)
            st.markdown(
                f'<div class="info-bar">📊 Estimated total candidates in {year}: '
                f'<b>{tc:,}</b></div>',
                unsafe_allow_html=True,
            )

            # Marks context hints
            if marks >= 250:
                hint = "🔥 Elite zone — Top IIT / CS branch territory"
                hint_col = "#f6e05e"
            elif marks >= 200:
                hint = "🟢 Excellent — Strong IIT / NIT Top chances"
                hint_col = "#68d391"
            elif marks >= 150:
                hint = "🔵 Good — NIT+ / IIIT prospects"
                hint_col = "#63b3ed"
            elif marks >= 100:
                hint = "🟣 Moderate — NIT / IIIT / GFTI range"
                hint_col = "#a78bfa"
            else:
                hint = "🔴 Lower percentile — Prepare for improvement"
                hint_col = "#fc8181"

            st.markdown(
                f'<div style="color:{hint_col};font-size:.9rem;'
                f'padding:8px 0;font-weight:600;">{hint}</div>',
                unsafe_allow_html=True,
            )

            predict_clicked = st.button("⚡ Predict My Rank", key="predict_btn")

        with col_out:
            if predict_clicked or "last_prediction" in st.session_state:
                if predict_clicked:
                    result = predict_rank(models, marks, year)
                    st.session_state["last_prediction"] = result
                    st.session_state["last_marks"] = marks
                    st.session_state["last_year"] = year
                else:
                    result = st.session_state["last_prediction"]
                    marks  = st.session_state.get("last_marks", marks)
                    year   = st.session_state.get("last_year", year)

                cat     = result["category"]
                badge   = BADGE_CLASS.get(cat, "badge-lower")

                # ── Result hero ──
                st.markdown(
                    f"""
                    <div class="result-hero">
                      <div class="metric-label">🏆 Predicted AIR Rank</div>
                      <div class="result-rank">{result['rank']:,}</div>
                      <div style="margin:10px 0 2px;color:#a0aec0;font-size:.85rem;">
                        Uncertainty Band (P10–P90)
                      </div>
                      <div style="color:#90cdf4;font-size:1.05rem;font-weight:700;">
                        {result['rank_p10']:,}  →  {result['rank_p90']:,}
                      </div>
                      <div style="margin-top:14px;">
                        <span class="category-badge {badge}">{cat}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Secondary metrics ──
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card">
                          <div class="metric-value">{result['percentile']:.2f}%</div>
                          <div class="metric-label">Percentile</div>
                        </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(
                        f"""<div class="metric-card">
                          <div class="metric-value">{result['rank_ratio']*100:.2f}%</div>
                          <div class="metric-label">Rank Ratio</div>
                          <div class="metric-sub">rank / total candidates</div>
                        </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(
                        f"""<div class="metric-card">
                          <div class="metric-value">{result['total_candidates']//1000}K</div>
                          <div class="metric-label">Total Candidates</div>
                          <div class="metric-sub">{year}</div>
                        </div>""", unsafe_allow_html=True)

                # ── Category probability bar chart ──
                st.markdown("<br>", unsafe_allow_html=True)
                prob_fig = go.Figure(go.Bar(
                    x=CATEGORY_LABELS,
                    y=result["cat_proba"] * 100,
                    marker=dict(
                        color=[CATEGORY_COLOR_MAP[c] for c in CATEGORY_LABELS],
                        opacity=0.85,
                    ),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                ))
                prob_fig.update_layout(
                    title=dict(text="🎲 College Tier Probability Distribution",
                               font=dict(size=13)),
                    xaxis_title="College Tier",
                    yaxis_title="Probability (%)",
                    **PLOTLY_TEMPLATE,
                    height=280,
                    margin=dict(t=40, b=60, l=40, r=20),
                )
                st.plotly_chart(prob_fig,
                                use_container_width=True,
                                config={"displayModeBar": False})

            else:
                # ── Placeholder state ──
                st.markdown(
                    """
                    <div style="height:300px;display:flex;flex-direction:column;
                         align-items:center;justify-content:center;
                         border:1px dashed rgba(99,179,237,0.25);
                         border-radius:16px;color:#4a5568;font-size:1rem;">
                      <span style="font-size:3rem;">🎯</span><br>
                      Enter your marks &amp; year, then click<br>
                      <b style="color:#63b3ed;">⚡ Predict My Rank</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical comparison chart ──
    st.markdown("---")
    st.markdown("#### 📈 How Your Marks Compare Historically")

    if "last_marks" in st.session_state:
        target_marks = st.session_state["last_marks"]
    else:
        target_marks = marks

    hist_fig = go.Figure()

    # Actual data scatter (faded)
    hist_fig.add_trace(go.Scatter(
        x=df["Marks"], y=df["Rank"],
        mode="markers",
        marker=dict(color="#63b3ed", size=3, opacity=0.25),
        name="All data points",
        hovertemplate="Marks: %{x}<br>Rank: %{y:,}<extra></extra>",
    ))

    # Year-wise median line
    yr_med = df.groupby("Year")[["Marks","Rank"]].median().reset_index()
    hist_fig.add_trace(go.Scatter(
        x=yr_med["Marks"], y=yr_med["Rank"],
        mode="markers",
        marker=dict(color="#f6ad55", size=9,
                    symbol="diamond",
                    line=dict(color="#fff", width=1)),
        name="Year median",
        hovertemplate="Year median Marks: %{x}<br>Rank: %{y:,}<extra></extra>",
    ))

    # User's mark vertical line
    hist_fig.add_vline(
        x=target_marks,
        line_dash="dash",
        line_color="#a78bfa",
        line_width=2,
        annotation_text=f"Your marks: {target_marks}",
        annotation_font=dict(color="#a78bfa", size=12),
        annotation_position="top right",
    )

    hist_fig.update_layout(
        title=dict(text=f"Marks={target_marks} vs All Historical Data",
                   font=dict(size=14)),
        xaxis_title="Marks",
        yaxis_title="AIR Rank",
        **PLOTLY_TEMPLATE,
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(hist_fig, use_container_width=True,
                    config={"displayModeBar": False})


def render_data_explorer(df: pd.DataFrame):
    """Interactive raw data explorer with filters."""
    st.markdown('<div class="section-title">🗃️ Data Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Filter and explore the full JEE dataset</div>',
                unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        yr_range = st.slider("Year Range", 2009, 2026, (2015, 2026))
    with fc2:
        marks_range = st.slider("Marks Range", 0, 300, (50, 300))
    with fc3:
        sel_cats = st.multiselect("College Tier",
                                  CATEGORY_LABELS,
                                  default=CATEGORY_LABELS)

    filtered = df[
        (df["Year"].between(*yr_range))
        & (df["Marks"].between(*marks_range))
        & (df["Category"].isin(sel_cats))
    ].copy()

    st.markdown(
        f'<div class="info-bar">Showing <b>{len(filtered):,}</b> '
        f'of <b>{len(df):,}</b> records</div>',
        unsafe_allow_html=True,
    )

    display_cols = ["Year", "Marks", "Percentile", "Rank",
                    "Total_Candidates", "RankRatio", "Category"]
    st.dataframe(
        filtered[display_cols]
            .sort_values("Rank")
            .reset_index(drop=True)
            .style.format({
                "Percentile"      : "{:.4f}",
                "RankRatio"       : "{:.6f}",
                "Total_Candidates": "{:,.0f}",
                "Rank"            : "{:,.0f}",
            }),
        use_container_width=True,
        height=400,
    )


# ── SECTION 10: SIDEBAR ──────────────────────────────────────
def render_sidebar(df: pd.DataFrame, models: dict):
    with st.sidebar:
        st.markdown(
            """<div style="text-align:center;padding:16px 0 8px;">
                 <span style="font-size:2.5rem;">🎓</span><br>
                 <span style="font-weight:800;font-size:1.1rem;
                       background:linear-gradient(135deg,#63b3ed,#a78bfa);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;">JEE Rank Intelligence</span><br>
                 <span style="color:#4a5568;font-size:.78rem;">2009 – 2026 · ML Edition</span>
               </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Overview", "📊 EDA & Insights", "🎯 Predict Rank", "🗃️ Data Explorer"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**📦 Dataset Info**")
        st.caption(f"Records : {len(df):,}")
        st.caption(f"Years   : 2009 – 2026")
        st.caption(f"Features: Marks, Percentile, Rank, Total_Candidates")

        st.markdown("---")
        st.markdown("**🤖 Model Info**")
        st.caption(f"RF R² (test) : {models['reg_r2']:.4f}")
        st.caption(f"RF MAE       : {models['reg_mae']:,.0f} rank units")
        st.caption(f"XGB Accuracy : {models['cls_acc']*100:.1f}%")
        st.caption("Training     : Automatic on startup")
        st.caption("SMOTE        : Applied (k=3) for 6 classes")

        st.markdown("---")
        st.markdown(
            '<div style="color:#4a5568;font-size:.75rem;text-align:center;">'
            'Trained on JEE data 2009–2026<br>'
            'RandomForest + XGBoost + SMOTE</div>',
            unsafe_allow_html=True,
        )

    return page


# ── SECTION 11: MAIN ORCHESTRATOR ───────────────────────────
def main():
    inject_css()

    # ── Loading spinner shown only on first run ──
    with st.spinner("⚙️  Loading dataset & training models — this takes ~15 seconds..."):
        df     = load_and_preprocess("jee_marks_percentile_rank_2009_2026.csv")
        models = train_all_models(df)

    page = render_sidebar(df, models)

    # ── Route pages ──
    if page == "🏠 Overview":
        render_hero()
        st.markdown("---")
        render_model_kpis(models, df)
        st.markdown("<br>", unsafe_allow_html=True)

        # Quick preview charts on homepage
        st.markdown(
            '<div class="section-title">📌 Quick Highlights</div>',
            unsafe_allow_html=True,
        )
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(chart_total_candidates_trend(df),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with h2:
            st.plotly_chart(chart_marks_bucket_avg_rank(df),
                            use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown(
            """<div class="insight-box">
              🚀 <b>How to use this app:</b>
              Use the sidebar to navigate to <b>EDA & Insights</b> for full data analysis,
              <b>Predict Rank</b> to get your personalised AI-powered rank estimate, or
              <b>Data Explorer</b> to filter and browse the raw dataset.
            </div>""", unsafe_allow_html=True)

    elif page == "📊 EDA & Insights":
        render_eda_section(df, models)

    elif page == "🎯 Predict Rank":
        render_prediction_section(models, df)

    elif page == "🗃️ Data Explorer":
        render_data_explorer(df)


if __name__ == "__main__":
    main()
