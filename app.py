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

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="JEE Rank Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #f5f0ff;
    letter-spacing: -1px;
    margin: 0;
}
.hero p {
    color: #a89fd8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* ── metric cards ── */
.kpi-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: #1a1a2e;
    border: 1px solid #3a3060;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    flex: 1;
    min-width: 160px;
    text-align: center;
}
.kpi-card .kpi-val {
    font-size: 2rem;
    font-weight: 800;
    color: #c9b8ff;
    font-family: 'DM Mono', monospace;
}
.kpi-card .kpi-label {
    font-size: 0.78rem;
    color: #7a72a8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── insight pill ── */
.insight-pill {
    background: #16213e;
    border-left: 4px solid #7c5cbf;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #ccc5f0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── prediction result ── */
.pred-box {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 0.5rem 0;
}
.pred-box.linear  { background: #1a0933; border: 2px solid #7c5cbf; }
.pred-box.poly    { background: #001a33; border: 2px solid #3a86ff; }
.pred-box .pred-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 0.4rem;
}
.pred-box .pred-val {
    font-size: 2.6rem;
    font-weight: 800;
    font-family: 'DM Mono', monospace;
}
.pred-box.linear .pred-val { color: #c9b8ff; }
.pred-box.poly   .pred-val { color: #3a86ff; }

/* ── tabs tweaks ── */
div[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
}

/* ── section title ── */
.sec-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f0ebff;
    border-bottom: 2px solid #3a3060;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("jee_marks_percentile_rank_2009_2026.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ─────────────────────────────────────────────
#  CLEAN DATA & TRAIN MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(data):
    # ── Data Cleaning ──
    clean = data.copy()
    clean = clean[clean['rank'] > 0]
    # Keep Rank=1 entries — highest marks (295-297) legitimately get Rank 1

    # ── Feature Engineering ──
    # Log-transform rank to linearise the exponential marks→rank relationship
    clean['log_rank'] = np.log1p(clean['rank'])

    # Build features: marks, year, total_candidates
    feature_cols = ['marks', 'year']
    if 'total_candidates' in clean.columns:
        feature_cols.append('total_candidates')

    X = clean[feature_cols]
    y_log = clean['log_rank']          # predict in log space
    y_raw = clean['rank']              # keep for display metrics

    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    # ── Linear Regression (in log-rank space) ──
    lin = LinearRegression()
    lin.fit(X_train, y_log_train)
    y_pred_lin_log = lin.predict(X_test)
    y_pred_lin = np.expm1(y_pred_lin_log)          # back to original scale
    y_pred_lin = np.clip(y_pred_lin, 1, None)

    # ── Polynomial Regression degree 3 (in log-rank space) ──
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test  = poly.transform(X_test)
    pol = LinearRegression()
    pol.fit(X_poly_train, y_log_train)
    y_pred_pol_log = pol.predict(X_poly_test)
    y_pred_pol = np.expm1(y_pred_pol_log)           # back to original scale
    y_pred_pol = np.clip(y_pred_pol, 1, None)

    # ── Metrics (evaluated on original rank scale) ──
    y_test_vals = y_raw_test.values
    metrics = {
        "lin_r2":  r2_score(y_test_vals, y_pred_lin),
        "lin_mae": mean_absolute_error(y_test_vals, y_pred_lin),
        "pol_r2":  r2_score(y_test_vals, y_pred_pol),
        "pol_mae": mean_absolute_error(y_test_vals, y_pred_pol),
    }
    residuals = {
        "lin": y_test_vals - y_pred_lin,
        "pol": y_test_vals - y_pred_pol,
    }

    # Store feature columns for prediction
    meta = {"feature_cols": feature_cols, "clean": clean}

    return lin, pol, poly, metrics, residuals, y_test_vals, y_pred_lin, y_pred_pol, meta

lin_model, poly_model, poly_feat, metrics, residuals, y_test, y_pred_lin, y_pred_poly, model_meta = train_models(df)

# ─────────────────────────────────────────────
#  PLOT THEME HELPER
# ─────────────────────────────────────────────
DARK_BG  = "#0d0d1a"
CARD_BG  = "#12122a"
ACCENT1  = "#7c5cbf"
ACCENT2  = "#3a86ff"
TEXT_CLR = "#c8c0e8"

def dark_fig(nrows=1, ncols=1, figsize=(7, 4)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    axes = [ax] if (nrows == 1 and ncols == 1) else ax.flatten()
    for a in (axes if isinstance(axes, list) else [axes]):
        a.set_facecolor(CARD_BG)
        a.tick_params(colors=TEXT_CLR, labelsize=9)
        for spine in a.spines.values():
            spine.set_edgecolor("#2a2a4a")
        a.xaxis.label.set_color(TEXT_CLR)
        a.yaxis.label.set_color(TEXT_CLR)
        a.title.set_color(TEXT_CLR)
    return fig, ax


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📐 JEE Rank Predictor</h1>
  <p>Exploratory analysis · Model comparison · Rank estimation — 2009–2026</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_eda, tab_insights, tab_model, tab_predict = st.tabs(
    ["📋 EDA", "📈 Insights", "🤖 Model", "🎯 Predict"]
)


# ══════════════════════════════════════════════
#  TAB 1 — EDA
# ══════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="sec-title">Dataset Overview</div>', unsafe_allow_html=True)

    # KPI row
    missing   = int(df.isnull().sum().sum())
    dupes     = int(df.duplicated().sum())
    yr_range  = f"{int(df['year'].min())}–{int(df['year'].max())}"
    n_years   = int(df['year'].nunique())

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val">{df.shape[0]:,}</div><div class="kpi-label">Total Rows</div></div>
      <div class="kpi-card"><div class="kpi-val">{df.shape[1]}</div><div class="kpi-label">Columns</div></div>
      <div class="kpi-card"><div class="kpi-val">{missing}</div><div class="kpi-label">Missing Values</div></div>
      <div class="kpi-card"><div class="kpi-val">{dupes}</div><div class="kpi-label">Duplicates</div></div>
      <div class="kpi-card"><div class="kpi-val">{n_years}</div><div class="kpi-label">Unique Years</div></div>
      <div class="kpi-card"><div class="kpi-val">{yr_range}</div><div class="kpi-label">Year Range</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**Column Info & Missing Values**")
        info_df = pd.DataFrame({
            "dtype":   df.dtypes.astype(str),
            "non-null": df.count(),
            "missing":  df.isnull().sum(),
            "missing %": (df.isnull().mean() * 100).round(2),
        })
        st.dataframe(info_df, use_container_width=True)

    with col_r:
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown("**Sample Rows**")
    st.dataframe(df.sample(min(10, len(df)), random_state=1).reset_index(drop=True),
                 use_container_width=True)

    # Distribution plots
    st.markdown("---")
    st.markdown('<div class="sec-title">Feature Distributions</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    dist_cols = st.columns(len(numeric_cols))
    for i, col in enumerate(numeric_cols):
        with dist_cols[i]:
            fig, ax = dark_fig(figsize=(4, 3))
            ax.hist(df[col].dropna(), bins=30, color=ACCENT1, edgecolor="none", alpha=0.85)
            ax.set_title(col.capitalize(), fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════
#  TAB 2 — INSIGHTS
# ══════════════════════════════════════════════
with tab_insights:
    st.markdown('<div class="sec-title">Key Insights</div>', unsafe_allow_html=True)

    # ── Row 1: Marks vs Rank  +  Year vs Rank
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Marks vs Rank**")
        fig, ax = dark_fig(figsize=(6, 4))
        sc = ax.scatter(df['marks'], df['rank'], alpha=0.45, s=12,
                        c=df['year'], cmap='plasma', edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
        cbar.ax.tick_params(colors=TEXT_CLR, labelsize=8)
        cbar.set_label("Year", color=TEXT_CLR, fontsize=8)
        ax.set_xlabel("Marks")
        ax.set_ylabel("Rank")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        <div class="insight-pill">
        ↳ Strong inverse relationship — more marks means a better (lower) rank.<br>
        ↳ The curve is non-linear: every extra mark in the 200–300 range moves the rank dramatically.<br>
        ↳ Colours show no obvious year-to-year drift in the pattern.
        </div>""", unsafe_allow_html=True)

    with r1c2:
        st.markdown("**Marks Distribution by Year (Box)**")
        years_sorted = sorted(df['year'].unique())
        fig, ax = dark_fig(figsize=(6, 4))
        data_by_year = [df.loc[df['year'] == y, 'marks'].dropna().values for y in years_sorted]
        bp = ax.boxplot(data_by_year, patch_artist=True,
                        medianprops=dict(color="#f5c518", lw=2),
                        whiskerprops=dict(color=TEXT_CLR),
                        capprops=dict(color=TEXT_CLR),
                        flierprops=dict(marker='.', color=ACCENT1, alpha=0.4, markersize=4))
        for patch in bp['boxes']:
            patch.set_facecolor(ACCENT1)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(years_sorted) + 1))
        ax.set_xticklabels([str(y) for y in years_sorted], rotation=45, fontsize=7)
        ax.set_xlabel("Year")
        ax.set_ylabel("Marks")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        <div class="insight-pill">
        ↳ Median marks and spread vary noticeably across years.<br>
        ↳ Some years have heavier tails, indicating a harder/easier paper.<br>
        ↳ Year alone is a weak predictor — but context matters.
        </div>""", unsafe_allow_html=True)

    # ── Row 2: Avg Rank per Year  +  Correlation Heatmap
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Average Rank by Year**")
        avg_rank = df.groupby('year')['rank'].median().reset_index()
        fig, ax = dark_fig(figsize=(6, 4))
        ax.bar(avg_rank['year'], avg_rank['rank'], color=ACCENT2, alpha=0.8, width=0.6)
        ax.set_xlabel("Year")
        ax.set_ylabel("Median Rank")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xticks(avg_rank['year'])
        ax.set_xticklabels(avg_rank['year'].astype(int), rotation=45, fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        <div class="insight-pill">
        ↳ Median rank fluctuates per year — likely tied to total candidates and cutoff shifts.<br>
        ↳ Growing candidate pool post-2018 pushes ranks higher overall.
        </div>""", unsafe_allow_html=True)

    with r2c2:
        st.markdown("**Correlation Heatmap**")
        fig, ax = dark_fig(figsize=(6, 4))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, mask=mask, linewidths=0.5,
                    annot_kws={"size": 10},
                    cbar_kws={"shrink": 0.8})
        ax.tick_params(axis='x', labelrotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("""
        <div class="insight-pill">
        ↳ Marks has the strongest (negative) correlation with rank — the primary driver.<br>
        ↳ Year shows weak correlation — confirms it adds marginal signal.<br>
        ↳ Percentile (if present) will be near-perfectly correlated with marks.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 — MODEL
# ══════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="sec-title">Model Performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-pill">
    <strong>How the model works:</strong> Instead of predicting rank directly, we predict <code>log(rank)</code> 
    and then convert back. This is because the marks→rank curve is exponential — a small change 
    in high marks causes a huge rank change. Log-transform linearises this, dramatically improving accuracy.
    Features used: Marks, Year, and Total Candidates.
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val">{metrics['lin_r2']:.4f}</div><div class="kpi-label">Linear R²</div></div>
      <div class="kpi-card"><div class="kpi-val">{metrics['lin_mae']:,.0f}</div><div class="kpi-label">Linear MAE</div></div>
      <div class="kpi-card"><div class="kpi-val">{metrics['pol_r2']:.4f}</div><div class="kpi-label">Poly R²</div></div>
      <div class="kpi-card"><div class="kpi-val">{metrics['pol_mae']:,.0f}</div><div class="kpi-label">Poly MAE</div></div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)

    # ── Actual vs Predicted
    with m1:
        st.markdown("**Actual vs Predicted — Linear**")
        fig, ax = dark_fig(figsize=(5.5, 4))
        ax.scatter(y_test, y_pred_lin, alpha=0.4, s=10, color=ACCENT1, label="Predictions")
        lims = [min(y_test.min(), y_pred_lin.min()), max(y_test.max(), y_pred_lin.max())]
        ax.plot(lims, lims, '--', color="#f5c518", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Rank")
        ax.set_ylabel("Predicted Rank")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_CLR)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with m2:
        st.markdown("**Actual vs Predicted — Polynomial (deg 3)**")
        fig, ax = dark_fig(figsize=(5.5, 4))
        ax.scatter(y_test, y_pred_poly, alpha=0.4, s=10, color=ACCENT2, label="Predictions")
        lims = [min(y_test.min(), y_pred_poly.min()), max(y_test.max(), y_pred_poly.max())]
        ax.plot(lims, lims, '--', color="#f5c518", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Rank")
        ax.set_ylabel("Predicted Rank")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_CLR)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Residuals
    st.markdown("---")
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**Residuals Distribution — Linear**")
        fig, ax = dark_fig(figsize=(5.5, 3.5))
        ax.hist(residuals['lin'], bins=40, color=ACCENT1, edgecolor="none", alpha=0.8)
        ax.axvline(0, color="#f5c518", linestyle="--", lw=1.5)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with r2:
        st.markdown("**Residuals Distribution — Polynomial**")
        fig, ax = dark_fig(figsize=(5.5, 3.5))
        ax.hist(residuals['pol'], bins=40, color=ACCENT2, edgecolor="none", alpha=0.8)
        ax.axvline(0, color="#f5c518", linestyle="--", lw=1.5)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Bar comparison
    st.markdown("---")
    st.markdown("**Side-by-Side Metric Comparison**")
    comp_df = pd.DataFrame({
        "Model": ["Linear", "Polynomial"],
        "R²":    [metrics['lin_r2'], metrics['pol_r2']],
        "MAE":   [metrics['lin_mae'], metrics['pol_mae']],
    })

    bc1, bc2 = st.columns(2)
    with bc1:
        fig, ax = dark_fig(figsize=(4, 3))
        bars = ax.bar(comp_df["Model"], comp_df["R²"],
                      color=[ACCENT1, ACCENT2], width=0.5, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("R² Score")
        ax.set_title("R² (higher is better)")
        for bar, val in zip(bars, comp_df["R²"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha='center', color=TEXT_CLR, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with bc2:
        fig, ax = dark_fig(figsize=(4, 3))
        bars = ax.bar(comp_df["Model"], comp_df["MAE"],
                      color=[ACCENT1, ACCENT2], width=0.5, alpha=0.85)
        ax.set_ylabel("MAE")
        ax.set_title("MAE (lower is better)")
        for bar, val in zip(bars, comp_df["MAE"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{val:,.0f}", ha='center', color=TEXT_CLR, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════
#  TAB 4 — PREDICT
# ══════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="sec-title">Predict Your JEE Rank</div>', unsafe_allow_html=True)

    p_col, r_col = st.columns([1, 1.2], gap="large")

    # Lookup the total_candidates for the selected year from the data
    year_candidates_map = df.groupby('year')['total_candidates'].first().to_dict() if 'total_candidates' in df.columns else {}

    with p_col:
        marks_input = st.slider(
            "📝 Marks (out of 300)", min_value=0.0, max_value=300.0,
            value=150.0, step=0.5,
        )
        year_input = st.slider(
            "📅 Exam Year", min_value=2009, max_value=2026,
            value=2024, step=1,
        )

        # Build input with the same features used during training
        feature_cols = model_meta["feature_cols"]
        inp_dict = {'marks': marks_input, 'year': year_input}
        if 'total_candidates' in feature_cols:
            # Use known total_candidates for this year, or extrapolate
            tc = year_candidates_map.get(year_input, 1400000)  # default to latest known
            inp_dict['total_candidates'] = tc
        inp = pd.DataFrame([inp_dict])[feature_cols]

        # Predict in log space, then convert back
        lin_pred_log  = lin_model.predict(inp)[0]
        lin_pred = int(np.clip(np.expm1(lin_pred_log), 1, None))

        poly_pred_log = poly_model.predict(poly_feat.transform(inp))[0]
        poly_pred = int(np.clip(np.expm1(poly_pred_log), 1, None))

        st.markdown(f"""
        <div class="pred-box linear">
          <div class="pred-label">🔵 Linear Regression</div>
          <div class="pred-val">{lin_pred:,}</div>
        </div>
        <div class="pred-box poly">
          <div class="pred-label">🟣 Polynomial Regression (deg 3)</div>
          <div class="pred-val">{poly_pred:,}</div>
        </div>
        """, unsafe_allow_html=True)

        diff = abs(lin_pred - poly_pred)
        st.caption(f"Models differ by **{diff:,}** rank positions for these inputs.")

    with r_col:
        # Show where this mark sits in the historical data
        st.markdown("**Where does this mark land historically?**")

        fig, ax = dark_fig(figsize=(6, 4.5))
        ax.scatter(df['marks'], df['rank'], alpha=0.25, s=8,
                   color="#888", edgecolors='none', label="Historical data")

        # Highlight selected mark range
        band = df[(df['marks'] >= marks_input - 5) & (df['marks'] <= marks_input + 5)]
        ax.scatter(band['marks'], band['rank'], alpha=0.9, s=24,
                   color="#f5c518", edgecolors='none', label=f"±5 marks of {marks_input:.0f}")

        # Prediction markers
        ax.axvline(marks_input, color=ACCENT1, lw=1.5, linestyle="--")
        ax.scatter([marks_input], [lin_pred],  s=120, color=ACCENT1,
                   zorder=5, label=f"Linear: {lin_pred:,}")
        ax.scatter([marks_input], [poly_pred], s=120, color=ACCENT2,
                   zorder=5, marker="D", label=f"Poly: {poly_pred:,}")

        ax.set_xlabel("Marks")
        ax.set_ylabel("Rank")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_CLR,
                  framealpha=0.7, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary stats for nearby points
        if not band.empty:
            st.markdown(f"**Historical ranks for marks {marks_input-5:.0f}–{marks_input+5:.0f}:**")
            band_stats = band['rank'].describe()[['min','25%','50%','75%','max']].astype(int)
            cols = st.columns(5)
            labels = ["Min", "25th %ile", "Median", "75th %ile", "Max"]
            for c, (label, val) in zip(cols, zip(labels, band_stats)):
                c.metric(label, f"{val:,}")
