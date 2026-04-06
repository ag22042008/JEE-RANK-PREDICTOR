# ============================================================
#  JEE RANK PREDICTOR — Streamlit Frontend (app.py)
#  Converted from JeeRankprediction.ipynb
# ============================================================
#
# HOW TO RUN:
#   pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn plotly seaborn matplotlib
#   streamlit run app.py
#
# DATASET REQUIRED:
#   Place  jee_marks_percentile_rank_2009_2026.csv  in the same folder as app.py
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
# Standard scientific Python stack
import numpy as np                        # Numerical computing (arrays, log transforms)
import pandas as pd                       # Tabular data handling (DataFrame)
import matplotlib.pyplot as plt           # Static plotting backend (used by seaborn)
import seaborn as sns                     # High-level statistical plots (boxplots, kde, heatmaps)
import plotly.express as px               # Interactive plots rendered inside Streamlit
import warnings
warnings.filterwarnings('ignore')         # Suppress non-critical sklearn/xgboost warnings

# Streamlit — turns a Python script into a web application
import streamlit as st

# Sklearn utilities
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.svm import SVC                # Support Vector Classifier

# XGBoost — gradient-boosted trees for classification
from xgboost import XGBClassifier

# SMOTE — Synthetic Minority Over-sampling to handle class imbalance
from imblearn.over_sampling import SMOTE


# ── 2. PAGE CONFIG ───────────────────────────────────────────
# Must be the FIRST Streamlit call; sets tab title, icon, and layout
st.set_page_config(
    page_title="JEE Rank Predictor",
    page_icon="🎯",
    layout="wide",           # Uses full browser width
    initial_sidebar_state="expanded"
)

# ── 3. CUSTOM CSS ────────────────────────────────────────────
# Inject raw CSS into the page for branding and aesthetics
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hero header */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35, #F7C59F, #EFEFD0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #888;
        font-size: 1.05rem;
        margin-top: 4px;
        font-weight: 300;
    }

    /* Section headers */
    .section-head {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #FF6B35;
        border-left: 4px solid #FF6B35;
        padding-left: 12px;
        margin-top: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #FF6B35;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Prediction result box */
    .result-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #FF6B35;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-rank {
        font-family: 'Syne', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        color: #FF6B35;
    }
    .result-range {
        color: #F7C59F;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    .result-category {
        display: inline-block;
        background: #FF6B35;
        color: white;
        border-radius: 24px;
        padding: 6px 20px;
        margin-top: 12px;
        font-weight: 500;
    }

    /* Info/explanation boxes */
    .explain-box {
        background: #0f0f23;
        border-left: 3px solid #F7C59F;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #ccc;
        margin: 8px 0 16px 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d0d1f;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── 4. HERO HEADER ───────────────────────────────────────────
# st.markdown renders raw HTML; unsafe_allow_html is needed for custom tags
st.markdown("""
<div style="padding: 1rem 0 2rem 0;">
    <div class="hero-title">🎯 JEE Rank Predictor</div>
    <div class="hero-sub">
        Predict your JEE rank using Machine Learning · Random Forest · XGBoost · SVM
    </div>
</div>
""", unsafe_allow_html=True)


# ── 5. SIDEBAR — FILE UPLOAD & INPUTS ────────────────────────
# The sidebar stays fixed on the left while main content scrolls
with st.sidebar:
    st.markdown("### 📂 Load Dataset")

    # File uploader widget — accepts only CSV files
    # Returns a UploadedFile object or None
    uploaded_file = st.file_uploader(
        "Upload jee_marks_percentile_rank_2009_2026.csv",
        type=["csv"]
    )

    st.markdown("---")
    st.markdown("### 🔮 Make a Prediction")

    # Number input: marks scored in JEE (0–360)
    input_marks = st.number_input(
        "Marks Scored", min_value=0.0, max_value=360.0, value=150.0, step=0.5
    )

    # Select box for JEE year (common years in dataset)
    input_year = st.selectbox(
        "JEE Year", options=list(range(2009, 2027)), index=15
    )

    # Number input for total candidates appearing that year
    input_candidates = st.number_input(
        "Total Candidates", min_value=100000, max_value=1600000,
        value=1200000, step=10000
    )

    # Button triggers model training + prediction
    predict_btn = st.button("🚀 Predict My Rank", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Navigation")
    # Radio acts as a page/tab selector stored in session state
    page = st.radio(
        "Go to section",
        ["🏠 Overview", "📊 EDA", "🤖 Models", "🏆 Results"]
    )


# ── 6. DATA LOADING & CACHING ────────────────────────────────
# @st.cache_data memoises the function: runs once, then reuses output
# This prevents reloading CSV on every Streamlit rerun
@st.cache_data
def load_data(file):
    """Load CSV from uploaded file object or path string."""
    return pd.read_csv(file)


# ── 7. FEATURE ENGINEERING ───────────────────────────────────
def rank_ratio(dataframe):
    """
    Add RankRatio column = Rank / Total_Candidates.
    Normalises rank across years with different candidate pools.
    A ratio of 0.001 means top 0.1% regardless of year.
    """
    dataframe['RankRatio'] = dataframe['Rank'] / dataframe['Total_Candidates']
    return dataframe


def get_category(ratio):
    """
    Convert a numeric RankRatio into a human-readable performance tier.
    Used as the target variable for classification models.
    """
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


# ── 8. MODEL TRAINING ────────────────────────────────────────
# Cache the trained models so they're not retrained on every interaction
@st.cache_resource
def train_models(df):
    """
    Full ML pipeline:
      1. Feature engineering (RankRatio, Category)
      2. Regression  → predict exact rank (log-transformed target)
      3. Classification → predict performance tier
    Returns all fitted objects needed for inference.
    """

    # --- 8a. Feature engineering ---
    df = rank_ratio(df.copy())
    df['Category'] = df['RankRatio'].apply(get_category)

    # --- 8b. Regression setup ---
    # X drops derived/target columns; Y is log1p(Rank) to reduce skewness
    X = df.drop(['RankRatio', 'Category', 'Rank', 'Percentile'], axis=1)
    Y = np.log1p(df["Rank"])              # log(1 + Rank) → makes distribution more normal

    # StandardScaler: zero-mean, unit-variance normalisation
    # fit_transform on train; only transform on test (avoids data leakage)
    sc = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42   # 80/20 split, reproducible seed
    )
    X_train_sc = sc.fit_transform(X_train)      # Fit scaler on train set only
    X_test_sc  = sc.transform(X_test)           # Apply same scale to test set

    # --- 8c. Polynomial Linear Regression (degree=3) ---
    # PolynomialFeatures adds interaction terms: x1², x2², x1·x2, etc.
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_test_poly  = poly.transform(X_test_sc)

    lr = LinearRegression()
    lr.fit(X_train_poly, Y_train)               # Fit on polynomial-expanded features
    y_lr_pred = np.expm1(lr.predict(X_test_poly))  # Inverse log transform

    # --- 8d. Random Forest Regressor ---
    # 300 trees; max_depth=5 prevents overfitting; min_samples_leaf=5 smooths predictions
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=5,
        min_samples_split=3, min_samples_leaf=5, random_state=42
    )
    rf_reg.fit(X_train_sc, Y_train)
    y_rf_pred = np.expm1(rf_reg.predict(X_test_sc))
    y_test_orig = np.expm1(Y_test)              # Convert test labels back to original rank

    # --- 8e. Regression metrics ---
    r2_lr  = r2_score(y_test_orig, y_lr_pred)
    r2_rf  = r2_score(y_test_orig, y_rf_pred)

    n, p_lr = len(y_test_orig), X_test_poly.shape[1]
    p_rf    = X_test_sc.shape[1]

    # Adjusted R² penalises adding too many features (corrects R² for model complexity)
    adj_r2_lr = 1 - (1 - r2_lr) * (n - 1) / (n - p_lr - 1)
    adj_r2_rf = 1 - (1 - r2_rf) * (n - 1) / (n - p_rf - 1)

    mae_lr  = mean_absolute_error(y_test_orig, y_lr_pred)
    rmse_lr = np.sqrt(mean_squared_error(y_test_orig, y_lr_pred))
    mae_rf  = mean_absolute_error(y_test_orig, y_rf_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test_orig, y_rf_pred))

    # --- 8f. Prediction interval via tree percentiles ---
    # Collect individual tree predictions to estimate uncertainty range (10th–90th pct)
    all_preds_test = np.array([tree.predict(X_test_sc) for tree in rf_reg.estimators_])
    lower = np.expm1(np.percentile(all_preds_test, 10, axis=0))
    upper = np.expm1(np.percentile(all_preds_test, 90, axis=0))
    coverage = np.mean((y_test_orig >= lower) & (y_test_orig <= upper))

    # --- 8g. Classification setup ---
    # Predict performance category from Year + Marks + Rank (no Percentile leakage)
    X2 = df.drop(columns=['Percentile', 'RankRatio', 'Category', 'Total_Candidates'])
    Y2 = df['Category']

    # LabelEncoder converts string labels → integers (required by XGBoost)
    encoder = LabelEncoder()
    Y2_enc = encoder.fit_transform(Y2)

    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
        X2, Y2_enc, test_size=0.2, random_state=42
    )
    sc2 = StandardScaler()
    X_train2_sc = sc2.fit_transform(X_train2)
    X_test2_sc  = sc2.transform(X_test2)

    # SMOTE generates synthetic minority-class samples to balance training data
    smote = SMOTE(random_state=42)
    X_train2_res, Y_train2_res = smote.fit_resample(X_train2_sc, Y_train2)

    # --- 8h. XGBoost Classifier ---
    # learning_rate=0.02 (slow, careful); max_depth=3 (shallow → less overfit)
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.02,
        max_depth=3, subsample=0.8, random_state=42, eval_metric='logloss'
    )
    xgb.fit(X_train2_res, Y_train2_res)
    y_xgb_pred = xgb.predict(X_test2_sc)

    # --- 8i. SVM Classifier ---
    # RBF kernel handles non-linear boundaries; C=1 balances margin vs misclassification
    svc = SVC(kernel='rbf', C=1, gamma='scale')
    svc.fit(X_train2_res, Y_train2_res)
    y_svc_pred = svc.predict(X_test2_sc)

    # --- 8j. Random Forest Classifier ---
    rf_clf = RandomForestClassifier(
        n_estimators=25, max_depth=4, min_samples_leaf=4, random_state=42
    )
    rf_clf.fit(X_train2_res, Y_train2_res)
    y_rf_clf_pred = rf_clf.predict(X_test2_sc)

    # Pack everything into a dict for easy access in the UI
    return {
        # Fitted objects needed for new predictions
        "sc_reg": sc,
        "sc_clf": sc2,
        "rf_reg": rf_reg,
        "xgb": xgb,
        "encoder": encoder,

        # Regression metrics
        "reg_comparison": pd.DataFrame({
            "Model": ["Polynomial LR (deg=3)", "Random Forest Regressor"],
            "R² Score": [round(r2_lr, 4), round(r2_rf, 4)],
            "Adj R²": [round(adj_r2_lr, 4), round(adj_r2_rf, 4)],
            "MAE": [round(mae_lr, 2), round(mae_rf, 2)],
            "RMSE": [round(rmse_lr, 2), round(rmse_rf, 2)],
            "Coverage (10-90%)": [round(coverage, 4), "—"],
        }),

        # Classification metrics
        "clf_comparison": pd.DataFrame({
            "Model": ["XGBoost", "SVM (RBF)", "Random Forest"],
            "Accuracy": [
                round(accuracy_score(Y_test2, y_xgb_pred), 4),
                round(accuracy_score(Y_test2, y_svc_pred), 4),
                round(accuracy_score(Y_test2, y_rf_clf_pred), 4),
            ],
        }),

        # Raw predictions & true values for charts
        "y_test":   y_test_orig,
        "y_rf_pred": y_rf_pred,
        "df": df,
    }


# ── 9. MAIN CONTENT ROUTER ───────────────────────────────────
# Show appropriate section based on sidebar radio selection

if uploaded_file is None:
    # ── No file yet: show upload prompt ──
    st.info("👈  Please upload the JEE dataset CSV from the sidebar to get started.")
    st.markdown("""
    <div class="explain-box">
    <b>Expected columns:</b> Year · Marks · Percentile · Rank · Total_Candidates
    </div>
    """, unsafe_allow_html=True)
    st.stop()                           # Halt execution — nothing else renders


# ── 10. LOAD DATA ────────────────────────────────────────────
df = load_data(uploaded_file)           # Read CSV into a Pandas DataFrame


# ── 11. OVERVIEW PAGE ────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-head">Dataset Overview</div>', unsafe_allow_html=True)

    # Shape info
    rows, cols = df.shape
    c1, c2, c3, c4 = st.columns(4)

    # Display quick summary metrics using HTML cards
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{rows:,}</div>'
                    f'<div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{cols}</div>'
                    f'<div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    with c3:
        nulls = int(df.isnull().sum().sum())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nulls}</div>'
                    f'<div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
    with c4:
        dups = int(df.duplicated().sum())
        st.markdown(f'<div class="metric-card"><div class="metric-value">{dups}</div>'
                    f'<div class="metric-label">Duplicate Rows</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)   # Interactive scrollable table

    st.markdown('<div class="section-head">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)  # count/mean/std/min/quartiles/max


# ── 12. EDA PAGE ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown('<div class="section-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Relationships", "🌡 Correlations"])

    with tab1:
        st.markdown("#### Percentile Distribution")
        st.markdown('<div class="explain-box">Percentile is left-skewed: most candidates cluster in the 80–100 range, '
                    'creating a long left tail toward lower values.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            # Plotly histogram with 50 bins; nbins controls granularity
            fig = px.histogram(df, x='Percentile', nbins=50,
                               title='Percentile Histogram',
                               color_discrete_sequence=['#FF6B35'])
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Interactive box plot — shows median, IQR, and outliers
            fig2 = px.box(df, x='Percentile', title='Percentile Boxplot',
                          color_discrete_sequence=['#F7C59F'])
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Marks Distribution")
        fig3 = px.box(df, x='Marks', title='Marks Boxplot',
                      color_discrete_sequence=['#EFEFD0'])
        fig3.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Rank Distribution (Right-Skewed)")
        st.markdown('<div class="explain-box">Rank is heavily right-skewed — most students bunch at lower ranks while '
                    'a long tail extends to high rank numbers. That is why the model uses log1p(Rank) as the target.</div>',
                    unsafe_allow_html=True)
        fig4 = px.histogram(df, x='Rank', nbins=50, title='Rank Histogram',
                            color_discrete_sequence=['#a29bfe'])
        fig4.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.markdown("#### Marks vs Year (Violin Plot)")
        st.markdown('<div class="explain-box">Width of each violin = density of students at that mark value. '
                    'Box inside shows median and IQR. Year-by-year consistency validates the dataset.</div>',
                    unsafe_allow_html=True)
        # Violin plot reveals distribution shape and density simultaneously
        fig5 = px.violin(df, x="Year", y="Marks", box=True, color="Year",
                         title="Marks Distribution per Year")
        fig5.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Rank vs Total Candidates")
        st.markdown('<div class="explain-box">Vertical clustering = fixed candidates per year. '
                    'As total candidates grow, max possible rank also grows — same marks yield a worse rank in a bigger pool.</div>',
                    unsafe_allow_html=True)
        fig6 = px.scatter(df, x='Total_Candidates', y='Rank',
                          color='Marks', title='Rank vs Total Candidates (coloured by Marks)',
                          color_continuous_scale='Oranges')
        fig6.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("#### Rank vs Marks")
        st.markdown('<div class="explain-box">Non-linear inverse relationship: at high marks a 5-mark gain '
                    'dramatically improves rank; at low marks the same gain barely changes rank (high student density zone).</div>',
                    unsafe_allow_html=True)
        fig7 = px.scatter(df, x='Rank', y='Marks', color='Percentile',
                          title='Rank vs Marks (coloured by Percentile)',
                          color_continuous_scale='RdYlGn')
        fig7.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig7, use_container_width=True)

    with tab3:
        st.markdown("#### Full Correlation Heatmap")
        st.markdown('<div class="explain-box">Marks and Percentile are almost perfectly correlated (≈0.99), '
                    'confirming they carry the same information — only one is needed as a feature.</div>',
                    unsafe_allow_html=True)
        # Seaborn heatmap rendered via matplotlib figure embedded in Streamlit
        fig8, ax = plt.subplots(figsize=(7, 5))
        numeric_df = df.select_dtypes(include=[np.number])  # Drop non-numeric cols
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_facecolor('#0d0d1f')
        fig8.patch.set_facecolor('#0d0d1f')
        plt.xticks(color='white'); plt.yticks(color='white')
        st.pyplot(fig8)


# ── 13. MODELS PAGE ──────────────────────────────────────────
elif page == "🤖 Models":
    st.markdown('<div class="section-head">Training Models…</div>', unsafe_allow_html=True)

    with st.spinner("Training all models (this may take ~30 seconds)…"):
        results = train_models(df)       # Calls the cached training function

    st.success("✅ All models trained successfully!")

    st.markdown('<div class="section-head">Regression Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">'
                '<b>Polynomial LR:</b> adds x², x³ interaction terms for non-linear fits. '
                '<b>Random Forest:</b> ensemble of 300 trees; robust to outliers and skewness '
                'because each tree splits locally rather than minimising global squared error.'
                '</div>', unsafe_allow_html=True)
    # Highlight highest R² in green using Pandas Styler
    st.dataframe(
        results["reg_comparison"].style.highlight_max(
            subset=["R² Score", "Adj R²"], color="#2d5a27"
        ).highlight_min(subset=["MAE", "RMSE"], color="#2d5a27"),
        use_container_width=True
    )

    st.markdown('<div class="section-head">Classification Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">'
                'SMOTE was applied before training to balance rare elite-tier samples. '
                'XGBoost generally outperforms SVM and RF on tabular data with non-linear boundaries.'
                '</div>', unsafe_allow_html=True)
    st.dataframe(
        results["clf_comparison"].style.highlight_max(
            subset=["Accuracy"], color="#2d5a27"
        ),
        use_container_width=True
    )

    st.markdown('<div class="section-head">Actual vs Predicted Rank (Random Forest)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="explain-box">Points near the diagonal = accurate predictions. '
                'Scatter at high ranks is expected because fewer training samples exist in the tail.</div>',
                unsafe_allow_html=True)

    y_test  = results["y_test"]
    y_pred  = results["y_rf_pred"]

    # Build scatter comparing actual vs predicted ranks
    fig_ap = px.scatter(
        x=y_test, y=y_pred,
        labels={"x": "Actual Rank", "y": "Predicted Rank"},
        title="Actual vs Predicted Rank",
        opacity=0.5,
        color_discrete_sequence=['#FF6B35']
    )
    # Perfect-prediction reference line
    max_val = max(y_test.max(), y_pred.max())
    fig_ap.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                     line=dict(color='white', dash='dash'))
    fig_ap.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_ap, use_container_width=True)


# ── 14. RESULTS / PREDICTION PAGE ───────────────────────────
elif page == "🏆 Results":
    st.markdown('<div class="section-head">Live Rank Predictor</div>', unsafe_allow_html=True)
    st.info("Set your marks, year, and candidate count in the sidebar, then press **🚀 Predict My Rank**.")

    if predict_btn:
        with st.spinner("Training models and predicting…"):
            results = train_models(df)   # Uses cached models after first run

        sc     = results["sc_reg"]       # Scaler fitted on regression training set
        rf_reg = results["rf_reg"]       # Trained Random Forest Regressor
        xgb    = results["xgb"]          # Trained XGBoost Classifier
        sc_clf = results["sc_clf"]       # Scaler fitted on classification training set
        encoder= results["encoder"]      # LabelEncoder to decode category integers

        # ── Regression Prediction ──────────────────────────────
        X_new = [[input_year, input_marks, input_candidates]]
        X_new_sc = sc.transform(X_new)          # Scale with the same fitted scaler

        log_pred = rf_reg.predict(X_new_sc)[0]  # Model outputs log1p(Rank)
        rank_pred = int(np.expm1(log_pred))      # Convert back: expm1 = exp(x) - 1

        # Per-tree predictions to build confidence interval
        tree_preds = np.array([
            tree.predict(X_new_sc)[0] for tree in rf_reg.estimators_
        ])
        rank_lower = int(np.expm1(np.percentile(tree_preds, 10)))
        rank_upper = int(np.expm1(np.percentile(tree_preds, 90)))

        # ── Classification Prediction ──────────────────────────
        # Classification uses Year + Marks + Rank (predicted) as features
        X_clf = [[input_year, input_marks, rank_pred]]
        X_clf_sc = sc_clf.transform(X_clf)
        cat_enc  = xgb.predict(X_clf_sc)[0]                  # Integer label
        category = encoder.inverse_transform([cat_enc])[0]   # Human-readable tier

        # ── Display result ─────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div style="color:#aaa; font-size:0.9rem; margin-bottom:8px;">Predicted JEE Rank</div>
            <div class="result-rank">{rank_pred:,}</div>
            <div class="result-range">📊 Confidence Range: {rank_lower:,} – {rank_upper:,}</div>
            <div class="result-category">{category}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Supporting metrics columns ─────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            percentile_approx = round((1 - rank_pred / input_candidates) * 100, 2)
            st.metric("Approx. Percentile", f"{percentile_approx}%")
        with m2:
            st.metric("Rank Range Width",
                      f"{rank_upper - rank_lower:,}")
        with m3:
            ratio = rank_pred / input_candidates
            st.metric("Rank Ratio", f"{ratio:.4f}")

        # ── Gauge chart for Rank Ratio ─────────────────────────
        st.markdown('<div class="section-head">Rank Ratio Gauge</div>', unsafe_allow_html=True)
        st.markdown('<div class="explain-box">Rank Ratio = Your Rank / Total Candidates. '
                    'Closer to 0 → better. Elite tier is ≤ 0.005.</div>', unsafe_allow_html=True)

        fig_gauge = px.bar(
            x=["Your Ratio", "Elite Cutoff", "Top Tier", "Competitive"],
            y=[ratio, 0.005, 0.02, 0.10],
            color=["Your Ratio", "Elite Cutoff", "Top Tier", "Competitive"],
            color_discrete_map={
                "Your Ratio": "#FF6B35",
                "Elite Cutoff": "#00b894",
                "Top Tier": "#0984e3",
                "Competitive": "#636e72"
            },
            title="Your Rank Ratio vs Category Thresholds"
        )
        fig_gauge.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                showlegend=False)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ── 15. FOOTER ───────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2a2a4a; margin-top:3rem;">
<div style="text-align:center; color:#555; font-size:0.8rem; padding-bottom:1rem;">
    JEE Rank Predictor · Built with Streamlit · Models: Random Forest · XGBoost · SVM · Poly-LR
</div>
""", unsafe_allow_html=True)
