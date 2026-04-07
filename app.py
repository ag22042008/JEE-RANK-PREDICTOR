import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JEE Rank Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { color: #38bdf8; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }
    .result-box {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-box h1 { color: #6ee7b7; font-size: 3rem; margin: 0; }
    .result-box p  { color: #a7f3d0; margin: 0.3rem 0 0; }
    .range-box {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .range-box h3 { color: #fbbf24; font-size: 1.6rem; margin: 0; }
    .range-box p  { color: #94a3b8; margin: 0.2rem 0 0; font-size: 0.85rem; }
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 0.75rem;
        margin-bottom: 1rem;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Data & model cache ────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['RankRatio'] = df['Rank'] / df['Total_Candidates']
    def get_category(r):
        if r <= 0.005:   return 'Elite (Top 0.5%)'
        elif r <= 0.02:  return 'Top Tier (0.5% - 2%)'
        elif r <= 0.05:  return 'Highly Competitive (2% - 5%)'
        elif r <= 0.10:  return 'Competitive (5% - 10%)'
        else:            return 'Not Prepared (>10%)'
    df['Category'] = df['RankRatio'].apply(get_category)
    return df


@st.cache_resource
def train_models(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

    sc  = StandardScaler()
    sc2 = StandardScaler()
    enc = LabelEncoder()

    # ── Regression ──────────────────────────────────────────────────────────
    X = df.drop(['RankRatio', 'Category', 'Rank', 'Percentile'], axis=1)
    Y = np.log1p(df['Rank'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    # Polynomial LR
    poly  = PolynomialFeatures(degree=3)
    X_tr1 = poly.fit_transform(X_train_s)
    X_te1 = poly.transform(X_test_s)
    lr = LinearRegression()
    lr.fit(X_tr1, Y_train)
    y_lr = np.expm1(lr.predict(X_te1))

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=5,
        min_samples_split=3, min_samples_leaf=5, random_state=42
    )
    rf_reg.fit(X_train_s, Y_train)
    y_rf = np.expm1(rf_reg.predict(X_test_s))
    y_true = np.expm1(Y_test)

    r2_lr  = r2_score(y_true, y_lr)
    r2_rf  = r2_score(y_true, y_rf)
    n, p_lr = len(y_true), X_te1.shape[1]
    p_rf    = X_test_s.shape[1]
    adj_r2_lr = 1 - (1 - r2_lr) * (n - 1) / (n - p_lr - 1)
    adj_r2_rf = 1 - (1 - r2_rf) * (n - 1) / (n - p_rf - 1)

    # Prediction interval info from RF
    all_preds = np.array([t.predict(X_test_s) for t in rf_reg.estimators_])
    lower_r = np.expm1(np.percentile(all_preds, 10, axis=0))
    upper_r = np.expm1(np.percentile(all_preds, 90, axis=0))
    coverage = float(np.mean((y_true >= lower_r) & (y_true <= upper_r)))
    avg_width = float((upper_r - lower_r).mean())

    reg_metrics = {
        'r2_lr': r2_lr, 'adj_r2_lr': adj_r2_lr,
        'mae_lr': mean_absolute_error(y_true, y_lr),
        'mse_lr': mean_squared_error(y_true, y_lr),
        'rmse_lr': float(np.sqrt(mean_squared_error(y_true, y_lr))),
        'r2_rf': r2_rf, 'adj_r2_rf': adj_r2_rf,
        'mae_rf': mean_absolute_error(y_true, y_rf),
        'mse_rf': mean_squared_error(y_true, y_rf),
        'rmse_rf': float(np.sqrt(mean_squared_error(y_true, y_rf))),
        'coverage': coverage, 'avg_width': avg_width,
    }

    # ── Classification ───────────────────────────────────────────────────────
    X2 = df.drop(columns=['Percentile', 'RankRatio', 'Category', 'Total_Candidates'])
    Y2 = enc.fit_transform(df['Category'])

    X_tr2, X_te2, Y_tr2, Y_te2 = train_test_split(X2, Y2, test_size=0.2, random_state=42)
    X_tr2_s = sc2.fit_transform(X_tr2)
    X_te2_s = sc2.transform(X_te2)

    smote = SMOTE(random_state=42)
    X_tr2_sm, Y_tr2_sm = smote.fit_resample(X_tr2_s, Y_tr2)

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.02,
                        max_depth=3, subsample=0.8, random_state=42)
    xgb.fit(X_tr2_sm, Y_tr2_sm)

    svc = SVC(kernel='rbf', C=1, gamma='scale')
    svc.fit(X_tr2_sm, Y_tr2_sm)

    rf_clf = RandomForestClassifier(
        n_estimators=25, max_depth=4, min_samples_leaf=4, random_state=42
    )
    rf_clf.fit(X_tr2_sm, Y_tr2_sm)

    from sklearn.metrics import classification_report
    clf_metrics = {
        'xgb': {'acc': accuracy_score(Y_te2, xgb.predict(X_te2_s)),
                'report': classification_report(Y_te2, xgb.predict(X_te2_s), output_dict=True)},
        'svc': {'acc': accuracy_score(Y_te2, svc.predict(X_te2_s)),
                'report': classification_report(Y_te2, svc.predict(X_te2_s), output_dict=True)},
        'rf':  {'acc': accuracy_score(Y_te2, rf_clf.predict(X_te2_s)),
                'report': classification_report(Y_te2, rf_clf.predict(X_te2_s), output_dict=True)},
    }

    return {
        'sc': sc, 'sc2': sc2, 'enc': enc, 'poly': poly,
        'lr': lr, 'rf_reg': rf_reg, 'xgb': xgb, 'svc': svc, 'rf_clf': rf_clf,
        'reg_metrics': reg_metrics, 'clf_metrics': clf_metrics,
    }


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("CP-School-61.png", width=100,use_column_width=False )
st.sidebar.title("🎯 JEE Rank Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home & Predict", "📊 EDA", "🤖 Model Performance"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload Dataset**")
uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    df = load_data(uploaded)
    models = train_models(df)
    data_ready = True
else:
    st.sidebar.info("Upload `jee_marks_percentile_rank_2009_2026.csv` to begin.")
    data_ready = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home & Predict
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.markdown("## 🎯 JEE Rank Predictor")
    st.markdown(
        "Predict your **JEE Main rank** and **performance category** from marks, "
        "year, and total candidates using machine learning."
    )

    if not data_ready:
        st.warning("⬅️ Please upload your dataset from the sidebar to enable predictions.")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="section-header"><h3>Make a Prediction</h3></div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        marks = st.number_input("📝 Marks (out of 300)", min_value=0, max_value=300,
                                value=150, step=1)
    with c2:
        year  = st.number_input("📅 Year", min_value=2009, max_value=2030,
                                value=2024, step=1)
    with c3:
        total = st.number_input("👥 Total Candidates", min_value=100_000,
                                max_value=2_000_000, value=1_200_000, step=10_000,
                                format="%d")

    predict_btn = st.button("🚀 Predict Rank", use_container_width=True, type="primary")

    if predict_btn:
        sc     = models['sc']
        sc2    = models['sc2']
        rf_reg = models['rf_reg']
        xgb    = models['xgb']
        enc    = models['enc']

        # Regression prediction
        X_new = sc.transform([[year, marks, total]])
        log_pred = rf_reg.predict(X_new)[0]
        predicted_rank = int(np.expm1(log_pred))

        tree_preds = np.array([t.predict(X_new)[0] for t in rf_reg.estimators_])
        lower = int(np.expm1(np.percentile(tree_preds, 10)))
        upper = int(np.expm1(np.percentile(tree_preds, 90)))

      
     # Classification prediction
# Ensure the columns match the exact names used in X2 during training
input_df_clf = pd.DataFrame([[year, marks]], columns=['Year', 'Marks'])

# Apply scaling and predict
X_new2 = sc2.transform(input_df_clf)

# XGBoost sometimes returns float or numpy type → convert to int
cat_encoded = int(xgb.predict(X_new2)[0])

# Ensure encoder input shape is correct
category = enc.inverse_transform([cat_encoded])[0]
        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1.5, 1.5])
        with r1:
            st.markdown(f"""
            <div class="result-box">
                <p>🏆 Predicted Rank</p>
                <h1>{predicted_rank:,}</h1>
                <p>Random Forest Regressor</p>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="range-box">
                <p>📉 Lower Bound (10th %ile)</p>
                <h3>{lower:,}</h3>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="range-box">
                <p>📈 Upper Bound (90th %ile)</p>
                <h3>{upper:,}</h3>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <p>🎖️ Performance Category</p>
                <h2 style="font-size:1.1rem; color:#fbbf24;">{category}</h2>
                <p>XGBoost Classifier</p>
            </div>""", unsafe_allow_html=True)
            rank_ratio = predicted_rank / total
            percentile_est = (1 - rank_ratio) * 100
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.5rem;">
                <p>📊 Estimated Percentile</p>
                <h2>{percentile_est:.2f}</h2>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.info(
            f"**Interpretation:** With **{marks} marks** in **{year}** "
            f"(out of {total:,} candidates), your predicted rank is **{predicted_rank:,}** "
            f"with an 80% confidence interval of **{lower:,} – {upper:,}**. "
            f"You fall in the **{category}** category."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    if not data_ready:
        st.warning("⬅️ Please upload your dataset to view EDA.")
        st.stop()

    # Dataset overview
    st.markdown('<div class="section-header"><h3>Dataset Overview</h3></div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [
        (m1, "Total Records",   f"{len(df):,}"),
        (m2, "Years Covered",   f"{df['Year'].min()} – {df['Year'].max()}"),
        (m3, "Max Candidates",  f"{df['Total_Candidates'].max():,}"),
        (m4, "Missing Values",  str(int(df.isnull().sum().sum()))),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <p>{label}</p>
            <h2>{val}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "📦 Boxplots by Year", "🔗 Bivariate", "🌡️ Correlation"]
    )

    # ── Tab 1: Distributions ─────────────────────────────────────────────────
    with tab1:
        cols = st.columns(2)
        for ax_idx, (col_name, color, title) in enumerate([
            ('Percentile', 'purple', 'KDE – Percentile'),
            ('Marks',      'skyblue', 'Distribution – Marks'),
            ('Rank',       'tomato',  'Distribution – Rank (skewed)'),
            ('Total_Candidates', 'green', 'KDE – Total Candidates'),
        ]):
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.histplot(df[col_name], kde=True, color=color, ax=ax, bins=40)
            ax.set_title(title, color='white', fontsize=11)
            ax.tick_params(colors='#94a3b8')
            ax.spines[:].set_color('#334155')
            ax.set_xlabel(col_name, color='#94a3b8')
            ax.set_ylabel('Count', color='#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            cols[ax_idx % 2].pyplot(fig)
            plt.close()

        skew_data = {
            'Column': ['Percentile', 'Marks', 'Rank', 'Total_Candidates'],
            'Skewness': [
                round(df['Percentile'].skew(), 3),
                round(df['Marks'].skew(), 3),
                round(df['Rank'].skew(), 3),
                round(df['Total_Candidates'].skew(), 3),
            ]
        }
        st.dataframe(pd.DataFrame(skew_data), use_container_width=True)

    # ── Tab 2: Boxplots by Year ───────────────────────────────────────────────
    with tab2:
        for col_name in ['Marks', 'Rank']:
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.boxplot(data=df, x='Year', y=col_name, palette='Blues', ax=ax)
            ax.set_title(f'{col_name} Distribution by Year', color='white', fontsize=12)
            ax.tick_params(colors='#94a3b8', rotation=45)
            ax.spines[:].set_color('#334155')
            ax.set_xlabel('Year', color='#94a3b8')
            ax.set_ylabel(col_name, color='#94a3b8')
            st.pyplot(fig)
            plt.close()

    # ── Tab 3: Bivariate ─────────────────────────────────────────────────────
    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f172a')

        for ax, (x, y, c, ttl) in zip(axes, [
            ('Total_Candidates', 'Rank',  'Marks',     'Rank vs Total Candidates\n(coloured by Marks)'),
            ('Rank',             'Marks', 'Percentile', 'Rank vs Marks\n(coloured by Percentile)'),
        ]):
            ax.set_facecolor('#1e293b')
            sc_plot = ax.scatter(
                df[x], df[y], c=df[c],
                cmap='plasma', alpha=0.5, s=10
            )
            plt.colorbar(sc_plot, ax=ax, label=c)
            ax.set_title(ttl, color='white', fontsize=10)
            ax.tick_params(colors='#94a3b8')
            ax.spines[:].set_color('#334155')
            ax.set_xlabel(x, color='#94a3b8')
            ax.set_ylabel(y, color='#94a3b8')

        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Key observations:**
        - Rank scales with the number of candidates — more competitors = higher max rank.
        - Marks and rank share a strong **inverse, non-linear** relationship.
        - The rank distribution is highly compressed at the top and spread at the bottom.
        """)

    # ── Tab 4: Correlation ───────────────────────────────────────────────────
    with tab4:
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        sns.heatmap(
            numeric_df.corr(), annot=True, cmap='coolwarm',
            fmt='.2f', ax=ax,
            linewidths=0.5, linecolor='#0f172a',
            annot_kws={'color': 'white', 'size': 9}
        )
        ax.set_title('Correlation Heatmap', color='white', fontsize=13)
        ax.tick_params(colors='#94a3b8')
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")

    if not data_ready:
        st.warning("⬅️ Please upload your dataset to view model metrics.")
        st.stop()

    reg  = models['reg_metrics']
    clf  = models['clf_metrics']

    tab_r, tab_c = st.tabs(["📉 Regression Models", "🏷️ Classification Models"])

    # ── Regression tab ───────────────────────────────────────────────────────
    with tab_r:
        st.markdown('<div class="section-header"><h3>Rank Prediction (Regression)</h3></div>',
                    unsafe_allow_html=True)

        reg_df = pd.DataFrame({
            'Model':            ['Poly Linear Regression', 'Random Forest Regressor'],
            'R² Score':         [round(reg['r2_lr'], 4),   round(reg['r2_rf'], 4)],
            'Adj. R²':          [round(reg['adj_r2_lr'], 4), round(reg['adj_r2_rf'], 4)],
            'MAE':              [round(reg['mae_lr'], 1),   round(reg['mae_rf'], 1)],
            'RMSE':             [round(reg['rmse_lr'], 1),  round(reg['rmse_rf'], 1)],
        })
        st.dataframe(reg_df, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class="metric-card">
            <p>RF — Prediction Coverage (10–90%)</p>
            <h2>{reg['coverage']:.1%}</h2>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card">
            <p>RF — Average Range Width</p>
            <h2>{reg['avg_width']:,.0f}</h2>
        </div>""", unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        models_names = ['Poly LR', 'RF Regressor']
        r2_vals = [reg['r2_lr'], reg['r2_rf']]
        bars = ax.barh(models_names, r2_vals, color=['#38bdf8', '#34d399'], height=0.4)
        for bar, v in zip(bars, r2_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', ha='right', color='white', fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_title('R² Score Comparison', color='white')
        ax.tick_params(colors='#94a3b8')
        ax.spines[:].set_color('#334155')
        st.pyplot(fig)
        plt.close()

    # ── Classification tab ───────────────────────────────────────────────────
    with tab_c:
        st.markdown('<div class="section-header"><h3>Category Classification</h3></div>',
                    unsafe_allow_html=True)

        clf_df = pd.DataFrame({
            'Model':       ['XGBoost', 'SVC', 'Random Forest'],
            'Accuracy':    [round(clf['xgb']['acc'], 4),
                            round(clf['svc']['acc'], 4),
                            round(clf['rf']['acc'],  4)],
            'Precision (w)': [
                round(clf['xgb']['report']['weighted avg']['precision'], 4),
                round(clf['svc']['report']['weighted avg']['precision'], 4),
                round(clf['rf']['report']['weighted avg']['precision'],  4),
            ],
            'Recall (w)':  [
                round(clf['xgb']['report']['weighted avg']['recall'], 4),
                round(clf['svc']['report']['weighted avg']['recall'], 4),
                round(clf['rf']['report']['weighted avg']['recall'],  4),
            ],
            'F1 (w)':      [
                round(clf['xgb']['report']['weighted avg']['f1-score'], 4),
                round(clf['svc']['report']['weighted avg']['f1-score'], 4),
                round(clf['rf']['report']['weighted avg']['f1-score'],  4),
            ],
        })
        st.dataframe(clf_df, use_container_width=True)

        # Accuracy bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        clrs = ['#f59e0b', '#a78bfa', '#34d399']
        bars = ax.barh(clf_df['Model'], clf_df['Accuracy'], color=clrs, height=0.4)
        for bar, v in zip(bars, clf_df['Accuracy']):
            ax.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', ha='right', color='white', fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_title('Classification Accuracy Comparison', color='white')
        ax.tick_params(colors='#94a3b8')
        ax.spines[:].set_color('#334155')
        st.pyplot(fig)
        plt.close()

        st.markdown("**Category Labels**")
        st.table(pd.DataFrame({
            'Category': [
                'Elite (Top 0.5%)', 'Top Tier (0.5% - 2%)',
                'Highly Competitive (2% - 5%)',
                'Competitive (5% - 10%)', 'Not Prepared (>10%)',
            ],
            'Rank Ratio Threshold': [
                '≤ 0.005', '0.005 – 0.02', '0.02 – 0.05', '0.05 – 0.10', '> 0.10'
            ]
        }))
