import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="JEE Predictor", page_icon="🎓", layout="centered")

# Custom CSS Injection
st.markdown('''
<style>
/* Base styling */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Hide streamlit default header/footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Typography Gradient for Headers */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Form Styling / Glassmorphism */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

/* Custom Button */
div.stButton > button {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px -3px rgba(139, 92, 246, 0.5);
}

/* Metric Cards HTML/CSS */
.result-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease;
}
.result-card:hover {
    transform: translateY(-5px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.result-label {
    font-size: 1.1rem;
    color: #cbd5e1;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.result-value {
    font-size: 2.2rem;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin: 0;
}
.result-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎓 JEE Rank Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 1.1rem;'>Predict your estimated JEE Rank and Competitive Category based on your marks and the total number of candidates.</p>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def load_and_train_models():
    """
    Loads the dataset and trains the models according to the provided notebook.
    This runs only once when the app is started and caches the trained models.
    """
    # 1. Load Data
    try:
        # Assuming the CSV is in the same folder
        df = pd.read_csv('jee_marks_percentile_rank_2009_2026.csv')
    except Exception as e:
        return None, f"Error loading dataset: {e}. Please ensure 'jee_marks_percentile_rank_2009_2026.csv' is in the application folder."

    # 2. Feature Engineering
    df['RankRatio'] = df['Rank'] / df['Total_Candidates']

    def get_category(ratio):
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

    df['Category'] = df['RankRatio'].apply(get_category)

    # ----------------------------------------
    # 3. Train Regression Model (Random Forest)
    # ----------------------------------------
    X_reg = df.drop(['RankRatio', 'Category', 'Rank', 'Percentile'], axis=1) # features: Year, Marks, Total_Candidates
    # Ensure column order matches the prediction array (Year, Marks, Total_Candidates)
    X_reg = df[['Year', 'Marks', 'Total_Candidates']] 
    Y_reg = np.log1p(df["Rank"])

    X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg, Y_reg, random_state=42, test_size=0.2)

    sc_reg = StandardScaler()
    X_train_reg = sc_reg.fit_transform(X_train_reg)
    X_test_reg = sc_reg.transform(X_test_reg)

    rf_model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_split=3,
                                     min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train_reg, Y_train_reg)

    # ----------------------------------------
    # 4. Train Classification Model (XGBoost)
    # ----------------------------------------
    X_clf = df.drop(columns=['Percentile', 'RankRatio', 'Category', 'Total_Candidates'])
    # Keeping order as Year, Marks, Rank to be safe
    X_clf = df[['Year', 'Marks', 'Rank']]
    Y_clf = df['Category']

    label_encoder = LabelEncoder()
    Y_clf_encoded = label_encoder.fit_transform(Y_clf)

    X_train_clf, X_test_clf, Y_train_clf, Y_test_clf = train_test_split(X_clf, Y_clf_encoded, test_size=0.2, random_state=42)

    sc_clf = StandardScaler()
    X_train_clf_scaled = sc_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = sc_clf.transform(X_test_clf)

    # Apply SMOTE to balancing classes
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_clf_scaled, Y_train_clf)

    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.02, max_depth=3,
                              subsample=0.8, random_state=42)
    xgb_model.fit(X_train_resampled, Y_train_resampled)

    return {
        'rf': rf_model,
        'xgb': xgb_model,
        'sc_reg': sc_reg,
        'sc_clf': sc_clf,
        'le': label_encoder
    }, None

# Attempt to load and train the models
with st.spinner("Initializing models and loading data. This may take a moment on the first run..."):
    models, error_msg = load_and_train_models()

if error_msg:
    st.error(error_msg)
    st.info("Make sure you have downloaded the CSV and placed it beside this app.py file.")
    st.stop()

st.success("Models fully trained and ready!")

# ----------------------------------------
# Input Form
# ----------------------------------------
with st.form("prediction_form"):
    st.subheader("Enter Prediction Details")
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Year", min_value=2009, max_value=2030, value=2024, step=1)
        marks = st.number_input("Marks", min_value=0, max_value=360, value=150, step=1)
    with col2:
        total_candidates = st.number_input("Total Candidates", min_value=100000, max_value=2000000, value=1200000, step=10000)
    
    submit_button = st.form_submit_button("Predict Result")

if submit_button:
    # ----- 1. Rank Prediction (Regression) -----
    # Using the exact scaler and model we stored in cache
    X_new_reg = np.array([[year, marks, total_candidates]])
    X_new_reg_scaled = models['sc_reg'].transform(X_new_reg)
    
    # Expected log rank
    rf_pred_log = models['rf'].predict(X_new_reg_scaled)[0]
    predicted_rank_point = int(np.expm1(rf_pred_log))
    
    # Finding 10-90 percentile range from all trees in Random Forest
    all_tree_preds = np.array([tree.predict(X_new_reg_scaled)[0] for tree in models['rf'].estimators_])
    
    rf_lower_log = np.percentile(all_tree_preds, 10)
    rf_upper_log = np.percentile(all_tree_preds, 90)
    
    predicted_rank_lower = int(np.expm1(rf_lower_log))
    predicted_rank_upper = int(np.expm1(rf_upper_log))

    # ----- 2. Category Prediction (Classification) -----
    # Input order expected: Year, Marks, Rank
    X_new_clf = np.array([[year, marks, predicted_rank_point]])
    X_new_clf_scaled = models['sc_clf'].transform(X_new_clf)
    
    cat_encoded = models['xgb'].predict(X_new_clf_scaled)[0]
    predicted_category = models['le'].inverse_transform([cat_encoded])[0]

    # ----- Display Results -----
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>🎯 Your Estimated Results</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        # Using custom HTML for the rank result card
        rank_html = f'''
        <div class="result-card">
            <div class="result-label">Predicted Estimated Rank</div>
            <h3 class="result-value">#{predicted_rank_point:,}</h3>
            <div class="result-sub">Expected Range (10-90%):<br><strong>{predicted_rank_lower:,} - {predicted_rank_upper:,}</strong></div>
        </div>
        '''
        st.markdown(rank_html, unsafe_allow_html=True)
        
    with c2:
        # Using custom HTML for the category result card
        cat_html = f'''
        <div class="result-card">
            <div class="result-label">Competitive Category</div>
            <h3 class="result-value" style="background: linear-gradient(to right, #10b981, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {predicted_category}
            </h3>
            <div class="result-sub">Based on RankRatio classification</div>
        </div>
        '''
        st.markdown(cat_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("The 10-90 percentile range represents the confidence interval from the underlying Random Forest model's decision trees.", icon="ℹ️")
