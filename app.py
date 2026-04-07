import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

st.set_page_config(page_title="JEE Predictor", layout="wide")

# 🔥 SAME STYLE (NOT REMOVED)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}
.main { background-color: transparent; }
h1 { text-align: center; color: #38bdf8; }
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: black;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
}
.card {
    padding: 25px;
    border-radius: 18px;
    background: rgba(30, 41, 59, 0.8);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    margin-top: 20px;
}
.metric { font-size: 22px; font-weight: bold; color: #22c55e; }
.insight-good { color: #4ade80; }
.insight-mid { color: #facc15; }
.insight-bad { color: #f87171; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 JEE Rank + Category Predictor")

# 📂 Upload dataset (same as notebook flow)
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

    # Features (same assumption as your notebook)
    X = df[['Year', 'Marks']]
    y_rank = df['Rank']
    y_cat = df['Category']

    # Encoding
    enc = LabelEncoder()
    y_cat_encoded = enc.fit_transform(y_cat)

    # Split
    X_train, X_test, y_rank_train, y_rank_test = train_test_split(X, y_rank, test_size=0.2, random_state=42)
    _, _, y_cat_train, y_cat_test = train_test_split(X, y_cat_encoded, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train models (LIVE like notebook)
    reg_model = RandomForestRegressor()
    clf_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    reg_model.fit(X_train_scaled, y_rank_train)
    clf_model.fit(X_train_scaled, y_cat_train)

    st.success("✅ Model trained successfully (same as notebook)")

    # 🎯 Inputs
    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox("📅 Select Year", sorted(df['Year'].unique()))

    with col2:
        marks = st.slider("📝 Enter Marks", int(df['Marks'].min()), int(df['Marks'].max()), int(df['Marks'].mean()))

    if st.button("⚡ Predict Now"):
        input_df = pd.DataFrame([[year, marks]], columns=['Year', 'Marks'])
        X_scaled = scaler.transform(input_df)

        # Predictions
        rank_pred = int(reg_model.predict(X_scaled)[0])
        cat_encoded = int(np.round(clf_model.predict(X_scaled)[0]))
        category = enc.inverse_transform([cat_encoded])[0]

        lower = int(rank_pred * 0.85)
        upper = int(rank_pred * 1.15)

        # RESULT
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        st.markdown(f"<p class='metric'>Predicted Rank: {rank_pred}</p>", unsafe_allow_html=True)
        st.markdown(f"📉 Range: **{lower} - {upper}**")
        st.markdown(f"🏷️ Category: **{category}**")
        st.markdown("</div>", unsafe_allow_html=True)

        # INSIGHTS (KEPT + SAME FEEL)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Smart Insights")

        if rank_pred < 10000:
            st.markdown("<p class='insight-good'>🔥 Top Performer — IIT chances strong</p>", unsafe_allow_html=True)
            st.progress(90)
        elif rank_pred < 30000:
            st.markdown("<p class='insight-good'>💪 Very Good — Top NITs possible</p>", unsafe_allow_html=True)
            st.progress(75)
        elif rank_pred < 70000:
            st.markdown("<p class='insight-mid'>👍 Decent — Mid NIT / IIIT chances</p>", unsafe_allow_html=True)
            st.progress(55)
        else:
            st.markdown("<p class='insight-bad'>⚠️ Low Zone — Improve strategy</p>", unsafe_allow_html=True)
            st.progress(30)

        if marks < 120:
            st.info("💡 Focus on accuracy over attempts")
        elif marks > 200:
            st.success("🚀 Strong scoring pattern")

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ for JEE Aspirants</center>", unsafe_allow_html=True)
