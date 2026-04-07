import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
model_reg = joblib.load('rank_model.pkl')
model_clf = joblib.load('category_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.set_page_config(page_title="JEE Predictor", layout="wide")

# 🔥 ADVANCED CSS (kept + improved)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

.main {
    background-color: transparent;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

.stButton>button {
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: black;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.card {
    padding: 25px;
    border-radius: 18px;
    background: rgba(30, 41, 59, 0.8);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    margin-top: 20px;
}

.metric {
    font-size: 22px;
    font-weight: bold;
    color: #22c55e;
}

.insight-good { color: #4ade80; }
.insight-mid { color: #facc15; }
.insight-bad { color: #f87171; }

</style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 JEE Rank + Category Predictor")
st.markdown("### Predict smarter. Understand better.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("📅 Select Year", [2021, 2022, 2023, 2024])

with col2:
    marks = st.slider("📝 Enter Marks", 0, 300, 150)

# Prediction
if st.button("⚡ Predict Now"):

    input_df = pd.DataFrame([[year, marks]], columns=['Year', 'Marks'])
    X_scaled = scaler.transform(input_df)

    # Predictions
    rank_pred = int(model_reg.predict(X_scaled)[0])
    cat_encoded = int(np.round(model_clf.predict(X_scaled)[0]))
    category = encoder.inverse_transform([cat_encoded])[0]

    lower = int(rank_pred * 0.85)
    upper = int(rank_pred * 1.15)

    # 🎯 RESULT CARD
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🎯 Prediction Results")

    st.markdown(f"<p class='metric'>Predicted Rank: {rank_pred}</p>", unsafe_allow_html=True)
    st.markdown(f"📉 Expected Range: **{lower} - {upper}**")
    st.markdown(f"🏷️ Category: **{category}**")

    st.markdown("</div>", unsafe_allow_html=True)

    # 📊 INSIGHTS (ENHANCED)
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
        st.markdown("<p class='insight-bad'>⚠️ Low Zone — Consider improvement strategy</p>", unsafe_allow_html=True)
        st.progress(30)

    # Extra intelligence
    if marks < 120:
        st.info("💡 Tip: Focus on accuracy over attempts")
    elif marks > 200:
        st.success("🚀 Strong scoring pattern detected")

    st.markdown("</div>", unsafe_allow_html=True)

    # 📈 CONFIDENCE VISUAL
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Confidence")

    spread = upper - lower
    confidence = max(0, 100 - (spread / rank_pred * 100))

    st.write(f"Confidence Score: **{int(confidence)}%**")
    st.progress(int(confidence))

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ for JEE Aspirants</center>", unsafe_allow_html=True)
