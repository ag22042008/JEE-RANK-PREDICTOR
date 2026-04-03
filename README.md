# 📊 JEE Rank Predictor

A machine learning web app that predicts **JEE rank based on Marks and Year**, along with interactive data analysis and visual insights.

🔗 **Live App:** https://jee-rank-predictor-fabsethg6gqpczk7kqp8zr.streamlit.app/

---

## 🚀 Features

* 🎯 Predict JEE Rank using:

  * Marks (out of 360)
  * Exam Year
* 🤖 Multiple Models:

  * Linear Regression
  * Polynomial Regression
* 📊 Interactive Visualizations:

  * Marks vs Rank
  * Year vs Rank
  * Historical distribution
* 📈 Model Evaluation:

  * R² Score
  * Mean Absolute Error (MAE)
* 🧠 Insight Section:

  * Data-driven conclusions from EDA
* 🎨 Clean UI with navigation (EDA | Insights | Model | Prediction)

---

## 🧠 Key Insights

* 📉 **Marks vs Rank** shows a strong inverse relationship
* 📊 Rank distribution is **non-linear**, especially at higher marks
* 📅 Year has **minor influence** compared to marks
* 🤖 Polynomial model performs better than linear model

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
  * Scikit-learn

---

## 📂 Project Structure

```
📁 jee-rank-predictor
│
├── app.py                # Main Streamlit app
├── dataset.csv          # Dataset used
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/jee-rank-predictor.git
cd jee-rank-predictor
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 📊 Model Details

* **Input Features:** Marks, Year
* **Target:** Rank

### Models Used:

* Linear Regression
* Polynomial Regression (degree = 2)

### Evaluation Metrics:

* R² Score (model accuracy)
* Mean Absolute Error (prediction error)

---

## ⚠️ Limitations

* Rank prediction is **approximate**, not exact
* Model assumes consistent difficulty across years
* Does not account for:

  * Paper difficulty variation
  * Category-based ranks
  * Normalization differences

---

## 💡 Future Improvements

* 🔥 Log transformation for better accuracy
* 📊 Confidence intervals for predictions
* 🧠 Explainable AI (why this rank?)
* 📈 More advanced models (XGBoost, etc.)

---

## 👨‍💻 Author

**Aditya Gupta**

* Passionate about Machine Learning & Data Science
* Focused on building real-world ML applications

---

## ⭐ If you liked this project

Give it a ⭐ on GitHub and share feedback!
