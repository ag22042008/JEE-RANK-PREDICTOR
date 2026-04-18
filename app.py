"""
JEE Rank Predictor – app.py
Run: python app.py
Models are trained once at startup; no training happens per request.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from flask import Flask, render_template_string, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA & FEATURE ENGINEERING  (exactly as in the notebook)
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Loading dataset …")
df = pd.read_csv("jee_marks_percentile_rank_2009_2026.csv")

df["RankRatio"] = df["Rank"] / df["Total_Candidates"]

def get_category(ratio: float) -> str:
    if ratio <= 0.005:
        return "Elite (Top 0.5%)"
    elif ratio <= 0.02:
        return "Top Tier (0.5% – 2%)"
    elif ratio <= 0.05:
        return "Highly Competitive (2% – 5%)"
    elif ratio <= 0.10:
        return "Competitive (5% – 10%)"
    else:
        return "Not Prepared (>10%)"

df["Category"] = df["RankRatio"].apply(get_category)

# Year → expected total candidates (auto-fill in the UI)
YEAR_CANDIDATES = df.groupby("Year")["Total_Candidates"].first().to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN REGRESSION MODEL  (Rank prediction)
#     Features : [Year, Marks, Total_Candidates]
#     Target   : log1p(Rank)
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Training rank-prediction model …")
X_reg  = df[["Year", "Marks", "Total_Candidates"]]
Y_reg  = np.log1p(df["Rank"])

sc_reg = StandardScaler()
Xr_tr, Xr_te, Yr_tr, Yr_te = train_test_split(
    X_reg, Y_reg, test_size=0.2, random_state=42
)
Xr_tr_s = sc_reg.fit_transform(Xr_tr)
Xr_te_s = sc_reg.transform(Xr_te)

rf = RandomForestRegressor(
    n_estimators=300, max_depth=5,
    min_samples_split=3, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
rf.fit(Xr_tr_s, Yr_tr)
r2_reg = rf.score(Xr_te_s, Yr_te)
print(f"   ✅  Regression R² (log scale) = {r2_reg:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN CLASSIFICATION MODEL  (Category prediction)
#     Features : [Year, Marks, Rank]
#     Target   : Category (encoded)
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Training category-classification model …")
X_clf  = df[["Year", "Marks", "Rank"]]
encoder = LabelEncoder()
Y_clf  = encoder.fit_transform(df["Category"])

sc_clf = StandardScaler()
Xc_tr, Xc_te, Yc_tr, Yc_te = train_test_split(
    X_clf, Y_clf, test_size=0.2, random_state=42
)
Xc_tr_s = sc_clf.fit_transform(Xc_tr)
Xc_te_s = sc_clf.transform(Xc_te)

smote = SMOTE(random_state=42)
Xc_tr_res, Yc_tr_res = smote.fit_resample(Xc_tr_s, Yc_tr)

xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.02,
    max_depth=3, subsample=0.8,
    random_state=42, eval_metric="mlogloss"
)
xgb.fit(Xc_tr_res, Yc_tr_res)
acc_clf = xgb.score(Xc_te_s, Yc_te)
print(f"   ✅  Classifier accuracy = {acc_clf:.4f}")
print("🚀  Models ready – starting Flask server …\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_rank_and_category(year: int, marks: float, total_candidates: int):
    """Return predicted rank, lower/upper bound, and category string."""
    X_new    = np.array([[year, marks, total_candidates]], dtype=float)
    X_new_s  = sc_reg.transform(X_new)

    # Point estimate
    pred_log = rf.predict(X_new_s)[0]
    pred_rank = int(round(np.expm1(pred_log)))

    # 10-90 percentile interval across trees
    tree_preds = np.array([t.predict(X_new_s)[0] for t in rf.estimators_])
    lower_rank = int(round(np.expm1(np.percentile(tree_preds, 10))))
    upper_rank = int(round(np.expm1(np.percentile(tree_preds, 90))))

    # Category using predicted rank
    X_cat   = np.array([[year, marks, pred_rank]], dtype=float)
    X_cat_s = sc_clf.transform(X_cat)
    cat_enc = xgb.predict(X_cat_s)[0]
    category = encoder.inverse_transform([cat_enc])[0]

    return pred_rank, lower_rank, upper_rank, category

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>JEE Rank Predictor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  :root {
    --bg:       #0a0e1a;
    --card:     #111827;
    --card2:    #1a2235;
    --border:   #1e2d47;
    --accent:   #4f8ef7;
    --accent2:  #7c3aed;
    --green:    #10b981;
    --yellow:   #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── HERO HEADER ── */
  header {
    width: 100%;
    padding: 28px 24px 22px;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    text-align: center;
  }
  header .badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
  }
  header h1 {
    font-size: clamp(24px, 4vw, 40px);
    font-weight: 800;
    background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
  }
  header p {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
  }

  /* ── MAIN LAYOUT ── */
  main {
    width: 100%;
    max-width: 900px;
    padding: 36px 16px 60px;
    display: grid;
    gap: 24px;
  }

  /* ── INPUT CARD ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 32px;
  }
  .card-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-title span.icon { font-size: 20px; }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  @media (max-width: 560px) { .form-grid { grid-template-columns: 1fr; } }

  label {
    display: flex;
    flex-direction: column;
    gap: 7px;
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .7px;
  }

  input, select {
    background: #0d1526;
    border: 1.5px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-size: 16px;
    font-family: 'Inter', sans-serif;
    padding: 12px 16px;
    outline: none;
    transition: border-color .2s, box-shadow .2s;
    width: 100%;
  }
  input:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(79,142,247,.18);
  }
  select option { background: #0d1526; }

  /* Range slider */
  .marks-wrapper { display: flex; align-items: center; gap: 12px; }
  .marks-num {
    font-size: 22px;
    font-weight: 800;
    color: var(--accent);
    min-width: 46px;
    text-align: right;
  }
  input[type=range] {
    -webkit-appearance: none;
    flex: 1;
    height: 6px;
    background: var(--border);
    border: none;
    border-radius: 99px;
    padding: 0;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    cursor: pointer;
    box-shadow: 0 0 8px rgba(79,142,247,.5);
    border: none;
  }

  /* Candidate info chip */
  .cand-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(79,142,247,.1);
    border: 1px solid rgba(79,142,247,.25);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    color: #7dd3fc;
    margin-top: 8px;
  }

  /* ── PREDICT BUTTON ── */
  .btn-predict {
    width: 100%;
    margin-top: 28px;
    padding: 16px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: .5px;
    color: #fff;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
    box-shadow: 0 4px 24px rgba(79,142,247,.35);
    transition: transform .15s, box-shadow .15s, opacity .15s;
    position: relative;
    overflow: hidden;
  }
  .btn-predict:active { transform: scale(.98); }
  .btn-predict:hover  { box-shadow: 0 6px 32px rgba(79,142,247,.5); }
  .btn-predict:disabled { opacity: .6; cursor: not-allowed; }

  /* spinning loader inside button */
  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2.5px solid rgba(255,255,255,.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin .65s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── RESULTS CARD ── */
  #results {
    display: none;
    animation: fadeUp .4s ease;
  }
  @keyframes fadeUp {
    from { opacity:0; transform: translateY(18px); }
    to   { opacity:1; transform: translateY(0); }
  }

  .result-hero {
    text-align: center;
    padding: 28px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 26px;
  }
  .result-hero .label { font-size: 12px; color: var(--muted); letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 8px; }
  .result-hero .rank-num {
    font-size: clamp(48px, 10vw, 80px);
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
  }
  .result-hero .rank-range {
    margin-top: 10px;
    font-size: 14px;
    color: var(--muted);
  }
  .result-hero .rank-range strong { color: var(--text); }

  /* category pill */
  .cat-pill {
    display: inline-block;
    padding: 7px 20px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 700;
    margin-top: 18px;
  }
  .cat-elite          { background: linear-gradient(90deg,#10b981,#059669); color:#fff; }
  .cat-top            { background: linear-gradient(90deg,#3b82f6,#1d4ed8); color:#fff; }
  .cat-highly         { background: linear-gradient(90deg,#8b5cf6,#6d28d9); color:#fff; }
  .cat-competitive    { background: linear-gradient(90deg,#f59e0b,#b45309); color:#fff; }
  .cat-not            { background: linear-gradient(90deg,#ef4444,#b91c1c); color:#fff; }

  /* Stats row */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }
  @media (max-width: 540px) { .stats-row { grid-template-columns: 1fr 1fr; } }

  .stat-box {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
  }
  .stat-box .s-val {
    font-size: 22px;
    font-weight: 800;
    color: var(--text);
  }
  .stat-box .s-lbl {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .7px;
    margin-top: 4px;
  }

  /* confidence bar */
  .conf-section { margin-top: 22px; }
  .conf-title { font-size: 13px; color: var(--muted); margin-bottom: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .7px; }
  .conf-track {
    background: var(--border);
    border-radius: 99px;
    height: 10px;
    position: relative;
    overflow: visible;
  }
  .conf-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width .8s cubic-bezier(.4,0,.2,1);
  }
  .conf-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
  }

  /* disclaimer */
  .disclaimer {
    margin-top: 22px;
    padding: 14px 18px;
    background: rgba(245,158,11,.07);
    border: 1px solid rgba(245,158,11,.2);
    border-radius: 10px;
    font-size: 12.5px;
    color: #fbbf24;
    line-height: 1.6;
  }

  /* error */
  #error-msg {
    display: none;
    padding: 14px 18px;
    background: rgba(239,68,68,.1);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 10px;
    color: #fca5a5;
    font-size: 14px;
    margin-top: 14px;
  }

  /* ── INFO BANNER ── */
  .info-strip {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 24px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
  }
  .info-item { text-align: center; }
  .info-item .iv { font-size: 20px; font-weight: 800; color: var(--accent); }
  .info-item .il { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; }

  footer {
    padding: 18px;
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    border-top: 1px solid var(--border);
    width: 100%;
    max-width: 900px;
  }
</style>
</head>
<body>

<header>
  <div class="badge">AI-Powered Predictor</div>
  <h1>JEE Rank Predictor</h1>
  <p>Trained on 2009–2026 data &nbsp;·&nbsp; Random Forest + XGBoost ensemble</p>
</header>

<main>

  <!-- Stats strip -->
  <div class="info-strip">
    <div class="info-item"><div class="iv">1,620</div><div class="il">Data Points</div></div>
    <div class="info-item"><div class="iv">2009–2026</div><div class="il">Years Covered</div></div>
    <div class="info-item"><div class="iv">300 Trees</div><div class="il">Random Forest</div></div>
    <div class="info-item"><div class="iv">5 Tiers</div><div class="il">Category Labels</div></div>
  </div>

  <!-- Input card -->
  <div class="card">
    <div class="card-title"><span class="icon">📝</span> Enter Your Details</div>

    <div class="form-grid">
      <!-- Marks slider -->
      <label style="grid-column: 1 / -1;">
        Your JEE Marks (out of 300)
        <div class="marks-wrapper">
          <input type="range" id="marksRange" min="0" max="300" value="150"
                 oninput="syncMarks(this.value)">
          <span class="marks-num" id="marksDisplay">150</span>
        </div>
      </label>

      <!-- Marks number input -->
      <label>
        Or type exact marks
        <input type="number" id="marksInput" min="0" max="300" value="150"
               placeholder="0 – 300" oninput="syncMarksInput(this.value)">
      </label>

      <!-- Year -->
      <label>
        Exam Year
        <select id="yearSelect" onchange="updateCandidates()">
          {% for yr in years %}
          <option value="{{ yr }}" {% if yr == 2026 %}selected{% endif %}>{{ yr }}</option>
          {% endfor %}
        </select>
      </label>
    </div>

    <!-- Candidate chip -->
    <div class="cand-chip" id="candChip">
      👥 <span id="candText">Expected candidates in 2026: <strong>14,00,000</strong></span>
    </div>

    <button class="btn-predict" id="predictBtn" onclick="predict()">
      <span id="btnText">🔮 Predict My JEE Rank</span>
      <div class="spinner" id="spinner"></div>
    </button>

    <div id="error-msg"></div>
  </div>

  <!-- Results card -->
  <div class="card" id="results">
    <div class="card-title"><span class="icon">📊</span> Prediction Results</div>

    <div class="result-hero">
      <div class="label">Predicted JEE Rank</div>
      <div class="rank-num" id="rankNum">–</div>
      <div class="rank-range">
        Confidence Range: <strong id="rankRange">–</strong>
      </div>
      <div id="catPill" class="cat-pill">–</div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="s-val" id="sMarks">–</div>
        <div class="s-lbl">Your Marks</div>
      </div>
      <div class="stat-box">
        <div class="s-val" id="sYear">–</div>
        <div class="s-lbl">Exam Year</div>
      </div>
      <div class="stat-box">
        <div class="s-val" id="sCands">–</div>
        <div class="s-lbl">Total Candidates</div>
      </div>
    </div>

    <!-- Confidence width bar -->
    <div class="conf-section">
      <div class="conf-title">Prediction Confidence (narrower range = higher confidence)</div>
      <div class="conf-track">
        <div class="conf-fill" id="confBar" style="width:0%"></div>
      </div>
      <div class="conf-labels">
        <span id="confLow">–</span>
        <span id="confHigh">–</span>
      </div>
    </div>

    <div class="disclaimer">
      ⚠️  This is an ML-based estimate trained on historical JEE data (2009–2026).
      Actual ranks may differ due to paper difficulty, normalisation, and other factors.
      Use this as a reference, not an official result.
    </div>
  </div>

</main>

<footer>
  JEE Rank Predictor &nbsp;·&nbsp; Data: 2009–2026 &nbsp;·&nbsp; Model: Random Forest Regressor + XGBoost Classifier
</footer>

<script>
  // Year → total candidates map injected from Python
  const YEAR_CANDS = {{ year_cands | tojson }};

  function fmt(n) {
    return n.toLocaleString('en-IN');
  }

  function syncMarks(v) {
    document.getElementById('marksDisplay').textContent = v;
    document.getElementById('marksInput').value = v;
  }

  function syncMarksInput(v) {
    let val = Math.min(300, Math.max(0, parseInt(v) || 0));
    document.getElementById('marksRange').value = val;
    document.getElementById('marksDisplay').textContent = val;
  }

  function updateCandidates() {
    const yr = parseInt(document.getElementById('yearSelect').value);
    const cands = YEAR_CANDS[yr] || 1400000;
    document.getElementById('candText').innerHTML =
      `Expected candidates in ${yr}: <strong>${fmt(cands)}</strong>`;
  }

  function categoryClass(cat) {
    if (cat.includes('Elite'))            return 'cat-elite';
    if (cat.includes('Top Tier'))         return 'cat-top';
    if (cat.includes('Highly'))           return 'cat-highly';
    if (cat.includes('Competitive'))      return 'cat-competitive';
    return 'cat-not';
  }

  async function predict() {
    const marks = parseFloat(document.getElementById('marksInput').value);
    const year  = parseInt(document.getElementById('yearSelect').value);

    if (isNaN(marks) || marks < 0 || marks > 300) {
      showError("Please enter marks between 0 and 300.");
      return;
    }

    // Show spinner
    document.getElementById('btnText').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('error-msg').style.display = 'none';
    document.getElementById('results').style.display = 'none';

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ marks, year })
      });
      const data = await res.json();

      if (data.error) { showError(data.error); return; }

      // Populate results
      document.getElementById('rankNum').textContent   = fmt(data.rank);
      document.getElementById('rankRange').textContent = `${fmt(data.lower)} – ${fmt(data.upper)}`;
      document.getElementById('sMarks').textContent    = marks;
      document.getElementById('sYear').textContent     = year;
      document.getElementById('sCands').textContent    = fmt(data.total_candidates);
      document.getElementById('confLow').textContent   = fmt(data.lower);
      document.getElementById('confHigh').textContent  = fmt(data.upper);

      // Category pill
      const pill = document.getElementById('catPill');
      pill.textContent  = data.category;
      pill.className    = 'cat-pill ' + categoryClass(data.category);

      // Confidence bar: narrower interval relative to total = higher confidence
      const spread   = data.upper - data.lower;
      const maxSpread = data.total_candidates;
      const confPct  = Math.max(5, Math.min(100, 100 - (spread / maxSpread) * 100));
      document.getElementById('confBar').style.width = confPct.toFixed(1) + '%';

      document.getElementById('results').style.display = 'block';
    } catch(e) {
      showError("Network error – make sure the server is running.");
    } finally {
      document.getElementById('btnText').style.display = 'block';
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('predictBtn').disabled = false;
    }
  }

  function showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = '⚠️ ' + msg;
    el.style.display = 'block';
    document.getElementById('btnText').style.display = 'block';
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('predictBtn').disabled = false;
  }

  // Init
  updateCandidates();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    years = sorted(YEAR_CANDIDATES.keys(), reverse=True)
    return render_template_string(HTML, years=years, year_cands=YEAR_CANDIDATES)


@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        body  = request.get_json(force=True)
        marks = float(body.get("marks", 0))
        year  = int(body.get("year",  2026))

        if not (0 <= marks <= 300):
            return jsonify({"error": "Marks must be between 0 and 300."}), 400
        if year not in YEAR_CANDIDATES:
            return jsonify({"error": f"Year {year} not in training data."}), 400

        total_candidates = YEAR_CANDIDATES[year]
        rank, lower, upper, category = predict_rank_and_category(year, marks, total_candidates)

        return jsonify({
            "rank":             rank,
            "lower":            lower,
            "upper":            upper,
            "category":         category,
            "total_candidates": total_candidates,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
