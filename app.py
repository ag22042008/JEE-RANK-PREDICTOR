import pandas as pd
import plotly.express as px
import streamlit as st
import base64

# =========================
# PAGE CONFIG (MOBILE SAFE)
# =========================
st.set_page_config(
    page_title="IPL 2025 Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# BACKGROUND IMAGE
# =========================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.6);
            z-index: -1;
        }}

        h1, h2, h3, h4, p {{
            color: #ffffff;
        }}

        /* Mobile responsiveness */
        @media (max-width: 768px) {{
            h1 {{ font-size: 1.8rem; text-align: center; }}
            h2 {{ font-size: 1.4rem; }}
            h3 {{ font-size: 1.2rem; }}
            p {{ font-size: 0.95rem; }}
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("ChatGPT Image Feb 1, 2026, 10_26_21 AM.png")

# =========================
# TITLE
# =========================
st.title("🏏 IPL 2025 Interactive Data Analysis Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    bowlers = pd.read_csv("IPL2025Bowlers.csv")
    matches.fillna(0, inplace=True)
    return matches, bowlers

ipl2025, bowlers = load_data()

# =========================
# MOBILE-SAFE NAVIGATION
# =========================
st.markdown("### 📍 Navigation")

section = None
if st.button("📊 Dataset Overview"):
    section = "dataset"
if st.button("🪙 Toss & Results"):
    section = "toss"
if st.button("🔥 High Scoring Matches"):
    section = "runs"
if st.button("📈 Points & Batting"):
    section = "batting"
if st.button("🎯 Bowling & Insights"):
    section = "bowling"

# =========================
# DATASET OVERVIEW
# =========================
if section == "dataset":
    st.header("📊 Dataset Overview & Structural Analysis")

    st.markdown("""
**Match Dataset**
- 74 matches, 22 columns
- Numerical, categorical & text features
- Missing values mostly due to no-result / abandoned matches
- Suitable for scoring, toss impact & result analysis

**Bowling Dataset**
- 108 bowlers, 13 attributes
- No missing values
- Reliable for economy, wickets & workload analysis
    """)

# =========================
# TOSS & RESULTS
# =========================
elif section == "toss":
    st.header("🪙 Toss & Match Results")

    toss = ipl2025["toss_decision"].value_counts().reset_index()
    toss.columns = ["Decision", "Count"]

    fig = px.bar(toss, x="Decision", y="Count", color="Decision",
                 title="Toss Decision: Bat vs Field")
    st.plotly_chart(fig, use_container_width=True)

    toss_win = ipl2025[ipl2025["toss_winner"] == ipl2025["match_winner"]]
    win_rate = round(len(toss_win) / len(ipl2025) * 100, 2)

    st.markdown(f"""
**Conclusion**
- Toss advantage exists but is not decisive.
- Toss winners won **{win_rate}%** of matches.
    """)

# =========================
# HIGH SCORING MATCHES
# =========================
elif section == "runs":
    st.header("🔥 Top 10 Highest Scoring Matches")

    ipl2025["total_score"] = (
        ipl2025["first_ings_score"] + ipl2025["second_ings_score"]
    )

    top10 = ipl2025.sort_values("total_score", ascending=False).head(10)

    fig = px.bar(
        top10,
        x="match_id",
        y="total_score",
        color="total_score",
        color_continuous_scale="Inferno",
        hover_data=["team1", "team2", "venue", "stage"],
        title="Top 10 Highest Scoring Matches – IPL 2025"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Analysis**
- These matches form the extreme high-scoring end of IPL 2025.
- All exceed tournament average by a large margin.

**Conclusion**
- A few ultra-high scoring matches dominate the season narrative.
- Bowling discipline in middle & death overs is critical.
    """)

# =========================
# POINTS & BATTING
# =========================
elif section == "batting":
    st.header("📈 Points Table & Batting Analysis")

    pts = ipl2025[ipl2025["match_winner"] != 0]["match_winner"].value_counts() * 2
    ties = ipl2025[ipl2025["match_winner"] == 0]

    for _, r in ties.iterrows():
        pts[r["team1"]] = pts.get(r["team1"], 0) + 1
        pts[r["team2"]] = pts.get(r["team2"], 0) + 1

    pts = pts.sort_values(ascending=False).reset_index()
    pts.columns = ["Team", "Points"]

    fig = px.bar(pts, x="Team", y="Points", color="Points",
                 title="IPL 2025 Points Table")
    st.plotly_chart(fig, use_container_width=True)

    bat = (
        ipl2025[ipl2025["top_scorer"] != 0]
        .groupby("top_scorer")["highscore"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(bat, y="top_scorer", x="highscore",
                 color="highscore",
                 title="Top Individual Scores")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# BOWLING & ADVANCED INSIGHTS
# =========================
elif section == "bowling":
    st.header("🎯 Bowling Analysis & Insights")

    eco = bowlers.sort_values("ECO").head(10)
    fig = px.bar(eco, y="Player Name", x="ECO", color="Team",
                 title="Most Economical Bowlers")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** Bumrah & Unadkat dominate run control.")

    fig = px.scatter(
        bowlers, x="WKT", y="ECO", size="OVR",
        color="Team", hover_name="Player Name",
        title="Wickets vs Economy Rate"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** Expensive bowlers rarely get long spells.")

    team_eco = bowlers.groupby("Team")["ECO"].mean().reset_index()
    fig = px.bar(team_eco, x="Team", y="ECO", color="ECO",
                 title="Team-wise Average Economy")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** MI & RCB show strong bowling balance.")

    death = bowlers[bowlers["OVR"] > 15]
    fig = px.scatter(death, x="ECO", y="SR", color="Team",
                     hover_name="Player Name",
                     title="Death Overs: Economy vs Strike Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** Bumrah is the most reliable death bowler.")

    impact = bowlers[bowlers["OVR"] < 16].head(10)
    fig = px.bar(impact, x="Player Name", y="WKT", color="Team",
                 title="Impact Bowlers")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** Short spells can be match-deciding.")

    underrated = (
        bowlers[(bowlers["MAT"] < 8) & (bowlers["MAT"] > 3)]
        .sort_values("ECO")
        .head(5)
    )

    fig = px.bar(underrated, y="Player Name", x="ECO",
                 color="Team", title="Underrated Bowlers")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** Jaydev Unadkat stands out as underrated.")
