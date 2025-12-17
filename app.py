import os
import sys
import time
import subprocess
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from scrape_shl2 import scrape_shl_catalog, save_to_csv
from openrouter_api import generate_content, create_embeddings

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="SHL Intelligent Recommendation System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# GLOBAL STYLES
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #0A2540;
}

.sub-title {
    font-size: 16px;
    color: #5E6E85;
    margin-bottom: 25px;
}

.card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.badge {
    display: inline-block;
    background-color: #EEF4FF;
    color: #1E40AF;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 4px;
}

.relevance {
    background: #F8FAFC;
    padding: 14px;
    border-left: 4px solid #2563EB;
    border-radius: 6px;
    font-size: 14px;
}

section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ENV + API SETUP
# =========================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment")
    st.stop()

API_ENDPOINT = "http://localhost:8000"

# =========================
# API PROCESS MANAGEMENT
# =========================
def start_api_server():
    try:
        process = subprocess.Popen(
            [sys.executable, "api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        return process
    except Exception as e:
        st.error(f"Failed to start API: {e}")
        return None

def check_api_health():
    try:
        r = requests.get(f"{API_ENDPOINT}/health")
        return r.status_code == 200
    except:
        return False

# =========================
# API CALLS
# =========================
def get_recommendations_from_api(query, max_results=10):
    try:
        r = requests.post(
            f"{API_ENDPOINT}/recommend",
            json={"query": query, "max_results": max_results}
        )
        if r.status_code == 200:
            return r.json()["recommended_assessments"]
        return []
    except Exception as e:
        st.error(f"API error: {e}")
        return []

# =========================
# JOB DESCRIPTION SCRAPER
# =========================
def scrape_job_description(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        content = soup.find("div", class_=["job-description", "description"])
        if content:
            return content.get_text(" ", strip=True)

        return " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    except Exception as e:
        return f"Error: {e}"

# =========================
# MAIN APP
# =========================
def main():

    # -------------------------
    # Start / Check API
    # -------------------------
    if not check_api_health():
        st.info("Starting backend AI service...")
        start_api_server()
        if not check_api_health():
            st.error("Failed to start API service")
            st.stop()

    # -------------------------
    # HEADER
    # -------------------------
    st.markdown("""
    <div class="main-title">🧠 SHL Intelligent Recommendation System</div>
    <div class="sub-title">
    AI-powered engine to match job roles with optimal SHL assessments
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # SIDEBAR CONTROLS
    # -------------------------
    with st.sidebar:
        st.markdown("## 🔍 Controls")
        input_mode = st.radio("Input Type", ["Text Query", "Job Description URL"])
        max_results = st.slider("Max Recommendations", 3, 15, 10)

    # -------------------------
    # INPUT SECTION
    # -------------------------
    query = None

    if input_mode == "Text Query":
        query = st.text_area(
            "Describe role requirements",
            height=160,
            placeholder="Hiring Java developers with collaboration skills. Assessments under 40 minutes."
        )
        run = st.button("🔎 Generate Recommendations")

    else:
        url = st.text_input("Paste Job Description URL")
        run = st.button("🌐 Fetch & Analyze")

        if run and url:
            with st.spinner("Extracting job description..."):
                query = scrape_job_description(url)
                st.text_area("Extracted Job Description", query, height=200)

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    if run and query:
        with st.spinner("🧠 AI analyzing role & constraints..."):
            results = get_recommendations_from_api(query, max_results)

        if not results:
            st.warning("No assessments found. Try refining your query.")
            return

        # -------------------------
        # METRICS
        # -------------------------
        durations = [r["duration"] for r in results if r["duration"]]
        test_types = [t for r in results for t in r["test_type"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Recommendations", len(results))
        c2.metric("Unique Test Types", len(set(test_types)))
        c3.metric("Avg Duration (mins)", int(np.mean(durations)) if durations else "N/A")

        st.markdown("## 📊 Recommended Assessments")

        # -------------------------
        # CARDS
        # -------------------------
        for i, a in enumerate(results):
            st.markdown(f"""
            <div class="card">
                <h3>{i+1}. <a href="{a['url']}" target="_blank">{a['description'].split('.')[0]}</a></h3>

                <div>
                    {" ".join([f"<span class='badge'>{t}</span>" for t in a['test_type']])}
                </div>

                <p><b>⏱ Duration:</b> {a['duration']} minutes</p>
                <p><b>🌍 Remote Testing:</b> {a['remote_support']}</p>
                <p><b>📊 Adaptive / IRT:</b> {a['adaptive_support']}</p>

                <div class="relevance">
                    <b>Why this assessment?</b><br>
                    {". ".join(a['description'].split('.')[1:])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -------------------------
        # VISUALIZATION
        # -------------------------
        df_vis = pd.Series(test_types).value_counts().reset_index()
        df_vis.columns = ["Test Type", "Count"]

        fig = px.bar(
            df_vis,
            x="Test Type",
            y="Count",
            title="Assessment Type Distribution",
            color="Test Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("---")
    st.markdown(
        "SHL Intelligent Assessment System | Built with Streamlit, FastAPI & LLMs"
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
