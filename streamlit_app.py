"""
Streamlit dashboard for the Customer Segmentation & Retention project
=======================================================================
Same idea as app.py (Flask) + static/index.html, rebuilt as a single
Streamlit app. Click "Run Analysis" -> re-runs Telco + Retail + Uplift
(always) and Netflix (if you upload its CSV) -> renders fresh results.

Run locally:
  streamlit run streamlit_app.py

Deploy on Render:
  Start command: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
"""

import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import churn_segmentation_project as telco
import netflix_churn_segmentation as netflix
import retail_segmentation as retail
import uplift_retention_model as uplift

DASHBOARD_DATA_PATH = "dashboard_data.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Customer Segmentation & Retention", layout="wide")


# ---------------------------------------------------------------------------
# Analysis runner (same logic as app.py's /api/run)
# ---------------------------------------------------------------------------
def safe_run(label: str, fn, *args):
    try:
        result = fn(*args)
        return result, None
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def cross_domain_comparison(domain_summaries: dict) -> dict:
    churn_rate_by_domain, auc_by_domain, top_driver_by_domain = {}, {}, {}
    for summary in domain_summaries.values():
        if not summary:
            continue
        churn_rate_by_domain[summary["domain"]] = summary["churn_rate"]
        auc_by_domain[summary["domain"]] = summary["model_auc"]
        if summary.get("top_drivers"):
            top_driver_by_domain[summary["domain"]] = summary["top_drivers"][0]["feature"]
    return {
        "churn_rate_by_domain": churn_rate_by_domain,
        "auc_by_domain": auc_by_domain,
        "top_driver_by_domain": top_driver_by_domain,
    }


def run_analysis(netflix_file) -> dict:
    netflix_path = None
    if netflix_file is not None:
        netflix_path = os.path.join(UPLOAD_DIR, netflix_file.name)
        with open(netflix_path, "wb") as f:
            f.write(netflix_file.getbuffer())

    progress = st.progress(0, text="Running Telco...")
    telco_summary, telco_err = safe_run("Telco", telco.run)
    progress.progress(33, text="Running Retail...")
    retail_summary, retail_err = safe_run("Retail", retail.run)
    progress.progress(60, text="Running Uplift...")
    uplift_summary, uplift_err = safe_run("Uplift", uplift.run)

    netflix_summary, netflix_err = None, "skipped (no CSV uploaded)"
    if netflix_path:
        progress.progress(80, text="Running Netflix...")
        netflix_summary, netflix_err = safe_run("Netflix", netflix.run, netflix_path)

    progress.progress(100, text="Done.")
    progress.empty()

    domain_summaries = {"telco": telco_summary, "netflix": netflix_summary, "retail": retail_summary}
    comparison = cross_domain_comparison(domain_summaries)

    data = {
        "_demo": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domains": domain_summaries,
        "domain_errors": {"telco": telco_err, "netflix": netflix_err, "retail": retail_err},
        "uplift": uplift_summary,
        "uplift_error": uplift_err,
        "cross_domain_comparison": comparison,
    }

    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    return data


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_domain(summary: dict, error: str | None):
    if error and not summary:
        st.info(f"Skipped or failed: {error}")
        return
    if not summary:
        st.info("No data yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn rate", f"{summary['churn_rate']*100:.1f}%")
    c1.caption(f"{summary['n_customers']:,} customers")
    if summary.get("top_drivers"):
        c2.metric("Top churn driver", summary["top_drivers"][0]["feature"])
    if summary.get("clv_summary"):
        c3.metric("Avg CLV", f"${summary['clv_summary']['mean']:,.1f}")

    auc = summary.get("model_auc", {})
    if auc:
        st.caption("Model AUC — " + " · ".join(f"{k}: {v}" for k, v in auc.items()))

    if summary.get("top_drivers"):
        st.subheader("Top churn drivers")
        drivers_df = pd.DataFrame(summary["top_drivers"]).set_index("feature")
        st.bar_chart(drivers_df["importance"])

    if summary.get("segments"):
        st.subheader("Segments")
        st.dataframe(pd.DataFrame(summary["segments"]), use_container_width=True)

    if summary.get("priority_matrix"):
        st.subheader("Retention priority matrix")
        pm_df = pd.DataFrame(
            {"action": list(summary["priority_matrix"].keys()), "count": list(summary["priority_matrix"].values())}
        ).set_index("action")
        st.bar_chart(pm_df["count"])


def render_uplift(uplift_summary, uplift_error):
    if uplift_error and not uplift_summary:
        st.info(f"Skipped or failed: {uplift_error}")
        return
    if not uplift_summary:
        st.info("No data yet.")
        return
    # Shape of uplift_summary isn't known here (uplift_retention_model.py
    # wasn't available when this file was written) -- render generically.
    if isinstance(uplift_summary, dict):
        top_level = {k: v for k, v in uplift_summary.items() if not isinstance(v, (dict, list))}
        if top_level:
            cols = st.columns(len(top_level))
            for col, (k, v) in zip(cols, top_level.items()):
                col.metric(k.replace("_", " ").title(), v)
    st.json(uplift_summary)


def render_comparison(comparison: dict):
    if not comparison:
        return
    st.subheader("Cross-domain comparison")
    if comparison.get("churn_rate_by_domain"):
        st.caption("Churn rate by domain")
        st.bar_chart(pd.Series(comparison["churn_rate_by_domain"]))
    if comparison.get("top_driver_by_domain"):
        st.caption("Top churn driver by domain")
        st.table(pd.Series(comparison["top_driver_by_domain"], name="top driver"))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Customer Segmentation & Retention Analysis")
st.caption("Live dashboard — Streamlit + the same analysis modules as the Flask version")

with st.sidebar:
    st.header("Run analysis")
    netflix_file = st.file_uploader("Netflix CSV (optional)", type="csv")
    run_clicked = st.button("Run Analysis", type="primary")
    st.caption("First run downloads real datasets (Telco, UK retail, Hillstrom email experiment) and trains models — can take 30–90 seconds.")

if run_clicked:
    with st.spinner("Running analysis..."):
        st.session_state["dashboard_data"] = run_analysis(netflix_file)

if "dashboard_data" not in st.session_state:
    if os.path.exists(DASHBOARD_DATA_PATH):
        with open(DASHBOARD_DATA_PATH) as f:
            st.session_state["dashboard_data"] = json.load(f)

data = st.session_state.get("dashboard_data")

if not data:
    st.info("No analysis has been run yet. Click **Run Analysis** in the sidebar.")
else:
    st.caption(f"Last generated: {data.get('generated_at', 'unknown')}")
    tabs = st.tabs(["Telco", "Retail", "Netflix", "Uplift", "Cross-domain"])

    with tabs[0]:
        render_domain(data["domains"].get("telco"), data.get("domain_errors", {}).get("telco"))
    with tabs[1]:
        render_domain(data["domains"].get("retail"), data.get("domain_errors", {}).get("retail"))
    with tabs[2]:
        render_domain(data["domains"].get("netflix"), data.get("domain_errors", {}).get("netflix"))
    with tabs[3]:
        render_uplift(data.get("uplift"), data.get("uplift_error"))
    with tabs[4]:
        render_comparison(data.get("cross_domain_comparison"))
