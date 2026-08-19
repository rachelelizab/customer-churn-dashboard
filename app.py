"""
Flask backend for the Customer Segmentation & Retention dashboard
====================================================================
Wraps the four existing analysis scripts (churn_segmentation_project.py,
netflix_churn_segmentation.py, retail_segmentation.py, uplift_retention_model.py)
behind a small API, and serves the frontend (static/index.html) that talks to
it. This turns the project from "run scripts, then bake results into a static
HTML file" into an actual frontend/backend app: click a button in the browser,
the backend re-runs the analysis, the frontend re-renders with fresh data.

Endpoints:
  GET  /                    -> serves static/index.html
  GET  /api/dashboard-data  -> returns the last computed results as JSON
                                (404 with a message if nothing has run yet)
  POST /api/run             -> re-runs Telco + Retail + Uplift (always), and
                                Netflix if a CSV file was uploaded with the
                                request. Returns the fresh results as JSON.

Requires: flask, plus everything the analysis scripts need
          (pandas, numpy, scikit-learn, openpyxl)
Install:  pip install flask pandas numpy scikit-learn openpyxl
Run:      python app.py
Then open: http://127.0.0.1:5000
"""

import json
import os
import traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

import churn_segmentation_project as telco
import netflix_churn_segmentation as netflix
import retail_segmentation as retail
import uplift_retention_model as uplift

app = Flask(__name__, static_folder="static", static_url_path="")

DASHBOARD_DATA_PATH = "dashboard_data.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def safe_run(label: str, fn, *args):
    print(f"Running: {label}")
    try:
        result = fn(*args)
        print(f"[OK] {label}")
        return result, None
    except Exception as e:
        print(f"[FAILED] {label}: {e}")
        traceback.print_exc()
        return None, str(e)


def cross_domain_comparison(domain_summaries: dict) -> dict:
    churn_rate_by_domain, auc_by_domain, top_driver_by_domain = {}, {}, {}
    for summary in domain_summaries.values():
        if not summary:
            continue
        churn_rate_by_domain[summary["domain"]] = summary["churn_rate"]
        auc_by_domain[summary["domain"]] = summary["model_auc"]
        if summary["top_drivers"]:
            top_driver_by_domain[summary["domain"]] = summary["top_drivers"][0]["feature"]
    return {
        "churn_rate_by_domain": churn_rate_by_domain,
        "auc_by_domain": auc_by_domain,
        "top_driver_by_domain": top_driver_by_domain,
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/dashboard-data", methods=["GET"])
def get_dashboard_data():
    if not os.path.exists(DASHBOARD_DATA_PATH):
        return jsonify({"error": "No analysis has been run yet. Click 'Run Analysis'."}), 404
    with open(DASHBOARD_DATA_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/run", methods=["POST"])
def run_analysis():
    netflix_path = None
    uploaded = request.files.get("netflix_csv")
    if uploaded and uploaded.filename:
        netflix_path = os.path.join(UPLOAD_DIR, uploaded.filename)
        uploaded.save(netflix_path)

    telco_summary, telco_err = safe_run("Telco", telco.run)
    retail_summary, retail_err = safe_run("Retail", retail.run)
    uplift_summary, uplift_err = safe_run("Uplift", uplift.run)

    netflix_summary, netflix_err = None, "skipped (no CSV uploaded)"
    if netflix_path:
        netflix_summary, netflix_err = safe_run("Netflix", netflix.run, netflix_path)

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

    return jsonify(data)


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(debug=True, port=5000)
