"""
Cross-Domain Aggregator
=========================
Runs all four analysis modules and combines them into a single
dashboard_data.json that the frontend (dashboard.html, via build_dashboard.py)
reads. This is what actually closes "gap 2" from the project's research
framing: it puts the same RFM/churn/CLV pipeline's results from three
structurally different businesses (subscription/SaaS, streaming, marketplace)
plus the causal uplift analysis side by side for direct comparison, instead
of writing four disconnected reports.

Usage:
  python run_all.py [path_to_netflix_csv]

  The Telco, Retail, and Uplift datasets download automatically (all three
  are public URLs with no auth required). The Netflix dataset is the one
  exception -- it's on Kaggle and needs a manual download (see
  netflix_churn_segmentation.py for instructions) -- so pass its path as an
  argument if you have it. If omitted, the Netflix domain is skipped and the
  dashboard will show the other three plus uplift.

Requires: pandas, numpy, scikit-learn, openpyxl
"""

import json
import sys
import traceback
from datetime import datetime, timezone

import churn_segmentation_project as telco
import netflix_churn_segmentation as netflix
import retail_segmentation as retail
import uplift_retention_model as uplift


def safe_run(label: str, fn, *args):
    print(f"\n{'='*70}\nRUNNING: {label}\n{'='*70}")
    try:
        result = fn(*args)
        print(f"[OK] {label} finished.")
        return result, None
    except Exception as e:
        print(f"[FAILED] {label}: {e}")
        traceback.print_exc()
        return None, str(e)


def cross_domain_comparison(domain_summaries: dict) -> dict:
    """The actual gap-2 analysis: do churn drivers and rates look the same
    or structurally different across domains?"""
    churn_rate_by_domain = {}
    auc_by_domain = {}
    top_driver_by_domain = {}

    for key, summary in domain_summaries.items():
        if summary is None:
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


def main():
    netflix_csv = sys.argv[1] if len(sys.argv) > 1 else None

    telco_summary, telco_err = safe_run("Telco (Subscription/SaaS proxy)", telco.run)
    retail_summary, retail_err = safe_run("Retail (Ecommerce/Marketplace)", retail.run)
    uplift_summary, uplift_err = safe_run("Uplift/Persuadability (Email experiment)", uplift.run)

    netflix_summary, netflix_err = None, "skipped (no CSV path provided)"
    if netflix_csv:
        netflix_summary, netflix_err = safe_run("Netflix (Streaming, synthetic)", netflix.run, netflix_csv)

    domain_summaries = {
        "telco": telco_summary,
        "netflix": netflix_summary,
        "retail": retail_summary,
    }

    comparison = cross_domain_comparison(domain_summaries)

    dashboard_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domains": domain_summaries,
        "domain_errors": {
            "telco": telco_err,
            "netflix": netflix_err,
            "retail": retail_err,
        },
        "uplift": uplift_summary,
        "uplift_error": uplift_err,
        "cross_domain_comparison": comparison,
    }

    with open("dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\n{'='*70}\nDONE. Wrote dashboard_data.json\n{'='*70}")
    print("Next: python build_dashboard.py   (bakes this data into dashboard.html)")


if __name__ == "__main__":
    main()
