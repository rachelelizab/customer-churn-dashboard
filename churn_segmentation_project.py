"""
Customer Segmentation & Churn/Retention Analysis
==================================================
Dataset: IBM Telco Customer Churn (real, public, subscription-business data)
Source:  https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv
         (7,043 customers, one row per customer, real telecom subscription churn labels)

Why this dataset: it's a real subscription business (monthly billing, contracts,
add-on services) — the closest public analogue to Netflix/Spotify/Slack-style churn,
with an actual ground-truth "Churn" label so the model can be validated, not just
built. Swap the CSV_URL below for the UCI "Online Retail" dataset if you want the
ecommerce/RFM version instead (transactional, no churn label, so churn = inactivity
past a threshold).

Pipeline:
  1. Load + clean data
  2. Behavioral segmentation (RFM-style, adapted for subscription data)
  3. Churn prediction model (classification)
  4. Customer Lifetime Value estimate
  5. Priority matrix: who gets retention spend, who gets early access, who's a lost cause
  6. Plain-English interpretation printed at the end — this is the part that matters most

Requires: pandas, numpy, scikit-learn  (pip install pandas numpy scikit-learn)
Run:      python churn_segmentation_project.py
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DOMAIN_NAME = "Telco (Subscription/SaaS proxy)"


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_data(path_or_url: str = CSV_URL) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)

    # TotalCharges has blank strings for brand-new customers (tenure=0) instead of 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Engagement breadth: count of add-on services actively used (proxy for
    # "how embedded/sticky this customer is" — the SaaS/streaming equivalent
    # of purchase category breadth)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)

    return df


# ---------------------------------------------------------------------------
# 2. Behavioral segmentation (RFM adapted for a subscription business)
#    Subscription data has no discrete "purchases" to compute recency/frequency
#    from, so the adaptation is:
#      Recency proxy   -> tenure (months since acquisition; low tenure = newer/riskier)
#      Frequency proxy -> ServiceCount (breadth of engagement / stickiness)
#      Monetary        -> MonthlyCharges (recurring value) and TotalCharges (LTV to date)
# ---------------------------------------------------------------------------
def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    features = df[["tenure", "ServiceCount", "MonthlyCharges", "TotalCharges"]]
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Segment"] = km.fit_predict(scaled)

    # Label segments by their average tenure/monetary profile so output is
    # readable instead of just "Segment 0/1/2/3"
    profile = df.groupby("Segment")[["tenure", "ServiceCount", "MonthlyCharges", "TotalCharges"]].mean()
    profile["ChurnRate"] = df.groupby("Segment")["Churn"].mean()
    return df, profile.sort_values("TotalCharges", ascending=False)


# ---------------------------------------------------------------------------
# 3. Churn prediction model
# ---------------------------------------------------------------------------
def train_churn_model(df: pd.DataFrame):
    categorical = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "ServiceCount"]

    X = pd.get_dummies(df[categorical + numeric], columns=categorical, drop_first=True)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\n--- Logistic Regression ---")
    print("ROC-AUC:", round(logreg_auc, 3))
    print(classification_report(y_test, logreg.predict(X_test)))

    print("\n--- Random Forest ---")
    print("ROC-AUC:", round(rf_auc, 3))
    print(classification_report(y_test, rf.predict(X_test)))

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 churn drivers (Random Forest feature importance):")
    print(importances.head(10))

    # Score every customer with the RF model (used downstream for the priority matrix)
    df = df.loc[X.index].copy()
    df["ChurnProb"] = rf.predict_proba(X)[:, 1]
    auc_scores = {"logistic_regression": round(float(logreg_auc), 4), "random_forest": round(float(rf_auc), 4)}
    return df, importances, auc_scores


# ---------------------------------------------------------------------------
# 4. Customer Lifetime Value (simple, transparent estimate)
#    CLV = MonthlyCharges * expected remaining months
#    expected remaining months ~ 1 / monthly churn hazard, capped at a
#    reasonable horizon (24 months) so a near-zero churn probability doesn't
#    blow up the estimate.
# ---------------------------------------------------------------------------
def estimate_clv(df: pd.DataFrame, horizon_months: int = 24) -> pd.DataFrame:
    monthly_hazard = (df["ChurnProb"] / 12).clip(lower=1 / (horizon_months * 4))
    expected_remaining_months = (1 / monthly_hazard).clip(upper=horizon_months)
    df["CLV"] = df["MonthlyCharges"] * expected_remaining_months
    return df


# ---------------------------------------------------------------------------
# 5. Priority matrix: CLV tier x churn-risk tier -> recommended action
#    This is the "interpretation" step — the part that turns a model into a
#    business decision instead of just a number.
# ---------------------------------------------------------------------------
def build_priority_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # rank(method="first") breaks ties before qcut so repeated CLV values
    # (common since CLV is capped at a horizon) can't collapse bin edges and
    # raise a "Bin labels must be one fewer than the number of bin edges" error
    df["CLVTier"] = pd.qcut(df["CLV"].rank(method="first"), 3, labels=["Low CLV", "Mid CLV", "High CLV"])
    df["RiskTier"] = pd.cut(
        df["ChurnProb"], bins=[0, 0.3, 0.6, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    def action(row):
        if row["CLVTier"] == "High CLV" and row["RiskTier"] in ("Medium Risk", "High Risk"):
            return "PRIORITIZE: proactive retention offer"
        if row["CLVTier"] == "High CLV" and row["RiskTier"] == "Low Risk":
            return "REWARD: early access / loyalty perks (already staying)"
        if row["CLVTier"] == "Low CLV" and row["RiskTier"] == "High Risk":
            return "DO NOT INVEST: low value, unlikely to be saved profitably"
        return "MONITOR: no action needed yet"

    df["RecommendedAction"] = df.apply(action, axis=1)
    return df


def print_interpretation(df: pd.DataFrame, importances: pd.Series):
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    top_driver = importances.index[0]
    print(f"- Strongest churn driver in this dataset: {top_driver}")

    action_counts = df["RecommendedAction"].value_counts()
    print("\n- Customer counts by recommended action:")
    print(action_counts)

    prioritize = df[df["RecommendedAction"].str.startswith("PRIORITIZE")]
    print(f"\n- {len(prioritize)} customers are high-value AND at risk — this is where "
          f"retention budget should go first, not spread evenly across everyone at risk.")

    reward = df[df["RecommendedAction"].str.startswith("REWARD")]
    print(f"- {len(reward)} customers are high-value and already staying — good early-access "
          f"candidates, but retention spend here is largely wasted since they weren't leaving anyway.")

    writeoff = df[df["RecommendedAction"].str.startswith("DO NOT")]
    print(f"- {len(writeoff)} customers are low-value and high-risk — model says they'll likely "
          f"churn regardless of intervention; not worth discounting to keep.")


# ---------------------------------------------------------------------------
# 6. Summary export -- a single JSON-serializable dict the dashboard (and
#    cross-domain comparison) can consume without needing the raw CSV.
# ---------------------------------------------------------------------------
def build_summary(scored: pd.DataFrame, segment_profile: pd.DataFrame, importances: pd.Series, auc_scores: dict) -> dict:
    segments = []
    for seg_id, row in segment_profile.iterrows():
        segments.append({
            "segment": int(seg_id),
            "count": int(row.get("Count", (scored["Segment"] == seg_id).sum())),
            "churn_rate": round(float(row["ChurnRate"]), 4),
            "avg_tenure": round(float(row["tenure"]), 2),
            "avg_service_count": round(float(row["ServiceCount"]), 2),
            "avg_monthly_charges": round(float(row["MonthlyCharges"]), 2),
            "avg_total_charges": round(float(row["TotalCharges"]), 2),
        })

    return {
        "domain": DOMAIN_NAME,
        "n_customers": int(len(scored)),
        "churn_rate": round(float(scored["Churn"].mean()), 4),
        "model_auc": auc_scores,
        "top_drivers": [
            {"feature": feat, "importance": round(float(val), 4)}
            for feat, val in importances.head(8).items()
        ],
        "segments": segments,
        "priority_matrix": {
            k: int(v) for k, v in scored["RecommendedAction"].value_counts().items()
        },
        "clv_summary": {
            "mean": round(float(scored["CLV"].mean()), 2),
            "median": round(float(scored["CLV"].median()), 2),
            "total": round(float(scored["CLV"].sum()), 2),
        },
    }


def run(path_or_url: str = CSV_URL) -> dict:
    """Entry point used by run_all.py. Returns the JSON-serializable summary dict."""
    data = load_data(path_or_url)
    data, segment_profile = segment_customers(data)
    scored, feature_importances, auc_scores = train_churn_model(data)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    scored.to_csv("scored_customers.csv", index=False)
    return build_summary(scored, segment_profile, feature_importances, auc_scores)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_URL

    data = load_data(path)
    data, segment_profile = segment_customers(data)
    print("Segment profile (sorted by value):")
    print(segment_profile)

    scored, feature_importances, auc_scores = train_churn_model(data)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    print_interpretation(scored, feature_importances)

    scored.to_csv("scored_customers.csv", index=False)
    print("\nFull scored customer table written to scored_customers.csv")

    summary = build_summary(scored, segment_profile, feature_importances, auc_scores)
    with open("telco_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary written to telco_summary.json")
