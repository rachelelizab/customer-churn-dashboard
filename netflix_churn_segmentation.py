"""
Netflix-Style Customer Segmentation & Churn/Retention Analysis
================================================================
IMPORTANT NOTE ON DATA: Netflix does not publish real subscriber data (it's
private, obviously). What's used across every public "Netflix churn" project
-- on Kaggle, GitHub, YouTube tutorials, etc. -- is a SYNTHETIC dataset built
to mirror real subscription-streaming behavior. That's what this script
targets:

  Dataset: "Netflix Customer Churn Dataset" (Kaggle, ~5,000 synthetic users)
  https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset

  Typical columns:
    customer_id, age, gender, subscription_type (Basic/Standard/Premium),
    watch_hours, last_login_days, region, device, monthly_fee, churn,
    payment_method, number_of_profiles, avg_watch_time_per_day, favorite_genre

To get the file:
  1. Kaggle CLI (needs a free Kaggle account + API token at ~/.kaggle/kaggle.json):
       kaggle datasets download -d abdulwadood11220/netflix-customer-churn-dataset -p . --unzip
  2. Or download manually from the Kaggle page above and drop the CSV next to
     this script.

If your actual column names differ slightly, adjust the COLUMN NAMES section
right below -- everything downstream references these variables so one edit
fixes the whole script.

Pipeline (same structure as any subscription-churn project, applied here):
  1. Load + clean
  2. RFM-style segmentation: Recency = last_login_days, Frequency = watch_hours,
     Monetary = monthly_fee
  3. Churn prediction model (classification)
  4. Customer Lifetime Value estimate
  5. Priority matrix: retain / reward-with-early-access / write-off
  6. Plain-English interpretation

Requires: pandas, numpy, scikit-learn
Run:      python netflix_churn_segmentation.py netflix_customer_churn.csv
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

DOMAIN_NAME = "Streaming (Netflix-style, synthetic)"

# ---------------------------------------------------------------------------
# COLUMN NAMES -- edit these if your downloaded CSV uses different headers
# ---------------------------------------------------------------------------
COL_CHURN = "churned"                # 1/0 or Yes/No
COL_RECENCY = "last_login_days"      # days since last login (higher = more at risk)
COL_FREQUENCY = "watch_hours"        # total or avg watch hours (higher = more engaged)
COL_MONETARY = "monthly_fee"         # recurring subscription price
COL_PLAN = "subscription_type"       # Basic / Standard / Premium
CATEGORICAL_FEATURES = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
NUMERIC_FEATURES = ["age", COL_RECENCY, COL_FREQUENCY, COL_MONETARY, "number_of_profiles"]


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df[COL_CHURN].dtype == object:
        df[COL_CHURN] = df[COL_CHURN].str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(df[COL_CHURN])
    df[COL_CHURN] = df[COL_CHURN].astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. RFM-style segmentation
#    Recency = last_login_days (inverted: fewer days = better)
#    Frequency = watch_hours (engagement volume)
#    Monetary = monthly_fee (recurring value)
# ---------------------------------------------------------------------------
def segment_customers(df: pd.DataFrame, n_clusters: int = 4):
    features = df[[COL_RECENCY, COL_FREQUENCY, COL_MONETARY]].copy()
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Segment"] = km.fit_predict(scaled)

    profile = df.groupby("Segment")[[COL_RECENCY, COL_FREQUENCY, COL_MONETARY]].mean()
    profile["ChurnRate"] = df.groupby("Segment")[COL_CHURN].mean()
    profile["Count"] = df.groupby("Segment").size()
    return df, profile.sort_values(COL_MONETARY, ascending=False)


# ---------------------------------------------------------------------------
# 3. Churn prediction model
# ---------------------------------------------------------------------------
def train_churn_model(df: pd.DataFrame):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    y = df[COL_CHURN]

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

    df = df.loc[X.index].copy()
    df["ChurnProb"] = rf.predict_proba(X)[:, 1]
    auc_scores = {"logistic_regression": round(float(logreg_auc), 4), "random_forest": round(float(rf_auc), 4)}
    return df, importances, auc_scores


# ---------------------------------------------------------------------------
# 4. Customer Lifetime Value
#    CLV = monthly_fee * expected remaining months, where expected remaining
#    months is derived from the model's churn probability (capped at a
#    reasonable horizon so near-zero churn risk doesn't blow up the estimate).
# ---------------------------------------------------------------------------
def estimate_clv(df: pd.DataFrame, horizon_months: int = 24) -> pd.DataFrame:
    monthly_hazard = (df["ChurnProb"] / 12).clip(lower=1 / (horizon_months * 4))
    expected_remaining_months = (1 / monthly_hazard).clip(upper=horizon_months)
    df["CLV"] = df[COL_MONETARY] * expected_remaining_months
    return df


# ---------------------------------------------------------------------------
# 5. Priority matrix: who gets retention spend, who gets early access,
#    who's not worth saving.
# ---------------------------------------------------------------------------
def build_priority_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # rank(method="first") avoids tie-collapse errors in qcut (see churn_segmentation_project.py)
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

    print(f"- Strongest churn driver: {importances.index[0]}")

    print("\n- Customer counts by recommended action:")
    print(df["RecommendedAction"].value_counts())

    prioritize = df[df["RecommendedAction"].str.startswith("PRIORITIZE")]
    print(f"\n- {len(prioritize)} customers are high-value AND at risk -- retention "
          f"budget should go here first.")

    reward = df[df["RecommendedAction"].str.startswith("REWARD")]
    print(f"- {len(reward)} customers are high-value and already staying -- your early-access "
          f"/ beta-feature candidates. Retention spend on this group is largely wasted.")

    writeoff = df[df["RecommendedAction"].str.startswith("DO NOT")]
    print(f"- {len(writeoff)} customers are low-value and high-risk -- the model expects them "
          f"to churn regardless; not worth discounting to keep.")

    if COL_PLAN in df.columns:
        print("\n- Churn rate by plan:")
        print(df.groupby(COL_PLAN)[COL_CHURN].mean().sort_values(ascending=False))


# ---------------------------------------------------------------------------
# 6. Summary export -- for the dashboard / cross-domain comparison
# ---------------------------------------------------------------------------
def build_summary(scored: pd.DataFrame, segment_profile: pd.DataFrame, importances: pd.Series, auc_scores: dict) -> dict:
    segments = []
    for seg_id, row in segment_profile.iterrows():
        segments.append({
            "segment": int(seg_id),
            "count": int(row.get("Count", (scored["Segment"] == seg_id).sum())),
            "churn_rate": round(float(row["ChurnRate"]), 4),
            "avg_recency_days": round(float(row[COL_RECENCY]), 2),
            "avg_frequency": round(float(row[COL_FREQUENCY]), 2),
            "avg_monetary": round(float(row[COL_MONETARY]), 2),
        })

    plan_churn = {}
    if COL_PLAN in scored.columns:
        plan_churn = {str(k): round(float(v), 4) for k, v in scored.groupby(COL_PLAN)[COL_CHURN].mean().items()}

    return {
        "domain": DOMAIN_NAME,
        "n_customers": int(len(scored)),
        "churn_rate": round(float(scored[COL_CHURN].mean()), 4),
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
        "churn_by_plan": plan_churn,
    }


def run(csv_path: str = "netflix_customer_churn.csv") -> dict:
    """Entry point used by run_all.py. Returns the JSON-serializable summary dict."""
    data = load_data(csv_path)
    data, segment_profile = segment_customers(data)
    scored, feature_importances, auc_scores = train_churn_model(data)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    scored.to_csv("netflix_scored_customers.csv", index=False)
    return build_summary(scored, segment_profile, feature_importances, auc_scores)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "netflix_customer_churn.csv"

    data = load_data(csv_path)
    data, segment_profile = segment_customers(data)
    print("Segment profile (sorted by monetary value):")
    print(segment_profile)

    scored, feature_importances, auc_scores = train_churn_model(data)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    print_interpretation(scored, feature_importances)

    scored.to_csv("netflix_scored_customers.csv", index=False)
    print("\nFull scored customer table written to netflix_scored_customers.csv")

    summary = build_summary(scored, segment_profile, feature_importances, auc_scores)
    with open("netflix_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary written to netflix_summary.json")
