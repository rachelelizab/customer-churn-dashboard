"""
Ecommerce/Marketplace Customer Segmentation & Retention Analysis
==================================================================
Dataset: UCI "Online Retail" (real, public, transactional data)
Source:  https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
         (~541,909 invoice line items, real UK-based online gift retailer,
         Dec 2010 - Dec 2011, ~4,300 unique customers)

Why this dataset: it's the closest public analogue to Amazon/Shopify-style
marketplace behavior -- real invoices, real returns, real repeat purchasing.
Unlike Telco or the Netflix dataset, there is NO explicit "churn" label here,
because marketplace customers don't cancel a subscription -- they just stop
buying. So churn has to be *defined*, not read off a column: a customer is
treated as churned if they haven't purchased in more than CHURN_THRESHOLD_DAYS
since their last invoice (measured from a fixed "snapshot" date at the end of
the observation window).

This is true classic RFM (not an adaptation like the subscription scripts):
  Recency   = days since last invoice
  Frequency = number of distinct invoices (orders)
  Monetary  = total amount spent

Pipeline:
  1. Load + clean (drop missing CustomerID, remove returns/cancellations)
  2. Build customer-level RFM table
  3. Segment customers (KMeans on Recency/Frequency/Monetary)
  4. Predict inactivity-churn from behavior EXCLUDING recency itself (recency
     defines the label, so including it as a model feature would leak the
     answer -- a subtlety most tutorial versions of this project get wrong)
  5. Customer Lifetime Value estimate
  6. Priority matrix + interpretation

Requires: pandas, numpy, scikit-learn, openpyxl (for reading the .xlsx source)
Run:      python retail_segmentation.py
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

DOMAIN_NAME = "Ecommerce/Marketplace (Amazon/Shopify-style, real transactions)"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
CHURN_THRESHOLD_DAYS = 90  # no purchase in 90+ days => treated as churned/inactive


# ---------------------------------------------------------------------------
# 1. Load + clean raw invoice-line data
# ---------------------------------------------------------------------------
def load_raw(path_or_url: str = DATA_URL) -> pd.DataFrame:
    if str(path_or_url).lower().endswith(".csv"):
        df = pd.read_csv(path_or_url, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(path_or_url)

    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancellations/returns (InvoiceNo starting with 'C') and bad rows
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


# ---------------------------------------------------------------------------
# 2. Build the customer-level RFM table (true RFM, not an adaptation)
# ---------------------------------------------------------------------------
def build_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customers = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        ProductDiversity=("StockCode", "nunique"),
        Country=("Country", lambda x: x.mode().iloc[0]),
    ).reset_index()

    customers["TenureDays"] = (snapshot_date - customers["FirstPurchase"]).dt.days.clip(lower=1)
    customers["AvgOrderValue"] = customers["Monetary"] / customers["Frequency"]
    customers["Churn"] = (customers["Recency"] > CHURN_THRESHOLD_DAYS).astype(int)

    return customers


# ---------------------------------------------------------------------------
# 3. Segmentation (classic RFM clustering)
# ---------------------------------------------------------------------------
def segment_customers(df: pd.DataFrame, n_clusters: int = 4):
    features = df[["Recency", "Frequency", "Monetary"]]
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Segment"] = km.fit_predict(scaled)

    profile = df.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
    profile["ChurnRate"] = df.groupby("Segment")["Churn"].mean()
    profile["Count"] = df.groupby("Segment").size()
    return df, profile.sort_values("Monetary", ascending=False)


# ---------------------------------------------------------------------------
# 4. Churn model -- deliberately EXCLUDES Recency as a feature, since Recency
#    is what defines the Churn label. Including it would make the "model"
#    just re-derive its own label (trivial, near-perfect, useless AUC). This
#    is a common mistake in tutorial versions of this exact project.
# ---------------------------------------------------------------------------
def train_churn_model(df: pd.DataFrame):
    categorical = ["Country"]
    numeric = ["Frequency", "Monetary", "AvgOrderValue", "ProductDiversity", "TenureDays"]

    # Collapse rare countries so get_dummies doesn't explode into 40 columns
    top_countries = df["Country"].value_counts().nlargest(8).index
    df["Country"] = df["Country"].where(df["Country"].isin(top_countries), "Other")

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

    print("\n--- Logistic Regression (Recency excluded from features) ---")
    print("ROC-AUC:", round(logreg_auc, 3))
    print(classification_report(y_test, logreg.predict(X_test)))

    print("\n--- Random Forest (Recency excluded from features) ---")
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
# 5. Customer Lifetime Value
#    monthly_value = Monetary / (TenureDays/30), CLV = monthly_value * expected
#    remaining months, remaining months derived from churn hazard (same
#    approach as the subscription scripts, for consistency across domains).
# ---------------------------------------------------------------------------
def estimate_clv(df: pd.DataFrame, horizon_months: int = 24) -> pd.DataFrame:
    tenure_months = (df["TenureDays"] / 30).clip(lower=1)
    monthly_value = df["Monetary"] / tenure_months

    monthly_hazard = (df["ChurnProb"] / 12).clip(lower=1 / (horizon_months * 4))
    expected_remaining_months = (1 / monthly_hazard).clip(upper=horizon_months)

    df["CLV"] = monthly_value * expected_remaining_months
    return df


# ---------------------------------------------------------------------------
# 6. Priority matrix
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

    print(f"- Strongest (non-recency) churn driver: {importances.index[0]}")
    print("\n- Customer counts by recommended action:")
    print(df["RecommendedAction"].value_counts())

    prioritize = df[df["RecommendedAction"].str.startswith("PRIORITIZE")]
    print(f"\n- {len(prioritize)} customers are high-value AND at risk -- retention "
          f"budget should go here first.")

    print(f"\n- Churn (90+ day inactivity) rate overall: {round(df['Churn'].mean() * 100, 1)}%")
    print("\n- Churn rate by country (top 5 by customer count):")
    top5 = df["Country"].value_counts().nlargest(5).index
    print(df[df["Country"].isin(top5)].groupby("Country")["Churn"].mean().sort_values(ascending=False))


# ---------------------------------------------------------------------------
# 7. Summary export
# ---------------------------------------------------------------------------
def build_summary(scored: pd.DataFrame, segment_profile: pd.DataFrame, importances: pd.Series, auc_scores: dict) -> dict:
    segments = []
    for seg_id, row in segment_profile.iterrows():
        segments.append({
            "segment": int(seg_id),
            "count": int(row.get("Count", (scored["Segment"] == seg_id).sum())),
            "churn_rate": round(float(row["ChurnRate"]), 4),
            "avg_recency_days": round(float(row["Recency"]), 2),
            "avg_frequency": round(float(row["Frequency"]), 2),
            "avg_monetary": round(float(row["Monetary"]), 2),
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
        "churn_definition": f"No purchase in {CHURN_THRESHOLD_DAYS}+ days (inferred, not a real label)",
    }


def run(path_or_url: str = DATA_URL) -> dict:
    """Entry point used by run_all.py. Returns the JSON-serializable summary dict."""
    raw = load_raw(path_or_url)
    customers = build_customer_table(raw)
    customers, segment_profile = segment_customers(customers)
    scored, feature_importances, auc_scores = train_churn_model(customers)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    scored.to_csv("retail_scored_customers.csv", index=False)
    return build_summary(scored, segment_profile, feature_importances, auc_scores)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_URL

    raw = load_raw(path)
    customers = build_customer_table(raw)
    customers, segment_profile = segment_customers(customers)
    print("Segment profile (sorted by monetary value):")
    print(segment_profile)

    scored, feature_importances, auc_scores = train_churn_model(customers)
    scored = estimate_clv(scored)
    scored = build_priority_matrix(scored)
    print_interpretation(scored, feature_importances)

    scored.to_csv("retail_scored_customers.csv", index=False)
    print("\nFull scored customer table written to retail_scored_customers.csv")

    summary = build_summary(scored, segment_profile, feature_importances, auc_scores)
    with open("retail_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary written to retail_summary.json")
