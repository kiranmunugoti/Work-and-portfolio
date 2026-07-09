"""
Churn Prediction & Risk Segmentation
--------------------------------------------------------
Predicts customer churn using Logistic Regression and XGBoost on noisy,
multi-source behavioral/transactional data, then segments the customer
base with K-Means on RFM-style features to prioritize retention outreach
by segment rather than treating all "at risk" customers identically.

NOTE: Demonstration on synthetic data with realistic class imbalance and
noisy/missing features, mirroring the structure of the original
multi-source (CRM + billing + support-ticket) dataset. Company data is
proprietary, so this repo reproduces the modeling approach on generated
data.

Author: Sai Kiran Munugoti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from xgboost import XGBClassifier

RNG = np.random.default_rng(11)
N = 3000


def generate_customers(n=N):
    tenure_months = RNG.integers(1, 72, n)
    monthly_spend = RNG.gamma(4, 25, n)
    support_tickets = RNG.poisson(1.2, n)
    late_payments = RNG.poisson(0.6, n)
    engagement_score = RNG.normal(0.55, 0.2, n).clip(0, 1)
    discount_pct = RNG.choice([0, 5, 10, 20], size=n, p=[0.55, 0.2, 0.15, 0.1])

    # Latent churn propensity drives the label, with noise so it's not trivially separable
    logit = (
        -1.8
        - 0.035 * tenure_months
        + 0.012 * support_tickets * support_tickets
        + 0.55 * late_payments
        - 1.6 * engagement_score
        - 0.01 * discount_pct
        + RNG.normal(0, 0.9, n)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = RNG.binomial(1, churn_prob)

    # Inject missingness to mimic multi-source data reality
    df = pd.DataFrame({
        "tenure_months": tenure_months, "monthly_spend": monthly_spend,
        "support_tickets": support_tickets, "late_payments": late_payments,
        "engagement_score": engagement_score, "discount_pct": discount_pct,
        "churned": churned,
    })
    missing_idx = RNG.choice(df.index, size=int(0.03 * n), replace=False)
    df.loc[missing_idx, "engagement_score"] = np.nan
    df["engagement_score"] = df["engagement_score"].fillna(df["engagement_score"].median())
    return df


def train_models(df):
    X = df.drop(columns="churned")
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=500, class_weight="balanced")
    logreg.fit(X_train_s, y_train)
    logreg_proba = logreg.predict_proba(X_test_s)[:, 1]

    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    results = {}
    for name, proba, preds_input in [
        ("Logistic Regression", logreg_proba, logreg.predict(X_test_s)),
        ("XGBoost", xgb_proba, xgb.predict(X_test)),
    ]:
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, preds_input)
        results[name] = {"auc": auc, "acc": acc, "proba": proba}
        print(f"{name:>22s}  AUC={auc:.3f}   Accuracy={acc:.3f}")

    return results, y_test, xgb, X.columns


def rfm_segmentation(df, n_clusters=4):
    """K-Means segmentation on Recency/Frequency/Monetary-style features."""
    rfm = df[["tenure_months", "support_tickets", "monthly_spend"]].copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["segment"] = km.fit_predict(rfm_scaled)

    summary = df.groupby("segment").agg(
        avg_tenure=("tenure_months", "mean"),
        avg_spend=("monthly_spend", "mean"),
        churn_rate=("churned", "mean"),
        size=("churned", "count"),
    ).round(2).sort_values("churn_rate", ascending=False)
    print("\nSegment summary (sorted by churn rate):")
    print(summary.to_string())
    return df, summary


def plot_roc_and_segments(results, y_test, seg_summary, out_path="churn_model_chart.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        color = "#b3543e" if name == "XGBoost" else "#a3b899"
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Churn Models", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    seg_summary = seg_summary.sort_values("churn_rate")
    axes[1].barh(
        [f"Segment {i}" for i in seg_summary.index],
        seg_summary["churn_rate"] * 100,
        color="#d9a441",
    )
    axes[1].set_xlabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate by RFM Segment", fontweight="bold")

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    print(f"\nSaved chart to {out_path}")


def main():
    df = generate_customers()
    results, y_test, xgb_model, feature_names = train_models(df)
    df, seg_summary = rfm_segmentation(df)
    plot_roc_and_segments(results, y_test, seg_summary)

    importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
    print("\nXGBoost feature importances:")
    print(importances.sort_values(ascending=False).round(3).to_string())

    df.to_csv("churn_segments_output.csv", index=False)


if __name__ == "__main__":
    main()
