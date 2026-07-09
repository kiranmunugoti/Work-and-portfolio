"""
Healthcare Claims Fraud Detection
--------------------------------------------------------
Flags high-risk insurance claims using XGBoost on a realistically
imbalanced claims dataset (fraud is rare), with SQL-style feature
engineering (claim-vs-provider-baseline ratios, billing velocity) that
mirrors how the original pipeline pulled and shaped claims data before
modeling.

NOTE: Demonstration on synthetic claims data with realistic class
imbalance (~3% fraud rate). In production this supported claims fraud
review for a healthcare payer, contributing to a measurable improvement
in fraud catch rate and reduction in manual review cost. Company/member
data is protected and proprietary, so this repo reproduces the modeling
and evaluation approach on generated data only.

Author: Sai Kiran Munugoti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

RNG = np.random.default_rng(23)
N = 20000
FRAUD_RATE = 0.03


def generate_claims(n=N, fraud_rate=FRAUD_RATE):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_block(n_rows, fraud):
        billed_amount = RNG.gamma(3, 220 if not fraud else 340, n_rows)
        provider_baseline_ratio = RNG.normal(1.0 if not fraud else 1.9, 0.25, n_rows).clip(0.3, None)
        claims_per_member_30d = RNG.poisson(1.3 if not fraud else 3.1, n_rows)
        days_since_service_to_claim = RNG.normal(9 if not fraud else 25, 5, n_rows).clip(0, None)
        provider_new_flag = RNG.binomial(1, 0.05 if not fraud else 0.28, n_rows)
        weekend_service_flag = RNG.binomial(1, 0.1 if not fraud else 0.22, n_rows)
        diagnosis_procedure_mismatch = RNG.binomial(1, 0.02 if not fraud else 0.35, n_rows)
        return pd.DataFrame({
            "billed_amount": billed_amount,
            "provider_baseline_ratio": provider_baseline_ratio,
            "claims_per_member_30d": claims_per_member_30d,
            "days_since_service_to_claim": days_since_service_to_claim,
            "provider_new_flag": provider_new_flag,
            "weekend_service_flag": weekend_service_flag,
            "diagnosis_procedure_mismatch": diagnosis_procedure_mismatch,
            "is_fraud": fraud,
        })

    df = pd.concat([make_block(n_legit, 0), make_block(n_fraud, 1)], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_fraud_model(df):
    X = df.drop(columns="is_fraud")
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # Business-tuned threshold: prioritize recall on fraud for manual review queue
    threshold = 0.35
    preds = (proba >= threshold).astype(int)

    print(f"Decision threshold: {threshold}\n")
    print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix [rows=actual, cols=predicted]:")
    print(pd.DataFrame(cm, index=["Legit", "Fraud"], columns=["Pred Legit", "Pred Fraud"]).to_string())

    ap = average_precision_score(y_test, proba)
    print(f"\nAverage Precision (PR-AUC): {ap:.3f}")

    return model, X_test, y_test, proba, X.columns, ap


def plot_pr_and_importance(y_test, proba, model, feature_names, ap, out_path="fraud_model_chart.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    precision, recall, _ = precision_recall_curve(y_test, proba)
    axes[0].plot(recall, precision, color="#b3543e", linewidth=2.2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall Curve (AP={ap:.3f})", fontweight="bold")
    axes[0].set_ylim(0, 1.02)

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    axes[1].barh(importances.index, importances.values, color="#a3b899")
    axes[1].set_title("Feature Importance", fontweight="bold")
    axes[1].set_xlabel("Importance")

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    print(f"\nSaved chart to {out_path}")


def main():
    df = generate_claims()
    print(f"Dataset: {len(df):,} claims, fraud rate = {df['is_fraud'].mean():.2%}\n")

    model, X_test, y_test, proba, feature_names, ap = train_fraud_model(df)
    plot_pr_and_importance(y_test, proba, model, feature_names, ap)

    out = X_test.copy()
    out["actual_fraud"] = y_test.values
    out["fraud_probability"] = proba.round(4)
    out.to_csv("fraud_scored_claims.csv", index=False)


if __name__ == "__main__":
    main()
