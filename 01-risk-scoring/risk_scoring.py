"""
Multi-Dimensional Risk Scoring — Hospitality Portfolio
--------------------------------------------------------
Scores properties in a multi-site hospitality portfolio on operational,
financial, and behavioral risk using a weighted ensemble of engineered
features, integrating signal that in production would come from PMS
(property management system), POS, CRM, and booking-platform data.

NOTE: This is a demonstration built on synthetic data. In the original
production version, features were sourced from live PMS/POS/CRM feeds
via Azure Data Factory pipelines and scored nightly, feeding a Power BI
dashboard used by portfolio ops leadership to prioritize site visits and
interventions. Company data cannot be shared publicly, so this repo
reproduces the method end-to-end on generated data with the same
statistical structure.

Author: Sai Kiran Munugoti
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N_PROPERTIES = 40


def generate_portfolio_data(n=N_PROPERTIES):
    """Simulate property-level operational/financial/behavioral signals."""
    df = pd.DataFrame({
        "property_id": [f"PROP-{i:03d}" for i in range(1, n + 1)],
        "occupancy_rate": RNG.normal(0.72, 0.12, n).clip(0.2, 1.0),
        "revpar_variance_pct": RNG.normal(0, 15, n),          # +/- vs budget
        "maintenance_tickets_30d": RNG.poisson(6, n),
        "staff_turnover_rate": RNG.normal(0.18, 0.08, n).clip(0.02, 0.6),
        "guest_complaint_rate": RNG.gamma(2, 0.8, n),          # per 100 stays
        "late_payment_incidents": RNG.poisson(1.5, n),
        "avg_review_score": RNG.normal(4.1, 0.4, n).clip(2.0, 5.0),
        "days_since_last_audit": RNG.integers(10, 400, n),
    })
    return df


def engineer_risk_features(df):
    """Transform raw signals into risk-oriented features (higher = riskier)."""
    feat = pd.DataFrame(index=df.index)
    feat["occupancy_risk"] = 1 - df["occupancy_rate"]
    feat["revpar_risk"] = df["revpar_variance_pct"].apply(lambda x: max(-x, 0)) / 30
    feat["maintenance_risk"] = df["maintenance_tickets_30d"] / df["maintenance_tickets_30d"].max()
    feat["turnover_risk"] = df["staff_turnover_rate"]
    feat["complaint_risk"] = df["guest_complaint_rate"] / df["guest_complaint_rate"].max()
    feat["payment_risk"] = df["late_payment_incidents"] / (df["late_payment_incidents"].max() + 1e-6)
    feat["review_risk"] = (5 - df["avg_review_score"]) / 3
    feat["audit_staleness_risk"] = df["days_since_last_audit"] / df["days_since_last_audit"].max()
    return feat


def compute_composite_risk_score(feat, weights=None):
    """
    Weighted composite score (0-100) blending business-defined weights with
    a RandomForest-learned importance re-weighting, so the score reflects
    domain priorities while still adapting to which signals actually move
    together in the data (a lightweight, explainable alternative to a
    black-box classifier for a stakeholder-facing score).
    """
    default_weights = {
        "occupancy_risk": 0.15, "revpar_risk": 0.20, "maintenance_risk": 0.10,
        "turnover_risk": 0.15, "complaint_risk": 0.15, "payment_risk": 0.10,
        "review_risk": 0.10, "audit_staleness_risk": 0.05,
    }
    weights = weights or default_weights

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(feat), columns=feat.columns, index=feat.index)

    # Pseudo-label: a noisy blend used only to let the RF learn feature
    # interactions; the RF's feature_importances_ nudge the manual weights.
    pseudo_target = (scaled * pd.Series(weights)).sum(axis=1) + RNG.normal(0, 0.03, len(scaled))
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
    rf.fit(scaled, pseudo_target)
    learned_weights = pd.Series(rf.feature_importances_, index=feat.columns)
    blended_weights = 0.6 * pd.Series(weights) + 0.4 * (learned_weights / learned_weights.sum())
    blended_weights /= blended_weights.sum()

    score = (scaled * blended_weights).sum(axis=1) * 100
    return score.round(1), blended_weights


def assign_risk_tier(score):
    return pd.cut(score, bins=[-1, 33, 66, 101], labels=["Low", "Medium", "High"])


def plot_risk_distribution(df, out_path="risk_scores_chart.png"):
    df_sorted = df.sort_values("risk_score", ascending=False).head(15)
    colors = df_sorted["risk_tier"].map({"Low": "#8fae8b", "Medium": "#d9a441", "High": "#b3543e"})

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_sorted["property_id"], df_sorted["risk_score"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Risk Score (0–100)")
    ax.set_title("Top 15 Properties by Portfolio Risk Score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 100)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    print(f"Saved chart to {out_path}")


def main():
    df = generate_portfolio_data()
    feat = engineer_risk_features(df)
    df["risk_score"], weights = compute_composite_risk_score(feat)
    df["risk_tier"] = assign_risk_tier(df["risk_score"])

    print("\nBlended feature weights (business priors + learned adjustment):")
    print(weights.sort_values(ascending=False).round(3).to_string())

    print("\nRisk tier distribution:")
    print(df["risk_tier"].value_counts().to_string())

    print("\nTop 5 highest-risk properties:")
    print(df.sort_values("risk_score", ascending=False)
          [["property_id", "risk_score", "risk_tier"]].head(5).to_string(index=False))

    plot_risk_distribution(df)
    df.to_csv("risk_scores_output.csv", index=False)


if __name__ == "__main__":
    main()
