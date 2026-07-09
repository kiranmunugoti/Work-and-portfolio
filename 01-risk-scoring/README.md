# Multi-Dimensional Risk Scoring — Hospitality Portfolio

Risk scoring model for a multi-property hospitality portfolio, blending
operational, financial, and behavioral signals into a single explainable
0–100 risk score per property, with automatic tiering (Low / Medium / High).

In production, features were sourced from PMS, POS, CRM, and booking-platform
data via Azure Data Factory pipelines and refreshed nightly, feeding a Power BI
dashboard used by portfolio operations leadership to prioritize site visits
and interventions.

This repo reproduces the scoring methodology end-to-end on **synthetic data**
(real portfolio data is proprietary and cannot be shared publicly).

## Approach

- Engineers 8 risk-oriented features from raw operational signals
  (occupancy, RevPAR variance, maintenance load, staff turnover, guest
  complaints, late payments, review scores, audit staleness).
- Blends **business-defined weights** with a RandomForest-learned
  re-weighting, so the score stays explainable to stakeholders while still
  adapting to which signals move together in the data — a lightweight
  alternative to a black-box classifier for a metric leadership needs to
  trust and act on directly.
- Outputs a ranked, tiered property list plus a chart of the highest-risk
  properties.

## Run it

```bash
pip install -r requirements.txt
python risk_scoring.py
```

Outputs `risk_scores_chart.png` and `risk_scores_output.csv`.
