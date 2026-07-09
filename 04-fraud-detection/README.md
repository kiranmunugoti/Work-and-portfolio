# Healthcare Claims Fraud Detection

Flags high-risk insurance claims using XGBoost on a realistically imbalanced
claims dataset (~3% fraud rate), with claim-vs-provider-baseline and billing
velocity features that mirror the kind of feature engineering used to shape
claims data before modeling.

In production this supported fraud/claims-risk review for a healthcare
payer. This repo reproduces the modeling and evaluation approach on
**synthetic claims data** only (member and claims data is protected and
cannot be shared).

## Approach

- Generates ~20K synthetic claims with a realistic ~3% fraud rate and
  fraud-correlated feature shifts (provider billing ratio, claim lag,
  new-provider flag, diagnosis/procedure mismatch, weekend service).
- Trains XGBoost with `scale_pos_weight` to handle the class imbalance.
- Uses a business-tuned decision threshold (0.35, not the default 0.5) to
  favor recall on fraud for a manual-review queue, and reports full
  precision/recall/F1 plus a confusion matrix at that threshold.
- Plots the precision-recall curve (PR-AUC) and feature importances.

## Run it

```bash
pip install -r requirements.txt
python fraud_detection.py
```

Outputs `fraud_model_chart.png` and `fraud_scored_claims.csv`.
