# Churn Prediction & Risk Segmentation

Predicts customer churn with Logistic Regression and XGBoost on noisy,
multi-source behavioral/transactional data, then segments the customer base
with K-Means on RFM-style features (recency/tenure, frequency of support
contact, monetary spend) to prioritize retention outreach by segment rather
than treating every "at-risk" customer the same way.

Reproduces the modeling approach on **synthetic data** with realistic class
imbalance, noise, and injected missingness (real customer data is
proprietary).

## Approach

- Generates customer-level features with a latent churn-propensity function,
  noise, and ~3% missingness in one feature to mimic multi-source data
  reality.
- Trains and compares Logistic Regression (baseline, balanced class weights)
  against XGBoost, evaluated by ROC-AUC and accuracy on a held-out set.
- Runs K-Means (k=4) on RFM-style features to segment customers, then
  reports churn rate per segment — surfacing which segment to prioritize
  for retention spend.
- Plots ROC curves for both models side-by-side with churn rate by segment.

## Run it

```bash
pip install -r requirements.txt
python churn_segmentation.py
```

Outputs `churn_model_chart.png` and `churn_segments_output.csv`.
