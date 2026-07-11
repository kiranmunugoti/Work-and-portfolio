# Work and Portfolio

Contains project work and personal project ideas — data science / ML case studies plus a personal portfolio site.

## Contents

| Folder / File | Description |
|---|---|
| [`01-risk-scoring/`](./01-risk-scoring) | Risk scoring framework built for a hospitality portfolio context, combining multiple weighted risk dimensions into a single composite score to support portfolio-level decision-making. Techniques: Multi-factor scoring, weighted aggregation, risk segmentation |
| [`02-financial-forecasting/`](./02-financial-forecasting) | 
| End-to-end forecasting pipeline for financial risk metrics using an ensemble of classical time-series methods.
  Techniques: ARIMA, Holt-Winters exponential smoothing, ensemble forecasting |
| [`03-churn-segmentation/`](./03-churn-segmentation) | 
| Customer churn prediction paired with Recency-Frequency-Monetary (RFM) segmentation to identify at-risk, high-value customers.
  Techniques: Logistic Regression, XGBoost, RFM segmentation, feature engineering.|
| [`04-fraud-detection/`](./04-fraud-detection) | 
| Model to flag anomalous and potentially fraudulent healthcare insurance claims for downstream investigation.
  Techniques: Classification modeling, anomaly detection, feature engineering on claims data.|
| [`index.html`](./index.html) | Personal portfolio site — see below |

> Fill in the one-line descriptions above with whatever each project actually covers (dataset, model type, key result) — happy to help draft those too if you share what's in each folder.

## Portfolio Site

`index.html` at the repo root is a single-page, dark-themed portfolio built with AngularJS, custom CSS, and EmailJS for the contact form.

**Live site (once GitHub Pages is enabled):**
`https://kiranmunugoti.github.io/Work-and-portfolio/`

### Run locally

```bash
git clone https://github.com/kiranmunugoti/Work-and-portfolio.git
cd Work-and-portfolio
open index.html   # macOS
# or: xdg-open index.html   (Linux)
# or: start index.html      (Windows)
```

Or serve it so relative paths/assets behave the same as they would in production:

```bash
npx serve .
# or
python3 -m http.server 8000
```

### Configure the contact form (EmailJS)
Feel free to reach out at kiranmunugotis@gmail.com or connect via the portfolio site for collaboration, questions, or opportunities.

> The EmailJS public key is designed to be exposed client-side, so this is safe for a static site — just don't put any other secrets directly in this file.

## License

MIT — see [LICENSE](./LICENSE).
