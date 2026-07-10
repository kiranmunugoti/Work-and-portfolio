# Work and Portfolio

Contains project work and personal project ideas — data science / ML case studies plus a personal portfolio site.

## Contents

| Folder / File | Description |
|---|---|
| [`01-risk-scoring/`](./01-risk-scoring) | _Add a 1–2 line description of this project_ |
| [`02-financial-forecasting/`](./02-financial-forecasting) | _Add a 1–2 line description of this project_ |
| [`03-churn-segmentation/`](./03-churn-segmentation) | _Add a 1–2 line description of this project_ |
| [`04-fraud-detection/`](./04-fraud-detection) | _Add a 1–2 line description of this project_ |
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

The contact form uses [EmailJS](https://www.emailjs.com/) and currently has placeholder credentials. To make it functional:

1. Create a free account at emailjs.com
2. Set up an Email Service and Email Template
3. In `index.html`, replace these placeholders with your own values:

   ```js
   var EMAILJS_PUBLIC_KEY  = 'YOUR_PUBLIC_KEY';
   var EMAILJS_SERVICE_ID  = 'YOUR_SERVICE_ID';
   var EMAILJS_TEMPLATE_ID = 'YOUR_TEMPLATE_ID';
   ```

> The EmailJS public key is designed to be exposed client-side, so this is safe for a static site — just don't put any other secrets directly in this file.

### Updating content

- **Profile photo:** update the `<img src="...">` inside `.avatar-box`.
- **Projects shown on the site:** edit the AngularJS `projects` data in the `<script>` section near the bottom of `index.html`.
- **Skills:** edit the skills data structure in the same script section.

### Enable GitHub Pages

1. Go to **Settings → Pages**
2. Under "Build and deployment," set **Source** to `Deploy from a branch`
3. Choose the `main` branch and `/ (root)` folder
4. Save — the site will be live at the URL above within a minute or two

## License

MIT — see [LICENSE](./LICENSE).
