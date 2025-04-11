# Outreach prospecting Ahrefs App

This Streamlit app fetches, filters, enriches, and flags backlink data using the Ahrefs API.

## 🔧 Features

- Upload your own list of target URLs
- Set how many backlinks to fetch per URL
- Filter by Domain Rating, traffic, and link type
- Enrich with additional metrics using Ahrefs batch API
- Optionally flag backlinks found on a comparison domain list (e.g., gambling domains)
- Download the final processed and flagged CSV

## 📦 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Running Locally

```bash
streamlit run ahrefs_backlink_app_user_api.py
```

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** and select this repo
4. Set `ahrefs_backlink_app_user_api.py` as the entry point
5. Deploy and enjoy!

## 🛡️ API Key Security

- This app does **not** store your API key.
- Each user inputs their own Ahrefs API key in the sidebar for use in that session.

## 📁 File Uploads

- **Input URLs CSV** — Should have URLs in the **5th column (column E)**
- **Gambling Domains CSV (optional)** — Should have domains in the **first column**

## 📤 Output

- Filtered and enriched backlinks are shown in the app and available to download as CSV.
