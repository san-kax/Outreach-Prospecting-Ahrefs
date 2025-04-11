import streamlit as st
import pandas as pd
import requests
import time
from io import StringIO
from urllib.parse import urlparse

# === Streamlit UI ===
st.set_page_config(page_title="Ahrefs Backlink Analyzer", layout="wide")
st.title("🔗 Ahrefs Backlink Analysis Tool")

# === Sidebar Inputs ===
st.sidebar.header("Settings")
input_file = st.sidebar.file_uploader("Upload Input URLs CSV", type=["csv"])
gambling_file = st.sidebar.file_uploader("Upload Gambling Domains CSV", type=["csv"])

api_key = st.sidebar.text_input("Enter your Ahrefs API Key", type="password")
max_urls = st.sidebar.number_input("Max input URLs to process", min_value=1, max_value=500, value=10)
limit_backlinks = st.sidebar.number_input("Backlinks per URL (limit)", min_value=1, max_value=1000, value=20)
run_button = st.sidebar.button("Run Analysis")

# === Helper Functions ===
def extract_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def fetch_backlinks(target_url, limit, headers):
    api_url = (
        f"https://api.ahrefs.com/v3/site-explorer/all-backlinks?"
        f"target={requests.utils.quote(target_url)}&limit={limit}&mode=prefix&select={requests.utils.quote(SELECT_FIELDS)}"
    )
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        backlinks = response.json().get("backlinks", [])
        for b in backlinks:
            b["target_url"] = target_url
            b["referring_domain"] = extract_domain(b.get("url_from", ""))
        return backlinks
    else:
        st.error(f"❌ Error {response.status_code} fetching {target_url}")
        return []

def chunk_list(data, size=100):
    for i in range(0, len(data), size):
        yield data[i:i+size]

def fetch_batch_metrics(domains_chunk, headers):
    url = "https://api.ahrefs.com/v3/batch-analysis/batch-analysis"
    fields = ["domain_rating", "url_rating", "org_keywords", "org_keywords_1_3", "org_keywords_4_10", "org_traffic_top_by_country"]
    targets_chunk = [{"url": d, "mode": "subdomains", "protocol": "both"} for d in domains_chunk]
    payload = {"select": fields, "targets": targets_chunk, "volume_mode": "monthly"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"❌ Error fetching batch metrics: {response.status_code}")
        return None

def parse_batch_results(api_data, input_domains):
    rows = []
    if "targets" in api_data:
        for i, item in enumerate(api_data["targets"]):
            domain = input_domains[i].strip().rstrip("/")
            rows.append({
                "referring_domain": domain,
                "Domain Rating": item.get("domain_rating", "N/A"),
                "URL Rating": item.get("url_rating", "N/A"),
                "Org Keywords": item.get("org_keywords", "N/A"),
                "Org Keywords 1-3": item.get("org_keywords_1_3", "N/A"),
                "Org Keywords 4-10": item.get("org_keywords_4_10", "N/A"),
                "Org Traffic Top By Country": item.get("org_traffic_top_by_country", "N/A")
            })
    return rows

# === API Field Definitions ===
SELECT_FIELDS = ",".join([
    "url_from", "anchor", "title", "url_to", "domain_rating_source",
    "traffic_domain", "positions", "is_dofollow", "is_nofollow", "is_content"
])

# === Run Analysis ===
if run_button:
    if not api_key:
        st.error("⚠️ Please enter your Ahrefs API key.")
        st.stop()

    HEADERS = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if input_file is not None:
        df_input = pd.read_csv(input_file)
        target_urls = df_input.iloc[:, 4].dropna().unique()[:max_urls]

        all_backlinks = []
        with st.spinner("🔍 Fetching backlinks..."):
            for url in target_urls:
                backlinks = fetch_backlinks(url, limit_backlinks, HEADERS)
                all_backlinks.extend(backlinks)

        if not all_backlinks:
            st.warning("No backlinks found.")
            st.stop()

        df_backlinks = pd.DataFrame(all_backlinks)
        df_deduped = df_backlinks.drop_duplicates(subset="referring_domain", keep="first")
        st.success("✅ Backlinks fetched and deduplicated")
        st.dataframe(df_deduped.head())

        # Filter step
        df_filtered = df_deduped[
            (df_deduped["domain_rating_source"] >= 20) &
            (df_deduped["traffic_domain"] >= 300) &
            (df_deduped["is_dofollow"] == True) &
            (df_deduped["is_content"] == True)
        ]
        st.success("✅ Backlinks filtered")
        st.dataframe(df_filtered.head())

        # Metrics enrichment
        referring_domains = df_filtered["referring_domain"].dropna().unique()
        all_metrics = []
        with st.spinner("📊 Enriching domains with Ahrefs metrics..."):
            for chunk in chunk_list(referring_domains, 100):
                api_data = fetch_batch_metrics(chunk, HEADERS)
                if api_data:
                    parsed = parse_batch_results(api_data, chunk)
                    all_metrics.extend(parsed)
                time.sleep(1)

        df_metrics = pd.DataFrame(all_metrics)
        df_filtered["referring_domain"] = df_filtered["referring_domain"].astype(str).str.strip().str.rstrip("/")
        df_merged = df_filtered.merge(df_metrics, on="referring_domain", how="left")

        # Fix column type for display
        cols_to_fix = [
            "Domain Rating", "URL Rating", "Org Keywords",
            "Org Keywords 1-3", "Org Keywords 4-10", "Org Traffic Top By Country"
        ]
        for col in cols_to_fix:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].astype(str)

        st.success("✅ Metrics added")
        st.dataframe(df_merged.head())

        # Flag gambling domains (optional)
        if gambling_file is not None:
            df_compare = pd.read_csv(gambling_file)
            gambling_domains = df_compare.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
            df_merged["Found in Gambling.com"] = df_merged["referring_domain"].isin(gambling_domains)
            df_merged["Found in Gambling.com"] = df_merged["Found in Gambling.com"].apply(lambda x: "TRUE" if x else "FALSE")
            st.success("🏷️ Gambling.com flag added")

        # Output & download
        st.download_button("Download Final CSV", df_merged.to_csv(index=False), file_name="ahrefs_backlinks_flagged.csv", mime="text/csv")
        st.success("✅ Done! You can download the output above.")
