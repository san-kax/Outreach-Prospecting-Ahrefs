
import streamlit as st
import pandas as pd
import requests
import time
from io import StringIO
from urllib.parse import urlparse
import os

# === Streamlit UI ===
st.set_page_config(page_title="Ahrefs Backlink Analyzer", layout="wide")
st.title("📬 Outreach Prospecting Tool")

# === Sidebar Inputs ===
st.sidebar.header("Settings")
input_file = st.sidebar.file_uploader("Upload Input URLs CSV", type=["csv"])
gambling_file = st.sidebar.file_uploader("Upload Gambling Domains CSV", type=["csv"])
outreach_file = st.sidebar.file_uploader("Upload Already Outreach CSV", type=["csv"])
tld_file = st.sidebar.file_uploader("Upload TLD Blocklist (Excel)", type=["xlsx"])

api_key = st.sidebar.text_input("Enter your Ahrefs API Key", type="password")
max_urls = st.sidebar.number_input("Max input URLs to process", min_value=1, max_value=500, value=10)
limit_backlinks = st.sidebar.number_input("Backlinks per URL (limit)", min_value=1, max_value=1000, value=20)
run_button = st.sidebar.button("Run Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "🛑 **[Google Sheet: Rejected Domains](https://docs.google.com/spreadsheets/d/1td29sxdkKAXbzioI6rXxPqUkrFrEnmSH/edit?gid=1937666042#gid=1937666042)**\n"
    "Add new rejected domains here to keep the filter up to date."
)

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

SELECT_FIELDS = ",".join([
    "url_from", "anchor", "title", "url_to", "domain_rating_source",
    "traffic_domain", "positions", "is_dofollow", "is_nofollow", "is_content"
])

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

        df_filtered = df_deduped[
            (df_deduped["domain_rating_source"] >= 20) &
            (df_deduped["traffic_domain"] >= 300) &
            (df_deduped["is_dofollow"] == True) &
            (df_deduped["is_content"] == True)
        ]
        st.success("✅ Backlinks filtered")
        st.dataframe(df_filtered.head())

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

        cols_to_fix = [
            "Domain Rating", "URL Rating", "Org Keywords",
            "Org Keywords 1-3", "Org Keywords 4-10", "Org Traffic Top By Country"
        ]
        for col in cols_to_fix:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].astype(str)

        st.success("✅ Metrics added")
        st.dataframe(df_merged.head())

        if gambling_file is not None:
            df_compare = pd.read_csv(gambling_file)
            gambling_domains = df_compare.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
            df_merged["Found in Gambling.com"] = df_merged["referring_domain"].isin(gambling_domains)
            df_merged["Found in Gambling.com"] = df_merged["Found in Gambling.com"].apply(lambda x: "TRUE" if x else "FALSE")
            st.success("🏷️ Gambling.com flag added")

        if outreach_file is not None:
            df_outreach = pd.read_csv(outreach_file)
            if "Opportunity" in df_outreach.columns:
                outreached_domains = df_outreach["Opportunity"].dropna().str.strip().str.lower().unique()
                df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
                df_merged = df_merged[~df_merged["referring_domain"].isin(outreached_domains)]
                st.success("🚫 Removed already outreached domains")
            else:
                st.warning("⚠️ 'Opportunity' column not found in uploaded outreach file.")

        REJECTED_CSV_URL = "https://docs.google.com/spreadsheets/d/1td29sxdkKAXbzioI6rXxPqUkrFrEnmSH/export?format=csv&gid=1937666042"
        try:
            df_rejected = pd.read_csv(REJECTED_CSV_URL)
            rejected_domains = df_rejected.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
            df_merged = df_merged[~df_merged["referring_domain"].isin(rejected_domains)]
            st.success("❌ Rejected domains from Google Sheet filtered out")
        except Exception as e:
            st.warning(f"⚠️ Could not fetch rejected domains from Google Sheet: {e}")

        def extract_tld(domain):
            try:
                parts = domain.lower().split('.')
                if len(parts) >= 2:
                    return '.' + parts[-1]
                return ''
            except:
                return ''

        if tld_file is not None:
            df_tlds = pd.read_excel(tld_file)
            blocked_tlds = df_tlds.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].astype(str)
            df_merged["tld"] = df_merged["referring_domain"].apply(extract_tld)
            df_merged = df_merged[~df_merged["tld"].isin(blocked_tlds)]
            df_merged.drop(columns=["tld"], inplace=True)
            st.success("🔻 Blocked TLDs filtered out from result")

        st.session_state["df_merged"] = df_merged
        st.download_button("Download Final CSV", df_merged.to_csv(index=False), file_name="ahrefs_backlinks_flagged.csv", mime="text/csv")
        st.success("✅ Done! You can download the output above.")

# === Pitchbox Integration ===
st.sidebar.header("📤 Pitchbox Upload (One-by-One)")
pb_api_key = st.sidebar.text_input("Pitchbox API Key", type="password")
pb_campaign_id = st.sidebar.text_input("Pitchbox Campaign ID")
upload_button = st.sidebar.button("Upload to Pitchbox")

if upload_button:
    if not pb_api_key or not pb_campaign_id:
        st.error("⚠️ Please enter both API key and campaign ID.")
    elif "df_merged" not in st.session_state:
        st.error("⚠️ You must run the backlink analysis before uploading to Pitchbox.")
    elif "Found in Gambling.com" not in st.session_state["df_merged"].columns:
        st.error("🔍 Can't find gambling flag. Please ensure you ran the analysis with gambling domain list.")
    else:
        try:
            df_merged = st.session_state["df_merged"]
            filtered_domains = df_merged[df_merged["Found in Gambling.com"] == "FALSE"]["referring_domain"].dropna().unique()

            if len(filtered_domains) == 0:
                st.warning("No domains eligible for Pitchbox upload (all flagged as gambling).")
            else:
                st.info(f"Preparing to upload {len(filtered_domains)} domains to Pitchbox...")

                # Step 1: Authenticate and get token
                auth_url = "https://apiv2.pitchbox.com/docs#section/Authentication/JWT"
                auth_response = requests.post(auth_url, json={"api_key": pb_api_key})
                if auth_response.status_code != 200:
                    st.error(f"❌ Authentication failed: {auth_response.text}")
                else:
                    jwt = auth_response.json().get("access_token")
                    headers = {
                        "Authorization": f"Bearer {jwt}",
                        "Content-Type": "application/json"
                    }

                    # Step 2: Loop through each domain and POST individually
                    progress = st.progress(0)
                    success_count = 0
                    failure_log = []

                    for i, domain in enumerate(filtered_domains):
                        payload = {
                            "url": f"https://{domain}",
                            "campaign": int(pb_campaign_id),
                            "contacts": [],
                            "personalization": {}
                        }

                        response = requests.post("https://api.pitchbox.com/api/opportunities", json=payload, headers=headers)

                        if response.status_code == 200:
                            success_count += 1
                            st.write(f"✅ Added: {domain}")
                        else:
                            failure_log.append((domain, response.status_code, response.text))
                            st.error(f"❌ Failed: {domain} — {response.status_code} - {response.text}")

                        progress.progress((i + 1) / len(filtered_domains))

                    # Final summary
                    st.success(f"Upload complete: {success_count} succeeded, {len(failure_log)} failed.")
                    if failure_log:
                        with st.expander("View Failed Uploads"):
                            for domain, code, msg in failure_log:
                                st.text(f"{domain} → {code}: {msg}")
        except Exception as e:
            st.exception(f"Unexpected error: {e}")
