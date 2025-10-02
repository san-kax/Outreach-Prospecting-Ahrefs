import streamlit as st
import pandas as pd
import requests
import time
from urllib.parse import urlparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================
# App Setup
# ==========================
st.set_page_config(page_title="Ahrefs Backlink Analyzer", layout="wide")
st.title("📬 Outreach Prospecting Tool")

# ==========================
# Helpers
# ==========================
SELECT_FIELDS = ",".join([
    "url_from", "anchor", "title", "url_to", "domain_rating_source",
    "traffic_domain", "positions", "is_dofollow", "is_nofollow", "is_content"
])

@st.cache_data(show_spinner=False)
def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def to_domain_only(value: str) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().lower()
    if v.startswith("http://") or v.startswith("https://"):
        return extract_domain(v)
    if v.startswith("www."):
        v = v[4:]
    return v

@st.cache_data(show_spinner=False)
def extract_tld(domain: str) -> str:
    try:
        parts = str(domain).lower().split('.')
        if len(parts) >= 2:
            return '.' + parts[-1]
        return ''
    except Exception:
        return ''

# -------- Ahrefs -------

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
        return []


def parallel_fetch_backlinks(urls, limit, headers, max_workers=8):
    results = []
    urls = list(urls)
    max_workers = max(1, min(max_workers, len(urls)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_backlinks, u, limit, headers): u for u in urls}
        progress = st.progress(0.0)
        done_count = 0
        for fut in as_completed(futures):
            data = fut.result() or []
            results.extend(data)
            done_count += 1
            progress.progress(done_count / len(urls))
    return results


def chunk_list(data, size=100):
    for i in range(0, len(data), size):
        yield data[i:i+size]


def fetch_batch_metrics(domains_chunk, headers):
    url = "https://api.ahrefs.com/v3/batch-analysis/batch-analysis"
    fields = [
        "domain_rating", "url_rating", "org_keywords", "org_keywords_1_3",
        "org_keywords_4_10", "org_traffic_top_by_country"
    ]
    targets_chunk = [{"url": d, "mode": "subdomains", "protocol": "both"} for d in domains_chunk]
    payload = {"select": fields, "targets": targets_chunk, "volume_mode": "monthly"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.warning(f"Ahrefs batch error {response.status_code}")
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

# -------- Airtable -------

@st.cache_data(show_spinner=False)
def airtable_fetch_all(base_id: str, table_name: str, api_key: str, view: str | None = None, fields: list[str] | None = None) -> list[dict]:
    """Fetch all rows from an Airtable table (REST API), handling pagination."""
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Airtable: empty API key")
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_name)}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {}
    if view:
        params["view"] = view
    if fields:
        for i, f in enumerate(fields):
            params[f"fields[{i}]"] = f
    out = []
    while True:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 401:
            raise RuntimeError("Airtable 401 AUTHENTICATION_REQUIRED — paste the *token VALUE* (starts with 'pat…'), and ensure it has data.records:read + base access.")
        if r.status_code != 200:
            raise RuntimeError(f"Airtable error {r.status_code}: {r.text}")
        data = r.json()
        out.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
        time.sleep(0.2)
    return out

@st.cache_data(show_spinner=False)
def airtable_column_to_series(records: list[dict], field_name: str) -> pd.Series:
    values = []
    for rec in records:
        fields = rec.get("fields", {})
        v = fields.get(field_name)
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "name" in item:
                    values.append(item["name"])  # select options
                else:
                    values.append(str(item))
        else:
            values.append(str(v))
    s = pd.Series(values, dtype="string").dropna()
    return s

@st.cache_data(show_spinner=False)
def airtable_lookup_by_values(base_id: str, table_name: str, field_name: str, values: list[str], api_key: str, batch_size: int = 25) -> pd.Series:
    """Fast membership check: fetch only rows where {field_name} equals any of the given values.
    Uses filterByFormula with OR(LOWER({Field})='v1', ...). Returns a Series of matched values.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Airtable: empty API key")
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_name)}"
    headers = {"Authorization": f"Bearer {api_key}"}

    def esc(s: str) -> str:
        return s.replace("'", "\'")

    out = []
    values = [v for v in values if v]
    for batch in chunk_list(values, batch_size):
        formula_parts = [f"LOWER({{{field_name}}})='{esc(v.lower())}'" for v in batch]
        formula = "OR(" + ",".join(formula_parts) + ")"
        params = {"filterByFormula": formula, "fields[]": [field_name]}
        while True:
            r = requests.get(url, headers=headers, params=params)
            if r.status_code == 401:
                raise RuntimeError("Airtable 401 AUTHENTICATION_REQUIRED — check token + base access.")
            if r.status_code != 200:
                raise RuntimeError(f"Airtable error {r.status_code}: {r.text}")
            data = r.json()
            out.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break
            params["offset"] = offset
    return airtable_column_to_series(out, field_name)

# ==========================
# Sidebar — Inputs
# ==========================
st.sidebar.header("Settings")

# Input URLs
input_file = st.sidebar.file_uploader("Upload Input URLs CSV", type=["csv"])

# Ahrefs
api_key = st.sidebar.text_input("Enter your Ahrefs API Key", type="password")
max_urls = st.sidebar.number_input("Max input URLs to process", min_value=1, max_value=500, value=10)
limit_backlinks = st.sidebar.number_input("Backlinks per URL (limit)", min_value=1, max_value=1000, value=20)
ahrefs_workers = st.sidebar.slider("Parallel URL workers", min_value=1, max_value=16, value=6)
ahrefs_batch_delay = st.sidebar.number_input("Ahrefs batch delay (seconds)", min_value=0.0, max_value=3.0, value=0.2, step=0.1)

st.sidebar.markdown("---")

# Airtable Toggle & Config
use_airtable = st.sidebar.checkbox("Use Airtable for filters (existing / brand flag / rejected / blocklist)", value=True)
if use_airtable:
    st.sidebar.subheader("Airtable")
    at_api_key = st.sidebar.text_input("Airtable API Key", type="password", value=os.getenv("AIRTABLE_API_KEY", ""))

    # Presets (labels -> triplets)
    EXISTING_PRESETS = {
        "Prospect-Data (appHdhjsWVRxaCvcR)": ("appHdhjsWVRxaCvcR", "tbliCOQZY9RICLsLP", "Domain"),
        "Prospect-Data-1 (appVyIiM5boVyoBhf)": ("appVyIiM5boVyoBhf", "tbliCOQZY9RICLsLP", "Domain"),
        "GDC-Database (appUoOvkqzJvyyMvC)": ("appUoOvkqzJvyyMvC", "tbliCOQZY9RICLsLP", "Domain"),
        "WB-Database (appueIgn44RaVH6ot)": ("appueIgn44RaVH6ot", "tbl3vMYv4RzKfuBf4", "Domain"),
        "Freebets-Database (appFBasaCUkEKtvpV)": ("appFBasaCUkEKtvpV", "tblmTREzfIswOuA0F", "Domain"),
    }

    st.sidebar.markdown("**Existing domains — select Airtable sources to check & EXCLUDE**")

# Checkbox dropdown helper (uses popover if available, otherwise an expander)
def checkbox_list_dropdown(label: str, options: list[str], default: list[str]) -> list[str]:
    selected: list[str] = []
    has_popover = hasattr(st.sidebar, "popover")
    container = st.sidebar.popover(label) if has_popover else st.sidebar.expander(label, expanded=False)
    with container:
        select_all = st.checkbox("Select all", value=len(default) == len(options), key=f"{label}_all")
        base_set = set(options) if select_all else set(default)
        for i, opt in enumerate(options):
            checked = st.checkbox(opt, value=(opt in base_set), key=f"{label}_{i}")
            if checked:
                selected.append(opt)
    return selected

existing_options = list(EXISTING_PRESETS.keys())
default_existing = [
    "Prospect-Data (appHdhjsWVRxaCvcR)",
    "GDC-Database (appUoOvkqzJvyyMvC)",
    "WB-Database (appueIgn44RaVH6ot)",
    "Freebets-Database (appFBasaCUkEKtvpV)",
]

selected_existing_labels = checkbox_list_dropdown("Select sources", existing_options, default_existing)
selected_existing_cfg = [EXISTING_PRESETS[l] for l in selected_existing_labels]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Brand/Gambling flag source**")
    use_selected_for_flag = st.sidebar.checkbox(
        "Use the SAME selection above to set 'Found in Gambling.com' flag",
        value=True,
    )
    if use_selected_for_flag:
        brand_flag_cfg = selected_existing_cfg.copy()
    else:
        brand_flag_cfg = [
            ("appUoOvkqzJvyyMvC", "tbliCOQZY9RICLsLP", "Domain"),
            ("appFBasaCUkEKtvpV", "tblmTREzfIswOuA0F", "Domain"),
            ("appueIgn44RaVH6ot", "tbl3vMYv4RzKfuBf4", "Domain"),
        ]

    st.sidebar.markdown("---")
    enable_rejected = st.sidebar.checkbox("Exclude 'Outreach-Rejected-Sites' (appTf6MmZDgouu8SN)", value=True)
    enable_blocklist = st.sidebar.checkbox("Exclude 'GDC-Disavow-List' (appJTJQwjHRaAyLkw)", value=True)

    rejected_cfg = [("appTf6MmZDgouu8SN", "tbliCOQZY9RICLsLP", "Rejected Domains")] if enable_rejected else []
    blocklist_cfg = [("appJTJQwjHRaAyLkw", "tbliCOQZY9RICLsLP", "Disavow-Domains")] if enable_blocklist else []

    # Speed mode for Airtable lookups
    fast_match = st.sidebar.checkbox("Fast Airtable matching (query only candidates)", value=True, help="Avoids downloading entire tables by using filterByFormula to match only your candidate domains.")

    with st.sidebar.expander("Advanced: add custom tables (baseId,tableIdOrName,fieldName)"):
        st.caption("One per line. Examples are prefilled above; this lets power users add more without code changes.")
        custom_existing_txt = st.text_area("Additional EXISTING sources (exclude)", height=80, value="")
        custom_brand_txt = st.text_area("Additional BRAND flag sources", height=80, value="")
        custom_rejected_txt = st.text_area("Additional REJECTED sources (exclude)", height=80, value="")
        custom_blocklist_txt = st.text_area("Additional BLOCKLIST sources (exclude)", height=80, value="")

        def parse_cfg(text: str) -> list:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#', 1)[0].strip()
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    rows.append((parts[0], parts[1], parts[2]))
            return rows

        selected_existing_cfg += parse_cfg(custom_existing_txt)
        brand_flag_cfg += parse_cfg(custom_brand_txt)
        rejected_cfg += parse_cfg(custom_rejected_txt)
        blocklist_cfg += parse_cfg(custom_blocklist_txt)

    # Connection tester
    if st.sidebar.button("Test Airtable connection"):
        try:
            api_key = at_api_key.strip()
            r = requests.get("https://api.airtable.com/v0/meta/bases", headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 200:
                bases = r.json().get("bases", [])
                base_ids = [b.get("id") for b in bases]
                st.success(f"Token OK. Accessible bases: {', '.join(base_ids) or 'none'}")
            elif r.status_code == 401:
                st.error("401 AUTHENTICATION_REQUIRED: paste the *token VALUE* (starts with 'pat') and ensure it has data.records:read + base access.")
            else:
                st.error(f"Unexpected response {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Connection test failed: {e}")
else:
    gambling_file = st.sidebar.file_uploader("Upload Gambling Domains CSV", type=["csv"])
    outreach_file = st.sidebar.file_uploader("Upload Already Outreach CSV", type=["csv"])
    tld_file = st.sidebar.file_uploader("Upload TLD Blocklist (Excel)", type=["xlsx"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("🛑 If not using Airtable, keep the Google Sheet/CSV uploads here.")

run_button = st.sidebar.button("Run Analysis")

# ==========================
# Core Flow
# ==========================
if run_button:
    if not api_key:
        st.error("⚠️ Please enter your Ahrefs API key.")
        st.stop()

    HEADERS = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if input_file is None:
        st.error("⚠️ Please upload an input CSV containing target URLs.")
        st.stop()

    # ---- read input urls ----
    df_input = pd.read_csv(input_file)
    target_urls = df_input.iloc[:, 1].dropna().unique()[:max_urls]

    # ---- fetch backlinks (parallel) ----
    with st.spinner("🔍 Fetching backlinks in parallel..."):
        all_backlinks = parallel_fetch_backlinks(target_urls, limit_backlinks, HEADERS, max_workers=ahrefs_workers)

    if not all_backlinks:
        st.warning("No backlinks found.")
        st.stop()

    df_backlinks = pd.DataFrame(all_backlinks)
    df_deduped = df_backlinks.drop_duplicates(subset="referring_domain", keep="first")
    st.success("✅ Backlinks fetched and deduplicated")
    st.dataframe(df_deduped.head())

    # ---- base quality filter ----
    df_filtered = df_deduped[
        (df_deduped["domain_rating_source"] >= 20) &
        (df_deduped["traffic_domain"] >= 300) &
        (df_deduped["is_dofollow"] == True) &
        (df_deduped["is_content"] == True)
    ]
    st.success("✅ Backlinks filtered")
    st.dataframe(df_filtered.head())

    # ---- enrich with Ahrefs batch metrics ----
    referring_domains = df_filtered["referring_domain"].dropna().unique()
    all_metrics = []
    with st.spinner("📊 Enriching domains with Ahrefs metrics..."):
        for chunk in chunk_list(referring_domains, 100):
            api_data = fetch_batch_metrics(chunk, HEADERS)
            if api_data:
                parsed = parse_batch_results(api_data, chunk)
                all_metrics.extend(parsed)
            time.sleep(max(0.0, ahrefs_batch_delay))

    df_metrics = pd.DataFrame(all_metrics)
    df_filtered["referring_domain"] = df_filtered["referring_domain"].astype(str).str.strip().str.rstrip("/")
    df_merged = df_filtered.merge(df_metrics, on="referring_domain", how="left")

    for col in [
        "Domain Rating", "URL Rating", "Org Keywords",
        "Org Keywords 1-3", "Org Keywords 4-10", "Org Traffic Top By Country"
    ]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(str)

    st.success("✅ Metrics added")
    st.dataframe(df_merged.head())

    # ==========================
    # FILTERS: Airtable or local files
    # ==========================
    if use_airtable:
        if not at_api_key:
            st.error("⚠️ Please provide Airtable API key.")
            st.stop()

        try:
            cand = list({to_domain_only(d) for d in df_merged["referring_domain"].astype(str).tolist()})

            # Existing domains to EXCLUDE
            if fast_match:
                existing_series = []
                for (base_id, table_id, field_name) in selected_existing_cfg:
                    s = airtable_lookup_by_values(base_id, table_id, field_name, cand, at_api_key)
                    existing_series.append(s)
                existing_series = pd.concat(existing_series, ignore_index=True) if existing_series else pd.Series([], dtype="string")
            else:
                existing_records = []
                for cfg in selected_existing_cfg:
                    base_id, table_id, field_name = cfg
                    recs = airtable_fetch_all(base_id, table_id, at_api_key, fields=[field_name])
                    existing_records += recs
                existing_series = airtable_column_to_series(existing_records, selected_existing_cfg[0][2]) if selected_existing_cfg else pd.Series([], dtype="string")
            existing_domains = set(existing_series.map(to_domain_only).dropna().unique())

            # Brand flag
            if fast_match:
                brand_series = []
                for (base_id, table_id, field_name) in brand_flag_cfg:
                    s = airtable_lookup_by_values(base_id, table_id, field_name, cand, at_api_key)
                    brand_series.append(s)
                brand_series = pd.concat(brand_series, ignore_index=True) if brand_series else pd.Series([], dtype="string")
            else:
                brand_records = []
                for cfg in brand_flag_cfg:
                    base_id, table_id, field_name = cfg
                    recs = airtable_fetch_all(base_id, table_id, at_api_key, fields=[field_name])
                    brand_records += recs
                brand_series = airtable_column_to_series(brand_records, brand_flag_cfg[0][2]) if brand_flag_cfg else pd.Series([], dtype="string")
            brand_domains = set(brand_series.map(to_domain_only).dropna().unique())

            # Rejected & Blocklist (these are small; fetch-all is fine, but also support fast mode)
            def gather(cfgs):
                if not cfgs:
                    return set()
                if fast_match:
                    ser_list = []
                    for (base_id, table_id, field_name) in cfgs:
                        s = airtable_lookup_by_values(base_id, table_id, field_name, cand, at_api_key)
                        ser_list.append(s)
                    ser = pd.concat(ser_list, ignore_index=True) if ser_list else pd.Series([], dtype="string")
                else:
                    recs_all = []
                    for base_id, table_id, field_name in cfgs:
                        recs_all += airtable_fetch_all(base_id, table_id, at_api_key, fields=[field_name])
                    ser = pd.Series([], dtype="string") if not recs_all else airtable_column_to_series(recs_all, cfgs[0][2])
                return set(ser.map(to_domain_only).dropna().unique())

            rejected_domains = gather(rejected_cfg)
            blocklisted_domains = gather(blocklist_cfg)
        except Exception as e:
            st.error(f"Airtable fetch failed: {e}")
            st.stop()

        # Brand flag
        df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
        df_merged["Found in Gambling.com"] = df_merged["referring_domain"].isin(brand_domains)
        df_merged["Found in Gambling.com"] = df_merged["Found in Gambling.com"].apply(lambda x: "TRUE" if x else "FALSE")

        # Exclusions
        before = len(df_merged)
        df_merged = df_merged[~df_merged["referring_domain"].isin(existing_domains)]
        removed_existing = before - len(df_merged)

        before2 = len(df_merged)
        if rejected_domains:
            df_merged = df_merged[~df_merged["referring_domain"].isin(rejected_domains)]
        removed_rejected = before2 - len(df_merged)

        before3 = len(df_merged)
        if blocklisted_domains:
            df_merged = df_merged[~df_merged["referring_domain"].isin(blocklisted_domains)]
        removed_block = before3 - len(df_merged)

        st.success(f"🧹 Excluded existing: {removed_existing} • rejected: {removed_rejected} • blocklist: {removed_block}")

    else:
        # Legacy local-file flow
        gambling_domains = pd.Series([], dtype="string")
        if 'gambling_file' in locals() and gambling_file is not None:
            df_compare = pd.read_csv(gambling_file)
            gambling_domains = df_compare.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
            df_merged["Found in Gambling.com"] = df_merged["referring_domain"].isin(gambling_domains)
            df_merged["Found in Gambling.com"] = df_merged["Found in Gambling.com"].apply(lambda x: "TRUE" if x else "FALSE")
            st.success("🏷️ Gambling.com flag added")

        if 'outreach_file' in locals() and outreach_file is not None:
            df_outreach = pd.read_csv(outreach_file)
            if "opportunity" in df_outreach.columns:
                outreached_domains = df_outreach["opportunity"].dropna().str.strip().str.lower().unique()
                df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
                df_merged = df_merged[~df_merged["referring_domain"].isin(outreached_domains)]
                st.success("🚫 Removed already outreached domains")
            else:
                st.warning("⚠️ 'opportunity' column not found in uploaded outreach file.")

        if 'tld_file' in locals() and tld_file is not None:
            df_tlds = pd.read_excel(tld_file)
            blocked_tlds = df_tlds.iloc[:, 0].dropna().str.strip().str.lower().unique()
            df_merged["referring_domain"] = df_merged["referring_domain"].astype(str)
            df_merged["tld"] = df_merged["referring_domain"].apply(extract_tld)
            df_merged = df_merged[~df_merged["tld"].isin(blocked_tlds)]
            df_merged.drop(columns=["tld"], inplace=True)
            st.success("🔻 Blocked TLDs filtered out from result")

    # ---- Final Output ----
    st.session_state["df_merged"] = df_merged
    st.download_button("Download Final CSV", df_merged.to_csv(index=False), file_name="ahrefs_backlinks_flagged.csv", mime="text/csv")
    st.success("✅ Done! You can download the output above.")

# === Pitchbox Integration (Bulk Upload with JWT) ===
st.sidebar.header("📤 Pitchbox Upload (Bulk)")
pb_api_key = st.sidebar.text_input("Pitchbox API JWT", type="password")
pb_campaign_id = st.sidebar.text_input("Pitchbox Campaign ID")
upload_button = st.sidebar.button("Upload to Pitchbox")

if upload_button:
    if not pb_api_key or not pb_campaign_id:
        st.error("⚠️ Please enter both API key (JWT) and campaign ID.")
    elif "df_merged" not in st.session_state:
        st.error("⚠️ You must run the backlink analysis before uploading to Pitchbox.")
    elif "Found in Gambling.com" not in st.session_state["df_merged"].columns:
        st.error("🔍 Can't find gambling flag. Ensure you ran the analysis and (if using Airtable) configured the tables/fields.")
    else:
        try:
            df_merged = st.session_state["df_merged"]
            filtered_domains = df_merged[df_merged["Found in Gambling.com"] == "FALSE"]["referring_domain"].dropna().unique()

            if len(filtered_domains) == 0:
                st.warning("No domains eligible for Pitchbox upload (all flagged as gambling).")
            else:
                st.info(f"Preparing to upload {len(filtered_domains)} domains to Pitchbox...")

                payload = [
                    {"url": f"https://{domain}", "campaign": int(pb_campaign_id), "contacts": []}
                    for domain in filtered_domains
                ]

                headers = {"Authorization": f"Bearer {pb_api_key}", "Content-Type": "application/json"}
                response = requests.post("https://apiv2.pitchbox.com/api/opportunities", json=payload, headers=headers)

                if response.status_code == 200:
                    st.success("✅ Bulk upload completed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Upload failed — {response.status_code}")
                    st.text(response.text)

        except Exception as e:
            st.exception(f"Unexpected error: {e}")
