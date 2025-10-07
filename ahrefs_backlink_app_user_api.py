# outreach_prospecting_tool.py
import os
import time
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ==========================
# App Setup
# ==========================
st.set_page_config(page_title="Ahrefs Backlink Analyzer", layout="wide")
st.title("📬 Outreach Prospecting Tool")

# Keep last results between reruns so Export/Download still work
if "df_merged" not in st.session_state:
    st.session_state["df_merged"] = None

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
        parts = str(domain).lower().split(".")
        if len(parts) >= 2:
            return "." + parts[-1]
        return ""
    except Exception:
        return ""

def normalize_tld_series(s: pd.Series) -> pd.Series:
    """Ensure TLDs look like '.de' (adds leading dot, lower-cases, strips)."""
    s = s.astype(str).str.strip().str.lower()
    return s.apply(lambda x: x if x.startswith(".") else (("." + x) if x else x))

def mask_token(tok: str) -> str:
    return "pat…" + tok[-4:] if tok and tok.startswith("pat") and len(tok) > 6 else "(not set)"

def validate_ahrefs_key(api_key: str) -> bool:
    """Validate Ahrefs API key format"""
    if not api_key:
        return False
    # Ahrefs keys typically start with 'ahrefs_' and are longer
    return len(api_key) > 20 and api_key.startswith("ahrefs_")

def get_gcp_credentials():
    """Safely get GCP credentials from secrets"""
    try:
        return Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], 
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
    except KeyError:
        st.error("GCP service account not configured in secrets")
        st.stop()

def safe_read_csv(file_uploader, error_msg="Error reading CSV file"):
    """Safely read CSV with proper error handling"""
    try:
        if file_uploader is None:
            return None
        return pd.read_csv(file_uploader)
    except Exception as e:
        st.error(f"{error_msg}: {str(e)}")
        return None

# -------- Ahrefs -------
def fetch_backlinks(target_url, limit, headers):
    api_url = (
        "https://api.ahrefs.com/v3/site-explorer/all-backlinks?"
        f"target={requests.utils.quote(target_url)}&limit={limit}&mode=prefix&select={requests.utils.quote(SELECT_FIELDS)}"
    )
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        backlinks = response.json().get("backlinks", [])
        for b in backlinks:
            b["target_url"] = target_url
            b["referring_domain"] = extract_domain(b.get("url_from", ""))
        return backlinks
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
            raise RuntimeError("401 AUTHENTICATION_REQUIRED — token invalid OR not allowed for this base/table.")
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
                    values.append(item["name"])
                else:
                    values.append(str(item))
        else:
            values.append(str(v))
    s = pd.Series(values, dtype="string").dropna()
    return s

@st.cache_data(show_spinner=False)
def airtable_lookup_by_values(base_id: str, table_name: str, field_name: str, values: list[str], api_key: str, batch_size: int = 25) -> pd.Series:
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Airtable: empty API key")
    url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_name)}"
    headers = {"Authorization": f"Bearer {api_key}"}

    def esc(s: str) -> str:
        # Properly escape single quotes for filterByFormula strings
        return s.replace("'", "\\'")

    out = []
    values = [v for v in values if v]
    for batch in chunk_list(values, batch_size):
        formula_parts = [f"LOWER({{{field_name}}})='{esc(v.lower())}'" for v in batch]
        formula = "OR(" + ",".join(formula_parts) + ")"
        params = {"filterByFormula": formula, "fields[]": [field_name], "maxRecords": len(batch)}
        while True:
            r = requests.get(url, headers=headers, params=params)
            if r.status_code == 401:
                raise RuntimeError("401 AUTHENTICATION_REQUIRED — token invalid OR not allowed for this base/table.")
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
if api_key and not validate_ahrefs_key(api_key):
    st.sidebar.error("Invalid Ahrefs API key format")

max_urls = st.sidebar.number_input("Max input URLs to process", min_value=1, max_value=500, value=10)
limit_backlinks = st.sidebar.number_input("Backlinks per URL (limit)", min_value=1, max_value=1000, value=20)
ahrefs_workers = st.sidebar.slider("Parallel URL workers", min_value=1, max_value=16, value=6)
ahrefs_batch_delay = st.sidebar.number_input("Ahrefs batch delay (seconds)", min_value=0.0, max_value=3.0, value=0.2, step=0.1)

st.sidebar.markdown("---")

# Airtable Toggle & Config
use_airtable = st.sidebar.checkbox("Use Airtable for filters (existing / brand flag / rejected / blocklist)", value=True)
# Read PAT from secrets (no typing each run)
AIRTABLE_PAT = (st.secrets.get("at_api_key") or "").strip()

if use_airtable:
    st.sidebar.subheader("Airtable")
    if not AIRTABLE_PAT:
        st.sidebar.error("Add your Airtable Personal Access Token to `st.secrets[\"at_api_key\"]`.")
    else:
        st.sidebar.caption(f"Using token from secrets: {mask_token(AIRTABLE_PAT)}")

    # Presets (labels -> triplets)
    EXISTING_PRESETS = {
        "Prospect-Data (appHdhjsWVRxaCvcR)": ("appHdhjsWVRxaCvcR", "tbliCOQZY9RICLsLP", "Domain"),
        "Prospect-Data-1 (appVyIiM5boVyoBhf)": ("appVyIiM5boVyoBhf", "tbliCOQZY9RICLsLP", "Domain"),
        "GDC-Database (appUoOvkqzJvyyMvC)": ("appUoOvkqzJvyyMvC", "tbliCOQZY9RICLsLP", "Domain"),
        "WB-Database (appueIgn44RaVH6ot)": ("appueIgn44RaVH6ot", "tbl3vMYv4RzKfuBf4", "Domain"),
        "Freebets-Database (appFBasaCUkEKtvpV)": ("appFBasaCUkEKtvpV", "tblmTREzfIswOuA0F", "Domain"),
    }

    st.sidebar.markdown("**Existing domains — select Airtable sources to check & EXCLUDE**")
    existing_options = list(EXISTING_PRESETS.keys())
    default_existing = [
        "Prospect-Data (appHdhjsWVRxaCvcR)",
        "GDC-Database (appUoOvkqzJvyyMvC)",
        "WB-Database (appueIgn44RaVH6ot)",
        "Freebets-Database (appFBasaCUkEKtvpV)",
    ]

    # Checkbox dropdown
    selected_existing_labels = []
    with st.sidebar.expander("Select sources", expanded=False):
        select_all = st.checkbox("Select all", value=True, key="existing_all")
        base_defaults = set(default_existing)
        for i, opt in enumerate(existing_options):
            default_val = True if select_all else (opt in base_defaults)
            if st.checkbox(opt, value=default_val, key=f"existing_{i}"):
                selected_existing_labels.append(opt)
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

    rejected_cfg = [("appTf6MmZDgouu8SN", "tbliCOQZY9RICLsLP", "Domain")] if enable_rejected else []
    blocklist_cfg = [("appJTJQwjHRaAyLkw", "tbliCOQZY9RICLsLP", "Domain")] if enable_blocklist else []

    # Speed mode for Airtable lookups
    fast_match = st.sidebar.checkbox(
        "Fast Airtable matching (query only candidates)",
        value=True,
        help="Avoids downloading entire tables by using filterByFormula to match only your candidate domains."
    )

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
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    rows.append((parts[0], parts[1], parts[2]))
            return rows

        selected_existing_cfg += parse_cfg(custom_existing_txt)
        brand_flag_cfg += parse_cfg(custom_brand_txt)
        rejected_cfg += parse_cfg(custom_rejected_txt)
        blocklist_cfg += parse_cfg(custom_blocklist_txt)

    # Connection tester — test by reading 1 record from each configured base/table
    if st.sidebar.button("Test Airtable connection"):
        if not AIRTABLE_PAT:
            st.error("No token found in secrets['at_api_key'].")
        else:
            tested = set()
            any_ok = False
            problems = []
            # Collect a few tables to try
            test_cfgs = []
            test_cfgs += selected_existing_cfg
            test_cfgs += brand_flag_cfg
            test_cfgs += rejected_cfg
            test_cfgs += blocklist_cfg
            # Fallback to one known preset if nothing selected
            if not test_cfgs:
                test_cfgs = [EXISTING_PRESETS["Prospect-Data (appHdhjsWVRxaCvcR)"]]
            headers = {"Authorization": f"Bearer {AIRTABLE_PAT}"}
            for base_id, table_id_or_name, field_name in test_cfgs:
                key = (base_id, table_id_or_name)
                if key in tested:
                    continue
                tested.add(key)
                url = f"https://api.airtable.com/v0/{base_id}/{requests.utils.quote(table_id_or_name)}"
                params = {"maxRecords": 1, "fields[]": [field_name]}
                r = requests.get(url, headers=headers, params=params)
                if r.status_code == 200:
                    any_ok = True
                    st.success(f"✅ OK for base {base_id}, table '{table_id_or_name}'")
                elif r.status_code == 404:
                    problems.append(f"404 for base {base_id}, table '{table_id_or_name}' — table/field name may be wrong or you lack access.")
                elif r.status_code in (401, 403):
                    problems.append(f"{r.status_code} for base {base_id} — token lacks access to this base.")
                else:
                    problems.append(f"{r.status_code} for base {base_id}, table '{table_id_or_name}': {r.text}")
            if problems:
                st.error("Some checks failed:\n" + "\n".join(problems))
            if any_ok:
                st.info("If some bases failed, add those base IDs to the PAT's allowed bases and ensure scope `data.records:read`.")

else:
    # Legacy local-file flow (no Airtable)
    gambling_file = st.sidebar.file_uploader("Upload Gambling Domains CSV", type=["csv"])
    outreach_file = st.sidebar.file_uploader("Upload Already Outreach CSV", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("🛑 If not using Airtable, keep the Google Sheet/CSV uploads here.")

# --- TLD Blocklist (CSV or Excel) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 TLD Blocklist (optional)")
tld_block_file = st.sidebar.file_uploader(
    "Upload TLD blocklist (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="First column should contain TLDs like .de, .ru or de, ru",
)

# --------------------------
# Google Sheets Export UI
# --------------------------
st.sidebar.markdown("---")
st.sidebar.header("🧾 Google Sheets Export")
gs_key_or_url = st.sidebar.text_input("Spreadsheet URL or key", help="Paste full URL or just the spreadsheet key")
gs_worksheet = st.sidebar.text_input("Worksheet name", value="Outreach")
export_mode = st.sidebar.selectbox("Write mode", ["Replace sheet (overwrite)", "Append rows"], index=1)
export_content = st.sidebar.radio("Export content", ["Domains only", "Full results (all columns)"], index=0)

export_ready = (
    bool(gs_key_or_url.strip())
    and isinstance(st.session_state.get("df_merged"), pd.DataFrame)
    and not st.session_state["df_merged"].empty
)
export_button = st.sidebar.button("Export now", disabled=not export_ready, use_container_width=True)

if not export_ready:
    with st.sidebar:
        if not gs_key_or_url.strip():
            st.caption("↖ Paste your Google Sheet URL or key to enable export.")
        elif not isinstance(st.session_state.get("df_merged"), pd.DataFrame):
            st.caption("Run the analysis first — nothing to export yet.")
        else:
            st.caption("No rows to export.")

# Google Sheets helpers
def _open_sheet(key_or_url: str):
    if not key_or_url:
        raise RuntimeError("Spreadsheet URL/key is empty.")
    creds = get_gcp_credentials()  # Use the safe function
    gc = gspread.authorize(creds)
    if "http" in key_or_url:
        return gc.open_by_url(key_or_url)
    return gc.open_by_key(key_or_url)

def _get_or_create_worksheet(sh, title: str, cols: int = 26):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=1000, cols=cols)

def _update_replace(ws, df: pd.DataFrame):
    values = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()
    ws.clear()
    ws.update("A1", values)

def _append_rows(ws, df: pd.DataFrame):
    try:
        existing = ws.get_all_values()
    except Exception:
        existing = []
    if not existing:
        ws.update("A1", [df.columns.tolist()])
    ws.append_rows(df.astype(str).fillna("").values.tolist(), value_input_option="RAW")

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
    df_input = safe_read_csv(input_file, "Error reading input URLs CSV")
    if df_input is None:
        st.stop()
    
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
        if not AIRTABLE_PAT:
            st.error("⚠️ Please add `at_api_key` to Streamlit secrets.")
            st.stop()

        try:
            cand = list({to_domain_only(d) for d in df_merged["referring_domain"].astype(str).tolist()})

            # Existing domains to EXCLUDE
            if selected_existing_cfg:
                if fast_match:
                    existing_series = []
                    for (base_id, table_id, field_name) in selected_existing_cfg:
                        s = airtable_lookup_by_values(base_id, table_id, field_name, cand, AIRTABLE_PAT)
                        existing_series.append(s)
                    existing_series = pd.concat(existing_series, ignore_index=True) if existing_series else pd.Series([], dtype="string")
                else:
                    existing_records = []
                    for base_id, table_id, field_name in selected_existing_cfg:
                        recs = airtable_fetch_all(base_id, table_id, AIRTABLE_PAT, fields=[field_name])
                        existing_records += recs
                    existing_series = airtable_column_to_series(existing_records, selected_existing_cfg[0][2]) if selected_existing_cfg else pd.Series([], dtype="string")
            else:
                existing_series = pd.Series([], dtype="string")
            existing_domains = set(existing_series.map(to_domain_only).dropna().unique())

            # Brand flag
            if brand_flag_cfg:
                if fast_match:
                    brand_series = []
                    for (base_id, table_id, field_name) in brand_flag_cfg:
                        s = airtable_lookup_by_values(base_id, table_id, field_name, cand, AIRTABLE_PAT)
                        brand_series.append(s)
                    brand_series = pd.concat(brand_series, ignore_index=True) if brand_series else pd.Series([], dtype="string")
                else:
                    brand_records = []
                    for base_id, table_id, field_name in brand_flag_cfg:
                        recs = airtable_fetch_all(base_id, table_id, AIRTABLE_PAT, fields=[field_name])
                        brand_records += recs
                    brand_series = airtable_column_to_series(brand_records, brand_flag_cfg[0][2]) if brand_flag_cfg else pd.Series([], dtype="string")
            else:
                brand_series = pd.Series([], dtype="string")
            brand_domains = set(brand_series.map(to_domain_only).dropna().unique())

            # Rejected & Blocklist
            def gather(cfgs):
                if not cfgs:
                    return set()
                if fast_match:
                    ser_list = []
                    for (base_id, table_id, field_name) in cfgs:
                        s = airtable_lookup_by_values(base_id, table_id, field_name, cand, AIRTABLE_PAT)
                        ser_list.append(s)
                    ser = pd.concat(ser_list, ignore_index=True) if ser_list else pd.Series([], dtype="string")
                else:
                    recs_all = []
                    for base_id, table_id, field_name in cfgs:
                        recs_all += airtable_fetch_all(base_id, table_id, AIRTABLE_PAT, fields=[field_name])
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
        if 'gambling_file' in locals() and gambling_file is not None:
            df_compare = safe_read_csv(gambling_file, "Error reading gambling domains CSV")
            if df_compare is not None:
                gambling_domains = df_compare.iloc[:, 0].dropna().str.strip().str.lower().unique()
                df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
                df_merged["Found in Gambling.com"] = df_merged["referring_domain"].isin(gambling_domains)
                df_merged["Found in Gambling.com"] = df_merged["Found in Gambling.com"].apply(lambda x: "TRUE" if x else "FALSE")
                st.success("🏷️ Gambling.com flag added")

        if 'outreach_file' in locals() and outreach_file is not None:
            df_outreach = safe_read_csv(outreach_file, "Error reading outreach CSV")
            if df_outreach is not None:
                if "opportunity" in df_outreach.columns:
                    outreached_domains = df_outreach["opportunity"].dropna().str.strip().str.lower().unique()
                    df_merged["referring_domain"] = df_merged["referring_domain"].str.lower()
                    df_merged = df_merged[~df_merged["referring_domain"].isin(outreached_domains)]
                    st.success("🚫 Removed already outreached domains")
                else:
                    st.warning("⚠️ 'opportunity' column not found in uploaded outreach file.")

    # --- TLD Blocklist (works in both modes) ---
    blocked_tlds = set()
    if tld_block_file is not None:
        try:
            ext = os.path.splitext(tld_block_file.name)[1].lower()
            if ext == ".csv":
                tld_df = pd.read_csv(tld_block_file)
            else:
                tld_df = pd.read_excel(tld_block_file)
            tld_series = normalize_tld_series(tld_df.iloc[:, 0])
            blocked_tlds = set(tld_series.dropna().unique())
        except Exception as e:
            st.warning(f"Couldn't read TLD blocklist: {e}")

    if blocked_tlds:
        df_merged["__tld"] = df_merged["referring_domain"].astype(str).apply(extract_tld)
        before_tld = len(df_merged)
        df_merged = df_merged[~df_merged["__tld"].isin(blocked_tlds)].copy()
        df_merged.drop(columns="__tld", inplace=True)
        st.success(f"🔻 Blocked TLDs filtered out: {before_tld - len(df_merged)} removed")

    # ---- Final Output ----
    st.session_state["df_merged"] = df_merged.copy()
    st.success(f"✅ Analysis complete. {len(df_merged)} rows ready.")

# === Results panel (always visible if we have results) ===
_last = st.session_state.get("df_merged")
if isinstance(_last, pd.DataFrame) and not _last.empty:
    st.divider()
    st.subheader("Results")
    st.dataframe(_last.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Latest CSV",
        _last.to_csv(index=False),
        file_name="ahrefs_backlinks_flagged.csv",
        mime="text/csv",
        key="download_latest_csv",
    )
else:
    st.info("Run the analysis to generate results before downloading/exporting.")

# ==========================
# Export to Google Sheets
# ==========================
if export_button:
    try:
        df_out = st.session_state.get("df_merged")
        if not isinstance(df_out, pd.DataFrame) or df_out.empty:
            st.error("Run the analysis first — nothing to export yet.")
        else:
            if export_content == "Domains only":
                df_out = (
                    df_out[["referring_domain"]]
                    .rename(columns={"referring_domain": "Domain"})
                    .dropna()
                    .drop_duplicates()
                    .sort_values("Domain")
                    .reset_index(drop=True)
                )

            sh = _open_sheet(gs_key_or_url)
            ws = _get_or_create_worksheet(sh, gs_worksheet, cols=max(26, len(df_out.columns) + 2))

            if export_mode.startswith("Replace"):
                _update_replace(ws, df_out)
                st.success(f"Exported {len(df_out)} rows (replaced sheet).")
            else:
                _append_rows(ws, df_out)
                st.success(f"Appended {len(df_out)} rows.")
    except Exception as e:
        st.error(f"Google Sheets export failed: {e}")

# === Pitchbox Integration (Bulk Upload with JWT) ===
st.sidebar.header("📤 Pitchbox Upload (Bulk)")
pb_api_key = st.sidebar.text_input("Pitchbox API JWT", type="password")
pb_campaign_id = st.sidebar.text_input("Pitchbox Campaign ID")
upload_button = st.sidebar.button("Upload to Pitchbox")

if upload_button:
    if not pb_api_key or not pb_campaign_id:
        st.error("⚠️ Please enter both API key (JWT) and campaign ID.")
    elif not isinstance(st.session_state.get("df_merged"), pd.DataFrame):
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
