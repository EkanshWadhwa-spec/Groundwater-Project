"""
Public USGS National Water Information System (NWIS) Open Data Fetcher
Dataset Source: https://waterdata.usgs.gov/nwis (OPEN-DATA-USGS)

Provides reproducible download helpers for publicly accessible USGS Water Services
REST endpoints (site metadata and groundwater level time-series) with automatic
fallback to the curated local CSV snapshots in `data/` for offline environments.
"""

import io
import os
from typing import List, Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import pandas as pd

USGS_SITE_ENDPOINT = "https://waterservices.usgs.gov/nwis/site/"
USGS_GWLEVELS_ENDPOINT = "https://waterservices.usgs.gov/nwis/gwlevels/"
DEFAULT_SITE_IDS = [
    "01646500",
    "342500118250001",
    "293830098255501",
    "405012100123401",
    "280255082260501",
    "332450111584001",
    "364500119300001",
    "292512098450101",
    "412530096120001",
    "285500081250001",
]


def _parse_usgs_rdb(rdb_text: str) -> pd.DataFrame:
    """Parses USGS tab-delimited RDB responses (skipping '#' comment and format rows)."""
    lines = [ln for ln in rdb_text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    # Line 0 = column headers, Line 1 = RDB field-width specifiers, Line 2+ = data
    data_text = "\n".join([lines[0]] + lines[2:])
    return pd.read_csv(io.StringIO(data_text), sep="\t", dtype={"site_no": str})


def fetch_nwis_sites(site_ids: Optional[List[str]] = None, timeout: int = 10) -> Dict[str, Any]:
    """
    Attempts to download live site metadata from USGS NWIS (`format=rdb`).
    Returns a status dict with `source` ('usgs_live' or 'offline_cache') and `records`.
    """
    sites = site_ids or DEFAULT_SITE_IDS
    sites_param = ",".join(sites)
    url = f"{USGS_SITE_ENDPOINT}?format=rdb&sites={sites_param}&siteOutput=expanded"
    try:
        req = Request(url, headers={"User-Agent": "OPEN-DATA-USGS-Groundwater-Assistant/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            rdb_text = resp.read().decode("utf-8", errors="replace")
        df = _parse_usgs_rdb(rdb_text)
        if not df.empty:
            return {"status": "ok", "source": "usgs_live", "url": url, "dataframe": df}
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        return {"status": "fallback", "source": "offline_cache", "url": url, "error": str(exc), "dataframe": pd.DataFrame()}

    return {"status": "fallback", "source": "offline_cache", "url": url, "error": "Empty USGS RDB response", "dataframe": pd.DataFrame()}


def fetch_nwis_gwlevels(site_no: str, period: str = "P730D", timeout: int = 10) -> Dict[str, Any]:
    """
    Attempts to download groundwater level measurements from USGS NWIS (`gwlevels`, JSON format).
    """
    url = f"{USGS_GWLEVELS_ENDPOINT}?format=json&sites={site_no}&period={period}"
    try:
        req = Request(url, headers={"User-Agent": "OPEN-DATA-USGS-Groundwater-Assistant/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return {"status": "ok", "source": "usgs_live", "url": url, "raw_json": raw}
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        return {"status": "fallback", "source": "offline_cache", "url": url, "error": str(exc)}


def get_dataset_provenance(data_dir: str) -> Dict[str, Any]:
    """Returns metadata describing the OPEN-DATA-USGS local snapshots and public endpoints."""
    return {
        "academic_id": "OPEN-DATA-USGS",
        "portal_url": "https://waterdata.usgs.gov/nwis",
        "site_api": USGS_SITE_ENDPOINT,
        "gwlevels_api": USGS_GWLEVELS_ENDPOINT,
        "local_files": [
            f for f in [
                "usgs_groundwater_sites.csv",
                "usgs_water_levels_timeseries.csv",
                "usgs_water_quality.csv",
                "training_queries.csv",
                "benchmark_test_queries.json",
            ]
            if os.path.exists(os.path.join(data_dir, f))
        ],
        "restricted_data_required": False,
    }
