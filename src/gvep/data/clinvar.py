"""ClinVar BRCA1 slice (incl. Variants of Uncertain Significance / VUS).

ClinVar is the public archive of clinically interpreted human variants. We pull a slice
of BRCA1 SNVs with their clinical significance — especially the VUS, which are the whole
point of a triage tool (we want to see how the model scores variants experts can't yet
classify). This is fetched "for later" (the explanation layer, Milestone 5).

Source: NCBI E-utilities (esearch + esummary, JSON). We extract, per variant:
clinical significance, review status, and hg19/GRCh37 coordinates (to match Findlay).
Alleles come from the SPDI string (ref/alt are identical across builds for an SNV).

Network failures here are non-fatal: ClinVar is supplementary to Milestone 1.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from gvep.config import CACHE_DIR

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# SNVs in BRCA1. "single nucleotide variant"[Type of variation] keeps it to SNVs.
SEARCH_TERM = 'BRCA1[gene] AND "single nucleotide variant"[Type of variation]'
BATCH = 200

# A variant is a VUS if its clinical significance is "uncertain".
_VUS_TOKENS = ("uncertain",)


def _email() -> str:
    return os.environ.get("NCBI_EMAIL", "anonymous@example.com")


def _get(endpoint: str, params: dict) -> requests.Response:
    params = {**params, "email": _email(), "tool": "gvep"}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    resp = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=60)
    resp.raise_for_status()
    return resp


def _parse_record(rec: dict) -> dict | None:
    """Pull the fields we want out of one ClinVar esummary docsum."""
    # Clinical significance lives under different keys across API versions.
    sig_block = rec.get("germline_classification") or rec.get("clinical_significance") or {}
    clin_sig = (sig_block.get("description") or "").strip()
    review = (sig_block.get("review_status") or "").strip()

    vset = rec.get("variation_set") or []
    if not vset:
        return None
    v = vset[0]
    name = v.get("variation_name") or rec.get("title", "")

    # ref/alt from canonical SPDI: "NC_000017.11:43044294:G:A" -> ref=G, alt=A
    ref = alt = None
    spdi = v.get("canonical_spdi", "")
    parts = spdi.split(":")
    if len(parts) == 4 and len(parts[2]) == 1 and len(parts[3]) == 1:
        ref, alt = parts[2].upper(), parts[3].upper()

    # GRCh37 position from the location list
    pos_hg19 = chrom = None
    for loc in v.get("variation_loc", []):
        if loc.get("assembly_name") == "GRCh37":
            pos_hg19 = loc.get("start")
            chrom = loc.get("chr")
            break

    if not (ref and alt and pos_hg19):
        return None

    return {
        "clinvar_id": rec.get("uid", ""),
        "name": name,
        "chrom": str(chrom),
        "pos": int(pos_hg19),
        "ref": ref,
        "alt": alt,
        "clin_sig": clin_sig,
        "review_status": review,
        "is_vus": any(t in clin_sig.lower() for t in _VUS_TOKENS),
    }


def fetch_clinvar_brca1(max_records: int = 5000) -> pd.DataFrame:
    """Fetch a BRCA1 SNV slice from ClinVar. Returns empty DataFrame on failure."""
    cols = ["clinvar_id", "name", "chrom", "pos", "ref", "alt",
            "clin_sig", "review_status", "is_vus"]
    try:
        ids_json = _get("esearch.fcgi", {
            "db": "clinvar", "term": SEARCH_TERM,
            "retmax": max_records, "retmode": "json",
        }).json()
        uids = ids_json.get("esearchresult", {}).get("idlist", [])
        print(f"[gvep] ClinVar: {len(uids):,} BRCA1 SNV records found")

        rows: list[dict] = []
        for i in range(0, len(uids), BATCH):
            chunk = uids[i : i + BATCH]
            summ = _get("esummary.fcgi", {
                "db": "clinvar", "id": ",".join(chunk), "retmode": "json",
            }).json()
            result = summ.get("result", {})
            for uid in result.get("uids", []):
                parsed = _parse_record(result[uid])
                if parsed:
                    rows.append(parsed)
            time.sleep(0.34)  # be polite to NCBI (≈3 req/s without an API key)

        df = pd.DataFrame(rows, columns=cols)
        # Keep BRCA1's chromosome and well-formed SNVs only.
        df = df[df["chrom"] == "17"].reset_index(drop=True)
        # Cache the raw search for reproducibility/debugging.
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "clinvar_uids.json").write_text(json.dumps(uids))
        print(f"[gvep] ClinVar: {len(df):,} usable SNVs, {int(df['is_vus'].sum()):,} VUS")
        return df
    except Exception as exc:  # noqa: BLE001 — supplementary data, never fatal
        print(f"[gvep] WARNING: ClinVar fetch failed ({exc}). Skipping (non-fatal).")
        return pd.DataFrame(columns=cols)
