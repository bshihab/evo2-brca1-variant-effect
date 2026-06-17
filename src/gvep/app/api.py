"""Milestone 6 — FastAPI backend.

Exposes the explanation layer as a small HTTP API:
  GET  /health                       -> liveness check
  GET  /explain?pos=&ref=&alt=       -> trust-aware explanation for one variant
  GET  /prioritize?top=              -> VUS triage: ClinVar uncertain variants ranked
                                        by predicted disruptiveness

This wraps gvep.explain (no GPU needed — it serves the variants we already scored).
Run:  uvicorn gvep.app.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from gvep.explain import explain_variant, prioritize

app = FastAPI(
    title="BRCA1 Variant Effect — research/triage prototype",
    description="Zero-shot Evo 2 pathogenicity scoring with honest, category-aware "
    "confidence. NOT a clinical diagnostic.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disclaimer": "research/triage prototype, not a diagnostic"}


@app.get("/explain")
def explain(pos: int, ref: str, alt: str) -> dict:
    """Trust-aware explanation for chr17:<pos> <ref>><alt> (GRCh37)."""
    e = explain_variant(pos, ref, alt)
    if e is None:
        raise HTTPException(
            status_code=404,
            detail=f"chr17:{pos} {ref}>{alt} is not in the scored BRCA1 set.",
        )
    return e


@app.get("/prioritize")
def prioritize_endpoint(top: int = 25) -> dict:
    """Rank ClinVar Variants of Uncertain Significance by predicted disruptiveness."""
    ranked = prioritize(top=top)
    return {"count": len(ranked), "variants": ranked}
