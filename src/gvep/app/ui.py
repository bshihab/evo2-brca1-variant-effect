"""Milestone 6 — Streamlit demo UI.

A clickable front-end over the explanation layer: enter a BRCA1 variant and see the
zero-shot score, a calibrated probability, the prediction, and — crucially — the
honest, category-aware confidence caveat. Plus a VUS-prioritization view that ranks
uncertain variants for triage.

Run:  streamlit run src/gvep/app/ui.py
(For simplicity this calls gvep.explain directly; the same logic is exposed over HTTP
in gvep/app/api.py.)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gvep.explain import _context, explain_variant, prioritize

st.set_page_config(page_title="BRCA1 Variant Effect (research/triage)", layout="wide")

# Tier -> Streamlit status renderer
_TIER_STYLE = {
    "moderate–high": st.success,
    "moderate": st.info,
    "low": st.warning,
    "very low": st.error,
    "not assessable": st.warning,
}


def _show_explanation(e: dict) -> None:
    prot = f" · {e['protein_variant']}" if e["protein_variant"] else ""
    st.subheader(f"{e['variant']}  ({e['gene']}, {e['consequence']}{prot})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted P(disruptive)", f"{e['prob_disruptive']:.0%}")
    c2.metric("Prediction", e["prediction"].split(" (")[0])
    c3.metric("Evo 2 Δ score", f"{e['evo2_delta']:+.5f}")

    render = _TIER_STYLE.get(e["trust_tier"], st.warning)
    fpr = e["category_fpr_at_90_sens"]
    fpr_txt = f" Category false-alarm rate at high sensitivity: {fpr:.0%}." \
        if isinstance(fpr, (int, float)) else ""
    render(f"**Confidence: {e['trust_tier'].upper()}** — {e['trust_caveat']}.{fpr_txt}")

    c4, c5 = st.columns(2)
    c4.markdown(f"**ClinVar:** {e['clinvar']}")
    c5.markdown(f"**Findlay experiment (benchmark truth):** {e['findlay_experimental_class']}")


def main() -> None:
    st.title("🧬 BRCA1 Variant Effect Prediction")
    st.caption("Zero-shot Evo 2 scoring with honest, category-aware confidence.")
    st.error("⚠️ **Research / triage-prioritization prototype — NOT a clinical "
             "diagnostic.** Every output is an AI prediction with documented, "
             "category-specific limitations. Do not use for any medical decision.")

    df, _, _ = _context()
    tab1, tab2 = st.tabs(["🔎 Explain a variant", "📋 VUS prioritization"])

    with tab1:
        st.markdown("Enter a BRCA1 SNV (GRCh37 / hg19, chromosome 17).")
        examples = {
            "— pick an example —": None,
            "Splice variant, model confident & correct": (41267740, "T", "A"),
            "Pathogenic start-codon (model MISSES it)": (41276111, "C", "G"),
            "Benign synonymous": (41276108, "A", "G"),
            "Nonsense / stop-gain (can't be graded)": (41197777, "C", "T"),
        }
        pick = st.selectbox("Example variants", list(examples))
        d = examples[pick] or (41276111, "C", "G")
        col = st.columns(4)
        pos = col[0].number_input("Position (hg19)", value=int(d[0]), step=1)
        ref = col[1].text_input("Ref allele", value=d[1]).upper()
        alt = col[2].text_input("Alt allele", value=d[2]).upper()
        col[3].markdown("&nbsp;")
        if col[3].button("Explain", type="primary"):
            e = explain_variant(int(pos), ref, alt)
            if e is None:
                st.warning(f"chr17:{pos} {ref}>{alt} is not in the scored BRCA1 set "
                           "(only the ~3,900 Findlay benchmark variants are available "
                           "in this demo).")
            else:
                _show_explanation(e)

    with tab2:
        st.markdown("Variants of Uncertain Significance (from ClinVar) that we have "
                    "Evo 2 scores for, **ranked by predicted disruptiveness** — a triage "
                    "queue for expert review. Note the confidence column: many top hits "
                    "are missense, where the model is least reliable.")
        top = st.slider("How many to show", 5, 50, 20)
        ranked = prioritize(top=top)
        if ranked:
            table = pd.DataFrame([{
                "variant": e["variant"],
                "consequence": e["consequence"],
                "P(disruptive)": round(e["prob_disruptive"], 3),
                "prediction": e["prediction"].split(" (")[0],
                "confidence": e["trust_tier"],
                "ClinVar": e["clinvar"],
            } for e in ranked])
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("No scored ClinVar VUS available.")


main()
