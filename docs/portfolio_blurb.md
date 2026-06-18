# Portfolio blurbs

Copy-paste descriptions of this project for a résumé, LinkedIn, or a portfolio site.
Repo: https://github.com/bshihab/evo2-brca1-variant-effect

---

## One-liner

Zero-shot pathogenicity prediction for BRCA1 cancer-risk variants using the Evo 2 genomic
foundation model — with a rigorous, honest evaluation layer and a triage demo app.

---

## Résumé bullets

**Genomic Variant Effect Prediction (BRCA1) — Python, PyTorch, Evo 2, Modal, FastAPI/Streamlit**
- Built an end-to-end zero-shot variant-effect pipeline using the Evo 2 DNA foundation model;
  **reproduced the published benchmark (AUROC 0.74)** on ~3,900 experimentally-labeled BRCA1 variants.
- Engineered a resilient cloud-GPU workflow on Modal (FP8 on an L4; server-side checkpointing and
  `--detach` to survive budget caps and dropped connections).
- Authored a rigorous evaluation/"honesty" layer — stratified AUROC, AUPRC, calibration, and
  benchmarking vs. CADD/SIFT/PolyPhen-2/phyloP — that **exposed limitations the headline metric
  hid** (e.g. near-chance performance on missense variants; ~3 false alarms per true hit).
- Shipped a triage demo (FastAPI + Streamlit) giving per-variant **category-aware confidence**,
  flagging predictions the model can't be trusted to make.

---

## LinkedIn / portfolio paragraph

I built a research prototype that uses **Evo 2**, a genomic foundation model, to predict whether
single-letter DNA changes in the **BRCA1** breast-cancer gene are harmful — the kind of "Variant of
Uncertain Significance" that leaves patients without answers. Using zero-shot delta-likelihood
scoring, it reproduced the published benchmark (AUROC ~0.74) on ~3,900 experimentally-validated
variants.

The part I'm most proud of isn't the headline number — it's the **honesty layer**. I broke that
0.74 apart to show where the model actually fails: it's near-chance on the missense variants that
matter most clinically, it produces ~3 false alarms per true hit at usable sensitivity, and it's
beaten by older tools like CADD on this benchmark. The demo turns this into **per-variant,
category-aware confidence** — it will tell you when *not* to trust its own prediction. Along the
way I learned cloud-GPU engineering (Modal, FP8, resilient/resumable jobs) and the evaluation
rigor (calibration, class-imbalance, honest benchmarking) that separates a real tool from a demo.

> ⚠️ Research/triage prototype — explicitly **not** a clinical diagnostic.

---

## Talking points (for interviews)

- *"Why is the headline AUROC misleading?"* → between-category structure inflates it; missense
  alone is 0.60. (Shows you understand stratified evaluation.)
- *"How do you know it's reliable?"* → you don't, per category — that's why confidence is reported
  per category and some categories are "not assessable." (Shows responsible ML framing.)
- *"What was the hardest part?"* → resilient cloud-GPU engineering under a budget cap; designing
  for failure (checkpointing, server-side persistence). (Shows real systems experience.)
