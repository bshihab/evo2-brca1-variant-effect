# RESULTS — Validation & Honesty Layer (Milestone 3)

> **Not a clinical diagnostic.** These are research findings about a zero-shot model's
> behavior on one gene (BRCA1), using the **budget 1B** Evo 2 model. Read the limitations.

This is the heart of the project. The headline AUROC of ~0.74 is real — but the whole point
of this milestone is to look *underneath* it and report, honestly, where the model works and
where it fails. The short version: **the aggregate number flatters the model, and a rigorous
breakdown shows the 1B zero-shot predictor is not competitive with established tools on this
benchmark — though it has a characteristic strength (splice variants) and clear, explainable
failure modes.**

All numbers from `results/metrics/honesty_metrics.json`; figures in `results/figures/m3_*.png`.

---

## 1. Headline metrics

| Metric | Value | Reference |
|---|---|---|
| AUROC (LOF vs FUNC) | **0.737** (95% CI 0.715–0.757) | published Evo 2 1B ≈ 0.73 ✅ |
| AUROC (LOF vs FUNC+INT) | 0.729 | — |
| AUPRC (LOF vs FUNC) | 0.564 | random baseline = 0.226 |

We reproduced the published 1B number, which validates the whole pipeline. AUPRC (0.564) is
well above the 0.226 random baseline but far from 1.0 — a first hint that real-world precision
is limited. **See `m3_roc_pr.png`.**

---

## 2. The aggregate hides the most important weakness

The single biggest honesty finding. Performance is **not uniform** across variant types
(`m3_by_consequence.png`):

| Consequence | n | % LOF | AUROC | Note |
|---|---|---|---|---|
| Splice region | 414 | 25% | **0.852** | Evo 2's relative strength |
| Intronic | 465 | 2% | 0.629 | weak |
| **Missense** | **1,917** | **23%** | **0.604** | **weak — and this is the category that matters most** |
| Nonsense | 136 | 100% | n/a | only one class present |
| Canonical splice | 135 | 95% | n/a | only one class present |
| Synonymous | 532 | 1% | n/a | too few LOF |
| 5′ UTR | 45 | 0% | n/a | no LOF |

Two things to take from this:

1. **The headline 0.74 is partly "category detection," not "harm detection."** The model
   correctly gives near-zero scores to the overwhelmingly-benign intronic/synonymous variants
   and alarming scores to the overwhelmingly-harmful nonsense/canonical-splice variants. That
   easy between-category structure inflates the pooled AUROC. **Within the hard, clinically
   central category — missense (the bulk of real VUS) — AUROC collapses to 0.604**, barely
   above a coin flip.
2. **In several categories AUROC can't even be computed**, because the experiment contains only
   one class there (e.g. every Nonsense variant is LOF). A single aggregate number silently
   papers over this.

**Clinical meaning:** the variants a clinician most needs help with are uncertain *missense*
changes. That is exactly where this model is weakest.

---

## 3. At a useful sensitivity, precision is poor (class-imbalance honesty)

A triage tool must catch most harmful variants, so we fix **sensitivity at 90%** and look at the
cost. Because LOF is the minority class (22.6%), the picture is sobering:

- Overall **false-positive rate = 77%**
- **Precision = 25.4%** → **~2.9 false alarms for every true LOF caught.**

So "AUROC 0.74" coexists with "to catch 90% of harmful variants, 3 of every 4 flags are false."
This is the concrete way aggregate AUROC can mislead on imbalanced data. False-positive rates are
high across *every* category (57–81%, `m3_by_consequence.png`, right panel).

---

## 4. Severity failure mode: worse on the subtle variants

We split LOF into **severe** vs **mild** (by experimental function score) and re-measured
separation from FUNC (`m3_severity_benchmark.png`, left):

| LOF severity | AUROC vs FUNC |
|---|---|
| Severe (n=412) | 0.775 |
| Mild (n=411) | 0.698 |

Performance **degrades by 0.077 on mild variants** — the model is meaningfully better at
flagging obviously-broken variants than subtle ones. Unfortunately, subtle/partial loss of
function is often the clinically ambiguous case where help is most needed.

---

## 5. Calibration

Raw delta scores are **not** probabilities (they're tiny per-token log-likelihood differences).
After a cross-validated logistic recalibration, the reliability diagram (`m3_calibration.png`) is
reasonable (**Brier = 0.175**), but the takeaway is that **the raw score must be recalibrated
before it can be read as a "probability of pathogenic"** — an important caveat for any downstream
use or explanation layer.

---

## 6. Benchmark vs established tools — the humbling result

Using the predictor scores bundled in the Findlay dataset (`m3_severity_benchmark.png`, right):

| Predictor | AUROC (all) | AUROC (missense only) |
|---|---|---|
| **Evo 2 1B (Δ, zero-shot)** | **0.737** | **0.604** |
| CADD | 0.817 | 0.756 |
| phyloP (conservation) | 0.793 | 0.737 |
| SIFT | — | 0.708 |
| PolyPhen-2 | — | 0.669 |

**On this benchmark, the Evo 2 1B zero-shot score is beaten by CADD and even by plain
evolutionary conservation (phyloP)** — and on missense specifically it trails *every* established
tool. That is an honest, uncomfortable result, and it's important to report it rather than bury it.

**Fair context (caveats that cut both ways):**
- **We used the smallest (1B) model** for cost reasons. The published Evo 2 **40B** scores
  substantially higher on BRCA1 (~0.87+ in the paper); much of this gap is likely model size.
- **Evo 2 is zero-shot**; CADD/SIFT/PolyPhen-2 are **supervised** (trained on labels) and may
  carry some circularity advantage on this kind of benchmark. phyloP, however, is an unsupervised
  conservation score — and it still beats the 1B, which is the most sobering comparison.
- Evo 2's *conceptual* advantage is non-coding/splice reasoning, and indeed its **best category
  is splice region (0.852)** — but on this particular dataset CADD also handles those well.

---

## What this means (honest bottom line)

- **As a standalone triage tool, the Evo 2 1B zero-shot score is not ready.** At clinically
  useful sensitivity it produces ~3 false alarms per true hit, it is weakest on the missense VUS
  that matter most, and it underperforms cheaper, established predictors on this benchmark.
- **The headline AUROC was misleading on its own** — it was inflated by easy between-category
  structure and hid poor minority-class precision, a missense weakness, and a severity-dependent
  failure mode. Surfacing those is the value of this milestone.
- **The method is not without merit:** it reproduces the literature, needs *zero* task labels, and
  shows a real, explainable strength on splice variants — pointing to where a larger model (7B/40B)
  or an embedding-based classifier (Milestone 4) might genuinely add value.

The point of a research/triage prototype is to know *exactly* how much to trust it, per category.
This milestone delivers that — including the parts that aren't flattering.
