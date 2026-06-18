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

Raw delta scores are **not** probabilities (they're tiny per-token log-likelihood differences,
~1e-3). They must be recalibrated before they can be read as a "probability of pathogenic" — and
the recalibration must **standardize** the feature first, or the logistic collapses to the base
rate (a subtle bug we caught: its Brier equalled the no-skill prevalence baseline exactly).

After a *properly scaled*, cross-validated logistic recalibration, the reliability diagram
(`m3_calibration.png`) tracks the diagonal well across the full 0–1 range (**Brier = 0.142** vs a
no-skill baseline of 0.175). So the recalibrated probability is usable — but only as a rough,
gene-specific estimate, and the raw score alone is not interpretable as a probability.

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

---

# Milestone 4 — Can a supervised classifier on embeddings beat zero-shot?

**Short answer: no — and the rigorous evaluation is what reveals it.**

Instead of Evo 2's delta *score*, we extracted its internal **embedding** (1,920-dim) at the
variant position (layer `blocks.20.mlp.l3`) and used the **(variant − reference) difference** as
features. We trained a logistic regression and a small neural net (PCA-50 front-end), evaluated
with **grouped 5-fold cross-validation by genomic position** (so the test fold contains positions
never seen in training — no memorizing positions), and compared to the zero-shot baseline on the
same 3,644 variants. (`m4_classifier.png`.)

| Method | AUROC (grouped CV) | vs zero-shot |
|---|---|---|
| **Zero-shot Δ score** | **0.737** | — |
| Logistic regression on embeddings | 0.605 | −0.131 |
| Neural net on embeddings | 0.620 | −0.117 |
| (missense only) zero-shot 0.604 → emb 0.567 | | −0.037 |

**The overfitting check is the lesson.** The neural net reached **train AUROC = 1.000 but CV =
0.620** — an overfit gap of **0.38**. A naive evaluation (reporting train accuracy, or
non-grouped CV that leaks positions) would have shown a *spectacular-looking* number and been
completely misleading. Honest, leakage-free evaluation exposed that the model memorized rather
than generalized.

**Why no gain here (interpretation):**
- The honest task — generalize to *entirely new positions* — is hard with limited within-gene data.
- A single-position embedding-difference may be a weak feature; the zero-shot delta is already a
  strong, distilled signal that's tough to beat with a light classifier.
- With 1,920 features and ~1,300 position-groups, models overfit before they generalize.

**Caveats (this is one setup, not a verdict on embeddings):** a richer feature (mean-pooling, or
multiple layers), the larger 7B/40B model, or training across *many* genes could change the
result. What this milestone shows cleanly is the **methodology**: a fancier model is not
automatically better, and only honest cross-validation tells you the truth. **For this prototype,
the simple zero-shot score remains the better predictor.**

---

# Milestone 7 (stretch) — Evo 2 vs AlphaMissense

We added **AlphaMissense** (Google DeepMind's state-of-the-art *missense* predictor) to the
benchmark on the BRCA1 missense variants — the category where Evo 2 1B was weakest. AlphaMissense
scores come from its public hg19 release (Zenodo 8360242), filtered to the BRCA1 region.
1,908 of our 1,917 missense variants had an AlphaMissense score. (`m7_alphamissense.png`.)

| Predictor | AUROC (BRCA1 missense, LOF vs FUNC) |
|---|---|
| **AlphaMissense** | **0.904** |
| CADD | 0.757 |
| phyloP | 0.739 |
| SIFT | 0.707 |
| PolyPhen-2 | 0.664 |
| **Evo 2 1B (zero-shot)** | **0.608** |

**AlphaMissense wins decisively (0.90 vs 0.61)** — and it's worth being clear-eyed about why,
in both directions:

- **Why AlphaMissense wins:** it's a *supervised, missense-specialized* model trained on enormous
  data with protein-structure signal — exactly engineered for this task. On its home turf, it
  should win, and it does, by a wide margin.
- **The fair counterpoint (scope):** AlphaMissense **only scores missense variants.** It cannot
  touch the non-coding, intronic, or splice variants — and **splice was Evo 2's best category
  (AUROC 0.85 in M3).** Evo 2 is a *general-purpose, zero-shot* DNA model whose value is breadth
  (any variant type, no task labels), not peak missense accuracy.
- **Model size:** this is the **1B budget** Evo 2; the published 40B is far stronger on BRCA1.

**Honest takeaway:** for scoring *missense* pathogenicity specifically, a specialized tool like
AlphaMissense is the right choice and far outperforms the Evo 2 1B. Evo 2's niche is different —
broad, label-free scoring across *all* variant classes (including the non-coding ones no missense
tool can address). A serious triage system would likely **combine** them: a specialist for
missense, a foundation model for everything else.

---

# Milestone 7+ — A hybrid Evo 2 + AlphaMissense ensemble (and it works)

The benchmarks above revealed a clear complementarity: AlphaMissense dominates *missense* but
**only covers missense**; Evo 2 covers *every* variant type and is strongest on *splice*. So we
built a **routing ensemble**: send each variant to the tool that's good at it.

    hybrid = AlphaMissense   if the variant is missense
             Evo 2 (Δ)       otherwise (non-coding / splice / intronic / …)

To combine the two on a common scale across the routing boundary, each score is mapped to a
**probability of LOF** with a 5-fold cross-validated calibrator (leakage-free). Evaluated on the
**full benchmark (all variant types, n=3,644):**

| Method | AUROC (full set) |
|---|---|
| Evo 2 alone | 0.717 |
| AlphaMissense alone (can only cover the 52% that are missense) | 0.769 |
| **Hybrid (Evo 2 + AlphaMissense)** | **0.875** |

**The hybrid beats both tools decisively** (`m7_ensemble.png`). And the per-segment view shows
exactly why — no magic, just routing:

- **Missense (52% of variants):** Evo 2 **0.58 → AlphaMissense 0.89** — a large upgrade.
- **Non-missense (48%):** kept on Evo 2 at **0.87** — variants AlphaMissense *cannot score at all*.

**Why this is the satisfying result of the project.** After two honest *negative* findings (the
zero-shot score beat the embedding classifier; a specialist beat the 1B on missense), this is a
real *positive* one — and it came from taking those negatives seriously. We didn't just run a
bigger model; we **identified each tool's blind spot from our own evaluation and engineered a
combined predictor that fixes both.** That's the actual workflow of applied ML for genomics:
know your model's failure modes, then route around them.

*(Caveat, kept honest: the routing rule and calibration are simple and tuned on BRCA1; a
production system would validate the routing across genes and consider a learned combiner with
proper held-out testing.)*
