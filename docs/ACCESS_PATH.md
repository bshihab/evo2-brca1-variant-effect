# Evo 2 Access Path — Research & Decision Record

*Researched June 2026, at project start. Records what we found, what we chose, and why,
so the decision is reproducible and auditable later.*

## TL;DR decision

**Run Evo 2 `1b_base` with FP8 / Transformer Engine on a cloud NVIDIA GPU via Modal,
default GPU = L4 (Ada, FP8-capable, cheap), H100 fallback.** Apple Silicon cannot run the
model locally. Weights are Apache-2.0. Fits free-tier credits (~$1–2 for the full run).

> **CORRECTION (Milestone 2 research, 2026-06-15):** The original plan below said "bf16,
> no FP8." That does NOT work — the `evo2` package hard-requires Transformer Engine + FP8
> to even load the 1B model (GitHub issue #208 errors without TE). FP8 hardware exists on
> Ada/Hopper/Blackwell only (NOT Ampere — so A100/A10G are out). The fix is to embrace FP8
> on an FP8-capable GPU. Modal's **L4 is Ada (compute 8.9) and supports FP8**, so we run the
> 1B *properly* on a cheap GPU — better than the bf16 workaround. See "Scoring API" below.

## Constraints that drove the choice

- **Hardware:** Apple Silicon Mac only (no local NVIDIA GPU).
- **Budget:** free tiers only.
- **Preference:** Modal as the cloud GPU path; start with the 1B model.

## Key findings

### 1. Evo 2 is CUDA-only — Apple Silicon can't run it locally
Evo 2 uses the custom **StripedHyena 2** architecture with CUDA kernels. Requirements:
Linux (or WSL2), CUDA 12.1+, cuDNN 9.3+, Python 3.11/3.12, Torch 2.6/2.7. There is **no
documented Mac/MPS path.** ⇒ All real inference happens on a cloud GPU. The Mac is for
code, the data layer, analysis, and the demo UI.

### 2. The counterintuitive twist: the 1B *officially* wants a Hopper GPU, the 7B doesn't
- **7B:** runs in **bfloat16 on any supported GPU** — no FP8, no Transformer Engine.
- **1B / 20B / 40B:** documented as needing **FP8 via Transformer Engine + a Hopper (H100)
  GPU** "for numerical accuracy," because those models use FP8 in some layers.

**But** NVIDIA's own BRCA1 tutorial runs the 1B/7B on an **RTX A6000** (Ampere, compute 8.6,
*no* FP8). So the **1B does run on a non-Hopper GPU in bf16** — you just accept some numerical
imprecision vs. the FP8 reference. For a POC this is acceptable, and the imprecision becomes a
*measurable, honest* talking point: we compare our bf16 AUROC against the published number to
confirm bf16 didn't materially hurt separation.

### 3. Three viable access paths
| Path | Pros | Cons |
|---|---|---|
| **Arc `evo2` (HF weights + pip) on Modal** ✅ chosen | Full control; forward pass exposes logits → log-likelihoods (required for delta scoring); reproducible Modal script; Apache-2.0 | Must build a CUDA container |
| NVIDIA BioNeMo / NIM (self-host) | Same code, NVIDIA's framework | Heavier stack than we need |
| NVIDIA hosted API (build.nvidia.com) | Zero install, free credits | **Unverified** whether it exposes per-token likelihoods (may be generation/embeddings only); rate limits |

We chose self-host on Modal because **delta-likelihood scoring needs the logits**, which the
forward pass reliably provides. The hosted API remains a fallback if container builds prove painful.

### 4. Cost sanity check (free-tier feasibility)
Scoring ~3,893 variants × 2 sequences (ref + variant) ≈ ~7,800 forward passes of a **1B** model
over **8,192-bp** windows. On a modest GPU this is on the order of minutes to under an hour of
GPU time — comfortably within Modal's free monthly credit. Running **bf16 without Transformer
Engine** also avoids the slow, finicky TE + Flash-Attention build.

## Licensing

- **Evo 2 weights** (`arcinstitute/evo2_1b_base`): **Apache-2.0** — confirmed on the model card.
- **Evo 2 code:** open-source (Arc Institute "fully open" release).
- **Findlay 2018 dataset:** published in Nature; available via **MaveDB `urn:mavedb:00000045-b`**.
- **ClinVar:** NCBI, public domain.

All compatible with an open, MIT-licensed portfolio project, provided we attribute (see README).

## Upgrade path
1. **1B / bf16 / L4 or A10G** ← start here.
2. **7B / bf16** — also non-Hopper; likely better accuracy, more GPU time.
3. **1B or 7B with FP8 on H100** — to reproduce published numbers exactly (costs more credit).

## Scoring API (confirmed at Milestone 2)

From the official Evo 2 BRCA1 notebook:
```python
from evo2.models import Evo2
model = Evo2('evo2_1b_base')
ref_scores = model.score_sequences(ref_seqs)   # list[str] -> per-seq log-likelihood
var_scores = model.score_sequences(var_seqs)
delta = np.array(var_scores) - np.array(ref_scores)
auroc = roc_auc_score(brca1_df['class'] == 'LOF', -delta)   # published 1B AUROC ≈ 0.73
```
**Published 1B benchmark: AUROC ≈ 0.73** for LOF classification — our Milestone 3 sanity target.

Install (from the evo2 README "full install", required for 1B):
```bash
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```
On Modal this becomes a micromamba image (see src/gvep/scoring/modal_app.py). The image build
(TE + flash-attn) is the main risk and may need an iteration or two.

## Open item to verify at Milestone 2
- Whether `evo2` runs FP8 on the L4 (Ada) out of the box, or insists on Hopper. Mitigation:
  GPU is a one-line parameter — same image runs on L4 or H100, so a fallback needs no rebuild.

## Sources
- Evo 2 GitHub (requirements, model-size GPU notes): <https://github.com/ArcInstitute/evo2>
- Evo 2 1B model card (Apache-2.0, 8192 ctx): <https://huggingface.co/arcinstitute/evo2_1b_base>
- NVIDIA BioNeMo BRCA1 zero-shot tutorial: <https://docs.nvidia.com/bionemo-framework/2.5/user-guide/examples/bionemo-evo2/zeroshot_brca1/>
- Evo 2 on NVIDIA BioNeMo (announcement): <https://blogs.nvidia.com/blog/evo-2-biomolecular-ai/>
- Evo 2 paper (Nature 2026): <https://www.nature.com/articles/s41586-026-10176-5>
