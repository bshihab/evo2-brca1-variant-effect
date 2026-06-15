# data/

This directory holds datasets. **The actual data files are gitignored** — they're
large and fully regenerable from the loaders in `src/gvep/data/` (Milestone 1).

```
raw/         original downloads (Findlay 2018, ClinVar, BRCA1 reference region)
processed/   cleaned, per-variant tables (ref/var windows, coords, labels)
cache/       intermediate / memoized artifacts (e.g. cached Evo 2 scores)
```

Sources (see PRIMER.md and docs/ACCESS_PATH.md for context):
- **Findlay et al. (2018)** BRCA1 saturation genome editing — MaveDB `urn:mavedb:00000045-b`.
- **ClinVar** (BRCA1 slice, incl. VUS) — NCBI.
- **Reference genome** — GRCh38 chr17 BRCA1 region.

Run `make data` (once Milestone 1 lands) to populate these folders.
