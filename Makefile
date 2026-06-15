# Simple, reproducible entry points for each milestone.
# Each target maps to a CLI subcommand in src/gvep/cli.py.
# Run `make help` to see what's available.

PY := .venv/bin/python

.PHONY: help setup lock data score validate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install local dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

lock:  ## Freeze the resolved dependency versions for reproducibility
	$(PY) -m pip freeze > requirements-lock.txt

data:  ## [Milestone 1] Fetch + build the variant datasets (Findlay, ClinVar, reference)
	$(PY) -m gvep.cli data

score:  ## [Milestone 2] Run Evo 2 delta-likelihood scoring on Modal (cloud GPU)
	$(PY) -m gvep.cli score

validate:  ## [Milestone 3] Compute metrics + honesty/calibration analysis
	$(PY) -m gvep.cli validate

clean:  ## Remove generated data/results (keeps raw downloads)
	rm -rf data/processed/* data/cache/* results/figures/* results/metrics/*
