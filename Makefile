# Simple, reproducible entry points for each milestone.
# Each target maps to a CLI subcommand in src/gvep/cli.py.
# Run `make help` to see what's available.

PY := .venv/bin/python

.PHONY: help setup lock data smoke score sanity validate clean

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

smoke:  ## [Milestone 2] Cheap Modal validation: load Evo 2 + score 2 seqs on the GPU
	.venv/bin/modal run -m gvep.scoring.modal_app::smoke

score:  ## [Milestone 2] Run Evo 2 delta-likelihood scoring on Modal (full dataset)
	.venv/bin/modal run -m gvep.scoring.modal_app::main

sanity:  ## [Milestone 2] Plot delta distributions + quick AUROC (after scoring)
	$(PY) -m gvep.cli sanity

validate:  ## [Milestone 3] Compute metrics + honesty/calibration analysis
	$(PY) -m gvep.cli validate

explain:  ## [Milestone 5] Demo per-variant trust-aware explanations
	$(PY) -m gvep.cli explain

api:  ## [Milestone 6] Serve the FastAPI backend (http://localhost:8000/docs)
	.venv/bin/uvicorn gvep.app.api:app --reload

ui:  ## [Milestone 6] Launch the Streamlit demo UI
	.venv/bin/streamlit run src/gvep/app/ui.py

clean:  ## Remove generated data/results (keeps raw downloads)
	rm -rf data/processed/* data/cache/* results/figures/* results/metrics/*
