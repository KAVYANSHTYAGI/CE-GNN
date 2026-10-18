PY ?= python
.PHONY: test data-% reproduce-% summary
test:            ## unit tests (CPU, no data needed)
	$(PY) -m pytest -q
data-%:          ## make data-ncaltech101  -> preprocess one dataset
	$(PY) scripts/data/prepare_dataset.py --dataset $* --config configs/$*.yaml --workers 16
reproduce-%:     ## make reproduce-dvsgesture -> preprocess + train + evaluate (needs a GPU)
	bash scripts/run_pipeline.sh $*
summary:         ## rebuild results/SUMMARY.md from the shipped CSVs
	$(PY) scripts/analysis/summarize_results.py --results results
