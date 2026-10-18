# Results

Every number in the README / paper tables comes from the CSV/JSON files here (single evaluation passes,
exactly as used in the paper). `SUMMARY.md` is regenerated with `make summary`.

```
results/<dataset>/
  main_compare/comparison.csv                 synthetic-rate accuracy per (model, rate factor)
  main_natural_eval/natural_rate_per_bucket.csv   natural-rate accuracy per (model, bucket)
  rate_label_leakage/leakage.json             ANOVA / mutual information / rate-only classifiers
  lambda_train.txt                            reference (median) rate on the train split
results/ncaltech101/                          additionally: ablations, RA-TTA, R-Canon, TENT, temporal reality check
```

N-Caltech101 was evaluated over several passes (`film_ablation_natural_eval` is the pass used in the paper's
tables; the same baseline checkpoint scores 60.5–61.6 across passes — see docs/REPRODUCING.md §3).
The `ckpt` column inside some CSVs records the original (pre-release) checkpoint paths.
