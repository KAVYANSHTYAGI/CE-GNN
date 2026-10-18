# Preprocessing

```
 raw files ──► DatasetAdapter ──► (x, y, t_ms, polarity) ──► events_to_sample ──► PyG Data ──► .pt
 (per dataset)  cegnn/data/readers/*.py                       cegnn/data/sample.py
```

Everything dataset-specific (file layout, binary format, split, label map) lives in **one file per
dataset** under `cegnn/data/readers/`. Everything dataset-independent (rate statistics, subsampling,
graph construction, the `.pt` schema) lives in `cegnn/data/sample.py`. Nothing else knows about
datasets, so adding a new one is a single file ([ADDING_A_DATASET.md](ADDING_A_DATASET.md)).

## Keeping the natural rate

Many event-GNN pipelines rescale every recording's timestamps into a fixed window (AEGNN uses 300 ms),
which removes the recording's event rate by construction. CEGNN's preprocessing does the opposite:

* timestamps are converted to **milliseconds with origin at the first event** and are **not normalised**;
* the raw-stream rate is measured **before any subsampling** and stored as `log_lambda_raw`
  (this scalar later drives every FiLM layer);
* a dataset-level constant `t_unit` (graph-time units per ms, default 1 ⇒ 1 unit = 1 ms ≈ 1 px) puts time
  on a scale comparable to space; it is the *same* for every sample.

## Output schema (one `.pt` file = one recording)

| field | shape | meaning |
|---|---|---|
| `x` | (N, 1) | polarity ∈ {−1, +1} |
| `pos` | (N, 3) | `[x_px, y_px, t]`, `t` in graph-time units |
| `y` | (1,) | class label |
| `log_lambda_raw` | (1,) | ln(events / s) of the **raw** recording — the conditioning signal |
| `lambda_raw`, `n_events_raw`, `duration_ms` | (1,) | raw rate, raw event count, raw duration |
| `lambda_global` | (1,) | alias of `log_lambda_raw` (kept for old code) |
| `lambda_local` | (N, 1) | per-event log-rate relative to the clip median, clipped to [−4, 4] |
| `edge_index`, `edge_attr` | (2, E), (E, 3) | radius graph; `edge_attr = (Δx, Δy, Δt) / r` |
| `file_id` | str | source file / sample id |

## Graph construction

1. Keep every `K`-th event (after time sorting), capped at `n_samples` nodes (`subsample_K`, `n_samples` in the config).
2. Connect events within an isotropic radius `r` in (x, y, t) space, at most `d_max` neighbours (`radius_graph`).
3. Recordings that end up with no edges (or < 2 events) are skipped and counted in the summary line.

Defaults (all datasets): `r = 6`, `d_max = 32`, `K = 25`, `n_samples = 10 000`. Only the SA radii of the
model differ per sensor (see `configs/`).

## Rate resampling (training augmentation & evaluation shifts)

`cegnn.data.dataset.RateResample(f)` divides the time axis by `f` (`f > 1` ⇒ faster ⇒ higher rate) and
updates `log_lambda_raw += ln f`, `duration_ms /= f`; the graph is then rebuilt. This is how the
synthetic-rate grid {0.125 … 8×} and the curriculum's rate augmentation {0.5, 1, 2} are produced — on
the fly, with no re-preprocessing.

## Equivalence with the original scripts

The modular pipeline was checked **tensor-for-tensor** against the original per-dataset scripts it
replaced: for all five datasets, the first 4 train + 4 test samples were regenerated and compared on every
field (`x, pos, y, log_lambda_raw, lambda_raw, n_events_raw, duration_ms, lambda_global, lambda_local,
edge_index, edge_attr`) — 40 samples, 0 mismatches, including the deterministic splits.
