# Negative results

Everything we tried that did not help is kept, and still runnable, under `scripts/negative_results/`
(plus `cegnn/adapt/rcanon.py`, `cegnn/models/ratemoe.py`, `cegnn/models/rate_conditioned.py`).
N-Caltech101, natural-rate mean accuracy, from the paper's main table (baseline = 60.9):

| Idea | Natural mean | What happened |
|---|---:|---|
| **Rate-invariance training** (`ra-aug-only`) | 60.3 | Synthetic-rate robustness ↑ (16.0 → 47.0) but natural-rate accuracy slightly *below* baseline: invariance trades the accuracy that rate carries for robustness to shifts deployment never produces. |
| **Concatenate the rate to the pooled feature** (`train_rate_cond_concat.py`) | 60.0 | A head-only scalar is easy for the network to ignore; no gain. |
| **FiLM only after the global pool** (`--variant global_only`) | 60.3 | Same parameters as most of the full FiLM, zero benefit ⇒ *where* the rate enters matters (neighbourhood-forming stages). |
| **R-Canon** — learn a per-recording time-rescale to canonicalise the rate before a frozen classifier | 44.7 | Collapses to a near-constant ρ\* ≈ e⁻¹; time-rescaling is structurally the wrong canonicalisation. |
| **RA-TTA on a rate-sensitive base** (baseline, FiLM without aug) | 40.2 | An adapter cannot replace training-time rate handling; destabilises weak bases by ~20 pp. |
| **TENT** entropy-minimisation on the RA-TTA prompt | — | +3 pp on the weak RA-TTA-on-baseline combination, −8 pp on the strong one. |
| **Rate-adaptive graph construction** (adaptive radius / density-aware subsampling, `train_rate_aug_adaptive_graph.py`, `decompose_graph_levers.py`) | — | Early N-Cars study: simple rate augmentation beat the adaptive-graph variants; the fully adaptive one (RA-GNN) was the worst. |
| **RateMoE-GNN** (rate-routed mixture of EdgeConv experts, `train_ratemoe.py`) | — | Abandoned early in favour of simple conditioning. |
| **FiLM on AEGNN** (`ports/aegnn`) | 38.7 vs 39.2 | Full-length run: no natural-rate gain on this backbone (see `ports/README.md`). |

These runs are not clutter: each one tests one alternative explanation (capacity, augmentation range,
removing the rate instead of using it), and their failure is what supports conditioning on the rate early
in the network, with identity initialisation and a warm-started curriculum.
