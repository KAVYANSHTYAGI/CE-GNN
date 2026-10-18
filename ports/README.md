# Cross-architecture ports

The paper asks whether Rate-FiLM and the curriculum are specific to PointNet++. These folders contain
exactly what we changed in two third-party backbones, as (a) a patch against a pinned upstream commit
and (b) an `overlay/` of new files that mirrors the upstream layout. No upstream code is vendored.

| Port | Upstream (pinned) | What we add | Result (natural-rate Top-1) |
|---|---|---|---|
| `eventmamba/` | [rhwxmx/EventMamba](https://github.com/rhwxmx/EventMamba) `7f61fd3` | FiLM after each stage's Mamba/global-conv block, DVS128-Gesture windowing that keeps rate metadata, lighting-stratified evaluation | baseline **67.4** → no-aug 70.2 → curriculum **74.1** (+6.7 pp) |
| `aegnn/` | [uzh-rpg/aegnn](https://github.com/uzh-rpg/aegnn) `d96e13b` | FiLM after GraphRes blocks 2, 5 and 7 (SplineConv); fix of AEGNN's per-graph edge-offset normalisation (it made time-rescaling a no-op) | baseline 39.2 → no-aug 39.0 → curriculum 38.7 (**no gain**, full 80/80/30-epoch run) |
| `evtplus/` | [AlbertoSabater/EventTransformerPlus](https://github.com/AlbertoSabater/EventTransformerPlus) `b4625e1` | FiLM variant of the Event Transformer+ classifier (exploratory, not in the paper) | no completed run |

> The AEGNN numbers are the final full-length run (`aegnn/results/full_run_results.json`). The earlier
> 4-epoch sanity run (`verify_run_results.json`: baseline 18.9 → curriculum 33.3) showed a large gain, but
> with the full schedule the baseline reaches 39.2 and the gap disappears — so the short-run gain should not
> be quoted as a result.

## Build a runnable checkout

```bash
bash ports/apply_ports.sh eventmamba            # clones upstream into external/eventmamba @ pinned commit,
                                                # applies the patch, overlays our files, prints the export line
export EVENTMAMBA_REPO=$PWD/external/eventmamba
sbatch ports/eventmamba/slurm/submit_setup_smoketest.sh      # 2-epoch smoke test (data prep + 3 stages)
sbatch ports/eventmamba/slurm/submit_full.sh                 # full run
sbatch ports/eventmamba/slurm/submit_lighting_eval.sh        # lighting-stratified evaluation
```

Same for `aegnn` (`AEGNN_REPO`) and `evtplus` (`EVT_REPO`). Each port has its own conda env (the upstream
repos' requirements; EventMamba additionally needs `mamba_ssm`, `causal-conv1d`, `triton`). The SLURM
scripts are the ones we ran, with machine paths replaced by the environment variables above.

Each port trains three models with `train_film_curriculum.py --run {baseline,stage1,stage2,eval_all}`:
`baseline` → `stage1` (FiLM, natural rates) → `stage2` (curriculum: warm start + rate augmentation).
