#!/bin/bash
# run_ratemoe_ablations.sh — RateMoE ablation suite for N-Cars.
#
# Runs the main K=4 model plus K-sweep and feature/routing/τ ablations,
# saving each to results/{name}/best.ckpt. After completion, prints the
# eval_synthetic_rates.py command to evaluate all variants head-to-head.
#
# Usage:
#   bash scripts/negative_results/run_ratemoe_ablations.sh           # full suite, EPOCHS=60
#   EPOCHS=10 bash scripts/negative_results/run_ratemoe_ablations.sh # quick check

set -euo pipefail

cd "$(dirname "$0")/../.."

EPOCHS=${EPOCHS:-60}
SEED=${SEED:-42}
CONFIG=${CONFIG:-configs/ncars.yaml}

echo "=== RateMoE ablation suite ==="
echo "  EPOCHS=$EPOCHS  SEED=$SEED  CONFIG=$CONFIG"
echo "  started: $(date)"
echo

run() {
    local desc="$1"; shift
    echo "------------------------------------------------------------"
    echo ">>> $desc"
    echo "    cmd: python scripts/negative_results/train_ratemoe.py $*"
    echo "------------------------------------------------------------"
    python -u scripts/negative_results/train_ratemoe.py --config "$CONFIG" \
        --epochs "$EPOCHS" --seed "$SEED" "$@"
    echo
}

# Main and K-sweep
run "RateMoE K=4 (main)"      --name ratemoe --num_experts 4 --top_k 2
run "RateMoE K=1 (no MoE)"    --name ratemoe --num_experts 1 --top_k 1
run "RateMoE K=2"             --name ratemoe --num_experts 2 --top_k 2
run "RateMoE K=8 top-2"       --name ratemoe --num_experts 8 --top_k 2

# Ablations on K=4
run "Ablation: no rate input"   --name ratemoe --num_experts 4 --top_k 2 --no_rate_input
run "Ablation: random routing"  --name ratemoe --num_experts 4 --top_k 2 --random_routing
run "Ablation: frozen τ"        --name ratemoe --num_experts 4 --top_k 2 --freeze_tau
run "Ablation: all experts on"  --name ratemoe --num_experts 4 --top_k 2 --all_experts

echo "=== ALL ABLATIONS COMPLETE ==="
echo "finished: $(date)"
echo
echo "Next: run head-to-head comparison vs ms_only baseline:"
echo
echo "python scripts/eval/eval_synthetic_rates.py --config $CONFIG --name ratemoe_compare \\"
echo "  --models ms_only:results/ms_only/best.ckpt:fixed \\"
echo "           ratemoe_K4:results/ratemoe_K4/best.ckpt:fixed \\"
echo "           ratemoe_K1:results/ratemoe_K1/best.ckpt:fixed \\"
echo "           ratemoe_K2:results/ratemoe_K2/best.ckpt:fixed \\"
echo "           ratemoe_K8:results/ratemoe_K8/best.ckpt:fixed \\"
echo "           ratemoe_K4_no_rate:results/ratemoe_K4_no_rate/best.ckpt:fixed \\"
echo "           ratemoe_K4_rand_route:results/ratemoe_K4_rand_route/best.ckpt:fixed \\"
echo "           ratemoe_K4_frozen_tau:results/ratemoe_K4_frozen_tau/best.ckpt:fixed \\"
echo "           ratemoe_K4_all_exp:results/ratemoe_K4_all_exp/best.ckpt:fixed"
