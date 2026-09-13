#!/usr/bin/env bash
# Run all five models over BOTH splits under identical evaluation settings.
#
# The point of this script is that no model gets its own settings. Everything
# that shapes a metric - data path, stride, max_data_num, past/future length -
# comes from configs/eval_test2025_*.yaml, which are byte-identical apart from
# exp_name, ckpt_path, method and cache_path. The only thing that varies between
# the two passes below is which split is being read.
#
# The validation pass is re-measured here rather than reused from the paper,
# because the original validation runs each used a different stride and
# max_data_num, so those numbers cannot be placed next to a test number.
#
#   cd unitraj && bash tools/run_test2025_evals.sh [test|val|both]

set -u
cd "$(dirname "$0")/.." || exit 1

REPO=/home/ubuntu/projects/AIS_Trajectory_Prediction
DATA="$REPO/unitraj/data/processed_ais_4hours_optimized"
MODELS=(wayformer baseline ais_acnet gat_lstm traisformer)
WHICH=${1:-both}

run_split () {
  local split=$1 path=$2
  for m in "${MODELS[@]}"; do
    echo "=============== $split / $m ==============="
    python stratified_eval.py \
      --config-name "eval_test2025_${m}" \
      val_data_path="[$path]" \
      exp_name="${split}_${m}" \
      2>&1 | grep -vE "UserWarning|warnings.warn|pkg_resources|torch.load|ckpt = torch" \
           | tail -25
    echo
  done
}

[ "$WHICH" = test ] || [ "$WHICH" = both ] && run_split test2025 "$DATA/test"
[ "$WHICH" = val  ] || [ "$WHICH" = both ] && run_split val2024  "$DATA/val"

echo "All requested evaluations finished."
