#!/usr/bin/env bash
# Single-run helper — produces the 4-way comparison for one task.
#
#   ./scripts/run_single.sh <task> [default|aligned]
#
# Examples:
#   ./scripts/run_single.sh code aligned
#   ./scripts/run_single.sh math default
#   ./scripts/run_single.sh mmlu aligned

set -eu
TASK="${1:?usage: $0 <code|math|mmlu> [default|aligned]}"
LOSS="${2:-aligned}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_BASE="${OUT_BASE:-results}"

case "$TASK" in
  code)  TASK_ARGS=(--task code --code_train mbpp --code_eval both \
                    --n_train 2000 --n_calibration 50 --n_eval 100)
         ALIGNED_SRC="mbpp_solutions" ;;
  math)  TASK_ARGS=(--task math --math_eval both \
                    --n_train 7473 --n_calibration 50 --n_eval 100)
         ALIGNED_SRC="answer_only" ;;
  mmlu)  TASK_ARGS=(--task mmlu \
                    --n_train 2000 --n_calibration 500 --n_eval 200)
         ALIGNED_SRC="answer_only" ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

case "$LOSS" in
  default) EXTRA=();             TAG="base" ;;
  aligned) EXTRA=(--calibration_source "$ALIGNED_SRC"); TAG="aligned" ;;
  *) echo "unknown loss: $LOSS (default|aligned)" >&2; exit 2 ;;
esac

OUT_DIR="$OUT_BASE/${TASK}_${TAG}"
mkdir -p "$OUT_DIR"

echo "=== Running: task=$TASK  loss=$LOSS  model=$MODEL  out=$OUT_DIR ==="
python ssd_subspace.py \
  --model "$MODEL" \
  --output_dir "$OUT_DIR" \
  "${TASK_ARGS[@]}" \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT_DIR/run.log"
