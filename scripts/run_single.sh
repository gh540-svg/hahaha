#!/usr/bin/env bash
# Single-run helper — produces the 4-way comparison for one task.
#
#   ./scripts/run_single.sh <task>
#
# Examples:
#   ./scripts/run_single.sh code
#   ./scripts/run_single.sh math
#   ./scripts/run_single.sh mmlu

set -eu
TASK="${1:?usage: $0 <code|math|mmlu>}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_BASE="${OUT_BASE:-results}"

case "$TASK" in
  code)  TASK_ARGS=(--task code --code_train mbpp --code_eval both \
                    --n_train 464 --n_calibration 50 --n_eval 100) ;;
  math)  TASK_ARGS=(--task math --math_eval both \
                    --n_train 7473 --n_calibration 50 --n_eval 100) ;;
  mmlu)  TASK_ARGS=(--task mmlu \
                    --n_train 2000 --n_calibration 500 --n_eval 200) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

OUT_DIR="$OUT_BASE/${TASK}"
mkdir -p "$OUT_DIR"

echo "=== Running: task=$TASK  model=$MODEL  out=$OUT_DIR ==="
python ssd_subspace.py \
  --model "$MODEL" \
  --output_dir "$OUT_DIR" \
  "${TASK_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/run.log"
