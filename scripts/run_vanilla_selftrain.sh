#!/usr/bin/env bash
# Run vanilla self-training experiment across all 3 domains.
#
# The model generates its own answers (temperature=0.7, no top_p filtering),
# trains LoRA on those outputs, and evaluates on all 6 datasets.
#
# Usage:
#   # Run all 3 domains (math, code, mmlu):
#   bash scripts/run_vanilla_selftrain.sh
#
#   # Run a single domain:
#   bash scripts/run_vanilla_selftrain.sh --domains math
#
#   # Use a different model:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_vanilla_selftrain.sh
#
# Training data sizes (same as paper):
#   math: 7473 (full GSM8K train)
#   code: 464  (full MBPP train+val)
#   mmlu: 2000
#
# Eval datasets (100 each, BBH 300):
#   Math:  GSM8K test, SVAMP
#   Code:  MBPP sanitized test, CodeAlpaca (NLL + AST)
#   QA:    MMLU test, BBH (6 subtasks x 50)
#
# Estimated wall time on a single RTX-4090 with Qwen2.5-0.5B-Instruct: ~3-4h

set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-results/vanilla_selftrain}"
DOMAINS="${1:-math,code,mmlu}"

# Parse --domains flag if provided
if [ "$DOMAINS" = "--domains" ] && [ -n "${2:-}" ]; then
  DOMAINS="$2"
fi

mkdir -p "$OUT_DIR"

echo "=== Vanilla Self-Training Experiment ==="
echo "  Model:   $MODEL"
echo "  Domains: $DOMAINS"
echo "  Output:  $OUT_DIR"
echo "========================================="

python scripts/vanilla_selftrain.py \
  --model "$MODEL" \
  --domains "$DOMAINS" \
  --epochs 5 \
  --lora_r 8 \
  --lr 1e-5 \
  --n_eval 100 \
  --seed 42 \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run.log"

echo
echo "=== DONE. Results saved to $OUT_DIR/results.json ==="
