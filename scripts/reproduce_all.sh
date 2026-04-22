#!/usr/bin/env bash
# Reproduce the paper's main table: 6 runs =
# 3 domains (code/math/mmlu) × 2 loss variants (default/aligned).
#
# Each run emits results.json with all 4 methods
# (baseline, inference_hooks, ssd_plain, ssd_enhanced).
#
# Total wall time on a single RTX-3090/4090 with Qwen2.5-0.5B-Instruct: ~10-14h.

set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

export MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUT_BASE="${OUT_BASE:-results}"

for TASK in code math mmlu; do
  for LOSS in default aligned; do
    bash scripts/run_single.sh "$TASK" "$LOSS"
  done
done

echo
echo "=== DONE. Collected results below. ==="
python - <<'PY'
import json, os, glob
rows = []
for f in sorted(glob.glob("results/*/results.json")):
    with open(f) as fh: d = json.load(fh)
    cfg = d["config"]; r = d["results"]
    tag = os.path.basename(os.path.dirname(f))
    for method in ["baseline", "inference_hooks", "ssd_plain", "ssd_enhanced"]:
        m = r.get(method, {})
        primary = m.get("accuracy") or m.get("pass@1") or m.get("mmlu_accuracy") or m.get("gsm8k_accuracy") or m.get("mbpp_pass@1")
        rows.append((tag, method, primary))
for t, m, v in rows:
    print(f"{t:40s}  {m:16s}  {v if v is None else f'{v:.4f}'}")
PY
