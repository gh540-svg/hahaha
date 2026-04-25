#!/usr/bin/env bash
# Reproduce the paper's main table: 3 runs =
# 3 domains (code/math/mmlu) with correctness-aligned CE.
#
# Each run emits results.json with all 4 methods
# (baseline, inference_hooks, ssd_plain, ssd_enhanced).
#
# Total wall time on a single RTX-3090/4090 with Qwen2.5-0.5B-Instruct: ~5-7h.

set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

export MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUT_BASE="${OUT_BASE:-results}"

for TASK in code math mmlu; do
  bash scripts/run_single.sh "$TASK"
done

echo
echo "=== DONE. Collected results below. ==="
python - <<'PY'
import json, os, glob

methods = ["baseline", "inference_hooks", "ssd_plain", "ssd_enhanced"]

for f in sorted(glob.glob("results/*/results.json")):
    with open(f) as fh: d = json.load(fh)
    task = d["config"].get("task", os.path.basename(os.path.dirname(f)))
    r = d["results"]

    if task == "math":
        hdr = f"{'Method':20s} {'GSM8K':>8s} {'SVAMP':>8s}"
        print(f"\n  {task.upper()}")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for method in methods:
            m = r.get(method, {})
            g = m.get("gsm8k_accuracy", m.get("accuracy"))
            s = m.get("svamp_accuracy", m.get("accuracy"))
            gstr = f"{g:.2%}" if g is not None else "N/A"
            sstr = f"{s:.2%}" if s is not None else "N/A"
            print(f"  {method:20s} {gstr:>8s} {sstr:>8s}")

    elif task == "code":
        hdr = f"{'Method':20s} {'MBPP':>8s} {'CA_NLL':>8s} {'CA_AST':>8s}"
        print(f"\n  {task.upper()}")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for method in methods:
            m = r.get(method, {})
            mbpp = m.get("mbpp_pass@1", m.get("pass@1"))
            nll = m.get("ca_nll", m.get("nll"))
            ast_ = m.get("ca_ast_parse_rate", m.get("ast_parse_rate"))
            mstr = f"{mbpp:.2%}" if mbpp is not None else "N/A"
            nstr = f"{nll:.3f}" if nll is not None else "N/A"
            astr = f"{ast_:.2%}" if ast_ is not None else "N/A"
            print(f"  {method:20s} {mstr:>8s} {nstr:>8s} {astr:>8s}")

    elif task == "mmlu":
        hdr = f"{'Method':20s} {'MMLU':>8s} {'BBH':>8s}"
        print(f"\n  {task.upper()}")
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")
        for method in methods:
            m = r.get(method, {})
            mm = m.get("mmlu_accuracy", m.get("accuracy"))
            bb = m.get("bbh_accuracy")
            mstr = f"{mm:.2%}" if mm is not None else "N/A"
            bstr = f"{bb:.2%}" if bb is not None else "N/A"
            print(f"  {method:20s} {mstr:>8s} {bstr:>8s}")
PY
