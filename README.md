# SSD-Subspace

**Capability-selective self-distillation via KV subspace projection.**

A small language model can be made better at a targeted capability — code, math, multiple-choice QA — by (1) finding a low-rank subspace of its own K/V activations that matters for *correctness*, (2) projecting K/V onto that subspace during self-sampling, (3) fine-tuning on the resulting self-samples with plain cross-entropy. **No teacher, no external verifier, no gold labels during fine-tuning.**

The key insight: **correctness-aligned CE** (gradient only on assertion tokens for code, answer tokens for math/MMLU) extracts a subspace that captures the capability signal, not surface-level token distributions.

---

## TL;DR

```bash
# Install
pip install -r requirements.txt

# Run one task (e.g. code) on the default student:
bash scripts/run_single.sh code

# Use a larger student — MODEL is an env var, any HF causal LM works:
MODEL=Qwen/Qwen2.5-7B-Instruct        bash scripts/run_single.sh code
MODEL=Qwen/Qwen2.5-14B-Instruct       bash scripts/run_single.sh code
MODEL=Qwen/Qwen3-4B-Instruct-2507     bash scripts/run_single.sh code
MODEL=Qwen/Qwen3-14B-Instruct-2507    bash scripts/run_single.sh code

# Reproduce the full main table (3 runs, ~5-7h on a single 24 GB GPU):
bash scripts/reproduce_all.sh
```

Each run writes `results.json` to `results/<task>/` with four methods:
`baseline`, `inference_hooks`, `ssd_plain`, `ssd_enhanced`.

---

## Repository layout

```
.
├── ssd_subspace.py        # main pipeline (all 4 methods, all 3 tasks)
├── utils.py               # gradient collection, SVD, projection helpers
├── requirements.txt       # pip dependencies
├── scripts/
│   ├── run_single.sh      # run one task
│   └── reproduce_all.sh   # run all 3 tasks + summary table
└── examples/              # sample outputs (see below)
```

Two Python files, two bash scripts. Nothing else to read.

---

## Installation

Tested on Python 3.10 / 3.11 with CUDA 12.1.

```bash
# 1. Clone
git clone https://github.com/gh540-svg/hahaha.git ssd-subspace
cd ssd-subspace

# 2. Create an env (conda or venv, either works)
conda create -n ssd python=3.11 -y
conda activate ssd

# 3. Install dependencies
pip install -r requirements.txt
```

First-run dataset downloads will pull ~5 GB of HuggingFace caches (MMLU, MBPP, CodeAlpaca, GSM8K, SVAMP, BBH).

### Students used in the paper

Scaling study across five Qwen Instruct checkpoints (all chat-tuned, standard-GQA pure-text models — drop-in compatible with the code):

| Model | Family | Params | fp16 weights | Peak VRAM (LoRA r=8) | `transformers` |
|---|---|---|---|---|---|
| **`Qwen/Qwen2.5-0.5B-Instruct`** *(default)* | Qwen2.5 | 0.5B | 1 GB | ~4 GB | ≥ 4.44 |
| **`Qwen/Qwen2.5-7B-Instruct`** | Qwen2.5 | 7B | 14 GB | ~28 GB | ≥ 4.44 |
| **`Qwen/Qwen2.5-14B-Instruct`** | Qwen2.5 | 14B | 28 GB | ~48–56 GB | ≥ 4.44 |
| **`Qwen/Qwen3-4B-Instruct-2507`** | Qwen3 | 4B | 8 GB | ~16–20 GB | ≥ 4.51 |
| **`Qwen/Qwen3-14B-Instruct-2507`** | Qwen3 | 14B | 28 GB | ~48–56 GB | ≥ 4.51 |

Pick your student by setting `MODEL`:

```bash
# Qwen2.5 family (any transformers ≥ 4.44)
MODEL=Qwen/Qwen2.5-0.5B-Instruct   bash scripts/reproduce_all.sh   # ~5-7 h on RTX-4090
MODEL=Qwen/Qwen2.5-7B-Instruct     bash scripts/reproduce_all.sh   # ~12-16 h on A100-80
MODEL=Qwen/Qwen2.5-14B-Instruct    bash scripts/reproduce_all.sh   # ~24-30 h on A100-80

# Qwen3 family (requires transformers ≥ 4.51 — pip install -r requirements.txt)
MODEL=Qwen/Qwen3-4B-Instruct-2507  bash scripts/reproduce_all.sh   # ~15-20 h on RTX-4090
MODEL=Qwen/Qwen3-14B-Instruct-2507 bash scripts/reproduce_all.sh   # ~30-40 h on A100-80
```

---

## The method in 4 steps

```
[1] Find     SVD on gradients of a correctness-aligned CE loss → per-layer
             projection matrices P_K, P_V. Default rank = half of KV dim
             (e.g. 64 for 0.5B, 256 for 7B). Also supports energy-threshold
             and fixed-rank modes.
[2] Hook     Register forward hooks at [last, middle] attention layers that
             apply K → K·P_K and V → V·P_V.
[3] Sample   With hooks active, student generates N completions from prompts.
             No gold labels used; completions may be wrong.
[4] Train    Hooks off. Fine-tune LoRA adapter on (prompt, completion) pairs
             with plain next-token CE.
```

Every run of `ssd_subspace.py` produces **four methods** that together isolate each design choice:

| Method | Hook at gen? | Hook at eval? | Fine-tuned? | What it tests |
|---|---|---|---|---|
| `baseline` | - | - | - | Pre-training reference. |
| `inference_hooks` | — | Yes | - | Training-free projection. |
| `ssd_plain` | - | - | Yes | Vanilla self-distillation (no subspace). |
| `ssd_enhanced` | Yes | - | Yes | SSD-Subspace (our method). |

---

## Quickstart — running one task

```bash
# Code: MBPP training (464 examples), assertion-only CE for subspace, dual-eval (MBPP + CodeAlpaca)
bash scripts/run_single.sh code

# Math: GSM8K training (7473 examples), answer-only CE, dual-eval (GSM8K + SVAMP)
bash scripts/run_single.sh math

# MMLU: MMLU training (2000 examples), answer-only CE, dual-eval (MMLU + BBH)
bash scripts/run_single.sh mmlu
```

Override the student via the `MODEL` env var:

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507  bash scripts/run_single.sh math
MODEL=Qwen/Qwen3-14B-Instruct-2507 bash scripts/run_single.sh mmlu
```

---

## Reproducing the paper's main table

```bash
bash scripts/reproduce_all.sh
```

This runs 3 tasks sequentially, then prints a collated metric table at the end. Results land in `results/<task>/results.json`.

Expected approximate numbers — reference scales (Qwen2.5-0.5B-Instruct, reported in the paper):

| Domain | Metric | `baseline` | `ssd_plain` | `ssd_enhanced` |
|---|---|---|---|---|
| Code (MBPP) | pass@1 | 11.7% | 30.7% | **20.2%** |
| Math | GSM8K / SVAMP | 11% / 16% | 11% / 12% | **19% / 27%** |
| MMLU | MMLU / BBH | 46% / 32% | 48% / 38% | **48.5% / 37.3%** |

`ssd_plain` is stronger on MBPP because hook-generated code samples can have damaged identifiers; `ssd_enhanced` wins on math and MMLU where the aligned subspace cleanly denoises generation. Effect sizes change with scale — see the paper's Table 4 for per-model breakdowns.

---

## Command-line options

All options live on `ssd_subspace.py`; `run_single.sh` just wraps the common ones.

### Core

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | Any HF causal LM with standard `self_attn.{k_proj, v_proj}`. |
| `--task` | `math` | `math`, `code`, or `mmlu` |
| `--n_train` | task-dependent | Number of prompts for SSD sample generation + fine-tuning |
| `--n_calibration` | task-dependent | Examples used for gradient SVD |
| `--n_eval` | `100`–`200` | Evaluation examples per dataset |
| `--rank_mode` | `half` | `half` = kv_dim//2 (default), `energy` = auto via `--rank_energy`, `fixed` = use `--rank` value. |
| `--rank` | `0` | Subspace rank (only used when `--rank_mode fixed`). |
| `--rank_energy` | `0.95` | Energy threshold (only used when `--rank_mode energy`). |
| `--epochs` | `5` | Fine-tuning epochs |
| `--lora_r` | `8` | LoRA rank for adapter |
| `--lr` | `1e-5` | Fine-tuning learning rate |
| `--seed` | `42` | RNG seed |
| `--layers` | `last_mid` | `"last_mid"` (auto: `[L-1, L/2]`) or `"23,12"` etc. |
| `--output_dir` | `results/ssd_subspace` | Where `results.json` is written |

### Task-specific

| Flag | Choices | Purpose |
|---|---|---|
| `--math_eval` | `svamp` / `gsm8k_test` / `both` | Math eval dataset(s) |
| `--code_train` | `mbpp` / `codealpaca` | Code training source |
| `--code_eval` | `mbpp_sanitized` / `codealpaca` / `both` | Code eval dataset(s) |

### Calibration

| Flag | Choices | Purpose |
|---|---|---|
| `--calibration_source` | `aligned` / `mbpp_solutions` / `answer_only` | Which tokens contribute gradient during subspace extraction. `aligned` (default) auto-selects per task. |
| `--project_mode` | `both` / `k_only` / `v_only` | Project K, V, or both |

The correctness-aligned CE masks all tokens except the ones that define correctness:

| Task | Tokens that contribute gradient |
|---|---|
| Code | Only tokens inside `assert ...` strings (assertion-only CE) |
| Math | Only the numeric answer after `####` (answer-only CE) |
| MMLU | Only the answer letter token (answer-only CE) |

---

## Output format — `results.json`

```jsonc
{
  "timestamp": "2026-...",
  "config": { /* all CLI args */ },
  "target_layers": [23, 12],
  "projections": {
    "23": { "energy_K": 0.96, "energy_V": 0.95 },
    "12": { "energy_K": 0.99, "energy_V": 0.94 }
  },
  "results": {
    "baseline":        { /* primary metric + dual-eval sub-metrics */ },
    "inference_hooks": { /* same shape */ },
    "ssd_plain":       { /* + train_losses, best_loss, best_epoch */ },
    "ssd_enhanced":    { /* same */ }
  }
}
```

Primary-metric keys by task and eval:

| Task / eval | Primary metric | Secondary |
|---|---|---|
| math / svamp \| gsm8k_test | `accuracy` | — |
| math / both | `gsm8k_accuracy` | `svamp_accuracy` |
| code / mbpp_sanitized | `pass@1` | — |
| code / codealpaca | `nll`, `ast_parse_rate` | `ppl` |
| code / both | `mbpp_pass@1` | `ca_nll`, `ca_ast_parse_rate` |
| mmlu | `mmlu_accuracy` | `bbh_accuracy`, `bbh_per_task_accuracy` |

`ssd_plain` and `ssd_enhanced` also record `train_losses[]`, `best_loss`, `best_epoch`. Trained LoRA adapters are saved to `<output_dir>/student_{ssd_plain,ssd_enhanced}_seed{SEED}/`, loadable with `peft.PeftModel`.

---

## Datasets

| Domain | Training | In-domain eval | Transfer eval |
|---|---|---|---|
| Code | MBPP train+val (464) | MBPP sanitized (pass@1) | CodeAlpaca-20k (NLL + AST) |
| Math | GSM8K train (7473) | GSM8K test (accuracy) | SVAMP (accuracy) |
| QA | MMLU aux_train (2000) | MMLU test (accuracy) | BBH, 6 subtasks (accuracy) |

All datasets are auto-downloaded via the HuggingFace `datasets` library on first use.

BBH subtasks used (all single-token answer formats):
`logical_deduction_three_objects`, `date_understanding`, `movie_recommendation`, `boolean_expressions`, `causal_judgement`, `sports_understanding`.

---

## Windows users

The bash scripts work under Git Bash / WSL. For native PowerShell you can run:

```powershell
python ssd_subspace.py --task mmlu `
    --n_train 2000 --n_calibration 500 --n_eval 200 `
    --output_dir results/mmlu
```

---

## FAQ

**Q: Does this need a teacher model?**
No. The student bootstraps from its own generations. Ground-truth answers only appear in calibration texts (for SVD) and eval targets — never as fine-tuning labels.

**Q: Why are `ssd_plain` samples sometimes as good as `ssd_enhanced`?**
When the task is already well-formed (MBPP code completion), the student's natural samples are often on-manifold. The subspace projection helps most when the natural distribution has substantial off-capability noise (math reasoning, MMLU letter answers).

**Q: Can I scale up the student model?**
Yes — the paper runs five Qwen Instruct checkpoints (Qwen2.5-0.5B / 7B / 14B and Qwen3-4B / 14B, all `-Instruct` variants) under the identical pipeline. Just pass `--model` or set `MODEL`. The default `--rank_mode half` uses kv_dim//2, which scales automatically across model sizes.

**Q: How does `--rank_mode half` work?**
The default rank selection uses half of the KV projection dimension (e.g. 64 for 0.5B with kv_dim=128, 256 for 7B with kv_dim=512). This scales proportionally across model sizes without hand-tuning. Alternative modes: `--rank_mode energy --rank_energy 0.95` auto-selects the smallest rank capturing 95% of gradient energy (can be as low as rank 2 for math on 7B), or `--rank_mode fixed --rank 64` for a fixed value.

---

## License

The code in this repository is released under MIT. Dataset licenses follow their respective sources on HuggingFace.
