# SSD-Subspace

**Capability-selective self-distillation via KV subspace projection.**

A small language model can be made better at a targeted capability — code, math, multiple-choice QA — by (1) finding a low-rank subspace of its own K/V activations that matters for *correctness*, (2) projecting K/V onto that subspace during self-sampling, (3) fine-tuning on the resulting self-samples with plain cross-entropy. **No teacher, no external verifier, no gold labels during fine-tuning.**

The key empirical finding: **the loss function used to extract the subspace is what defines the capability.** Switching from full next-token CE to a correctness-aligned CE (gradient only on assertion tokens for code, answer tokens for math/MMLU) unlocks the method.

---

## TL;DR

```bash
# Install
pip install -r requirements.txt

# Run one task (e.g. code with assertion-only CE):
bash scripts/run_single.sh code aligned

# Reproduce the full main table (6 runs, ~10-14h on a single 24 GB GPU):
bash scripts/reproduce_all.sh
```

Each run writes `results.json` to `results/<task>_<loss>/` with four methods:
`baseline`, `inference_hooks`, `ssd_plain`, `ssd_enhanced`.

---

## Repository layout

```
.
├── ssd_subspace.py        # main pipeline (all 4 methods, all 3 tasks)
├── utils.py               # gradient collection, SVD, projection helpers
├── requirements.txt       # pip dependencies
├── scripts/
│   ├── run_single.sh      # run one (task, loss) combination
│   └── reproduce_all.sh   # run all 6 combinations + summary table
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

### Hardware

| Model | fp16 weights | Peak VRAM (LoRA r=8 fine-tuning) |
|---|---|---|
| Qwen2.5-0.5B-Instruct *(default)* | 1 GB | ~4 GB |
| Qwen2.5-1.5B-Instruct | 3 GB | ~10 GB |
| Qwen2.5-3B-Instruct | 6 GB | ~18 GB |
| Llama-3.1-8B-Instruct | 16 GB | ~32 GB |

---

## The method in 4 steps

```
[1] Find     SVD on gradients of a calibration loss → per-layer rank-r
             projection matrices P_K, P_V.
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
| `baseline` | ❌ | ❌ | ❌ | Pre-training reference. |
| `inference_hooks` | — | ✅ | ❌ | Training-free projection. |
| `ssd_plain` | ❌ | ❌ | ✅ | Vanilla self-distillation (no subspace). |
| `ssd_enhanced` | ✅ | ❌ | ✅ | SSD-Subspace (our method). |

---

## Quickstart — running one task

```bash
# Code: MBPP training, assertion-only CE for subspace, dual-eval (MBPP + CodeAlpaca)
bash scripts/run_single.sh code aligned

# Math: GSM8K training, answer-only CE, dual-eval (GSM8K + SVAMP)
bash scripts/run_single.sh math aligned

# MMLU: MMLU training, answer-only CE, dual-eval (MMLU + BBH)
bash scripts/run_single.sh mmlu aligned

# Use "default" instead of "aligned" for the vanilla-CE baseline variant:
bash scripts/run_single.sh code default
```

Override the model via the `MODEL` env var:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_single.sh code aligned
```

---

## Reproducing the paper's main table

```bash
bash scripts/reproduce_all.sh
```

This runs 3 tasks × 2 loss variants = 6 experiments sequentially, then prints a collated metric table at the end. Results land in `results/<task>_<loss>/results.json`.

Expected approximate numbers with Qwen2.5-0.5B-Instruct:

| Domain | Metric | `baseline` | `ssd_plain` | `ssd_enhanced` (aligned) |
|---|---|---|---|---|
| Code (MBPP) | pass@1 | 11.7% | 30.7% | **20.2%** |
| Math | GSM8K / SVAMP | 11% / 16% | 11% / 12% | **19% / 27%** |
| MMLU | MMLU / BBH | 46% / 32% | 48% / 38% | **48.5% / 37.3%** |

`ssd_plain` is stronger on MBPP because hook-generated code samples can have damaged identifiers; `ssd_enhanced` wins on math and MMLU where the aligned subspace cleanly denoises generation.

---

## Command-line options

All options live on `ssd_subspace.py`; `run_single.sh` just wraps the common ones.

### Core

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal LM with standard `self_attn.{k_proj, v_proj}` |
| `--task` | `math` | `math`, `code`, or `mmlu` |
| `--n_train` | task-dependent | Number of prompts for SSD sample generation + fine-tuning |
| `--n_calibration` | task-dependent | Examples used for gradient SVD |
| `--n_eval` | `100`–`200` | Evaluation examples per dataset |
| `--rank` | `64` | Subspace rank |
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

### The key ablation knob

| Flag | Choices | Purpose |
|---|---|---|
| `--calibration_source` | `default` / `mbpp_solutions` / `answer_only` | Which tokens contribute gradient during subspace extraction |
| `--project_mode` | `both` / `k_only` / `v_only` | Project K, V, or both |

---

## The two loss variants (the central finding)

Both variants use next-token cross-entropy. They differ only in which tokens' labels are *not* masked to `-100`:

| Variant | `--calibration_source` | Valid tasks | Tokens that contribute gradient |
|---|---|---|---|
| Full CE | `default` | all | Every token in calibration text |
| Assertion-only CE | `mbpp_solutions` | code | Only tokens inside `assert …` strings |
| Answer-only CE | `answer_only` | math / mmlu | Only the numeric or letter answer |

Empirically, the correctness-aligned variants consistently outperform `default` for `ssd_enhanced` (MBPP: **6.6% → 20.2%**; GSM8K: **13% → 19%**; MMLU: **48% → 48.5%**).

---

## Output format — `results.json`

```jsonc
{
  "timestamp": "2026-…",
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

| Domain | In-domain (train + eval) | Transfer (eval only) |
|---|---|---|
| Code | MBPP (`mbpp`, `sanitized` split) | CodeAlpaca-20k (`sahil2801/CodeAlpaca-20k`) |
| Math | GSM8K (`openai/gsm8k`, `main`) | SVAMP (`ChilleD/SVAMP`) |
| QA | MMLU (`cais/mmlu`, `all`) | BBH (`lukaemon/bbh`), 6 subtasks |

All datasets are auto-downloaded via the HuggingFace `datasets` library on first use.

BBH subtasks used (all single-token answer formats):
`logical_deduction_three_objects`, `date_understanding`, `movie_recommendation`, `boolean_expressions`, `causal_judgement`, `sports_understanding`.

---

## Windows users

The bash scripts work under Git Bash / WSL. For native PowerShell you can run:

```powershell
python ssd_subspace.py --task mmlu --calibration_source answer_only `
    --n_train 2000 --n_calibration 500 --n_eval 200 `
    --output_dir results/mmlu_aligned
```

---

## FAQ

**Q: Does this need a teacher model?**
No. The student bootstraps from its own generations. Ground-truth answers only appear in calibration texts (for SVD) and eval targets — never as fine-tuning labels.

**Q: Why are `ssd_plain` samples sometimes as good as `ssd_enhanced`?**
When the task is already well-formed (MBPP code completion), the student's natural samples are often on-manifold. The subspace projection helps most when the natural distribution has substantial off-capability noise (math reasoning, MMLU letter answers).

**Q: Do I need GPUs for all 6 reproduction runs?**
Yes, realistically. CPU fine-tuning on 7473 GSM8K samples would take days. A single consumer GPU (RTX 3090 / 4090) handles Qwen2.5-0.5B comfortably.

**Q: Can I scale up the student model?**
Yes — just pass `--model` or set `MODEL`. Increase `--rank` to 128 or 256 for models with `d_kv ≥ 1024`. Reduce `--n_train` if you're memory-bound.

---

## License

The code in this repository is released under MIT. Dataset licenses follow their respective sources on HuggingFace.
