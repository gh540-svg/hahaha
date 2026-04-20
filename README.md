# SSD-Subspace: Capability-Selective Self-Distillation via KV Subspace Projection

Single-script pipeline (`ssd_subspace.py`) that tests whether a small student model
can self-distill a targeted capability by:

1. **Finding** a low-rank capability subspace of the student's K/V activations
2. **Projecting** K/V onto that subspace at selected attention layers during generation
3. **Harvesting** self-samples produced under the projection
4. **Training** the student on those self-samples (no projection during training)

Supported capabilities: **math** (GSM8K / SVAMP), **code** (MBPP / CodeAlpaca),
**question answering** (MMLU with BBH transfer).

---

## The 4 methods evaluated per run

Every run produces all 4 — no separate flags required.

| Method | Subspace hooks? | Training? | What it tests |
|---|---|---|---|
| `baseline` | ❌ | ❌ | Frozen student. Reference point. |
| `inference_hooks` | ✅ at eval time | ❌ | Training-free: does the subspace projection alone help at inference? |
| `ssd_plain` | ❌ anywhere | ✅ on unhooked self-samples | Vanilla self-distillation (no subspace). |
| `ssd_enhanced` | ✅ during sample generation, ❌ during training | ✅ on hook-generated self-samples | The full method. Student trains on its hook-concentrated outputs. |

---

## Requirements

- Python 3.10+
- `torch` (CUDA recommended, fp16 is default)
- `transformers`, `peft`, `datasets`, `tqdm`, `numpy`
- Disk: HuggingFace dataset cache (~5 GB for MMLU auxiliary_train, ~0.5 GB for MBPP/GSM8K/SVAMP/BBH)

First run will download datasets; subsequent runs hit the cache.

---

## Quickstart

From the project root (`capability-selective-distillation/`):

```bash
# Math: GSM8K training + dual eval on GSM8K test and SVAMP
python ssd_subspace.py \
    --task math --math_eval both \
    --n_train 7473 --n_calibration 50 --n_eval 100 \
    --output_dir results/my_math_run

# Code: MBPP training + dual eval on MBPP pass@1 and CodeAlpaca NLL/AST
python ssd_subspace.py \
    --task code --code_train mbpp --code_eval both \
    --n_train 2000 --n_calibration 50 --n_eval 100 \
    --output_dir results/my_code_run

# Question answering: MMLU training + dual eval on MMLU and BBH
python ssd_subspace.py \
    --task mmlu \
    --n_train 2000 --n_calibration 500 --n_eval 200 \
    --output_dir results/my_mmlu_run
```

Each run writes `results.json` into `--output_dir` with the 4 methods' metrics.

---

## All command-line flags

### Core

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Student model. Any HF causal LM works; edit layer count logic if different architecture. |
| `--task` | `math` | One of `math`, `code`, `mmlu`. |
| `--n_train` | `7473` | Number of training prompts used for SSD sample generation and fine-tuning. |
| `--n_calibration` | `50` | Calibration examples used for subspace gradient SVD. |
| `--n_eval` | `100` | Eval examples per dataset. |
| `--rank` | `64` | Rank of the capability subspace. |
| `--epochs` | `5` | Fine-tuning epochs on self-samples. |
| `--lr` | `1e-5` | Fine-tuning learning rate. |
| `--lora_r` | `8` | LoRA rank for fine-tuning adapters. |
| `--seed` | `42` | Torch/NumPy/random seed. |
| `--layers` | `last_mid` | Either `last_mid` (auto: `[n_layers-1, n_layers//2]`) or a comma-separated list like `"23,12"` or `"12"` or `"15,12"`. |
| `--output_dir` | `results/ssd_subspace` | Where `results.json`, training log files, and LoRA adapters are written. |
| `--device` | `cuda` | Torch device. |

### Task-specific eval / train

| Flag | Choices | Purpose |
|---|---|---|
| `--math_eval` | `svamp`, `gsm8k_test`, `both` | `svamp` = transfer, `gsm8k_test` = same-dataset, `both` = dual eval. |
| `--code_train` | `mbpp`, `codealpaca` | Training prompt source. MBPP → same-dataset, CodeAlpaca → transfer. |
| `--code_eval` | `mbpp_sanitized`, `codealpaca`, `both` | MBPP pass@1, CodeAlpaca NLL+AST-parse, or dual. |

### Ablation knobs

| Flag | Choices | Purpose |
|---|---|---|
| `--project_mode` | `both`, `k_only`, `v_only` | Which of K and V to project inside the hook. `k_only` preserves token content, useful for syntactically rigid tasks (code). |
| `--calibration_source` | `default`, `mbpp_solutions`, `answer_only` | See next section. |

---

## Subspace loss functions (`--calibration_source`)

The subspace is built by SVD on gradients of a CE loss over calibration texts.
**Which tokens contribute to the loss determines which KV directions the subspace captures** — this is the most important research knob.

| Option | Valid tasks | What contributes to the loss | What the subspace captures |
|---|---|---|---|
| `default` | all | Every token in the calibration text | "KV directions for reproducing training-distribution text" (broad, includes scaffolding) |
| `mbpp_solutions` | `code` | Only tokens inside `assert ...` test strings in each MBPP prompt | "KV directions for predicting the test assertions" — execution-correctness-aligned |
| `answer_only` | `math`, `mmlu` | Only the final answer tokens (the numeric answer after `#### ` in GSM8K; the letter after `Answer: ` in MMLU) | "KV directions for picking the correct final answer" |

Empirically, the correctness-aligned variants (`mbpp_solutions` / `answer_only`)
consistently outperform `default` for `ssd_enhanced`. See *Findings* below.

---

## What each run does (pipeline)

```
[1] baseline eval (frozen student, no hooks)
[2] compute capability subspace
     forward+backward on calibration texts → SVD → rank-r projection matrices P_K, P_V
     at each target layer
[3] inference_hooks eval (hooks active on frozen student)
[4] generate PLAIN self-samples (hooks off)
[5] generate ENHANCED self-samples (hooks on)
[6] train a LoRA adapter on the plain samples → eval → save as ssd_plain
[7] train a LoRA adapter on the enhanced samples → eval → save as ssd_enhanced
    (no hooks during training; standard next-token CE on whole sample)
[8] write results.json
```

Samples in steps [4] and [5] are `prompt + student-generated-completion` — **no
ground-truth answers are ever spliced into training data.** Ground-truth answers
only appear in calibration texts (step [2], discarded after SVD).

---

## Output format — `results.json`

```jsonc
{
  "timestamp": "...",
  "config": { /* all CLI args */ },
  "target_layers": [23, 12],
  "projections": {
    "23": { "energy_K": 0.96, "energy_V": 0.95 },
    "12": { "energy_K": 0.99, "energy_V": 0.94 }
  },
  "results": {
    "baseline":        { /* primary metric + any dual-eval sub-metrics */ },
    "inference_hooks": { /* same shape */ },
    "ssd_plain":       { /* + train_losses, best_loss, best_epoch */ },
    "ssd_enhanced":    { /* same */ }
  }
}
```

Metric keys by task/eval:

| Task / eval | Primary | Extras |
|---|---|---|
| math / svamp | `accuracy` | `correct`, `total` |
| math / gsm8k_test | `accuracy` | `correct`, `total` |
| math / both | `gsm8k_accuracy` | `svamp_accuracy`, `{correct,total}` each |
| code / mbpp_sanitized | `pass@1` | `correct`, `total` |
| code / codealpaca | `nll`, `ast_parse_rate` | `ppl`, `parses`, `total` |
| code / both | `mbpp_pass@1` | `ca_nll`, `ca_ast_parse_rate`, ... |
| mmlu | `mmlu_accuracy` | `bbh_accuracy`, `bbh_per_task_accuracy` (6 subtasks) |

For `ssd_plain` / `ssd_enhanced`, an additional `train_losses` array (per epoch),
`best_loss`, `best_epoch` are recorded.

LoRA adapters are saved into `<output_dir>/student_ssd_plain_seed{SEED}/` and
`<output_dir>/student_ssd_enhanced_seed{SEED}/` (loadable via `peft.PeftModel`).

---

## Example commands — reproducing our findings

### Math, correctness-aligned subspace (best for transfer)
```bash
python ssd_subspace.py \
    --task math --math_eval both \
    --calibration_source answer_only \
    --n_train 7473 --n_calibration 50 --n_eval 100 \
    --output_dir results/math_answer_only
```

### Code, execution-aware subspace (best ssd_enhanced for MBPP pass@1)
```bash
python ssd_subspace.py \
    --task code --code_train mbpp --code_eval both \
    --calibration_source mbpp_solutions \
    --n_train 2000 --n_calibration 50 --n_eval 100 \
    --output_dir results/code_execaware
```

### MMLU, answer-only subspace + BBH transfer eval
```bash
python ssd_subspace.py \
    --task mmlu \
    --calibration_source answer_only \
    --n_train 2000 --n_calibration 500 --n_eval 200 \
    --output_dir results/mmlu_answer_only
```

### Layer ablation — project only middle layer
```bash
python ssd_subspace.py \
    --task code --code_train mbpp --code_eval both \
    --layers "12" \
    --output_dir results/code_mid_only
```

### Project K only, preserve V (code-friendly)
```bash
python ssd_subspace.py \
    --task code --code_train mbpp --code_eval both \
    --project_mode k_only \
    --output_dir results/code_kproj_only
```

---

## Findings

Across all three domains:

- The **inference-time hook** helps only when the subspace is built from a
  correctness-aligned loss (`mbpp_solutions` / `answer_only`). With `default`
  CE, the hook often hurts because the subspace captures style, not correctness.

- **`ssd_enhanced`** gives the largest gains when the correctness-aligned
  subspace is paired with the projection:
  - Code MBPP pass@1: **6.6% → 20.2%** switching `default` → `mbpp_solutions`
  - MMLU: **48.0% → 48.5%**, BBH transfer **35.7% → 37.3%** switching `default` → `answer_only`
  - Math SVAMP transfer: `ssd_enhanced` reached **24%** vs baseline 16% even with `default`; `answer_only` results pending.

- The subspace's **rank-64 captured energy** jumps from ~94–98% (`default`) to
  ~99.5–100% (correctness-aligned). The gradient matrix is lower-rank because
  only a handful of tokens per example contribute — this is a feature, not a
  bug: the subspace becomes a sharper correctness filter.

- **No ground-truth answers are used during fine-tuning.** The improvement
  comes from training on *self-generated but subspace-concentrated* samples.
  The wrong content averages out; the on-manifold structure accumulates into
  the weights.

---

## File layout

```
./
├── ssd_subspace.py     # the main pipeline (all 4 methods + all 3 tasks)
├── utils.py            # gradient collection, SVD, projection helpers
└── README.md           # this file
```

Just two Python files. Run with plain `python ssd_subspace.py --task ...`
— no installation step beyond `pip install torch transformers peft datasets tqdm numpy`.
