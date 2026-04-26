"""
Vanilla self-training cross-domain experiment.

The model generates its own answers using unconstrained sampling (no top_p
filtering), trains LoRA on those self-generated outputs, and evaluates on
ALL 6 datasets across 3 domains.

Comparison:
  1. baseline         - frozen pretrained model, no training
  2. vanilla_selftrain - trained on model's own generated data (no top_p)

Eval datasets:
  Math:  GSM8K test, SVAMP
  Code:  MBPP sanitized, CodeAlpaca (NLL + AST parse)
  QA:    MMLU test, BBH (6 subtasks)

Usage:
  python vanilla_selftrain.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --domains math,code,mmlu --epochs 5 --output_dir results/vanilla
"""
import argparse, ast, gc, json, os, random, re, subprocess, sys, tempfile, time
from datetime import datetime
from typing import Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_svamp(n_eval: int) -> List[Dict[str, str]]:
    for path, split in [("ChilleD/SVAMP", "test"), ("ChilleD/SVAMP", "train")]:
        try:
            ds = load_dataset(path, split=split)
            break
        except Exception:
            continue
    if n_eval and n_eval > 0:
        ds = ds.select(range(min(n_eval, len(ds))))
    examples = []
    for ex in ds:
        body = str(ex.get("Body", ex.get("body", ""))).strip()
        q = str(ex.get("Question", ex.get("question", ex.get("input", "")))).strip()
        if body and q and body not in q: q = f"{body} {q}"
        elif body and not q: q = body
        ans = ex.get("Answer", ex.get("answer", ex.get("target", ex.get("label", ""))))
        examples.append({"question": q, "answer": str(ans)})
    return examples


def load_mbpp_sanitized_test() -> List[Dict]:
    ds = load_dataset("mbpp", "sanitized", split="test")
    records = []
    for ex in ds:
        records.append({
            "task_id": ex["task_id"],
            "text": ex.get("prompt", ex.get("text", "")),
            "code": ex["code"],
            "test_list": ex["test_list"],
            "test_setup_code": ex.get("test_setup_code", ""),
        })
    return records


def load_codealpaca_eval(n_eval: int) -> List[Dict]:
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception:
        ds = load_dataset("theblackcat102/codealpha-cleaned", split="train")
    n = len(ds)
    idxs = list(range(max(0, n - n_eval), n))
    records = []
    for i in idxs:
        ex = ds[i]
        instr = ex.get("instruction", ex.get("prompt", ""))
        code = ex.get("output", ex.get("completion", ""))
        if not instr or not code:
            continue
        records.append({"instruction": instr, "code": code})
    return records


MMLU_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(n_eval: int, split: str = "test") -> List[Dict]:
    ds = load_dataset("cais/mmlu", "all", split=split)
    if n_eval and n_eval > 0:
        ds = ds.shuffle(seed=42).select(range(min(n_eval, len(ds))))
    records = []
    for ex in ds:
        if len(ex["choices"]) != 4: continue
        records.append({
            "question": ex["question"],
            "choices": ex["choices"],
            "answer_idx": int(ex["answer"]),
            "answer": MMLU_LETTERS[int(ex["answer"])],
            "subject": ex.get("subject", ""),
        })
    return records


def load_mmlu_for_training(n_total: int) -> List[Dict]:
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    except Exception:
        ds = load_dataset("cais/mmlu", "all", split="dev")
    ds = ds.shuffle(seed=42)
    records = []
    for ex in ds:
        if len(ex["choices"]) != 4: continue
        records.append({
            "question": ex["question"],
            "choices": ex["choices"],
            "answer_idx": int(ex["answer"]),
            "answer": MMLU_LETTERS[int(ex["answer"])],
        })
        if len(records) >= n_total: break
    return records


BBH_SUBTASKS = [
    "logical_deduction_three_objects",
    "date_understanding",
    "movie_recommendation",
    "boolean_expressions",
    "causal_judgement",
    "sports_understanding",
]


def load_bbh(n_per_task: int = 50) -> List[Dict]:
    records = []
    for task in BBH_SUBTASKS:
        ds = None
        for path in [("lukaemon/bbh", task, "test"),
                     ("maveriq/bigbenchhard", task, "train")]:
            try:
                ds = load_dataset(path[0], path[1], split=path[2])
                break
            except Exception:
                continue
        if ds is None: continue
        ds = ds.select(range(min(n_per_task, len(ds))))
        for ex in ds:
            records.append({
                "task": task,
                "input": ex["input"],
                "target": str(ex["target"]).strip(),
            })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation functions
# ═══════════════════════════════════════════════════════════════════════════════

def extract_numeric(text):
    matches = re.findall(r"[-+]?\d*\.?\d+", str(text).replace(",", ""))
    return matches[-1] if matches else ""


def answers_match(pred, gold, tol=1e-4):
    if not pred or not gold: return False
    try: return abs(float(pred) - float(gold)) <= tol
    except ValueError: return pred.strip() == gold.strip()


def eval_svamp(model, tokenizer, examples, device, label=""):
    model.eval()
    correct = 0
    for ex in tqdm(examples, desc=f"SVAMP {label}", leave=False):
        prompt = f"Question: {ex['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if answers_match(extract_numeric(resp), extract_numeric(ex["answer"])):
            correct += 1
    total = len(examples)
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total}


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def execute_mbpp(code, test_list, setup_code="", timeout=5):
    script = (setup_code + "\n" if setup_code else "") + code + "\n"
    for t in test_list:
        script += t + "\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script); fp = f.name
        r = subprocess.run([sys.executable, fp], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(fp)
        except Exception: pass


def eval_mbpp(model, tokenizer, records, device, label=""):
    model.eval()
    passed = 0
    for ex in tqdm(records, desc=f"MBPP {label}", leave=False):
        prompt = (f"Instruction: {ex['text']}\nYour code should pass these tests:\n"
                  + "\n".join(ex["test_list"]) + "\nCode:")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(resp)
        for stop in ["\nclass ", "\nif __name__", "\n\n\n"]:
            idx = code.find(stop)
            if idx > 0: code = code[:idx]; break
        if execute_mbpp(code, ex["test_list"], ex.get("test_setup_code", "")):
            passed += 1
    return {"pass@1": passed / len(records), "correct": passed, "total": len(records)}


def eval_codealpaca(model, tokenizer, records, device, label=""):
    model.eval()
    nll_total, tok_total = 0.0, 0
    parses, generated = 0, 0
    for ex in tqdm(records, desc=f"CodeAlpaca {label}", leave=False):
        prompt = f"Instruction: {ex['instruction']}\nCode:"
        full = prompt + " " + ex["code"]
        enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512).to(device)
        enc_full = tokenizer(full, return_tensors="pt", truncation=True,
                              max_length=1024).to(device)
        prompt_len = enc_prompt["input_ids"].shape[1]
        full_len = enc_full["input_ids"].shape[1]
        if full_len > prompt_len:
            labels = enc_full["input_ids"].clone()
            labels[:, :prompt_len] = -100
            with torch.no_grad():
                logits = model(**enc_full).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            nll = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1)).item()
            completion_toks = int((shift_labels != -100).sum().item())
            if completion_toks > 0:
                nll_total += nll
                tok_total += completion_toks
        with torch.no_grad():
            gen = model.generate(**enc_prompt, max_new_tokens=256, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(gen[0][enc_prompt["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        code = extract_code(resp)
        for stop in ["\nclass ", "\nif __name__", "\n\n\n"]:
            idx = code.find(stop)
            if idx > 0:
                code = code[:idx]; break
        generated += 1
        try:
            ast.parse(code)
            parses += 1
        except Exception:
            pass
    mean_nll = (nll_total / tok_total) if tok_total else float("nan")
    return {
        "nll": mean_nll,
        "ppl": float(np.exp(mean_nll)) if tok_total else float("nan"),
        "ast_parse_rate": parses / generated if generated else 0.0,
        "parses": parses,
        "total": generated,
    }


def format_mc_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{MMLU_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


def eval_mmlu(model, tokenizer, records, device, label=""):
    model.eval()
    correct = 0
    for ex in tqdm(records, desc=f"MMLU {label}", leave=False):
        prompt = format_mc_prompt(ex["question"], ex["choices"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        pred = ""
        for ch in resp:
            if ch.upper() in MMLU_LETTERS:
                pred = ch.upper(); break
        if pred == ex["answer"]:
            correct += 1
    total = len(records)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _normalize_bbh(s: str) -> str:
    return s.strip().strip(".").strip().lower()


def _extract_bbh_answer(resp: str, task: str) -> str:
    s = resp.strip().split("\n", 1)[0].strip()
    if task == "boolean_expressions":
        low = s.lower()
        if "true" in low: return "True"
        if "false" in low: return "False"
        return s
    if task in ("causal_judgement", "sports_understanding"):
        low = s.lower()
        if low.startswith("yes") or " yes" in low[:10]:
            return "Yes" if task == "causal_judgement" else "yes"
        if low.startswith("no") or " no" in low[:10]:
            return "No" if task == "causal_judgement" else "no"
        return s
    m = re.search(r"\(([A-F])\)", s)
    if m: return f"({m.group(1)})"
    m = re.search(r"\b([A-F])\b", s)
    if m: return f"({m.group(1)})"
    return s


def eval_bbh(model, tokenizer, records, device, label=""):
    model.eval()
    correct = 0
    per_task_correct = {t: 0 for t in BBH_SUBTASKS}
    per_task_total = {t: 0 for t in BBH_SUBTASKS}
    for ex in tqdm(records, desc=f"BBH {label}", leave=False):
        prompt = f"{ex['input']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        pred = _extract_bbh_answer(resp, ex["task"])
        per_task_total[ex["task"]] += 1
        if _normalize_bbh(pred) == _normalize_bbh(ex["target"]):
            correct += 1
            per_task_correct[ex["task"]] += 1
    total = len(records)
    per_task_acc = {
        t: (per_task_correct[t] / per_task_total[t]) if per_task_total[t] else 0.0
        for t in BBH_SUBTASKS
    }
    return {"accuracy": correct / total if total else 0.0,
            "correct": correct, "total": total,
            "per_task_accuracy": per_task_acc}


# ═══════════════════════════════════════════════════════════════════════════════
# Sample generation (no top_p filtering)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_samples(model, tokenizer, prompts, device, batch_size=8,
                     max_new_tokens=256, temperature=0.7):
    """Generate samples with temperature sampling but NO top_p filtering."""
    model.eval()
    tokenizer.padding_side = "left"
    texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling", leave=False):
        batch_p = prompts[i:i+batch_size]
        inputs = tokenizer(batch_p, return_tensors="pt", truncation=True,
                           max_length=384, padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=True, temperature=temperature,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        for seq in out:
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
            decoded = decoded.replace("\x00", "")
            if len(decoded) > 8192:
                decoded = decoded[:8192]
            texts.append(decoded)
    tokenizer.padding_side = "right"
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        skipped = 0
        for t in texts:
            if not isinstance(t, str): skipped += 1; continue
            t = t.replace("\x00", "")
            if len(t) > 8192: t = t[:8192]
            if not t.strip(): skipped += 1; continue
            try:
                enc = tokenizer(t, truncation=True, max_length=max_length,
                                 padding="max_length", return_tensors="pt")
            except BaseException:
                skipped += 1; continue
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
        if skipped:
            print(f"  [TextDataset] skipped {skipped}/{len(texts)} malformed samples",
                  flush=True)

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def train_lora(model_name, tokenizer, samples, lora_config, epochs, lr, device, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device)
    student.gradient_checkpointing_enable()
    student = get_peft_model(student, lora_config)
    n_params = sum(p.numel() for p in student.parameters()) / 1e9
    bs = 1 if n_params > 2 else 4
    accum_steps = 4 // bs
    ds = TextDataset(samples, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    steps_per_epoch = len(loader)
    horizon = steps_per_epoch * epochs
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(steps_per_epoch // 2, 1), num_training_steps=horizon)
    losses = []
    student.train()
    step = 0
    for epoch in range(epochs):
        ep_loss, nb = 0.0, 0
        for bi, batch in enumerate(tqdm(loader, desc=f"ep{epoch+1}/{epochs}", leave=False)):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = student(input_ids=ids, attention_mask=mask, labels=ids)
            loss = out.loss / accum_steps
            if not torch.isfinite(loss):
                step += 1; continue
            loss.backward()
            if (bi + 1) % accum_steps == 0 or (bi + 1) == len(loader):
                grads_ok = all(torch.isfinite(p.grad).all()
                               for p in student.parameters() if p.grad is not None)
                if not grads_ok:
                    opt.zero_grad(); step += 1; continue
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                if step < horizon: sched.step()
                step += 1
            ep_loss += loss.item() * accum_steps; nb += 1
        avg = ep_loss / max(nb, 1)
        losses.append(avg)
        print(f"    ep={epoch+1}/{epochs}: loss={avg:.4f}", flush=True)
        torch.cuda.empty_cache()
    return student, losses


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_all_eval_data(n_eval):
    print("Loading all 6 eval datasets...", flush=True)
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    if n_eval > 0:
        gsm8k_test = gsm8k_test.select(range(min(n_eval, len(gsm8k_test))))
    gsm8k_records = []
    for ex in gsm8k_test:
        m = re.search(r"####\s*([\d,.-]+)", ex["answer"])
        ans = m.group(1).replace(",", "") if m else ex["answer"]
        gsm8k_records.append({"question": ex["question"], "answer": str(ans)})

    svamp_records = load_svamp(n_eval)
    mbpp_records = load_mbpp_sanitized_test()[:n_eval]
    ca_records = load_codealpaca_eval(n_eval)
    mmlu_records = load_mmlu(n_eval, split="test")
    bbh_records = load_bbh(n_per_task=50)

    print(f"  GSM8K: {len(gsm8k_records)} | SVAMP: {len(svamp_records)} | "
          f"MBPP: {len(mbpp_records)} | CA: {len(ca_records)} | "
          f"MMLU: {len(mmlu_records)} | BBH: {len(bbh_records)}", flush=True)
    return {
        "gsm8k": gsm8k_records, "svamp": svamp_records,
        "mbpp": mbpp_records, "codealpaca": ca_records,
        "mmlu": mmlu_records, "bbh": bbh_records,
    }


def eval_all_6(model, tokenizer, eval_data, device):
    r = {}
    print("    eval GSM8K...", flush=True)
    g = eval_svamp(model, tokenizer, eval_data["gsm8k"], device, label="gsm8k")
    r["gsm8k_accuracy"] = g["accuracy"]

    print("    eval SVAMP...", flush=True)
    s = eval_svamp(model, tokenizer, eval_data["svamp"], device, label="svamp")
    r["svamp_accuracy"] = s["accuracy"]

    print("    eval MBPP...", flush=True)
    m = eval_mbpp(model, tokenizer, eval_data["mbpp"], device, label="mbpp")
    r["mbpp_pass@1"] = m["pass@1"]

    print("    eval CodeAlpaca...", flush=True)
    c = eval_codealpaca(model, tokenizer, eval_data["codealpaca"], device, label="ca")
    r["ca_nll"] = c["nll"]
    r["ca_ppl"] = c["ppl"]
    r["ca_ast_parse_rate"] = c["ast_parse_rate"]

    print("    eval MMLU...", flush=True)
    mm = eval_mmlu(model, tokenizer, eval_data["mmlu"], device, label="mmlu")
    r["mmlu_accuracy"] = mm["accuracy"]

    print("    eval BBH...", flush=True)
    b = eval_bbh(model, tokenizer, eval_data["bbh"], device, label="bbh")
    r["bbh_accuracy"] = b["accuracy"]
    return r


def load_math_train_prompts(n_train):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    n = min(n_train, len(ds))
    return [f"Question: {ex['question']}\nAnswer:" for ex in ds.select(range(n))]


def load_code_train_prompts(n_train):
    prompts = []
    for split in ["train", "validation"]:
        ds = load_dataset("mbpp", split=split)
        for ex in ds:
            p = (f"{ex['text']}\nYour code should pass these tests:\n"
                 + "\n".join(ex["test_list"]))
            prompts.append(f"Instruction: {p}\nCode:")
    if n_train > len(prompts) and len(prompts) > 0:
        reps = (n_train // len(prompts)) + 1
        prompts = (prompts * reps)[:n_train]
    else:
        prompts = prompts[:n_train]
    return prompts


def load_mmlu_train_prompts(n_train):
    data = load_mmlu_for_training(n_train)
    return [format_mc_prompt(r["question"], r["choices"]) for r in data[:n_train]]


def run_domain(domain, model_name, tok, eval_data, args, lora_cfg):
    device = args.device
    print(f"\n{'='*60}", flush=True)
    print(f"DOMAIN: {domain.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    cfg = {"math": 7473, "code": 464, "mmlu": 2000}
    n_train = cfg[domain]

    if domain == "math":
        train_prompts = load_math_train_prompts(n_train)
    elif domain == "code":
        train_prompts = load_code_train_prompts(n_train)
    else:
        train_prompts = load_mmlu_train_prompts(n_train)

    print(f"  Train prompts: {len(train_prompts)}", flush=True)
    domain_results = {}

    # ── Step 1: Baseline eval ──
    print(f"\n  [{domain}] Baseline eval on all 6 datasets...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device)
    base_model.eval()
    baseline_r = eval_all_6(base_model, tok, eval_data, device)
    domain_results["baseline"] = baseline_r
    print(f"    baseline: gsm8k={baseline_r['gsm8k_accuracy']:.2%} "
          f"svamp={baseline_r['svamp_accuracy']:.2%} "
          f"mbpp={baseline_r['mbpp_pass@1']:.2%} ca_nll={baseline_r['ca_nll']:.3f} "
          f"mmlu={baseline_r['mmlu_accuracy']:.2%} bbh={baseline_r['bbh_accuracy']:.2%}",
          flush=True)

    # ── Step 2: Generate self-training samples (no top_p) ──
    print(f"\n  [{domain}] Generating vanilla samples (temperature=0.7, no top_p)...",
          flush=True)
    vanilla_samples = generate_samples(
        base_model, tok, train_prompts, device, batch_size=8, temperature=0.7)
    print(f"    Generated {len(vanilla_samples)} samples", flush=True)
    del base_model; cuda_cleanup()

    # ── Step 3: Train LoRA on self-generated samples ──
    print(f"\n  [{domain}] Training vanilla_selftrain (LoRA)...", flush=True)
    trained_model, train_losses = train_lora(
        model_name, tok, vanilla_samples, lora_cfg, args.epochs, args.lr, device)
    domain_results["train_losses"] = train_losses

    # ── Step 4: Evaluate trained model on all 6 datasets ──
    print(f"  [{domain}] vanilla_selftrain eval on all 6 datasets...", flush=True)
    r = eval_all_6(trained_model, tok, eval_data, device)
    domain_results["vanilla_selftrain"] = r
    print(f"    vanilla: gsm8k={r['gsm8k_accuracy']:.2%} svamp={r['svamp_accuracy']:.2%} "
          f"mbpp={r['mbpp_pass@1']:.2%} ca_nll={r['ca_nll']:.3f} "
          f"mmlu={r['mmlu_accuracy']:.2%} bbh={r['bbh_accuracy']:.2%}", flush=True)

    del trained_model; cuda_cleanup()
    return domain_results


def print_summary(all_results):
    datasets = ["gsm8k_accuracy", "svamp_accuracy", "mbpp_pass@1",
                "ca_nll", "mmlu_accuracy", "bbh_accuracy"]

    for domain in ["math", "code", "mmlu"]:
        if domain not in all_results:
            continue
        dr = all_results[domain]
        print(f"\n{'='*80}", flush=True)
        print(f"  Train domain: {domain.upper()}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  {'Method':18s} {'GSM8K':>8s} {'SVAMP':>8s} {'MBPP':>8s} "
              f"{'CA_NLL':>8s} {'MMLU':>8s} {'BBH':>8s}", flush=True)
        print(f"  {'-'*66}", flush=True)

        for method in ["baseline", "vanilla_selftrain"]:
            r = dr.get(method, {})
            vals = []
            for ds_key in datasets:
                v = r.get(ds_key, None)
                if v is None: vals.append("  N/A   ")
                elif ds_key == "ca_nll": vals.append(f"{v:8.3f}")
                else: vals.append(f"{v:7.2%} ")
            print(f"  {method:18s} {''.join(vals)}", flush=True)

        base = dr.get("baseline", {})
        trained = dr.get("vanilla_selftrain", {})
        deltas = []
        for ds_key in datasets:
            bv = base.get(ds_key)
            ev = trained.get(ds_key)
            if bv is None or ev is None:
                deltas.append("  N/A   ")
            elif ds_key == "ca_nll":
                deltas.append(f"{ev - bv:+8.3f}")
            else:
                deltas.append(f"{ev - bv:+7.2%} ")
        print(f"  {'delta':18s} {''.join(deltas)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Vanilla self-training cross-domain experiment")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", default="math,code,mmlu",
                    help="Comma-separated domains to run")
    ap.add_argument("--output_dir", default="results/vanilla_selftrain")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    print("=" * 60, flush=True)
    print("Vanilla Self-Training Cross-Domain Experiment", flush=True)
    print(f"Model: {args.model} | Epochs: {args.epochs}", flush=True)
    print(f"Generation: temperature=0.7, NO top_p (unconstrained sampling)", flush=True)
    print("=" * 60, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM")

    eval_data = load_all_eval_data(args.n_eval)
    domains = [d.strip() for d in args.domains.split(",")]

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
    }
    results_path = os.path.join(args.output_dir, "results.json")

    def save():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    for domain in domains:
        t0 = time.time()
        dr = run_domain(domain, args.model, tok, eval_data, args, lora_cfg)
        dr["time_sec"] = time.time() - t0
        all_results[domain] = dr
        save()
        print(f"\n  [{domain}] completed in {dr['time_sec']:.0f}s", flush=True)

    print("\n\n" + "=" * 80, flush=True)
    print("VANILLA SELF-TRAINING CROSS-DOMAIN SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print_summary(all_results)
    save()
    print(f"\nResults saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
