"""
SSD-Enhanced: combine training-free 2-layer subspace projection with SSD.

Method:
  Phase 1: Compute student's AutoKV capability subspace at last + middle layers.
  Phase 2: Register forward hooks that project K/V at those layers onto the subspace.
  Phase 3: With hooks active, sample outputs from the student (better outputs than
           plain student because KV is math-focused).
  Phase 4: Remove hooks. Train the student (LoRA) on those enhanced samples using
           standard SSD loss (next-token cross-entropy).

Comparisons (4 methods):
  1. baseline           — frozen student, no modification
  2. inference_hooks    — frozen student + hooks at inference time (training-free)
  3. ssd_plain          — SSD trained on plain self-samples (no hooks)
  4. ssd_enhanced       — SSD trained on hook-generated self-samples (our idea)

Evaluation: SVAMP for math, MBPP sanitized test for code.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import collect_kv_gradients, compute_projection_matrix, extract_subspace


# ── Eval: math ───────────────────────────────────────────────────────────────

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


# ── Eval: code (MBPP) ────────────────────────────────────────────────────────

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


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


import ast


def load_codealpaca_eval(n_eval: int) -> List[Dict]:
    """Held-out CodeAlpaca eval split. Uses the LAST n_eval examples so
    that when --code_train=codealpaca (uses the first 2000) there's no overlap.
    CodeAlpaca has no executable tests, so we evaluate by NLL + AST parse rate."""
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


def eval_codealpaca(model, tokenizer, records, device, label=""):
    """nll: mean per-token NLL of reference completion given the prompt (lower=better).
       ast_parse_rate: fraction of greedy generations that parse as valid Python."""
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
        # AST parse on greedy gen
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


# ── Eval: MMLU (4-choice letter) ────────────────────────────────────────────

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
    """Train/calibration source: MMLU auxiliary_train (99K diverse MC)."""
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


def format_mc_prompt(question: str, choices: List[str]) -> str:
    opts = "\n".join(f"{MMLU_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


def format_mc_full(question: str, choices: List[str], answer_letter: str) -> str:
    return format_mc_prompt(question, choices) + f" {answer_letter}"


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
    return {"accuracy": correct / total if total else 0.0,
            "correct": correct, "total": total}


# ── Eval: BBH (6 representative subtasks) ───────────────────────────────────

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


def _normalize_bbh(s: str) -> str:
    return s.strip().strip(".").strip().lower()


def _extract_bbh_answer(resp: str, task: str) -> str:
    s = resp.strip()
    # Take first line of non-empty output
    s = s.split("\n", 1)[0].strip()
    if task == "boolean_expressions":
        low = s.lower()
        if "true" in low: return "True"
        if "false" in low: return "False"
        return s
    if task in ("causal_judgement", "sports_understanding"):
        low = s.lower()
        if low.startswith("yes") or " yes" in low[:10]: return "Yes" if task == "causal_judgement" else "yes"
        if low.startswith("no") or " no" in low[:10]: return "No" if task == "causal_judgement" else "no"
        return s
    # multi-choice tasks expect "(A)"-style target
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
            "per_task_accuracy": per_task_acc,
            "per_task_correct": per_task_correct,
            "per_task_total": per_task_total}


# ── Hook manager for subspace projection ────────────────────────────────────

class SubspaceHooks:
    """Forward hooks that post-process k_proj/v_proj outputs at target layers.

    mode: 'both' projects both K and V (default), 'k_only' projects only K,
          'v_only' projects only V.
    """
    def __init__(self, model, projections: Dict[int, Dict[str, torch.Tensor]],
                 device: str, mode: str = "both"):
        self.handles = []
        self.mode = mode
        for layer_idx, p in projections.items():
            attn = model.model.layers[layer_idx].self_attn
            P_K = p["P_K"].to(device)
            P_V = p["P_V"].to(device)
            def make_hook(P):
                def hook(module, inp, out):
                    return out @ P.to(out.device).to(out.dtype)
                return hook
            if mode in ("both", "k_only"):
                self.handles.append(attn.k_proj.register_forward_hook(make_hook(P_K)))
            if mode in ("both", "v_only"):
                self.handles.append(attn.v_proj.register_forward_hook(make_hook(P_V)))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


# ── Compute student's own AutoKV projections ────────────────────────────────

def compute_student_projections(model, tokenizer, calibration_texts,
                                 target_layers, rank, device, max_length=256,
                                 label_char_spans=None, energy_threshold=None,
                                 half_rank=False):
    """Rank selection modes (checked in order):
      1. half_rank=True: use half of the full KV dimension
         (e.g. 64 for 0.5B kv_dim=128, 256 for 7B kv_dim=512)
      2. energy_threshold (e.g. 0.95): auto-chosen per (layer, K/V)
      3. fixed rank (int)
    """
    grads = collect_kv_gradients(model, tokenizer, calibration_texts,
                                   target_layers, max_length=max_length, device=device,
                                   label_char_spans=label_char_spans)
    extract_kwargs = {}
    if half_rank:
        extract_kwargs["half_rank"] = True
    elif energy_threshold is not None:
        extract_kwargs["energy_threshold"] = energy_threshold
    else:
        extract_kwargs["rank"] = rank
    projections = {}
    for layer in target_layers:
        V_k, S_k, r_k = extract_subspace(grads[layer]["K_grads"], **extract_kwargs)
        V_v, S_v, r_v = extract_subspace(grads[layer]["V_grads"], **extract_kwargs)
        projections[layer] = {
            "P_K": compute_projection_matrix(V_k),
            "P_V": compute_projection_matrix(V_v),
            "rank_K": int(r_k),
            "rank_V": int(r_v),
            "energy_K": float((S_k[:r_k]**2).sum() / (S_k**2).sum()),
            "energy_V": float((S_v[:r_v]**2).sum() / (S_v**2).sum()),
        }
    return projections


# ── Batched sample generation ───────────────────────────────────────────────

def generate_samples(model, tokenizer, prompts, device, batch_size=8,
                     max_new_tokens=256, temperature=0.7, top_p=0.9):
    model.eval()
    tokenizer.padding_side = "left"
    texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling", leave=False):
        batch_p = prompts[i:i+batch_size]
        inputs = tokenizer(batch_p, return_tensors="pt", truncation=True,
                           max_length=384, padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=True, temperature=temperature, top_p=top_p,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        for seq in out:
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
            # Strip null/control bytes that crash the Rust tokenizer downstream
            decoded = decoded.replace("\x00", "")
            if len(decoded) > 8192:
                decoded = decoded[:8192]
            texts.append(decoded)
    tokenizer.padding_side = "right"
    return texts


# ── SSD training ────────────────────────────────────────────────────────────

def _sanitize_sample(text, max_chars=8192):
    """Guard against inputs that crash the Rust-backed fast tokenizer
    (pyo3 PanicException on malformed/oversized strings)."""
    if not isinstance(text, str):
        return None
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        text = text[:max_chars]
    if not text.strip():
        return None
    return text


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        skipped = 0
        for t in texts:
            t = _sanitize_sample(t)
            if t is None:
                skipped += 1
                continue
            try:
                enc = tokenizer(t, truncation=True, max_length=max_length,
                                 padding="max_length", return_tensors="pt")
            except BaseException:
                skipped += 1
                continue
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
        if skipped:
            print(f"  [TextDataset] skipped {skipped}/{len(texts)} malformed samples",
                  flush=True)
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def train_on_samples(student_name, tokenizer, samples, lora_config,
                      epochs, lr, device, seed,
                      patience=1, min_delta=1e-4):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16, device_map=device)
    student = get_peft_model(student, lora_config)

    ds = TextDataset(samples, tokenizer, max_length=256)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    steps_per_epoch = len(loader)
    lr_horizon = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(steps_per_epoch // 2, 1),
        num_training_steps=lr_horizon)

    train_losses = []
    best_loss, best_epoch, best_state, no_improve = float("inf"), 0, None, 0
    student.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in tqdm(loader, desc=f"ep{epoch+1}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = student(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if global_step < lr_horizon: scheduler.step()
            epoch_loss += loss.item(); n_batches += 1; global_step += 1
        avg = epoch_loss / max(n_batches, 1)
        train_losses.append(avg)
        print(f"  ep={epoch+1}/{epochs}: loss={avg:.4f}", flush=True)
        if avg < best_loss - min_delta:
            best_loss, best_epoch = avg, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        torch.cuda.empty_cache()
        if no_improve >= patience: break
    if best_state is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return student, {"train_losses": train_losses, "best_loss": best_loss, "best_epoch": best_epoch}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", choices=["math", "code", "mmlu"], default="math")
    parser.add_argument("--n_train", type=int, default=7473,
                        help="Math: 7473 GSM8K. Code: 464 MBPP train+val.")
    parser.add_argument("--n_calibration", type=int, default=50)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--rank", type=int, default=0,
                        help="Subspace rank. 0 (default) = half of KV dim. "
                             "Set >0 to force a fixed rank.")
    parser.add_argument("--rank_mode", default="half",
                        choices=["half", "energy", "fixed"],
                        help="Rank selection: 'half' = kv_dim//2 (default), "
                             "'energy' = auto via --rank_energy threshold, "
                             "'fixed' = use --rank value directly.")
    parser.add_argument("--rank_energy", type=float, default=0.95,
                        help="Energy threshold for rank_mode='energy'. Default 0.95.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="last_mid",
                        help="'last_mid' or comma-separated list like '23,12'")
    parser.add_argument("--math_eval", default="svamp",
                        choices=["svamp", "gsm8k_test", "both"],
                        help="Math eval dataset. 'svamp' = transfer only, "
                             "'gsm8k_test' = same-dataset only, "
                             "'both' = dual eval (GSM8K test + SVAMP).")
    parser.add_argument("--code_train", default="mbpp",
                        choices=["mbpp", "codealpaca"],
                        help="Code train dataset. 'mbpp' = same-dataset, 'codealpaca' = transfer")
    parser.add_argument("--code_eval", default="mbpp_sanitized",
                        choices=["mbpp_sanitized", "codealpaca", "both"],
                        help="Code eval. 'mbpp_sanitized' → pass@1 on MBPP. "
                             "'codealpaca' → NLL + AST-parse on held-out CodeAlpaca. "
                             "'both' → run both evals per method, merge metrics.")
    parser.add_argument("--project_mode", default="both",
                        choices=["both", "k_only", "v_only"],
                        help="Which KV projections to apply in hooks.")
    parser.add_argument("--calibration_source", default="aligned",
                        choices=["aligned", "mbpp_solutions", "answer_only"],
                        help="Correctness-aligned CE: gradient only on "
                             "assertion tokens (code) or answer tokens "
                             "(math/MMLU). 'aligned' auto-selects per task; "
                             "'mbpp_solutions' / 'answer_only' override.")
    parser.add_argument("--output_dir", default="results/ssd_subspace")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.calibration_source == "aligned":
        if args.task == "code":
            args.calibration_source = "mbpp_solutions"
        else:
            args.calibration_source = "answer_only"

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print(f"SSD-Enhanced via subspace projection (task={args.task})")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load student
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device)
    student.eval()
    n_layers = student.config.num_hidden_layers
    if args.layers == "last_mid":
        target_layers = [n_layers - 1, n_layers // 2]
    else:
        target_layers = [int(x) for x in args.layers.split(",")]
    print(f"Target layers for projection: {target_layers}")

    # Load calibration + training + eval data
    if args.task == "math":
        ds_train = load_dataset("openai/gsm8k", "main", split="train")
        cal_ds = ds_train.select(range(min(args.n_calibration, len(ds_train))))
        cal_texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}"
                     for ex in cal_ds]
        n_train = min(args.n_train, len(ds_train))
        train_prompts = [f"Question: {ex['question']}\nAnswer:"
                         for ex in ds_train.select(range(n_train))]

        # Execaware analog for math: gradient only on the numeric answer that
        # follows the GSM8K "####" marker. Everything else (question, reasoning
        # chain) is masked to -100.
        cal_label_spans = None
        if args.calibration_source == "answer_only":
            cal_label_spans = []
            marker = "#### "
            for t in cal_texts:
                pos = t.rfind(marker)
                if pos < 0:
                    cal_label_spans.append([])
                    continue
                cal_label_spans.append([(pos + len(marker), len(t))])
            total = sum(len(s) for s in cal_label_spans)
            print(f"Calibration override: GSM8K + answer-only CE loss "
                  f"({total} numeric-answer spans over {len(cal_texts)} texts)")

        if args.math_eval == "svamp":
            eval_records = load_svamp(args.n_eval)
            eval_fn = lambda m: eval_svamp(m, tok, eval_records, args.device, label="svamp")
            metric_name = "accuracy"
            print(f"Math eval: SVAMP (transfer from GSM8K)")
        elif args.math_eval == "gsm8k_test":
            ds_test = load_dataset("openai/gsm8k", "main", split="test")
            if args.n_eval and args.n_eval > 0:
                ds_test = ds_test.select(range(min(args.n_eval, len(ds_test))))
            eval_records = []
            for ex in ds_test:
                m = re.search(r"####\s*([\d,.-]+)", ex["answer"])
                ans = m.group(1).replace(",", "") if m else ex["answer"]
                eval_records.append({"question": ex["question"], "answer": str(ans)})
            eval_fn = lambda m: eval_svamp(m, tok, eval_records, args.device, label="gsm8k")
            metric_name = "accuracy"
            print(f"Math eval: GSM8K test (same-dataset as train)")
        else:  # both
            svamp_records = load_svamp(args.n_eval)
            ds_test = load_dataset("openai/gsm8k", "main", split="test")
            if args.n_eval and args.n_eval > 0:
                ds_test = ds_test.select(range(min(args.n_eval, len(ds_test))))
            gsm8k_records = []
            for ex in ds_test:
                m = re.search(r"####\s*([\d,.-]+)", ex["answer"])
                ans = m.group(1).replace(",", "") if m else ex["answer"]
                gsm8k_records.append({"question": ex["question"], "answer": str(ans)})
            def dual_math_eval(m):
                r_g = eval_svamp(m, tok, gsm8k_records, args.device, label="gsm8k")
                r_s = eval_svamp(m, tok, svamp_records, args.device, label="svamp")
                return {
                    "accuracy": r_g["accuracy"],  # primary = in-domain GSM8K
                    "gsm8k_accuracy": r_g["accuracy"],
                    "gsm8k_correct": r_g["correct"],
                    "gsm8k_total": r_g["total"],
                    "svamp_accuracy": r_s["accuracy"],
                    "svamp_correct": r_s["correct"],
                    "svamp_total": r_s["total"],
                }
            eval_records = gsm8k_records + svamp_records
            eval_fn = dual_math_eval
            metric_name = "gsm8k_accuracy"
            print(f"Math eval: BOTH — GSM8K ({len(gsm8k_records)}) + SVAMP ({len(svamp_records)})")
    elif args.task == "code":
        if args.code_train == "mbpp":
            train_prompts, cal_texts = [], []
            for split in ["train", "validation"]:
                ds = load_dataset("mbpp", split=split)
                for ex in ds:
                    p = (f"{ex['text']}\nYour code should pass these tests:\n"
                         + "\n".join(ex["test_list"]))
                    train_prompts.append(f"Instruction: {p}\nCode:")
                    cal_texts.append(f"Instruction: {p}\nCode: {ex['code']}")
            print(f"Code train: MBPP (same-dataset as test)")
        else:  # codealpaca
            try:
                ds_ca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            except Exception:
                ds_ca = load_dataset("theblackcat102/codealpha-cleaned", split="train")
            train_prompts, cal_texts = [], []
            for ex in ds_ca:
                instr = ex.get("instruction", ex.get("prompt", ""))
                out = ex.get("output", ex.get("completion", ""))
                train_prompts.append(f"Instruction: {instr}\nCode:")
                cal_texts.append(f"Instruction: {instr}\nCode: {out}")
            print(f"Code train: CodeAlpaca (transfer to MBPP)")
        cal_texts = cal_texts[:args.n_calibration]
        train_prompts = train_prompts[:args.n_train]
        # Execution-aware calibration: MBPP reference solutions +
        # assertion-only CE loss (grad only on the `assert ...` tokens inside
        # the prompt). The resulting subspace is built from KV directions that
        # matter for predicting the *test assertions* — the quantity that
        # defines execution correctness.
        cal_label_spans = None
        if args.calibration_source == "mbpp_solutions":
            mbpp_cal, mbpp_cal_tests = [], []
            for split in ["train", "validation"]:
                ds = load_dataset("mbpp", split=split)
                for ex in ds:
                    p = (f"{ex['text']}\nYour code should pass these tests:\n"
                         + "\n".join(ex["test_list"]))
                    mbpp_cal.append(f"Instruction: {p}\nCode: {ex['code']}")
                    mbpp_cal_tests.append(list(ex["test_list"]))
            cal_texts = mbpp_cal[:args.n_calibration]
            cal_tests = mbpp_cal_tests[:args.n_calibration]
            cal_label_spans = []
            for t, tests in zip(cal_texts, cal_tests):
                spans = []
                for a in tests:
                    start = 0
                    while True:
                        pos = t.find(a, start)
                        if pos < 0: break
                        spans.append((pos, pos + len(a)))
                        start = pos + len(a)
                cal_label_spans.append(spans)
            total_spans = sum(len(s) for s in cal_label_spans)
            print(f"Calibration override: MBPP reference solutions + "
                  f"assertion-only CE loss ({total_spans} assertion spans over "
                  f"{len(cal_texts)} calibration texts)")
        if args.code_eval == "codealpaca":
            eval_records = load_codealpaca_eval(args.n_eval)
            eval_fn = lambda m: eval_codealpaca(m, tok, eval_records, args.device, label=args.task)
            metric_name = "ast_parse_rate"
            print(f"Code eval: held-out CodeAlpaca ({len(eval_records)} examples, "
                  f"metrics: nll + ast_parse_rate)")
        elif args.code_eval == "both":
            mbpp_records = load_mbpp_sanitized_test()
            ca_records = load_codealpaca_eval(args.n_eval)
            def dual_eval(m):
                r_m = eval_mbpp(m, tok, mbpp_records, args.device, label="mbpp")
                r_c = eval_codealpaca(m, tok, ca_records, args.device, label="codealpaca")
                merged = {**r_m, **r_c}
                merged["mbpp_pass@1"] = r_m["pass@1"]
                merged["mbpp_correct"] = r_m["correct"]
                merged["mbpp_total"] = r_m["total"]
                merged["ca_nll"] = r_c["nll"]
                merged["ca_ppl"] = r_c["ppl"]
                merged["ca_ast_parse_rate"] = r_c["ast_parse_rate"]
                merged["ca_parses"] = r_c["parses"]
                merged["ca_total"] = r_c["total"]
                return merged
            eval_records = mbpp_records + ca_records
            eval_fn = dual_eval
            metric_name = "mbpp_pass@1"
            print(f"Code eval: BOTH — MBPP ({len(mbpp_records)} examples, pass@1) "
                  f"+ CodeAlpaca ({len(ca_records)} examples, nll + ast_parse_rate)")
        else:
            eval_records = load_mbpp_sanitized_test()
            eval_fn = lambda m: eval_mbpp(m, tok, eval_records, args.device, label=args.task)
            metric_name = "pass@1"
    elif args.task == "mmlu":
        # Training + calibration pulled from MMLU auxiliary_train
        cal_label_spans = None
        mmlu_train = load_mmlu_for_training(max(args.n_train, args.n_calibration))
        cal_records = mmlu_train[:args.n_calibration]
        cal_texts = [format_mc_full(r["question"], r["choices"], r["answer"])
                     for r in cal_records]
        train_prompts = [format_mc_prompt(r["question"], r["choices"])
                         for r in mmlu_train[:args.n_train]]

        # Answer-only CE: mask everything except the trailing answer letter
        if args.calibration_source == "answer_only":
            cal_label_spans = []
            for r, t in zip(cal_records, cal_texts):
                marker = "Answer: "
                pos = t.rfind(marker)
                if pos < 0:
                    cal_label_spans.append([])
                    continue
                # Span from the space before the letter to end-of-text so that
                # a merged " A" / "A" token is captured either way.
                letter_start = pos + len("Answer:")
                letter_end = len(t)
                cal_label_spans.append([(letter_start, letter_end)])
            total_spans = sum(len(s) for s in cal_label_spans)
            print(f"Calibration override: MMLU + answer-only CE loss "
                  f"({total_spans} letter spans over {len(cal_texts)} texts)")

        # Dual eval: MMLU test + BBH (6 subtasks)
        mmlu_eval_records = load_mmlu(args.n_eval, split="test")
        bbh_eval_records = load_bbh(n_per_task=50)
        def dual_eval_gen(m):
            r_m = eval_mmlu(m, tok, mmlu_eval_records, args.device, label="mmlu")
            r_b = eval_bbh(m, tok, bbh_eval_records, args.device, label="bbh")
            merged = {}
            merged["mmlu_accuracy"] = r_m["accuracy"]
            merged["mmlu_correct"] = r_m["correct"]
            merged["mmlu_total"] = r_m["total"]
            merged["bbh_accuracy"] = r_b["accuracy"]
            merged["bbh_correct"] = r_b["correct"]
            merged["bbh_total"] = r_b["total"]
            merged["bbh_per_task_accuracy"] = r_b["per_task_accuracy"]
            merged["accuracy"] = r_m["accuracy"]  # primary metric = MMLU
            return merged
        eval_records = mmlu_eval_records + bbh_eval_records
        eval_fn = dual_eval_gen
        metric_name = "mmlu_accuracy"
        print(f"MMLU eval: {len(mmlu_eval_records)} test | "
              f"BBH eval: {len(bbh_eval_records)} across {len(BBH_SUBTASKS)} subtasks")
    print(f"Calibration: {len(cal_texts)} | Train prompts: {len(train_prompts)} | Eval: {len(eval_records)}")

    results = {}

    # 1) Baseline: frozen student, no hooks
    print(f"\n[1] Baseline eval (no hooks)...")
    r = eval_fn(student)
    results["baseline"] = r
    print(f"  baseline: {metric_name}={r[metric_name]:.4f}")

    # 2) Compute student's AutoKV projections at target layers
    use_half = (args.rank_mode == "half")
    energy_thr = args.rank_energy if args.rank_mode == "energy" else None
    fixed_rank = args.rank if args.rank_mode == "fixed" else 0
    if use_half:
        mode_desc = "half KV dim"
    elif energy_thr is not None:
        mode_desc = f"energy-threshold {args.rank_energy:.3f} (auto rank)"
    else:
        mode_desc = f"fixed rank {args.rank}"
    print(f"\n[2] Computing student's AutoKV subspace at layers {target_layers} ({mode_desc})...")
    projections = compute_student_projections(
        student, tok, cal_texts, target_layers, fixed_rank, args.device,
        label_char_spans=locals().get("cal_label_spans"),
        energy_threshold=energy_thr, half_rank=use_half)
    for layer, p in projections.items():
        print(f"  L{layer}: rank_K={p['rank_K']:3d} (energy {p['energy_K']:.3f}), "
              f"rank_V={p['rank_V']:3d} (energy {p['energy_V']:.3f})")

    # 3) Eval with hooks (training-free inference-time projection)
    print(f"\n[3] Eval with inference-time hooks (mode={args.project_mode})...")
    hooks = SubspaceHooks(student, projections, args.device, mode=args.project_mode)
    try:
        r = eval_fn(student)
    finally:
        hooks.remove()
    results["inference_hooks"] = r
    print(f"  inference_hooks: {metric_name}={r[metric_name]:.4f}")

    # 4) Generate PLAIN self-samples (no hooks)
    print(f"\n[4] Generating plain SSD samples (no hooks)...")
    plain_samples = generate_samples(student, tok, train_prompts, args.device,
                                        batch_size=8)
    print(f"  Generated {len(plain_samples)} plain samples")

    # 5) Generate ENHANCED self-samples (with hooks)
    print(f"\n[5] Generating enhanced SSD samples (with hooks, mode={args.project_mode})...")
    hooks = SubspaceHooks(student, projections, args.device, mode=args.project_mode)
    try:
        enhanced_samples = generate_samples(student, tok, train_prompts, args.device,
                                              batch_size=8)
    finally:
        hooks.remove()
    print(f"  Generated {len(enhanced_samples)} enhanced samples")

    # Free up the original frozen student; we'll reload for training
    del student; torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM")

    # 6) Train on plain samples (ssd_plain)
    print(f"\n[6] Training ssd_plain on plain samples...")
    student_plain, stats_plain = train_on_samples(
        args.model, tok, plain_samples, lora_config,
        args.epochs, args.lr, args.device, args.seed)
    r = eval_fn(student_plain)
    results["ssd_plain"] = {**r, **stats_plain}
    save_dir = os.path.join(args.output_dir, f"student_ssd_plain_seed{args.seed}")
    student_plain.save_pretrained(save_dir)
    print(f"  ssd_plain: {metric_name}={r[metric_name]:.4f}")
    del student_plain; torch.cuda.empty_cache()

    # 7) Train on enhanced samples (ssd_enhanced, OUR METHOD)
    print(f"\n[7] Training ssd_enhanced on hook-generated samples...")
    student_enh, stats_enh = train_on_samples(
        args.model, tok, enhanced_samples, lora_config,
        args.epochs, args.lr, args.device, args.seed)
    r = eval_fn(student_enh)
    results["ssd_enhanced"] = {**r, **stats_enh}
    save_dir = os.path.join(args.output_dir, f"student_ssd_enhanced_seed{args.seed}")
    student_enh.save_pretrained(save_dir)
    print(f"  ssd_enhanced: {metric_name}={r[metric_name]:.4f}")
    del student_enh; torch.cuda.empty_cache()

    # Final table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label:20s}: {metric_name}={r[metric_name]:.4f}")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                    "config": vars(args), "target_layers": target_layers,
                    "projections": {
                        str(l): {"rank_K": p["rank_K"], "rank_V": p["rank_V"],
                                  "energy_K": p["energy_K"], "energy_V": p["energy_V"]}
                        for l, p in projections.items()
                    },
                    "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
