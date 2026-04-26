"""
Evaluate a pretrained model on all 6 datasets using top_p=0.9 sampling.

No training — just measures accuracy when generating with nucleus sampling
(temperature=0.7, top_p=0.9) instead of greedy decoding.

Usage:
  python scripts/eval_topp.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --n_eval 100 --output_dir results/eval_topp
"""
import argparse, json, os, random, re, sys, time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ssd_subspace import (
    load_svamp, load_mbpp_sanitized_test, load_codealpaca_eval,
    load_mmlu, load_bbh,
    extract_numeric, answers_match, extract_code, execute_mbpp,
    format_mc_prompt, MMLU_LETTERS, BBH_SUBTASKS,
    _normalize_bbh, _extract_bbh_answer,
)

import ast as ast_module


# ═══════════════════════════════════════════════════════════════════════════════
# Eval functions with top_p sampling
# ═══════════════════════════════════════════════════════════════════════════════

def eval_svamp_topp(model, tokenizer, examples, device, temperature, top_p, label=""):
    model.eval()
    correct = 0
    for ex in tqdm(examples, desc=f"SVAMP {label}", leave=False):
        prompt = f"Question: {ex['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                  do_sample=True, temperature=temperature, top_p=top_p,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if answers_match(extract_numeric(resp), extract_numeric(ex["answer"])):
            correct += 1
    total = len(examples)
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total}


def eval_mbpp_topp(model, tokenizer, records, device, temperature, top_p, label=""):
    model.eval()
    passed = 0
    for ex in tqdm(records, desc=f"MBPP {label}", leave=False):
        prompt = (f"Instruction: {ex['text']}\nYour code should pass these tests:\n"
                  + "\n".join(ex["test_list"]) + "\nCode:")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                  do_sample=True, temperature=temperature, top_p=top_p,
                                  use_cache=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(resp)
        for stop in ["\nclass ", "\nif __name__", "\n\n\n"]:
            idx = code.find(stop)
            if idx > 0: code = code[:idx]; break
        if execute_mbpp(code, ex["test_list"], ex.get("test_setup_code", "")):
            passed += 1
    return {"pass@1": passed / len(records), "correct": passed, "total": len(records)}


def eval_codealpaca_topp(model, tokenizer, records, device, temperature, top_p, label=""):
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
            gen = model.generate(**enc_prompt, max_new_tokens=256,
                                  do_sample=True, temperature=temperature, top_p=top_p,
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
            ast_module.parse(code)
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


def eval_mmlu_topp(model, tokenizer, records, device, temperature, top_p, label=""):
    model.eval()
    correct = 0
    for ex in tqdm(records, desc=f"MMLU {label}", leave=False):
        prompt = format_mc_prompt(ex["question"], ex["choices"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5,
                                  do_sample=True, temperature=temperature, top_p=top_p,
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


def eval_bbh_topp(model, tokenizer, records, device, temperature, top_p, label=""):
    model.eval()
    correct = 0
    per_task_correct = {t: 0 for t in BBH_SUBTASKS}
    per_task_total = {t: 0 for t in BBH_SUBTASKS}
    for ex in tqdm(records, desc=f"BBH {label}", leave=False):
        prompt = f"{ex['input']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16,
                                  do_sample=True, temperature=temperature, top_p=top_p,
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
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Eval with top_p sampling (no training)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/eval_topp")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    print("=" * 60, flush=True)
    print("Evaluation with top_p Sampling (No Training)", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generation: temperature={args.temperature}, top_p={args.top_p}", flush=True)
    print("=" * 60, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load eval data
    print("Loading all 6 eval datasets...", flush=True)
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_eval > 0:
        gsm8k_test = gsm8k_test.select(range(min(args.n_eval, len(gsm8k_test))))
    gsm8k_records = []
    for ex in gsm8k_test:
        m = re.search(r"####\s*([\d,.-]+)", ex["answer"])
        ans = m.group(1).replace(",", "") if m else ex["answer"]
        gsm8k_records.append({"question": ex["question"], "answer": str(ans)})

    svamp_records = load_svamp(args.n_eval)
    mbpp_records = load_mbpp_sanitized_test()[:args.n_eval]
    ca_records = load_codealpaca_eval(args.n_eval)
    mmlu_records = load_mmlu(args.n_eval, split="test")
    bbh_records = load_bbh(n_per_task=50)

    print(f"  GSM8K: {len(gsm8k_records)} | SVAMP: {len(svamp_records)} | "
          f"MBPP: {len(mbpp_records)} | CA: {len(ca_records)} | "
          f"MMLU: {len(mmlu_records)} | BBH: {len(bbh_records)}", flush=True)

    # Load model once
    print(f"\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
    }

    T = args.temperature
    P = args.top_p

    # Eval all 6
    print(f"\nEvaluating with top_p={P}, temperature={T}...", flush=True)

    print("  [1/6] GSM8K...", flush=True)
    r = eval_svamp_topp(model, tok, gsm8k_records, args.device, T, P, label="gsm8k")
    results["gsm8k_accuracy"] = r["accuracy"]
    results["gsm8k_correct"] = r["correct"]
    results["gsm8k_total"] = r["total"]
    print(f"    GSM8K: {r['accuracy']:.2%} ({r['correct']}/{r['total']})", flush=True)

    print("  [2/6] SVAMP...", flush=True)
    r = eval_svamp_topp(model, tok, svamp_records, args.device, T, P, label="svamp")
    results["svamp_accuracy"] = r["accuracy"]
    results["svamp_correct"] = r["correct"]
    results["svamp_total"] = r["total"]
    print(f"    SVAMP: {r['accuracy']:.2%} ({r['correct']}/{r['total']})", flush=True)

    print("  [3/6] MBPP...", flush=True)
    r = eval_mbpp_topp(model, tok, mbpp_records, args.device, T, P, label="mbpp")
    results["mbpp_pass@1"] = r["pass@1"]
    results["mbpp_correct"] = r["correct"]
    results["mbpp_total"] = r["total"]
    print(f"    MBPP: {r['pass@1']:.2%} ({r['correct']}/{r['total']})", flush=True)

    print("  [4/6] CodeAlpaca...", flush=True)
    r = eval_codealpaca_topp(model, tok, ca_records, args.device, T, P, label="ca")
    results["ca_nll"] = r["nll"]
    results["ca_ppl"] = r["ppl"]
    results["ca_ast_parse_rate"] = r["ast_parse_rate"]
    print(f"    CodeAlpaca: NLL={r['nll']:.3f} PPL={r['ppl']:.3f} AST={r['ast_parse_rate']:.2%}",
          flush=True)

    print("  [5/6] MMLU...", flush=True)
    r = eval_mmlu_topp(model, tok, mmlu_records, args.device, T, P, label="mmlu")
    results["mmlu_accuracy"] = r["accuracy"]
    results["mmlu_correct"] = r["correct"]
    results["mmlu_total"] = r["total"]
    print(f"    MMLU: {r['accuracy']:.2%} ({r['correct']}/{r['total']})", flush=True)

    print("  [6/6] BBH...", flush=True)
    r = eval_bbh_topp(model, tok, bbh_records, args.device, T, P, label="bbh")
    results["bbh_accuracy"] = r["accuracy"]
    results["bbh_correct"] = r["correct"]
    results["bbh_total"] = r["total"]
    results["bbh_per_task"] = r["per_task_accuracy"]
    print(f"    BBH: {r['accuracy']:.2%} ({r['correct']}/{r['total']})", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY (top_p={P}, temperature={T})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Dataset':15s} {'Metric':>10s} {'Value':>10s}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'GSM8K':15s} {'accuracy':>10s} {results['gsm8k_accuracy']:>9.2%}", flush=True)
    print(f"  {'SVAMP':15s} {'accuracy':>10s} {results['svamp_accuracy']:>9.2%}", flush=True)
    print(f"  {'MBPP':15s} {'pass@1':>10s} {results['mbpp_pass@1']:>9.2%}", flush=True)
    print(f"  {'CodeAlpaca':15s} {'NLL':>10s} {results['ca_nll']:>9.3f}", flush=True)
    print(f"  {'CodeAlpaca':15s} {'AST parse':>10s} {results['ca_ast_parse_rate']:>9.2%}", flush=True)
    print(f"  {'MMLU':15s} {'accuracy':>10s} {results['mmlu_accuracy']:>9.2%}", flush=True)
    print(f"  {'BBH':15s} {'accuracy':>10s} {results['bbh_accuracy']:>9.2%}", flush=True)

    # Save
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
