#!/usr/bin/env python
"""Zero-shot baselines on MATH500 + JEEBench.

Establishes the headroom picture before any training: where the student
candidates start, where the victim ceiling sits, and whether the gap between
them is large enough to measure. This is the paper's Table 6 on our models.

Run inside .venv-vllm.
"""
import argparse, json, re, sys
from pathlib import Path

import pandas as pd
# vllm/transformers are imported inside main() so the GGUF harness can reuse
# load_tasks/extract_boxed/grade without pulling in the whole vLLM engine

MATH500 = "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl"
JEEBENCH_PARQUET = (
    "https://huggingface.co/datasets/daman1209arora/jeebench/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
MATH500_PARQUET = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)

INSTR = "\n\nReason step by step, then give your final answer inside \\boxed{}."


def load_tasks(limit=None):
    """Return a flat list of task dicts across both benchmarks."""
    tasks = []

    m = pd.read_parquet(MATH500_PARQUET)
    if limit:
        m = m.head(limit)
    for _, r in m.iterrows():
        tasks.append({
            "bench": "MATH500", "id": r["unique_id"],
            "prompt": r["problem"] + INSTR,
            "gold": str(r["answer"]), "type": "math",
        })

    j = pd.read_parquet(JEEBENCH_PARQUET)
    if limit:
        j = j.head(limit)
    for _, r in j.iterrows():
        # MCQ golds are letter strings ("B", "ABD"); others are numeric.
        t = r["type"]
        extra = ""
        if t == "MCQ":
            extra = " Give the single correct option letter."
        elif t == "MCQ(multiple)":
            extra = " Give all correct option letters together, e.g. \\boxed{ABD}."
        tasks.append({
            "bench": "JEEBench", "id": f"{r['subject']}-{r['index']}",
            "prompt": r["question"] + INSTR + extra,
            "gold": str(r["gold"]).strip(), "type": t,
        })
    return tasks


def extract_boxed(text):
    """Last \\boxed{...} in the text, brace-balanced."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i + 7 : j - 1].strip() if depth == 0 else None


def grade(pred, gold, typ):
    if pred is None:
        return False
    pred, gold = pred.strip(), gold.strip()

    if typ in ("MCQ", "MCQ(multiple)"):
        # compare as letter sets so order and separators don't matter
        pset = set(re.findall(r"[A-D]", pred.upper()))
        gset = set(re.findall(r"[A-D]", gold.upper()))
        return bool(gset) and pset == gset

    if typ in ("Integer", "Numeric"):
        try:
            pv = float(re.sub(r"[^\d.eE+-]", "", pred))
            gv = float(gold)
        except ValueError:
            return False
        # JEEBench numerics are specified to 2 decimals
        return abs(pv - gv) <= 0.01 + 1e-9 * abs(gv)

    # MATH500: try symbolic equivalence, fall back to normalized string match
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        norm = lambda s: re.sub(r"[\s{}$\\]|\\left|\\right|(?:\\!)", "", s).lower()
        return norm(pred) == norm(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="rows per benchmark (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=16384)
    # sampling differs per model family: Qwen3.5 thinking wants 1.0/0.95/20,
    # R1-Distill wants 0.6/0.95 and no top-k
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    # the paper's eval passes --seed 1234 (their eval is seeded, their training
    # is not); match it so runs are reproducible
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tasks = load_tasks(args.limit)
    print(f"{len(tasks)} tasks", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": t["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for t in tasks
    ]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_frac,
        trust_remote_code=True,
    )
    # single sample = pass@1, matching the paper's protocol
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        seed=args.seed)
    outs = llm.generate(prompts, sp)

    rows, trunc = [], 0
    for t, o in zip(tasks, outs):
        c = o.outputs[0]
        text = c.text
        if c.finish_reason == "length":
            trunc += 1
        pred = extract_boxed(text)
        rows.append({
            **{k: t[k] for k in ("bench", "id", "type", "gold")},
            "pred": pred,
            "correct": grade(pred, t["gold"], t["type"]),
            "gen_tokens": len(c.token_ids),
            "truncated": c.finish_reason == "length",
            "finish_reason": c.finish_reason,
            # keep raw output so failed extractions can be recovered and
            # grading can be redone offline without regenerating
            "text": text,
        })

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out, orient="records", lines=True)

    print(f"\n=== {args.model} ===")
    for b, g in df.groupby("bench"):
        # no_answer = model emitted no \boxed{} at all; distinguishes a wrong
        # answer from a scoring artifact
        noans = int(g.pred.isna().sum())
        print(f"{b:10s} acc={100*g.correct.mean():5.1f}%  n={len(g):4d}  "
              f"median_gen_tokens={int(g.gen_tokens.median()):5d}  "
              f"truncated={int(g.truncated.sum()):4d}  no_answer={noans:4d}")
    print(f"overall truncation: {trunc}/{len(df)}  "
          f"no_answer: {int(df.pred.isna().sum())}/{len(df)}  -> {args.out}")
    if trunc / max(len(df), 1) > 0.15:
        print("WARNING: >15% truncated — accuracy is capped by --max-tokens, "
              "not by model capability. Re-run with a higher limit.")


if __name__ == "__main__":
    main()
