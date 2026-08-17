#!/usr/bin/env python
"""Baseline the GGUF victim via llama-server's OpenAI-compatible endpoint.

The 27B victim is GGUF-only (it is the sole quantization that fits 24 GB), so it
cannot go through vLLM like the students. Same tasks, same extraction, same
grading as eval_baseline.py — only the generation backend differs.

Defaults to a stratified subset: we need the victim's score as a *ceiling*, not
to publish it, and a full 1015-problem run costs ~9 h versus ~2 h for 250.

Start the server first:
  llama-server -m <model.gguf> -ngl 999 -c 81920 -np 32 \
               -fa on -ctk q8_0 -ctv q8_0 --port 8077
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from eval_baseline import load_tasks, extract_boxed, grade  # noqa: E402


def stratified(tasks, n_per_bench, seed=0):
    """Sample within (bench, type) so the mix mirrors the full set.

    Avoids groupby.apply: pandas 3 drops the grouping column from the applied
    frame, which silently strips 'type' and breaks grading downstream.
    """
    df = pd.DataFrame(tasks)
    out = []
    for b, g in df.groupby("bench", sort=False):
        frac = min(1.0, n_per_bench / len(g))
        for _, sub in g.groupby("type", sort=False):
            k = max(1, round(len(sub) * frac))
            out.append(sub.sample(min(k, len(sub)), random_state=seed))
    picked = pd.concat(out)
    assert {"bench", "id", "type", "gold", "prompt"} <= set(picked.columns), \
        f"stratified() lost columns: {set(picked.columns)}"
    return picked.to_dict("records")


def one(task, url, max_tokens, temp, top_p, top_k, timeout):
    body = {
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": max_tokens, "temperature": temp,
        "top_p": top_p, "top_k": top_k, "stream": False,
    }
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        # llama-server may split reasoning into reasoning_content
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        fin = d["choices"][0].get("finish_reason", "")
        ntok = d.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        # must not raise: a crash here hides the real request error
        return {"bench": task.get("bench"), "id": task.get("id"),
                "type": task.get("type"), "gold": task.get("gold"),
                "pred": None, "correct": False, "gen_tokens": 0,
                "truncated": False, "finish_reason": f"ERROR: {type(e).__name__}: {e}",
                "text": ""}

    pred = extract_boxed(text)
    return {**{k: task[k] for k in ("bench", "id", "type", "gold")},
            "pred": pred, "correct": grade(pred, task["gold"], task["type"]),
            "gen_tokens": ntok, "truncated": fin == "length",
            "finish_reason": fin, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8077")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-bench", type=int, default=250,
                    help="0 = run everything")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="must not exceed llama-server's -np")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    try:
        requests.get(f"{args.url}/health", timeout=10).raise_for_status()
    except Exception as e:
        sys.exit(f"llama-server not reachable at {args.url}: {e}")

    tasks = load_tasks()
    if args.n_per_bench:
        tasks = stratified(tasks, args.n_per_bench)
    print(f"{len(tasks)} tasks, concurrency={args.concurrency}", flush=True)

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, t, args.url, args.max_tokens, args.temperature,
                          args.top_p, args.top_k, args.timeout) for t in tasks]
        for f in futs:
            rows.append(f.result())
            done += 1
            if done % 10 == 0:
                print(f"Processed prompts: {100*done//len(tasks)}% | "
                      f"{done}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_json(args.out, orient="records", lines=True)

    errs = df.finish_reason.astype(str).str.startswith("ERROR").sum()
    print(f"\n=== victim (GGUF) ===")
    for b, g in df.groupby("bench"):
        d = g[~g.truncated]
        print(f"{b:10s} acc={100*g.correct.mean():5.1f}%  "
              f"(completed-only {100*d.correct.mean():5.1f}%)  n={len(g):4d}  "
              f"median_gen_tokens={int(g.gen_tokens.median()):5d}  "
              f"truncated={int(g.truncated.sum())}")
    print(f"request errors: {errs}/{len(df)}  -> {args.out}")


if __name__ == "__main__":
    main()
