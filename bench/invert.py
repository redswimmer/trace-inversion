#!/usr/bin/env python
"""Inverter inference: (x, y[, b]) -> t_hat. vLLM offline batch on MERGED bf16 weights (docs/13 §4.7).

Input is a phase2_format.py file — prompt messages + chat_template_kwargs per row, with an
optional completion (= t_true) — so the served prompt is byte-identical to the training
prompt. Used on the 200-row holdout here and on split B in Phase 4; written once.

Output row: {idx, domain, x, y, b, t_true, t_hat, raw, gen_tokens, finish_reason}
  raw          the generated text exactly as vLLM returned it — ALWAYS saved
  t_hat        raw with a defensive </think> strip (load-bearing 1 in 5,000 in Phase 1)
  gen_tokens   vLLM's own token count;  finish_reason == "length" is the cap-hit signal

Gates (exit 1, file is still written): 0 empty t_hat · rows out == rows in · every idx in
--holdout when given.  Launch with .venv-vllm/bin on PATH (docs/06 §1.8).
"""
import argparse, json, re, sys
from pathlib import Path

RE_THINK = re.compile(r"^.*?</think>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged bf16 weights dir")
    ap.add_argument("--data", required=True, help="phase2_format.py output (prompt messages per row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="bench/phase2/holdout.json", help="'' to skip the idx gate")
    ap.add_argument("--limit", type=int, default=0)
    # the paper's inversion-eval sampling (docs/07 §1.1; docs/13 §4.7)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)

    # Probe the thinking switch by RENDERING and diffing, never by catching an exception
    # (docs/11 §5): unknown kwargs fall through to Jinja silently.
    _m = [{"role": "user", "content": "x"}]
    _on = tok.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
    _off = tok.apply_chat_template(_m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert _on != _off, "enable_thinking has no effect on this template — the training prompt relied on it"
    print(f"chat template: enable_thinking switch is real; generation prompt ends {_off[-24:]!r}", flush=True)

    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                       **r.get("chat_template_kwargs", {})) for r in rows]
    plen = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
    need = max(plen) + args.max_tokens
    max_len = args.max_model_len
    if need > max_len:
        max_len = (need + 255) // 256 * 256
        print(f"WARNING: longest prompt {max(plen)} + max_tokens {args.max_tokens} = {need} > "
              f"max_model_len {args.max_model_len}; serving at {max_len} so no row's cap is cut", flush=True)
    print(f"{len(rows)} rows  prompt tokens max {max(plen)} median {sorted(plen)[len(plen) // 2]}  "
          f"max_model_len {max_len}", flush=True)

    llm = LLM(model=args.model, max_model_len=max_len, gpu_memory_utilization=args.gpu_frac,
              seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        repetition_penalty=args.repetition_penalty, seed=args.seed)
    outs = llm.generate(prompts, sp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_think = n_cap = n_empty = 0
    with open(args.out + ".new", "w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            text = c.text
            if "</think>" in text:                 # defensive; enable_thinking=False should prevent it
                n_think += 1
                text = RE_THINK.sub("", text, count=1)
            text = text.strip()
            n_cap += c.finish_reason == "length"
            n_empty += not text
            f.write(json.dumps({
                "idx": r["idx"], "domain": r.get("domain"), "x": r.get("x"), "y": r.get("y"),
                "b": r.get("b"),
                "t_true": r["completion"][0]["content"] if r.get("completion") else None,
                "t_hat": text, "raw": c.text,
                "gen_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
            }) + "\n")
    Path(args.out + ".new").replace(args.out)
    print(f"\ngenerated {len(outs)}  cap-hit (finish_reason=length) {n_cap}  stripped_think {n_think}  "
          f"empty {n_empty}  -> {args.out}", flush=True)

    fails = []
    if n_empty:
        fails.append(f"{n_empty} empty t_hat")
    if len(outs) != len(rows):
        fails.append(f"{len(outs)} outputs for {len(rows)} rows")
    if args.holdout:
        hold = set(json.load(open(args.holdout))["idx"])
        bad = [r["idx"] for r in rows if r["idx"] not in hold]
        if bad:
            fails.append(f"{len(bad)} rows not in the holdout (idx {bad[:5]})")
    for x in fails:
        print(f"FAIL  {x}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
