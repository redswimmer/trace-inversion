#!/usr/bin/env python
"""C' — compress surrogate traces t' into summaries b' with pi. Completes D2.

Reads the generation output, keeps only rows that did NOT hit the cap (a severed
trace has no answer and teaches the inverter never to conclude — docs/12 §2; the
paper's own preprocess_r1_distill.py drops them the same way), runs pi over each
trace, and writes {x', y', b', t'}.

Offline vLLM batch, not a server: the batch is fixed and known up front.

Output schema is x' / y' / b' / t' plus a `summary` field that DUPLICATES `b` — the
alias exists only so bench/phase1_stats.py --mode summaries can read this file and a
pi-validation file with the same code. Five text fields, four quantities.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1"))
from prompts import PI_SYSTEM, PI_USER  # noqa: E402

RE_THINK = re.compile(r"^.*?</think>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=0, help="0 = all kept rows")
    ap.add_argument("--drop-bad", action="store_true",
                    help="drop malformed rows from an existing --out and exit. No model "
                         "load. Use after --fix stops making progress: a row that fails "
                         "repeatedly is failing for a reason in its input, not by chance.")
    ap.add_argument("--fix", action="store_true",
                    help="reuse --out's good rows and regenerate only the bad ones "
                         "(empty summary, or finish_reason=='length'). Use a different "
                         "--seed or the sampler may reproduce the same failure.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="pi targets 600-900; 2048 leaves room without letting the "
                         "cap shape the length distribution we are measuring")
    ap.add_argument("--max-len", type=int, default=16384, help="the paper's cutoff_len")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    # the repo pins these on the summarization call too (qwen2_5_summarization_*.sh)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    # --drop-bad touches no model: it is a filter over an existing output file.
    if args.drop_bad:
        rows = [json.loads(l) for l in open(args.out)]
        bad = [r for r in rows if not r["b"].strip() or r["finish_reason"] == "length"]
        good = [r for r in rows if r["b"].strip() and r["finish_reason"] != "length"]
        side = args.out.replace(".jsonl", "-dropped.jsonl")
        with open(side, "w") as f:
            for r in bad:
                f.write(json.dumps(r) + "\n")
        with open(args.out, "w") as f:
            for r in good:
                f.write(json.dumps(r) + "\n")
        print(f"dropped {len(bad)} malformed rows -> {side}\n"
              f"  empty summary   {sum(1 for r in bad if not r['b'].strip())}\n"
              f"  severed at cap  {sum(1 for r in bad if r['finish_reason'] == 'length')}\n"
              f"  idx {[r['idx'] for r in bad]}\n"
              f"{len(good)} rows remain -> {args.out}", flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.traces)]
    kept = [r for r in rows if not r["capped"]]
    if args.limit:
        kept = kept[: args.limit]
    print(f"{len(rows)} traces, {len(kept)} kept (cap-hit "
          f"{100*(1-len(kept)/max(len(rows),1)):.1f}%)", flush=True)

    # --fix: a severed or empty summary is the same defect class as a capped trace —
    # the inverter's conditioning input is malformed, and an empty b is NOT the
    # no-summary condition, it is a summary-condition row teaching the inverter to
    # expect nothing. Dropping them costs rows we need; regenerating costs seconds.
    all_kept = list(kept)
    good = {}
    if args.fix and Path(args.out).exists():
        for line in open(args.out):
            r = json.loads(line)
            if r["b"].strip() and r["finish_reason"] != "length":
                good[r["idx"]] = r
        kept = [r for r in kept if r["idx"] not in good]
        print(f"--fix: {len(good)} good rows reused, regenerating {len(kept)} "
              f"(seed {args.seed})", flush=True)
        if not kept:
            print("nothing to fix"); return

    tok = AutoTokenizer.from_pretrained(args.model)
    # Qwen3.5 is a reasoning model; if its template has a thinking switch, turn it
    # off. A <think> block in C''s output is not part of b' and would land in the
    # inverter's training data as style noise.
    # Probe by RENDERING, not by catching TypeError: apply_chat_template forwards
    # unknown kwargs into the Jinja context instead of raising, so an exception-based
    # probe reports "supported" for every model and silently does nothing.
    kw = {}
    _m = [{"role": "user", "content": "x"}]
    _on = tok.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
    _off = tok.apply_chat_template(_m, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    if _on != _off:
        kw["enable_thinking"] = False
        print("chat template: enable_thinking switch is real, disabling", flush=True)
    else:
        print("chat template: enable_thinking has no effect; relying on the "
              "<think> strip below", flush=True)

    prompts, over = [], 0
    budget = args.max_len - args.max_tokens - 64
    for r in kept:
        trace = r["trace"]
        ids = tok.encode(trace)
        if len(ids) > budget - len(tok.encode(PI_SYSTEM)) - 64:
            # truncation here would silently drop the end of the reasoning, which is
            # the part the summary most needs; count and report instead of hiding it
            over += 1
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": PI_SYSTEM},
             {"role": "user", "content": PI_USER.format(thinking=trace)}],
            tokenize=False, add_generation_prompt=True, **kw))
    if over:
        print(f"WARNING: {over} traces exceed the prompt budget and vLLM will "
              f"truncate or reject them", flush=True)

    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_frac, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens,
                        repetition_penalty=args.repetition_penalty, seed=args.seed)
    outs = llm.generate(prompts, sp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_trunc = n_think = 0
    fresh = {}
    with open(args.out + ".new", "w") as f:
        for r, o in zip(kept, outs):
            c = o.outputs[0]
            text = c.text
            if "</think>" in text:                 # defensive; kw should prevent it
                n_think += 1
                text = RE_THINK.sub("", text, count=1)
            n_trunc += c.finish_reason == "length"
            fresh[r["idx"]] = {
                "idx": r["idx"], "domain": r["domain"], "source": r["source"],
                "x": r["prompt"],        # x' — the OpenThoughts user turn
                "y": r["answer"],        # y' — surrogate's final answer
                "b": text.strip(),       # b' — the summary
                "t": r["trace"],         # t' — the surrogate's trace
                "summary": text.strip(),          # alias, for phase1_stats.py
                "summary_tokens": len(c.token_ids),
                "finish_reason": c.finish_reason,
            }
        # rebuild in the original kept order so --fix does not reshuffle D2
        order = [r["idx"] for r in all_kept] if args.fix else [r["idx"] for r in kept]
        for i in order:
            f.write(json.dumps(fresh.get(i) or good[i]) + "\n")
    Path(args.out + ".new").replace(args.out)
    still_bad = sum(1 for i in order
                    for r in [fresh.get(i) or good[i]]
                    if not r["b"].strip() or r["finish_reason"] == "length")
    print(f"\nsummarised {len(kept)}  truncated_at_cap {n_trunc}  "
          f"stripped_think_block {n_think}  rows_written {len(order)}  "
          f"still_bad {still_bad}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
