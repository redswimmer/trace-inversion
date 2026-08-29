#!/usr/bin/env python
"""D2 -> TRL conversational prompt-completion JSONL for the four inverters (docs/13 §4).

Per arm and setting, two files under bench/results/phase2/:
  {arm}-{setting}-train.jsonl    D2 minus the shared 200-row holdout
  {arm}-{setting}-holdout.jsonl  the holdout — TRL eval_dataset AND the inversion input

Row: {idx, domain, x, y, [b,] t, prompt, completion, chat_template_kwargs}
  prompt               [system, user] — the format-matched Appendix B prompt (phase2/prompts.py)
  completion           [assistant] = t'.strip(), NO <think> tags (docs/13 §4.2)
  chat_template_kwargs {"enable_thinking": false} — TRL 1.12 reads this per row
The b column is carried only in the summary files, so a no-summary file holds no b' text.

Every row is rendered through the Qwen3.5-4B chat template exactly as the trainer renders
it and counted. Rows over --max-length are DROPPED and counted, never truncated: keep_start
would sever the trace tail, the non-termination poison Phase 1 dropped rows to avoid.

The holdout — 200 idx, seed 20260828, drawn from the prompts BOTH arms kept — is written to
bench/phase2/holdout.json the first time and loaded (and re-verified) after that.
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase2"))
from prompts import SUM_SYSTEM, SUM_USER, NOSUM_SYSTEM, NOSUM_USER  # noqa: E402

D2 = {"7b": "bench/results/phase1/d2-7b.jsonl", "1.5b": "bench/results/phase1/d2-1.5b.jsonl"}
HOLDOUT = Path("bench/phase2/holdout.json")
OUT = Path("bench/results/phase2")
SEED, N_HOLDOUT = 20260828, 200
KW = {"enable_thinking": False}


def to_row(r, setting):
    x, y, t = r["x"].strip(), r["y"].strip(), r["t"].strip()
    if setting == "sum":
        b = r["b"].strip()
        user = SUM_USER.format(user_prompt=x, assistant_answer=y, reasoning_summary=b)
        system = SUM_SYSTEM
    else:
        user = NOSUM_USER.format(user_prompt=x, assistant_answer=y)
        system = NOSUM_SYSTEM
    out = {"idx": r["idx"], "domain": r["domain"], "x": x, "y": y}
    if setting == "sum":
        out["b"] = b
    out["t"] = t
    out["prompt"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out["completion"] = [{"role": "assistant", "content": t}]
    out["chat_template_kwargs"] = dict(KW)
    return out


def _ids(out):
    return out["input_ids"] if hasattr(out, "keys") else out


def render_ids(tok, row):
    """(prompt_ids, prompt+completion_ids) the way TRL's tokenize_fn produces them."""
    p = _ids(tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True,
                                     **row["chat_template_kwargs"]))
    pc = _ids(tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True,
                                      **row["chat_template_kwargs"]))
    return list(p), list(pc)


def format_rows(rows, setting, tok, max_length):
    """Returns (kept rows, dropped count, [(prompt_len, total_len)] for kept rows)."""
    kept, lens, dropped = [], [], 0
    for r in rows:
        assert "</think>" not in r["t"], f"idx {r['idx']}: t' carries a think tag (docs/13 §0 says none do)"
        row = to_row(r, setting)
        p, pc = render_ids(tok, row)
        if len(pc) > max_length:
            dropped += 1          # never truncate — see the module docstring
            continue
        kept.append(row)
        lens.append((len(p), len(pc)))
    return kept, dropped, lens


def load_holdout(d2):
    pool = sorted(set.intersection(*[{r["idx"] for r in rows} for rows in d2.values()]))
    if HOLDOUT.exists():
        h = json.load(open(HOLDOUT))
        assert h["seed"] == SEED and h["pool"] == len(pool) and len(h["idx"]) == N_HOLDOUT, h
        assert set(h["idx"]) <= set(pool), "holdout idx not in the paired pool"
        return set(h["idx"]), len(pool)
    idx = sorted(random.Random(SEED).sample(pool, N_HOLDOUT))
    HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": SEED, "pool": len(pool), "idx": idx}, open(HOLDOUT, "w"))
    print(f"wrote {HOLDOUT}: {N_HOLDOUT} idx from a pool of {len(pool)} (seed {SEED})")
    return set(idx), len(pool)


def write(path, rows):
    with open(str(path) + ".new", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(str(path) + ".new").replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="the inverter's tokenizer")
    ap.add_argument("--max-length", type=int, default=12288)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    d2 = {arm: [json.loads(l) for l in open(p)] for arm, p in D2.items()}
    hold, pool = load_holdout(d2)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"tokenizer": args.model, "max_length": args.max_length, "pool": pool,
             "holdout": len(hold), "files": {}}
    first = None
    for arm, rows in d2.items():
        for setting in ("sum", "nosum"):
            for split, sel in (("train", [r for r in rows if r["idx"] not in hold]),
                               ("holdout", [r for r in rows if r["idx"] in hold])):
                kept, dropped, lens = format_rows(sel, setting, tok, args.max_length)
                name = f"{arm}-{setting}-{split}"
                write(OUT / f"{name}.jsonl", kept)
                tot = [n for _, n in lens]
                stats["files"][name] = {
                    "rows_in": len(sel), "rows": len(kept), "dropped_over_length": dropped,
                    "tokens": sum(tot), "max_tokens": max(tot),
                    "median_tokens": sorted(tot)[len(tot) // 2],
                    "max_prompt_tokens": max(p for p, _ in lens),
                }
                s = stats["files"][name]
                print(f"{name:18s} rows {s['rows']:5d} (in {s['rows_in']}, dropped over "
                      f"{args.max_length}: {dropped})  tokens {s['tokens']:,}  max {s['max_tokens']}  "
                      f"median {s['median_tokens']}  max prompt {s['max_prompt_tokens']}", flush=True)
                if first is None:
                    first = kept[0]
    json.dump(stats, open(OUT / "format-stats.json", "w"), indent=1)
    print(f"\n=== row 0 of 7b-sum-train, rendered by the chat template (tokenizer {args.model}) ===")
    print(tok.apply_chat_template(first["prompt"] + first["completion"], tokenize=False,
                                  **first["chat_template_kwargs"]))
    print("=== end row 0 ===")


if __name__ == "__main__":
    main()
