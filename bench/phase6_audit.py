#!/usr/bin/env python
"""Phase 6 per-run audit (docs/17 §4.5): per-bench acc / completed-only acc /
truncated / no_answer / median gen tokens / strict+loose loops, merged into
the committed summary.json keyed by run name.

Loop test is Phase 4's own (import phase4_draws.loops): strict = coverage 0.2,
loose = coverage 0.0, over each row's full generated text.

--spot-check-numeric N prints N graded-correct and N graded-wrong JEEBench
Numeric rows (pred, gold, verdict) — run once, on baseline-think
(baselines.md flagged Numeric 66.4% vs MCQ 98.2%; a tolerance bug hides here).
"""
import argparse, json
from pathlib import Path

import pandas as pd

from phase4_draws import loops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--run", required=True, help="summary.json key, e.g. baseline-think")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wall-s", type=float, required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--spot-check-numeric", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_json(args.jsonl, lines=True)
    entry = {"model": args.model, "seed": args.seed, "wall_s": round(args.wall_s, 1),
             "n": len(df), "bench": {}}
    for b, g in df.groupby("bench"):
        done = g[~g.truncated]
        texts = g.text.fillna("")
        strict = int(sum(loops(t, coverage=0.2) for t in texts))
        loose = int(sum(loops(t) for t in texts))
        e = {
            "n": int(len(g)),
            "acc": round(100 * g.correct.mean(), 1),
            "acc_completed": round(100 * done.correct.mean(), 1) if len(done) else None,
            "truncated": int(g.truncated.sum()),
            "no_answer": int(g.pred.isna().sum()),
            "median_gen_tokens": int(g.gen_tokens.median()),
            "loops_strict": strict,
            "loops_loose": loose,
        }
        entry["bench"][b] = e
        print(f"{args.run:28s} {b:9s} acc={e['acc']:5.1f}  done={e['acc_completed']}  "
              f"trunc={e['truncated']:4d}  noans={e['no_answer']:4d}  "
              f"medtok={e['median_gen_tokens']:5d}  loops={strict}/{loose} (strict/loose)",
              flush=True)

    if args.spot_check_numeric:
        num = df[(df.bench == "JEEBench") & (df.type == "Numeric")]
        for verdict, sub in (("CORRECT", num[num.correct]), ("WRONG", num[~num.correct])):
            print(f"\n--- JEEBench Numeric spot-check: graded {verdict} ---")
            for _, r in sub.head(args.spot_check_numeric).iterrows():
                print(f"  id={r['id']:24s} pred={str(r['pred'])[:60]!r}  "
                      f"gold={str(r['gold'])[:40]!r}  correct={bool(r['correct'])}")

    sp = Path(args.summary)
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    summary[args.run] = entry
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {args.run} -> {sp}", flush=True)


if __name__ == "__main__":
    main()
