#!/usr/bin/env python
"""Re-extract and re-grade saved runs without regenerating anything.

Saving raw text means every grading fix applies retroactively to all runs at
once, for free. Applies the same extractor to every model so no family gets an
advantage from its own answer convention.

Usage:  python bench/regrade.py bench/results/*.jsonl
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_baseline import extract_boxed, grade  # noqa: E402


def main():
    paths = [p for p in sys.argv[1:] if "smoke" not in p]
    for p in paths:
        df = pd.read_json(p, lines=True)
        if "text" not in df.columns:
            print(f"{Path(p).stem:32s} SKIP — no raw text saved")
            continue

        before = df.correct.mean()
        df["pred"] = df.text.fillna("").map(extract_boxed)
        df["correct"] = [grade(r.pred, str(r.gold), r.type)
                         for r in df.itertuples()]
        after = df.correct.mean()

        df.to_json(p, orient="records", lines=True)
        print(f"\n{Path(p).stem}")
        print(f"  overall {100*before:5.1f}% -> {100*after:5.1f}%")
        for b, g in df.groupby("bench"):
            done = g[~g.truncated]
            na = done.pred.isna().mean() if len(done) else 0
            print(f"    {b:9s} acc={100*g.correct.mean():5.1f}%  "
                  f"no_answer|completed={100*na:4.1f}%")


if __name__ == "__main__":
    main()
