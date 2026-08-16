#!/usr/bin/env python
"""Validate baseline runs before trusting any number from them.

A completed eval that silently truncated or failed to parse answers produces a
plausible-looking score that is really an artifact. This checks for that, and
calibrates the harness against the one model whose scores the paper publishes.

Usage:  python bench/audit_results.py bench/results/*.jsonl
"""
import sys, json
from pathlib import Path
import pandas as pd

# Paper Table 6 — the only external ground truth we have for this harness.
REFERENCE = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {"MATH500": 81.4, "JEEBench": 32.6},
}

TRUNC_LIMIT = 0.10    # >10% truncated => score is capped by max_tokens
NOANS_LIMIT = 0.05    # >5% unparseable => extraction is the problem
DEGEN_LIMIT = 0.05    # >5% suspiciously short => degenerate generation


def audit(path):
    tag = Path(path).stem
    df = pd.read_json(path, lines=True)
    n = len(df)
    problems, notes = [], []

    print(f"\n{'='*70}\n{tag}   n={n}\n{'='*70}")

    # ---- per-benchmark accuracy and health
    for b, g in df.groupby("bench"):
        acc = 100 * g.correct.mean()
        tr = g.truncated.mean()
        done = g[~g.truncated]
        # accuracy over completed generations only: the capability ceiling,
        # unconfounded by the token budget
        acc_done = 100 * done.correct.mean() if len(done) else float("nan")
        # a missing \boxed{} only indicts extraction if the model actually
        # finished; on a truncated row it just means it was cut off first
        na_done = done.pred.isna().mean() if len(done) else 0.0
        print(f"  {b:9s} acc={acc:5.1f}%  (completed-only {acc_done:5.1f}%)  "
              f"truncated={100*tr:5.1f}%  no_answer|completed={100*na_done:4.1f}%  "
              f"tokens med={int(g.gen_tokens.median()):5d} "
              f"p95={int(g.gen_tokens.quantile(.95)):6d}")
        if tr > TRUNC_LIMIT:
            problems.append(f"{b}: {100*tr:.1f}% truncated — report the "
                            f"completed-only figure ({acc_done:.1f}%) alongside")
        if na_done > NOANS_LIMIT:
            problems.append(f"{b}: {100*na_done:.1f}% of COMPLETED generations "
                            f"had no \\boxed{{}} — real extraction bug")

    # ---- JEEBench breaks down by question type; a type at 0% suggests a
    #      grading bug rather than uniform incompetence
    jee = df[df.bench == "JEEBench"]
    if len(jee):
        print("  JEEBench by type:")
        for t, g in jee.groupby("type"):
            acc = 100 * g.correct.mean()
            print(f"    {t:16s} n={len(g):4d}  acc={acc:5.1f}%")
            if len(g) >= 20 and acc == 0.0:
                problems.append(f"JEEBench type '{t}' scored exactly 0/{len(g)} "
                                f"— likely a grading bug")

    # ---- degenerate generations
    if "text" in df.columns:
        short = (df.text.fillna("").str.len() < 50).mean()
        if short > DEGEN_LIMIT:
            problems.append(f"{100*short:.1f}% of outputs under 50 chars — "
                            f"possible degenerate generation")
        notes.append(f"raw text saved ({df.text.notna().sum()}/{n} rows) — "
                     f"regradable offline")
    else:
        notes.append("raw text NOT saved — failed extractions cannot be "
                     "recovered without re-running")

    # ---- calibration against the paper, where available
    for ref_name, ref in REFERENCE.items():
        if ref_name in tag:
            print(f"\n  CALIBRATION vs paper Table 6:")
            for b, expected in ref.items():
                g = df[df.bench == b]
                if not len(g):
                    continue
                got = 100 * g.correct.mean()
                delta = got - expected
                verdict = "OK" if abs(delta) <= 8 else "MISMATCH"
                print(f"    {b:9s} ours={got:5.1f}%  paper={expected:5.1f}%  "
                      f"delta={delta:+5.1f}  [{verdict}]")
                if abs(delta) > 8:
                    problems.append(
                        f"CALIBRATION {b}: ours {got:.1f}% vs paper {expected:.1f}% "
                        f"({delta:+.1f}) — harness may be miscalibrated")

    # ---- verdict
    print()
    for x in notes:
        print(f"  note: {x}")
    if problems:
        print(f"\n  ** {len(problems)} PROBLEM(S) — DO NOT ACCEPT AS-IS **")
        for p in problems:
            print(f"     - {p}")
    else:
        print("  ** PASSED — no red flags **")
    return len(problems) == 0


def main():
    paths = [p for p in sys.argv[1:] if "smoke" not in p]
    if not paths:
        print("no result files yet")
        return
    ok = [audit(p) for p in paths]
    print(f"\n{'='*70}\n{sum(ok)}/{len(ok)} runs passed validation")


if __name__ == "__main__":
    main()
