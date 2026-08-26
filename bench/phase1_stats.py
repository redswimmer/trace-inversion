#!/usr/bin/env python
"""Measure Phase 1 artifacts: trace-length distributions and the Table 1 style stats.

Runs in the MAIN THREAD over saved raw text, never inside the generation run, so a
change here is retroactive and costs no GPU time (docs/11 §5).

Two modes:
  --traces     generation output   -> length distribution, cap-hit rate
  --summaries  compression output  -> the four Table 1 statistics

Table 1 targets (docs/11 §4): median tokens 540-590, bold-header sections >90%,
first-person prose >95%, LaTeX >70%. The paper measured these "with light regex
heuristics"; the exact regexes are not published, so ours are stated explicitly below
and reported alongside a looser variant, because a single threshold that happens to
sit on a cliff would be indistinguishable from a real match.
"""
import argparse, json, re, sys

import numpy as np

# A section header: a bold run alone on its line. Spec asks for 3-6 of them.
RE_HEADER = re.compile(r"^[ \t]*\*\*[^*\n]+\*\*[ \t]*:?[ \t]*$", re.M)
RE_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|I'll|me|my|myself)\b")
RE_LATEX = re.compile(r"\\\(|\\\[|\\frac|\\sqrt|\\pmod|\\cdot|\\times|\\le|\\ge|"
                      r"\\sum|\\int|\\alpha|\\beta|\\theta|\\pi\b|\$[^$\n]{1,200}\$|"
                      r"\\[a-zA-Z]{2,}")
# spec forbids these outright
RE_NUMBERED = re.compile(r"^[ \t]*\d+[.)]\s", re.M)
RE_BULLET = re.compile(r"^[ \t]*[-*+]\s+", re.M)


def tok(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def dist(xs, label):
    xs = np.asarray(xs)
    if not len(xs):
        return f"{label}: (none)"
    return (f"{label}: n={len(xs)}  median={int(np.median(xs))}  mean={int(xs.mean())}  "
            f"p05={int(np.percentile(xs,5))}  p95={int(np.percentile(xs,95))}  "
            f"max={int(xs.max())}")


def traces(rows, tk, cap):
    n = len(rows)
    capped = [r for r in rows if r["capped"]]
    kept = [r for r in rows if not r["capped"]]
    errs = [r for r in rows if str(r["finish_reason"]).startswith("ERROR")]
    print(f"rows            {n}")
    print(f"cap-hit         {100*len(capped)/max(n,1):.1f}%  ({len(capped)}/{n})")
    print(f"  finish_reason=length  {sum(r['finish_reason']=='length' for r in rows)}")
    print(f"  no answer, other      {sum(r['capped'] and r['finish_reason']!='length' for r in rows)}")
    print(f"request errors  {len(errs)}")
    print(f"rows kept       {len(kept)}")

    # Trace tokens on KEPT rows only: a capped trace's length is the cap, not a
    # sample from the model's length distribution, so including them biases the
    # median down and pins the tail at exactly `cap`.
    tt = [len(tk.encode(r["trace"])) for r in kept]
    at = [len(tk.encode(r["answer"])) for r in kept]
    print(dist(tt, "trace tokens (kept)"))
    print(dist(at, "answer tokens (kept)"))
    print(dist([len(tk.encode(r["trace"])) for r in rows], "trace tokens (all, incl capped)"))
    print(f"\nreference points")
    print(f"  paper's stated R1 average          6,130.6")
    print(f"  docs/12 §2 measured on this corpus 6,005 mean / 4,379 median / 25.5% >8192")
    if tt:
        print(f"  ours (kept)                        {int(np.mean(tt))} mean / {int(np.median(tt))} median")
    by = {}
    for r, t in zip(kept, tt):
        by.setdefault(r["domain"], []).append(t)
    print("\nby domain (kept)")
    for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {d:12s} " + dist(v, "").lstrip(": "))


def summaries(rows, tk):
    txt = [r["summary"] for r in rows if r.get("summary", "").strip()]
    n = len(txt)
    print(f"summaries       {n}  (empty/failed {len(rows)-n})")
    toks = [len(tk.encode(t)) for t in txt]
    hdr3 = [len(RE_HEADER.findall(t)) >= 3 for t in txt]
    hdr1 = [len(RE_HEADER.findall(t)) >= 1 for t in txt]
    fp3 = [len(RE_FIRST_PERSON.findall(t)) >= 3 for t in txt]
    fp1 = [len(RE_FIRST_PERSON.findall(t)) >= 1 for t in txt]
    tex = [bool(RE_LATEX.search(t)) for t in txt]
    num = [bool(RE_NUMBERED.search(t)) for t in txt]
    bul = [bool(RE_BULLET.search(t)) for t in txt]
    nsec = [len(RE_HEADER.findall(t)) for t in txt]

    def pct(v):
        return 100 * float(np.mean(v)) if len(v) else 0.0

    print(f"\n=== Table 1 ===")
    print(f"{'statistic':32s}{'ours':>9s}{'target':>14s}  {'':4s}")
    rowsout = [
        ("median tokens", f"{int(np.median(toks))}", "540-590",
         540 <= np.median(toks) <= 590),
        ("bold-header sections (>=3)", f"{pct(hdr3):.1f}%", ">90%", pct(hdr3) > 90),
        ("first-person prose (>=3)", f"{pct(fp3):.1f}%", ">95%", pct(fp3) > 95),
        ("LaTeX", f"{pct(tex):.1f}%", ">70%", pct(tex) > 70),
    ]
    for name, got, want, ok in rowsout:
        print(f"{name:32s}{got:>9s}{want:>14s}  {'PASS' if ok else 'FAIL'}")
    print(f"\nlooser variants (threshold sensitivity)")
    print(f"  bold-header >=1 section      {pct(hdr1):.1f}%")
    print(f"  first-person >=1 marker      {pct(fp1):.1f}%")
    print(f"\nspec compliance (Appendix B forbids both)")
    print(f"  contains a numbered list     {pct(num):.1f}%")
    print(f"  contains a bullet list       {pct(bul):.1f}%")
    print(f"  sections per summary         median {int(np.median(nsec))}  "
          f"in 3-6 range {pct([3 <= s <= 6 for s in nsec]):.1f}%")
    print()
    print(dist(toks, "summary tokens"))
    print(f"  in 600-900 (the prompt's stated target) {pct([600 <= t <= 900 for t in toks]):.1f}%")
    return all(ok for *_, ok in rowsout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--mode", choices=["traces", "summaries"], required=True)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--trace-tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="same family/vocab as the 7B; docs/12 §2 measured 6,005 with it")
    ap.add_argument("--summary-tokenizer", default="Qwen/Qwen3.5-4B",
                    help="the compressor's own tokenizer")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file)]
    print(f"=== {args.file}  ({args.mode}, n={len(rows)}) ===")
    if args.mode == "traces":
        traces(rows, tok(args.trace_tokenizer), args.cap)
    else:
        ok = summaries(rows, tok(args.summary_tokenizer))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
