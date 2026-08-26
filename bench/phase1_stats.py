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


def paired_reference(rows, tk, cap):
    """Compare our traces to R1's ground truth ON THE SAME PROMPTS.

    Every earlier comparison here was unpaired — our rows against a differently
    drawn reference sample — so prompt difficulty varied between the two sides and
    landed in the answer. Pairing removes that entirely: OpenThoughts already ships
    R1's own <think> trace for each row, so row i has both.

    Report the KEPT comparison (both sides under the cap) as the headline: that set
    is what D2 actually contains. Above the cap our distribution is CENSORED, not
    measured — we know how often we cross 8192, never how far we would have gone —
    so no claim about our tail's shape past the cap is supportable.
    """
    from datasets import load_dataset
    ds = load_dataset("llamafactory/OpenThoughts-114k", split="train")
    ref = ds.select([r["idx"] for r in rows])
    ours, r1 = [], []
    for r, row in zip(rows, ref):
        a = row["messages"][2]["content"]
        t = a.split("</think>")[0].replace("<think>", "", 1) if "</think>" in a else a
        r1.append(len(tk.encode(t)))
        ours.append(len(tk.encode(r["trace"])))
    ours, r1 = np.array(ours), np.array(r1)
    keep = (~np.array([r["capped"] for r in rows])) & (r1 <= cap)

    q = lambda x, p: int(np.percentile(x, p))
    print(f"\n=== paired against R1's ground truth, identical prompts (n={len(rows)}) ===")
    print(f"{'':24s}{'ours':>9s}{'R1':>9s}{'delta':>9s}")
    for lbl, m in (("all rows", np.ones(len(rows), bool)),
                   ("KEPT (both < cap)", keep)):
        o, g = ours[m], r1[m]
        print(f"  {lbl}  n={int(m.sum())}")
        for p in (25, 50, 75, 90):
            print(f"    p{p:<21d}{q(o,p):>9d}{q(g,p):>9d}{100*(q(o,p)/q(g,p)-1):>8.1f}%")
        print(f"    p75/p25 (dispersion){q(o,75)/q(o,25):>13.2f}{q(g,75)/q(g,25):>9.2f}")
    print(f"\n  cap-hit      ours {100*np.mean([r['capped'] for r in rows]):.1f}%   "
          f"R1 on the same prompts {100*np.mean(r1 > cap):.1f}%")
    print(f"  ours shorter on {100*np.mean(ours < r1):.1f}% of prompts")
    print(f"  R1 p90 over all rows is {q(r1,90)}, past our {cap} cap: our tail is")
    print(f"  censored there, so only the CROSSING RATE is comparable, not tail shape.")

    # The drop is not neutral. Pairing lets us split "hard prompt" from "this
    # surrogate ran away here", which an unpaired sample cannot do. D2 is built from
    # the survivors, so whatever is idiosyncratic to the surrogate leaves the
    # inverter's training distribution — while Phase 4 serves it on split B, which is
    # essentially unfiltered (the victim truncated 0.0-0.4% in Phase 0). That is a
    # train/serve shift introduced by our own drop policy. Measured, not assumed.
    ours_drop = np.array([r["capped"] for r in rows])
    r1_drop = r1 > cap
    both = int((ours_drop & r1_drop).sum())
    only_ours = int((ours_drop & ~r1_drop).sum())
    only_r1 = int((~ours_drop & r1_drop).sum())
    n = len(rows)
    print(f"\n  drop decomposition (why D2 loses the rows it loses)")
    print(f"    we drop                          {int(ours_drop.sum()):5d} / {n}  "
          f"= {100*ours_drop.mean():.1f}% of prompts")
    print(f"    R1 would drop, same prompts      {int(r1_drop.sum()):5d}")
    print(f"    both (genuinely long prompts)    {both:5d}")
    print(f"    OURS ONLY — R1 completes these   {only_ours:5d}  "
          f"= {100*only_ours/max(int(ours_drop.sum()),1):.0f}% of our drops, "
          f"{100*only_ours/n:.1f}% of all prompts")
    print(f"    R1 only — we complete these      {only_r1:5d}")
    print(f"    -> D2 is the surviving {100*(1-ours_drop.mean()):.0f}%; the OURS-ONLY share leaves")
    print(f"       the inverter's training set for reasons specific to this surrogate.")
    print(f"       Cross-arm overlap (7B vs 1.5B drops) separates 'hard prompt' from")
    print(f"       'this surrogate loops here' — free, both arms run anyway.")

    # A distill running ~10 pts above R1 is the expected shape (measured 36.5 vs
    # 26.0 on the 7B probe). Much beyond that is not a harder corpus — the prompts
    # are identical — so it is the surrogate or the harness.
    gap = 100 * (ours_drop.mean() - r1_drop.mean())
    print(f"\n  paired cap-hit gap: ours - R1 = {gap:+.1f} pts (tolerance ±10)")
    return ([f"paired cap-hit gap {gap:+.1f} pts vs R1 on identical prompts "
             f"(tolerance ±10) — prompt difficulty is controlled, so this is ours"]
            if abs(gap) > 10.0 else [])


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

    # Gates. audit_results.py cannot serve this phase: it reads a baseline-eval
    # schema (bench/correct/truncated/pred/type) that a trace file does not have,
    # and its TRUNC_LIMIT 0.10 fails every Phase 1 arm by construction, because we
    # cap 25-45% of rows deliberately and drop them. So the gate lives here, next to
    # the measurement it gates, rather than as a second auditor with one caller.
    fails = []
    if errs:
        fails.append(f"{len(errs)} request errors — every row must be a real "
                     f"generation, not a captured exception")
    # An empty trace on a row we KEPT is the silent-wrongness case: capped==False
    # says the model terminated, so an empty trace means the split lost it, not
    # that the model produced nothing.
    empty = [r for r in kept if not r["trace"].strip()]
    if empty:
        fails.append(f"{len(empty)} KEPT rows have an empty trace — the "
                     f"reasoning_content/</think> split dropped it (idx "
                     f"{[r['idx'] for r in empty[:5]]})")
    zero = sorted({r["domain"] for r in rows} - set(by))
    if zero:
        fails.append(f"domains with zero kept rows: {zero} — a domain that never "
                     f"survives the cap is absent from D2 entirely")
    ch = 100 * len(capped) / max(n, 1)
    if not 10.0 <= ch <= 45.0:
        fails.append(f"cap-hit {ch:.1f}% outside the 10-45% band")
    return fails


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
    print(f"  IQR {int(np.percentile(toks,25))}-{int(np.percentile(toks,75))}  "
          f"in 600-900 (the prompt's stated target) {pct([600 <= t <= 900 for t in toks]):.1f}%")

    # Table 1 gives medians only, so we cannot match a spread we were never told.
    # But pi has to hold up across very different inputs -- a 900-token math trace and
    # an 8,000-token code trace -- and a wide or input-dependent spread means pi is
    # unstable before it ever becomes the inverter's training signal. Report, not gate.
    rr = [r for r in rows if r.get("summary", "").strip()]
    bydom = {}
    for r, t in zip(rr, toks):
        bydom.setdefault(r.get("domain", "?"), []).append(t)
    print("\nsummary tokens by domain (is pi stable across input types?)")
    for d, v in sorted(bydom.items(), key=lambda kv: -len(kv[1])):
        print(f"  {d:12s} " + dist(v, "").lstrip(": "))

    tl = [len(r.get("t", "")) for r in rr]           # trace chars, a free proxy
    if any(tl):
        qs = np.percentile([x for x in tl if x], [25, 50, 75])
        buckets = {"short": [], "mid": [], "long": [], "longest": []}
        for t, n in zip(toks, tl):
            k = ("short" if n <= qs[0] else "mid" if n <= qs[1]
                 else "long" if n <= qs[2] else "longest")
            buckets[k].append(t)
        print("\nsummary tokens by trace-length quartile")
        for k in ("short", "mid", "long", "longest"):
            print(f"  {k:12s} " + dist(buckets[k], "").lstrip(": "))
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
    ap.add_argument("--paired", action="store_true",
                    help="also compare against R1's ground-truth trace for the SAME "
                         "prompts (traces mode only) — removes prompt-difficulty variance")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file)]
    print(f"=== {args.file}  ({args.mode}, n={len(rows)}) ===")
    if args.mode == "traces":
        tk = tok(args.trace_tokenizer)
        fails = traces(rows, tk, args.cap)
        if args.paired:
            fails += paired_reference(rows, tk, args.cap)
        else:
            print("\n  (paired cap-hit gate NOT run — pass --paired)")
        print(f"\n=== gates ===")
        for f in fails:
            print(f"  FAIL  {f}")
        print("  ** PASSED **" if not fails
              else f"  ** {len(fails)} GATE FAILURE(S) — DO NOT ACCEPT AS-IS **")
        sys.exit(1 if fails else 0)
    else:
        ok = summaries(rows, tok(args.summary_tokenizer))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
