#!/usr/bin/env python
"""Measure Phase 1 artifacts: trace-length distributions and the Table 1 style stats.

Runs in the MAIN THREAD over saved raw text, never inside the generation run, so a
change here is retroactive and costs no GPU time (docs/11 §5).

Three modes:
  --traces     generation output   -> length distribution, cap-hit rate
  --summaries  compression output  -> the four Table 1 statistics
  --inverted   invert.py output    -> paired t_hat vs t_true lengths, cap-hit, empties (Phase 2)

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


def paired_reference(rows, tk, cap, r1_tokens=None, gap_tol=10.0):
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
    if r1_tokens is None:
        from datasets import load_dataset
        ds = load_dataset("llamafactory/OpenThoughts-114k", split="train")
        ref = ds.select([r["idx"] for r in rows])
        r1 = []
        for row in ref:
            a = row["messages"][2]["content"]
            t = a.split("</think>")[0].replace("<think>", "", 1) if "</think>" in a else a
            r1.append(len(tk.encode(t)))
    else:
        # Tests inject reference lengths. Without this the paired gate is the one
        # gate with no test, purely because it loads a 114k-row corpus — and it is
        # the gate we can least afford to be wrong, since it is the only one prompt
        # difficulty cannot confound.
        r1 = list(r1_tokens)
    ours = [len(tk.encode(r["trace"])) for r in rows]
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

    # PER-ARM, and the gate's job is to catch a RE-RUN diverging, not to characterise
    # this run — so it is centred on the measured gap, which would be wrong for a gate
    # meant to detect a bad arm on first contact.
    #
    # Tolerance = measured gap + 8 points. The 8 is sourced, not picked: our flip-rate
    # measurement puts ~4 points of resampling noise on OUR side alone (R1's side is
    # deterministic — OpenThoughts ships one fixed trace per row — so only we move),
    # plus binomial CI. Revise it when the noise estimate improves.
    #   7B  measured +7.8  -> 16
    #   1.5B measured +18.6 -> 27
    # A ±10 tolerance left the 7B just 2.2 points of margin against ~4 points of noise,
    # so a re-run landing at +11 would have failed for reasons already measured as noise.
    gap = 100 * (ours_drop.mean() - r1_drop.mean())
    print(f"\n  paired cap-hit gap: ours - R1 = {gap:+.1f} pts (tolerance ±{gap_tol:g})")
    return ([f"paired cap-hit gap {gap:+.1f} pts vs R1 on identical prompts "
             f"(tolerance ±{gap_tol:g}) — prompt difficulty is controlled, so this is ours"]
            if abs(gap) > gap_tol else [])


def traces(rows, tk, cap, band=(10.0, 45.0), err_rate_max=1.0):
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
    # A COUNT cannot scale across arms of different length, and a zero-count gate
    # contradicts the operational policy (transient 5xx are tolerated below 1%).
    # The 7B ran 7,669 rows with zero; the 1.5B had 12 in 9,314 = 0.13%, dispersed
    # across the run and flat across two serving configurations (0.12% at 32 slots,
    # 0.14% at 128). Report both numbers so a reader sees the count and the rate.
    er = 100 * len(errs) / max(n, 1)
    print(f"request-error rate  {er:.2f}%  ({len(errs)}/{n})  threshold {err_rate_max:g}%")
    if er > err_rate_max:
        fails.append(f"request-error rate {er:.2f}% ({len(errs)}/{n}) exceeds "
                     f"{err_rate_max:g}% — a rising rate is a failing server")
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
    # The band is PER-SURROGATE, not a global constant — that is the finding, not a
    # convenience. The 7B settles at 34.6% and the 1.5B at 46.5% on identical prompts,
    # so a single threshold either passes a broken 7B or fails a healthy 1.5B. The
    # ceiling that is NOT a matter of judgement: 5,000 kept from 15,999 usable rows
    # needs a keep rate of 31.25%, so split A is exhausted at 68.75% cap-hit.
    ch = 100 * len(capped) / max(n, 1)
    if not band[0] <= ch <= band[1]:
        fails.append(f"cap-hit {ch:.1f}% outside the {band[0]:g}-{band[1]:g}% band "
                     f"(split A exhausts at 68.8%)")
    return fails


def d2_gates(rows, target=5000):
    """Integrity of D2 itself, separate from pi's Table 1 quality.

    A severed or empty summary is the same defect class as a capped trace: the
    inverter's conditioning input is malformed. An empty b' is NOT the no-summary
    condition — it is a summary-condition row that teaches the inverter to expect
    nothing. Both were found on the 7B arm (1 empty, 18 length-capped of 5,012) and
    neither is visible to the --mode traces gates, which check traces, not summaries.
    """
    fails = []
    n = len(rows)
    for k in ("x", "y", "b", "t"):
        bad = [r["idx"] for r in rows if not str(r.get(k, "")).strip()]
        if bad:
            fails.append(f"{len(bad)} rows with empty {k!r} (idx {bad[:5]})")
    cut = [r["idx"] for r in rows if r.get("finish_reason") == "length"]
    if cut:
        fails.append(f"{len(cut)} summaries hit the generation cap — severed "
                     f"conditioning input (idx {cut[:5]})")
    dup = n - len({r["idx"] for r in rows})
    if dup:
        fails.append(f"{dup} duplicate idx — D2 must be one row per prompt")
    if n < target:
        fails.append(f"{n} rows, target {target}")
    print(f"\n=== D2 integrity ===")
    print(f"  rows {n} (target {target}) · duplicate idx {dup} · "
          f"empty x/y/b/t {[sum(1 for r in rows if not str(r.get(k,'')).strip()) for k in 'xybt']} · "
          f"summaries at cap {len(cut)}")
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
    return [f"Table 1: {name} = {got}, target {want}" for name, got, want, ok in rowsout if not ok]


def r1_answers(idxs):
    """Boxed answer in each OpenThoughts row's own R1 solution (the assistant text after </think>).
    idx is the row position in llamafactory/OpenThoughts-114k, as everywhere in this project."""
    from datasets import load_dataset
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from eval_baseline import extract_boxed
    ds = load_dataset("llamafactory/OpenThoughts-114k", split="train")
    idxs = sorted(idxs)
    out = {}
    for i, row in zip(idxs, ds.select(idxs)):
        a = row["messages"][2]["content"]
        out[i] = extract_boxed(a.split("</think>")[-1] if "</think>" in a else a)
    return out


def vs_r1(rows, out_json):
    """The model's answer vs R1's answer on the SAME prompt (docs/14 §4.6). Report only, never a
    filter: R1's boxed answer is the dataset's solution, not ground truth, and the symbolic grader
    rejects some equivalent forms (~3-4% of Phase 2 mismatches). Last \\boxed{} in each KEPT row's
    post-think answer against the last \\boxed{} in the OpenThoughts row's own R1 solution, graded
    in the main thread (math_verify uses SIGALRM). A collapse here is the first sign of a template
    or split error, hours rather than days into a run."""
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from eval_baseline import extract_boxed, grade
    kept = [r for r in rows if not r["capped"]]
    r1 = r1_answers([r["idx"] for r in kept])
    buckets = {"agree": [], "disagree": [], "no box in y": [], "no box in R1": []}
    per_row = {}
    for r in kept:
        yb = extract_boxed(r["answer"] if "answer" in r else r.get("y", ""))   # traces or D2 schema
        gb = r1.get(r["idx"])
        if yb is None:
            b = "no box in y"
        elif gb is None:
            b = "no box in R1"
        else:
            b = "agree" if grade(yb, gb, "MATH") else "disagree"
        buckets[b].append(r["idx"])
        per_row[r["idx"]] = {"y_boxed": yb, "r1_boxed": gb, "bucket": b,
                             "agree": (b == "agree") if b in ("agree", "disagree") else None,
                             "domain": r.get("domain")}
    n = len(buckets["agree"]) + len(buckets["disagree"])
    rate = 100 * len(buckets["agree"]) / max(n, 1)
    print(f"\n=== y vs R1's answer, same prompt — KEPT rows only (n={len(kept)} of {len(rows)}), "
          f"report only (docs/14 §4.6) ===")
    print("  " + "  ".join(f"{k} {len(v)}" for k, v in buckets.items())
          + f"   agreement on gradable {len(buckets['agree'])}/{n} ({rate:.1f}%)")
    # "no box in y" is mostly domain, not failure: code rows never box. Show every bucket per domain.
    by = {}
    for k in per_row.values():
        by.setdefault(k["domain"], []).append(k["bucket"])
    print(f"  {'domain':12s}{'kept':>6s}{'agree':>7s}{'disagr':>7s}{'noboxY':>8s}{'noboxR1':>8s}{'agree%':>8s}")
    for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a, dis = v.count("agree"), v.count("disagree")
        print(f"  {d:12s}{len(v):6d}{a:7d}{dis:7d}{v.count('no box in y'):8d}{v.count('no box in R1'):8d}"
              f"{(100*a/(a+dis)) if a+dis else float('nan'):8.1f}")
    if buckets["disagree"]:
        print(f"  disagree idx: {buckets['disagree'][:30]}  (read three: two models differ, "
              f"or a grader miss on an equivalent form)")
    if n and rate < 75:
        print("  ** under 75% on gradable rows — STOP AND ASK (docs/14 §6); reported, not gated **")
    json.dump(per_row, open(out_json, "w"), indent=1)
    print(f"  per-row (y_boxed, r1_boxed, bucket) -> {out_json}")


def inverted(rows, tk, cap, holdout=None, n_expected=None, tag="", out_json=None, r1=None):
    """Paired t_hat vs t_true lengths on the SAME rows — the inverter's acceptance evidence
    (docs/13 §4.7, §7). Token counts use the inverter's tokenizer (Qwen3.5-4B); phase1.md
    counts the same traces with R1-Distill's, so say which before comparing across docs.
    Cap-hit is vLLM's own finish_reason; gen_tokens is vLLM's own count."""
    th = np.array([len(tk.encode(r["t_hat"])) for r in rows])
    tt = np.array([len(tk.encode(r["t_true"])) for r in rows])
    gen = np.array([r["gen_tokens"] for r in rows])
    cap_hit = np.array([r["finish_reason"] == "length" for r in rows])
    empty = [r["idx"] for r in rows if not r["t_hat"].strip()]
    q = lambda x, p: int(np.percentile(x, p))
    print(f"rows {len(rows)}   tokenizer: the inverter's (Qwen3.5-4B)")
    print(dist(th, "t_hat tokens (re-tokenized)"))
    print(dist(gen, "gen_tokens (vLLM's own count, incl. any stripped think block)"))
    print(dist(tt, "t_true tokens"))
    ratio = float(np.median(th) / max(np.median(tt), 1))
    print(f"t_hat / t_true at the median  {ratio:.2f}    per-row ratio median "
          f"{float(np.median(th / np.maximum(tt, 1))):.2f}    t_hat shorter on {100*np.mean(th < tt):.1f}% of rows")
    print(f"cap-hit (finish_reason == length)  {100*cap_hit.mean():.1f}%  ({int(cap_hit.sum())}/{len(rows)})"
          f"    empty t_hat {len(empty)}")
    by = {}
    for r, a, b in zip(rows, th, tt):
        by.setdefault(r.get("domain", "?"), []).append((a, b))
    print("by domain: t_hat median / t_true median")
    for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a, b = np.array(v).T
        print(f"  {d:12s} n={len(v):3d}  {int(np.median(a)):5d} / {int(np.median(b)):5d}")
    print("\n| inverter | t_hat median / mean / p05 / p95 | t_true median / mean / p05 / p95 "
          "| t_hat/t_true at median | cap-hit @ cap | empty |")
    print(f"| {tag or '?'} | {q(th,50)} / {int(th.mean())} / {q(th,5)} / {q(th,95)} "
          f"| {q(tt,50)} / {int(tt.mean())} / {q(tt,5)} / {q(tt,95)} | {ratio:.2f} "
          f"| {100*cap_hit.mean():.1f}% ({int(cap_hit.sum())}) | {len(empty)} |")
    # Answer consistency, report-only (not a gate): a forged trace that does not land on the
    # answer it was given is the failure the paper's mechanism assumes away. Last \boxed{} in
    # t_hat against the last \boxed{} in y, graded with eval_baseline's symbolic path (main
    # thread — docs/11 §5). A trace can reach the answer without boxing it, so "no box" is a
    # bucket, not a failure count.
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from eval_baseline import extract_boxed, grade
    buckets = {"match": [], "mismatch": [], "no box in t_hat": [], "no box in y": []}
    per_row = {}
    for r in rows:
        gold, pred = extract_boxed(r["y"] or ""), extract_boxed(r["t_hat"])
        if gold is None:
            b = "no box in y"
        elif pred is None:
            b = "no box in t_hat"
        else:
            b = "match" if grade(pred, gold, "MATH") else "mismatch"
        buckets[b].append(r["idx"])
        per_row[r["idx"]] = {"gold": gold, "pred": pred, "bucket": b, "finish_reason": r["finish_reason"]}
    n_graded = len(buckets["match"]) + len(buckets["mismatch"])
    nobox_cap = sum(per_row[i]["finish_reason"] == "length" for i in buckets["no box in t_hat"])
    # On the surrogate holdout y is the SURROGATE's answer and is sometimes wrong, so "t_hat matches
    # y" mixes inverter fidelity with surrogate error. Grade both against the boxed answer in the
    # OpenThoughts row's own R1 solution — "R1's answer (the dataset's solution)", not ground truth —
    # to split the mismatches into "inverter wrong" and "inverter right, surrogate wrong".
    if r1 is not None:
        for r in rows:
            k = per_row[r["idx"]]; g = r1.get(r["idx"])
            k["r1"] = g
            k["y_vs_r1"] = None if (g is None or k["gold"] is None) else grade(k["gold"], g, "MATH")
            k["t_hat_vs_r1"] = None if (g is None or k["pred"] is None) else grade(k["pred"], g, "MATH")
        def _rate(key):
            v = [k[key] for k in per_row.values() if k[key] is not None]
            return f"{sum(v)}/{len(v)} ({100*sum(v)/max(len(v),1):.1f}%)"
        split = {"inverter wrong (y = R1, t_hat != R1)": 0, "inverter right, surrogate wrong (t_hat = R1, y != R1)": 0,
                 "both match R1 (equivalent forms)": 0, "neither matches R1": 0, "R1 has no box": 0}
        for i in buckets["mismatch"]:
            k = per_row[i]
            if k["r1"] is None: split["R1 has no box"] += 1
            elif k["y_vs_r1"] and not k["t_hat_vs_r1"]: split["inverter wrong (y = R1, t_hat != R1)"] += 1
            elif k["t_hat_vs_r1"] and not k["y_vs_r1"]: split["inverter right, surrogate wrong (t_hat = R1, y != R1)"] += 1
            elif k["t_hat_vs_r1"] and k["y_vs_r1"]: split["both match R1 (equivalent forms)"] += 1
            else: split["neither matches R1"] += 1
        print(f"  against R1's answer (the dataset's solution, not ground truth): y vs R1 {_rate('y_vs_r1')}   "
              f"t_hat vs R1 {_rate('t_hat_vs_r1')}   R1 unboxed {sum(k['r1'] is None for k in per_row.values())}")
        # PAIRED: the same rows, gradable on both sides — the unpaired denominators differ because
        # t_hat often does not box, and the rows where it does may be the easier ones (docs/11 §5).
        both = [k for k in per_row.values() if k["y_vs_r1"] is not None and k["t_hat_vs_r1"] is not None]
        if both:
            print(f"  PAIRED on the {len(both)} rows gradable on both sides: y vs R1 {sum(k['y_vs_r1'] for k in both)}/{len(both)} "
                  f"({100*sum(k['y_vs_r1'] for k in both)/len(both):.1f}%)   t_hat vs R1 {sum(k['t_hat_vs_r1'] for k in both)}/{len(both)} "
                  f"({100*sum(k['t_hat_vs_r1'] for k in both)/len(both):.1f}%)   "
                  f"y right & t_hat wrong {sum(k['y_vs_r1'] and not k['t_hat_vs_r1'] for k in both)}   "
                  f"t_hat right & y wrong {sum(k['t_hat_vs_r1'] and not k['y_vs_r1'] for k in both)}")
        print("  mismatch split: " + "  ".join(f"{k} {v}" for k, v in split.items()))
    print(f"\nanswer consistency (last boxed in t_hat vs y; report only):  "
          + "  ".join(f"{k} {len(v)}" for k, v in buckets.items())
          + (f"   match rate among graded {100*len(buckets['match'])/n_graded:.1f}% (n={n_graded})" if n_graded else ""))
    print(f"  no box in t_hat = {nobox_cap} severed at the cap + {len(buckets['no box in t_hat']) - nobox_cap} ending in prose")
    if buckets["mismatch"]:
        print(f"  mismatch idx: {buckets['mismatch']}  (read them: genuine vs grader miss — docs/11 §5)")
    if out_json:
        json.dump(per_row, open(out_json, "w"), indent=1)      # re-gradable without regenerating
        print(f"  per-row (gold, pred, bucket) -> {out_json}")

    # three rows for a human to read: shortest, median and longest t_true
    order = np.argsort(tt)
    print("\nread these (idx, t_true tokens, t_hat tokens, finish):")
    for lbl, i in (("short", order[0]), ("median", order[len(order) // 2]), ("long", order[-1])):
        r = rows[i]
        print(f"  {lbl:6s} idx {r['idx']:6d}  t_true {tt[i]:5d}  t_hat {th[i]:5d}  {r['finish_reason']}")

    fails = []
    if empty:
        fails.append(f"{len(empty)} empty t_hat (idx {empty[:5]}) — STOP AND ASK (docs/13 §7)")
    if n_expected is not None and len(rows) != n_expected:
        fails.append(f"{len(rows)} rows, expected {n_expected}")
    dup = len(rows) - len({r["idx"] for r in rows})
    if dup:
        fails.append(f"{dup} duplicate idx")
    if holdout is not None:
        bad = [r["idx"] for r in rows if r["idx"] not in holdout]
        if bad:
            fails.append(f"{len(bad)} rows not in the holdout (idx {bad[:5]})")
    if ratio < 0.5:
        fails.append(f"median t_hat is {ratio:.2f}x the paired t_true median — under half; the "
                     f"adapter did not take — STOP AND ASK (docs/13 §7)")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--mode", choices=["traces", "summaries", "inverted"], required=True)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--trace-tokenizer", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="same family/vocab as the 7B; docs/12 §2 measured 6,005 with it")
    ap.add_argument("--summary-tokenizer", default="Qwen/Qwen3.5-4B",
                    help="the compressor's own tokenizer")
    ap.add_argument("--cap-hit-band", type=float, nargs=2, default=[10.0, 45.0],
                    metavar=("LO", "HI"),
                    help="traces mode: acceptable cap-hit %%. PER-SURROGATE: 10 45 for the "
                         "7B (settles 34.6%%), 10 58 for the 1.5B (settles 46.5%%). Split A "
                         "exhausts at 68.8%%, which is the only hard ceiling.")
    ap.add_argument("--err-rate-max", type=float, default=1.0, metavar="PCT",
                    help="traces mode: max request-error RATE in %%. A count cannot scale "
                         "across arms of different length.")
    ap.add_argument("--paired-gap-tol", type=float, default=10.0, metavar="PTS",
                    help="traces mode: paired cap-hit gap tolerance, PER ARM. Set to the "
                         "measured gap + 8 (7B: 16, 1.5B: 27). The gate catches a RE-RUN "
                         "diverging, so it is centred on the measured value.")
    ap.add_argument("--validation", action="store_true",
                    help="summaries mode: this is a ~200-row pi validation sample, so "
                         "gate Table 1 only and skip the D2 integrity gates")
    ap.add_argument("--target", type=int, default=5000,
                    help="summaries mode: required D2 row count")
    ap.add_argument("--holdout", default="bench/phase2/holdout.json",
                    help="inverted mode: every idx must be in this file; '' to skip")
    ap.add_argument("--tag", default="", help="inverted mode: label for the table row")
    ap.add_argument("--no-r1", action="store_true",
                    help="inverted mode: skip grading y and t_hat against the OpenThoughts row's R1 answer")
    ap.add_argument("--paired", action="store_true",
                    help="also compare against R1's ground-truth trace for the SAME "
                         "prompts (traces mode only) — removes prompt-difficulty variance")
    ap.add_argument("--vs-r1", action="store_true",
                    help="traces mode, REPORT ONLY: grade each kept row's boxed answer against "
                         "the OpenThoughts row's own R1 answer (docs/14 §4.6). No filtering follows.")
    ap.add_argument("--vs-r1-out", default="",
                    help="per-row json for --vs-r1 (default <file>-vs-r1.json)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file)]
    print(f"=== {args.file}  ({args.mode}, n={len(rows)}) ===")
    if args.mode == "inverted":
        hold = set(json.load(open(args.holdout))["idx"]) if args.holdout else None
        r1 = None if args.no_r1 else r1_answers([r["idx"] for r in rows])
        fails = inverted(rows, tok(args.summary_tokenizer), args.cap, hold,
                         len(hold) if hold else None, args.tag,
                         out_json=args.file.replace(".jsonl", "") + "-consistency.json", r1=r1)
        print(f"\n=== gates ===")
        for f in fails:
            print(f"  FAIL  {f}")
        print("  ** PASSED **" if not fails
              else f"  ** {len(fails)} GATE FAILURE(S) — DO NOT ACCEPT AS-IS **")
        sys.exit(1 if fails else 0)
    if args.mode == "traces":
        tk = tok(args.trace_tokenizer)
        fails = traces(rows, tk, args.cap, tuple(args.cap_hit_band), args.err_rate_max)
        if args.paired:
            fails += paired_reference(rows, tk, args.cap, gap_tol=args.paired_gap_tol)
        else:
            print("\n  (paired cap-hit gate NOT run — pass --paired)")
        if args.vs_r1:
            vs_r1(rows, args.vs_r1_out or args.file.replace(".jsonl", "") + "-vs-r1.json")
        print(f"\n=== gates ===")
        for f in fails:
            print(f"  FAIL  {f}")
        print("  ** PASSED **" if not fails
              else f"  ** {len(fails)} GATE FAILURE(S) — DO NOT ACCEPT AS-IS **")
        sys.exit(1 if fails else 0)
    else:
        fails = summaries(rows, tok(args.summary_tokenizer))
        # Integrity gates run on a full D2 file, not on a ~200-row pi validation
        # sample where a short count is expected.
        if not args.validation:
            fails += d2_gates(rows, args.target)
        print(f"\n=== gates ===")
        for f in fails:
            print(f"  FAIL  {f}")
        print("  ** PASSED **" if not fails
              else f"  ** {len(fails)} GATE FAILURE(S) — DO NOT ACCEPT AS-IS **")
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
