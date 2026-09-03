#!/usr/bin/env python
"""Redraw bookkeeping for capped forgeries (docs/15 §4.4): seeds 1234 -> 1235 -> 1236, at most three
draws, then drop and report. Two commands plus --selftest. Nothing here opens a file named ORACLE.

  subset   --draw forged-<tag>-drawN.jsonl --prompts attack-<set>.jsonl --out <redraw prompts>
           the prompt rows (no t anywhere) whose idx hit the cap (finish_reason == "length") in that
           draw, in prompt-file order. Prints the count; an empty file means nothing capped.
  assemble --draws d1 [d2 [d3]] --prompts attack-<set>.jsonl --out forged-<tag>.jsonl
           the training artifact: for every idx the first draw that terminated (`draw` = 1, 2 or 3);
           an idx capped in every draw is DROPPED. Rows keep invert.py's fields minus t_true (always
           None on split B) plus `draw`. Writes <out>-draws.json — per draw: rows, capped, empty,
           stripped_think, loops, gen_tokens; the dropped idx; final rows and loops (docs/15 §6).
           Gates (exit 1, files still written): every draw's idx ⊆ the prompt file's and draw n == the
           capped set of draw n-1 · final idx == prompt idx minus the dropped, in order · 0 empty t_hat
           · 0 finish_reason == "length" · no t_true content · nothing dropped after fewer than 3 draws.

Loop test, as phase2.md §5.4 used it: a 40-char chunk repeated >= 3x within the trace's last 4,000 chars.
"""
import argparse, json, sys, tempfile
from pathlib import Path

CAP = "length"
FINAL_KEYS = {"idx", "domain", "x", "y", "b", "t_hat", "raw", "gen_tokens", "finish_reason", "draw"}


def no_oracle(*paths):
    bad = [str(p) for p in paths if p and "oracle" in str(p).lower()]
    assert not bad, f"a file named ORACLE on the attack path: {bad}"


def load(path):
    return [json.loads(l) for l in open(path)]


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".new", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(str(path) + ".new").replace(path)


def loops(text, chunk=40, window=4000, step=8):
    tail = text[-window:]
    # ponytail: O(n²) scan stepping 8 chars (~10 ms/row); a loop of any period >= 1 still repeats some chunk
    return any(tail.count(tail[i:i + chunk]) >= 3 for i in range(0, max(len(tail) - chunk, 0) + 1, step))


def summarize(rows, n):
    return {"draw": n, "rows": len(rows), "capped": sum(r["finish_reason"] == CAP for r in rows),
            "empty": sum(not r["t_hat"].strip() for r in rows),
            "stripped_think": sum("</think>" in r["raw"] for r in rows),
            "loops": sum(loops(r["t_hat"]) for r in rows),
            "capped_loops": sum(loops(r["t_hat"]) for r in rows if r["finish_reason"] == CAP),
            "gen_tokens": sum(r["gen_tokens"] for r in rows)}


def subset(draw, prompts, out):
    capped = {r["idx"] for r in load(draw) if r["finish_reason"] == CAP}
    rows = [r for r in load(prompts) if r["idx"] in capped]
    assert len(rows) == len(capped), f"{len(capped) - len(rows)} capped idx missing from the prompt file"
    write(out, rows)
    print(f"capped in {Path(draw).name}: {len(rows)} -> {out}", flush=True)
    return len(rows)


def assemble(draws, prompts, out):
    pidx = [r["idx"] for r in load(prompts)]
    pset = set(pidx)
    final, fails, record = {}, [], {"draws": [], "dropped_idx": []}
    prev_capped, t_true = None, 0
    for n, path in enumerate(draws, 1):
        rows = load(path)
        idx = {r["idx"] for r in rows}
        if not idx <= pset:
            fails.append(f"draw {n}: {len(idx - pset)} idx not in the prompt file")
        if prev_capped is not None and idx != prev_capped:
            fails.append(f"draw {n}: idx set != draw {n - 1}'s capped set ({len(idx)} vs {len(prev_capped)})")
        s = summarize(rows, n)
        s["file"] = str(path)
        if n > 1:
            s["rescued"] = s["rows"] - s["capped"]
        record["draws"].append(s)
        for r in rows:
            t_true += r.get("t_true") is not None
            if r["finish_reason"] != CAP and r["idx"] not in final:
                final[r["idx"]] = dict({k: v for k, v in r.items() if k != "t_true"}, draw=n)
        prev_capped = {r["idx"] for r in rows if r["finish_reason"] == CAP}
    record["dropped_idx"] = sorted(prev_capped)
    ordered = [final[i] for i in pidx if i in final]
    write(out, ordered)
    record.update(final_rows=len(ordered), final_loops=sum(loops(r["t_hat"]) for r in ordered),
                  rows_by_draw={n: sum(r["draw"] == n for r in ordered) for n in range(1, len(draws) + 1)},
                  prompt_rows=len(pidx))
    if t_true:
        fails.append(f"{t_true} draw rows carry a t_true — the generation path saw a trace")
    if [r["idx"] for r in ordered] != [i for i in pidx if i not in prev_capped]:
        fails.append("final idx != prompt idx minus the dropped, in order")
    if prev_capped and len(draws) < 3:
        fails.append(f"{len(prev_capped)} idx still capped after only {len(draws)} draw(s) — the policy is three")
    if any(r["finish_reason"] == CAP for r in ordered):
        fails.append("a capped row reached the final file")
    if any(not r["t_hat"].strip() for r in ordered):
        fails.append("an empty t_hat reached the final file")
    if ordered and {k for r in ordered for k in r} != FINAL_KEYS:
        fails.append(f"final keys {sorted({k for r in ordered for k in r})} != {sorted(FINAL_KEYS)}")
    record["gates"] = fails
    rec_path = str(out).replace(".jsonl", "") + "-draws.json"
    json.dump(record, open(rec_path, "w"), indent=1)
    for s in record["draws"]:
        print(f"draw {s['draw']}: rows {s['rows']}  capped {s['capped']}"
              + (f"  rescued {s['rescued']}" if "rescued" in s else f"  ({100 * s['capped'] / max(s['rows'], 1):.1f}% cap-hit)")
              + f"  empty {s['empty']}  stripped_think {s['stripped_think']}  loops {s['loops']} "
              f"(of which capped {s['capped_loops']})  gen_tokens {s['gen_tokens']:,}", flush=True)
    print(f"final: {len(ordered)} rows of {len(pidx)}  by draw {record['rows_by_draw']}  dropped {len(prev_capped)} "
          f"{record['dropped_idx'][:20]}  loops {record['final_loops']}  -> {out}  record {rec_path}")
    for f in fails:
        print(f"  FAIL  {f}")
    print("  ** PASSED **" if not fails else f"  ** {len(fails)} GATE FAILURE(S) — DO NOT ACCEPT AS-IS **")
    return fails


def selftest():
    d = tempfile.mkdtemp()
    P, D1, D2, D3, R, OUT = (f"{d}/{n}.jsonl" for n in ("prompts", "d1", "d2", "d3", "redraw", "final"))

    def row(i, fr, text="a plain trace that ends"):
        return {"idx": i, "domain": "math", "x": f"x{i}", "y": f"y{i}", "b": None, "t_true": None,
                "t_hat": text, "raw": text, "gen_tokens": 3, "finish_reason": fr}

    write(P, [{"idx": i, "x": f"x{i}", "y": f"y{i}", "prompt": [], "chat_template_kwargs": {}} for i in (5, 3, 9)])
    write(D1, [row(5, "stop"), row(3, CAP), row(9, CAP)])
    assert subset(D1, P, R) == 2 and [r["idx"] for r in load(R)] == [3, 9], "subset must keep prompt order"
    write(D2, [row(3, "stop"), row(9, CAP)])
    write(D3, [row(9, CAP, "prose then " + "loop chunk " * 400)])
    assert loops("loop chunk " * 400) and not loops("no repetition here " + "x" * 50)
    assert assemble([D1, D2, D3], P, OUT) == []
    out = load(OUT)
    assert [r["idx"] for r in out] == [5, 3] and [r["draw"] for r in out] == [1, 2] and not any("t_true" in r for r in out)
    rec = json.load(open(OUT.replace(".jsonl", "") + "-draws.json"))
    assert rec["dropped_idx"] == [9] and rec["draws"][2]["loops"] == 1 and rec["draws"][1]["rescued"] == 1
    assert assemble([D1], P, OUT), "dropping after one draw must fail"
    write(D2, [row(3, "stop"), row(5, "stop")])
    assert any("capped set" in f for f in assemble([D1, D2, D3], P, OUT)), "a draw over the wrong idx must fail"
    write(D2, [dict(row(3, "stop"), t_true="the victim's trace"), row(9, CAP)])
    assert any("t_true" in f for f in assemble([D1, D2, D3], P, OUT)), "a t_true on the generation path must fail"
    print("phase4_draws self-test passed: subset order, loop test, assemble (draw field, drops, record), "
          "and the fewer-than-three-draws / wrong-idx / t_true gates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["subset", "assemble"])
    ap.add_argument("--draw"), ap.add_argument("--draws", nargs="+"), ap.add_argument("--prompts")
    ap.add_argument("--out"), ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    no_oracle(a.draw, a.prompts, a.out, *(a.draws or []))
    if a.cmd == "subset":
        subset(a.draw, a.prompts, a.out)
    elif a.cmd == "assemble":
        sys.exit(1 if assemble(a.draws, a.prompts, a.out) else 0)
    else:
        ap.error("subset, assemble or --selftest")


if __name__ == "__main__":
    main()
