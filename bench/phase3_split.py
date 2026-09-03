#!/usr/bin/env python
"""Derive the attack file from the oracle file: the same rows, same order, minus `t`, nothing
else changed (docs/14 §4.5). This is the ONE place on the Phase 3 path that opens the ORACLE
file on purpose; it runs once, in the main thread, and exits 1 if the result could carry the
victim's trace. Phase 4 and the Answer-only / Summary+Answer conditions read the attack file
and nothing named ORACLE; the Victim-Trace condition reads the oracle file.

  victimB-ORACLE.jsonl  {idx, domain, source, x, y, b, t, summary, summary_tokens, finish_reason}
  victimB-attack.jsonl  the same rows without t

Asserts (exit 1): the attack file has exactly the D2-minus-t keys and no t/trace/raw; the idx
sequences are identical (set, count and order); every attack row equals its oracle row minus t;
and no oracle trace (stripped) occurs inside any field of its attack row — a mis-join that
copied the trace into any field fails here. Checked on the files as written to disk, not on the
in-memory rows.

Run:       .venv-vllm/bin/python bench/phase3_split.py --oracle <…ORACLE.jsonl> --attack <…attack.jsonl>
Self-test: .venv-vllm/bin/python bench/phase3_split.py --selftest
"""
import argparse, json, sys, tempfile
from pathlib import Path

ATTACK_KEYS = {"idx", "domain", "source", "x", "y", "b", "summary", "summary_tokens", "finish_reason"}
FORBIDDEN = {"t", "trace", "raw"}


def split(oracle, attack):
    rows = [json.loads(l) for l in open(oracle)]
    Path(attack).parent.mkdir(parents=True, exist_ok=True)
    with open(attack, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "t"}) + "\n")
    return len(rows)


def check(oracle, attack):
    o = [json.loads(l) for l in open(oracle)]
    a = [json.loads(l) for l in open(attack)]
    fails = []
    keys = {k for r in a for k in r}
    if keys != ATTACK_KEYS:
        fails.append(f"attack keys {sorted(keys)} != {sorted(ATTACK_KEYS)}")
    if keys & FORBIDDEN:
        fails.append(f"attack file carries {sorted(keys & FORBIDDEN)}")
    if [r["idx"] for r in o] != [r["idx"] for r in a]:
        fails.append(f"idx sequences differ (oracle {len(o)} rows, attack {len(a)})")
    if len({r["idx"] for r in a}) != len(a):
        fails.append("duplicate idx in the attack file")
    leaked, unequal = [], []
    for ro, ra in zip(o, a):
        t = str(ro.get("t", "")).strip()
        if not t:
            fails.append(f"idx {ro['idx']}: empty t in the oracle file")
            continue
        if any(t in str(v) for v in ra.values()):
            leaked.append(ro["idx"])
        if {k: v for k, v in ro.items() if k != "t"} != ra:
            unequal.append(ro["idx"])
    if leaked:
        fails.append(f"{len(leaked)} attack rows contain their oracle trace (idx {leaked[:5]})")
    if unequal:
        fails.append(f"{len(unequal)} attack rows are not the oracle row minus t (idx {unequal[:5]})")
    return fails, len(o), len(a)


def selftest():
    def row(i, **kw):
        r = {"idx": i, "domain": "math", "source": "s", "x": f"x{i}", "y": f"y{i}", "b": f"b{i}",
             "t": f"trace {i}\nline two", "summary": f"b{i}", "summary_tokens": 3, "finish_reason": "stop"}
        r.update(kw)
        return r

    def write(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def fails_for(rows_attack, rows_oracle):
        d = tempfile.mkdtemp()
        write(f"{d}/o.jsonl", rows_oracle)
        write(f"{d}/a.jsonl", rows_attack)
        return check(f"{d}/o.jsonl", f"{d}/a.jsonl")[0]

    oracle = [row(i) for i in range(3)]
    d = tempfile.mkdtemp()
    write(f"{d}/o.jsonl", oracle)
    split(f"{d}/o.jsonl", f"{d}/a.jsonl")
    ok = check(f"{d}/o.jsonl", f"{d}/a.jsonl")[0]
    assert ok == [], f"a correct split must pass: {ok}"
    good = [{k: v for k, v in r.items() if k != "t"} for r in oracle]

    f = fails_for(good[:2] + [dict(good[2], b=good[2]["b"] + "\n" + oracle[2]["t"])], oracle)
    assert any("contain their oracle trace" in x for x in f), f"trace copied into b must fail: {f}"
    f = fails_for(good[:2] + [dict(good[2], trace=oracle[2]["t"])], oracle)
    assert any("carries ['trace']" in x for x in f), f"a trace key must fail: {f}"
    f = fails_for(good[:2], oracle)
    assert any("idx sequences differ" in x for x in f), f"a missing row must fail: {f}"
    f = fails_for([good[0], good[2], good[1]], oracle)
    assert any("idx sequences differ" in x for x in f), f"reordered rows must fail: {f}"
    f = fails_for(good[:2] + [dict(good[2], y="edited")], oracle)
    assert any("not the oracle row minus t" in x for x in f), f"an altered field must fail: {f}"
    f = fails_for(good, oracle[:2] + [dict(oracle[2], t="  ")])
    assert any("empty t" in x for x in f), f"an empty oracle trace must fail: {f}"
    print("phase3_split self-test passed: correct split passes; copied trace, trace key, missing, "
          "reordered, altered and empty-t cases all fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="bench/results/phase3/victimB-ORACLE.jsonl")
    ap.add_argument("--attack", default="bench/results/phase3/victimB-attack.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert "ORACLE" in Path(args.oracle).name and "ORACLE" not in Path(args.attack).name, \
        "the oracle file is the one named ORACLE; the attack file must not be"
    n = split(args.oracle, args.attack)
    fails, n_o, n_a = check(args.oracle, args.attack)
    print(f"oracle rows {n_o}  attack rows {n_a}  written {n}  keys {sorted(ATTACK_KEYS)}")
    print(f"=== self-check ===")
    for f in fails:
        print(f"  FAIL  {f}")
    print("  ** PASSED: keys, idx sequence, row equality minus t, no oracle trace in any attack field **"
          if not fails else f"  ** {len(fails)} FAILURE(S) — do not use {args.attack} **")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
