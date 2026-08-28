#!/usr/bin/env python
"""Self-check for the Phase 1 trace gates in phase1_stats.py.

The gates decide whether a 24 h generation run is believable, so a gate that
silently stops firing is the expensive failure — this project's documented shape.
Each case below flips exactly one thing and asserts the matching gate fires.

Run: .venv-vllm/bin/python bench/test_phase1_gates.py
"""
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "phase1_stats", Path(__file__).with_name("phase1_stats.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


class FakeTok:
    """1 token per 4 chars. The gates never read token values, only lengths."""
    def encode(self, s):
        return [0] * (len(s) // 4 + 1)


def row(idx, domain="math", capped=False, trace="x" * 400, fin="stop"):
    return {"idx": idx, "domain": domain, "source": "numina_math", "prompt": "p",
            "raw": trace, "trace": trace, "answer": "" if capped else "a" * 40,
            "gen_tokens": 100, "finish_reason": fin, "capped": capped, "secs": 1.0}


def gates(rows):
    return st.traces(rows, FakeTok(), 8192)


# 30% capped, two domains, no errors — the shape a healthy arm has.
BASE = ([row(i) for i in range(70)]
        + [row(100 + i, capped=True, fin="length") for i in range(30)]
        + [row(200 + i, domain="code") for i in range(10)])

f = gates(BASE)
assert f == [], f"healthy arm should pass, got {f}"

f = gates(BASE + [row(900, fin="ERROR: ConnectionError: boom")])
assert any("request error" in x for x in f), f"error gate did not fire: {f}"

f = gates(BASE + [row(901, trace="   ")])
assert any("empty trace" in x for x in f), f"empty-trace gate did not fire: {f}"
# ...but an empty trace on a CAPPED row is normal and must NOT fire
f = gates(BASE + [row(902, capped=True, fin="length", trace="")])
assert f == [], f"empty trace on a capped row must not fire: {f}"

# a domain present in the run but with every row capped never reaches D2
f = gates(BASE + [row(903, domain="biology", capped=True, fin="length")])
assert any("biology" in x for x in f), f"zero-kept-domain gate did not fire: {f}"

f = gates([row(i) for i in range(95)] + [row(500 + i, capped=True) for i in range(5)])
assert any("outside the 10-45%" in x for x in f), f"low cap-hit gate did not fire: {f}"

f = gates([row(i) for i in range(50)] + [row(500 + i, capped=True) for i in range(50)])
assert any("outside the 10-45%" in x for x in f), f"high cap-hit gate did not fire: {f}"

# boundaries are inclusive, so 10% and 45% must PASS, not fail
for k in (10, 45):
    f = gates([row(i) for i in range(100 - k)] + [row(500 + i, capped=True) for i in range(k)])
    assert not any("outside" in x for x in f), f"cap-hit {k}% should be inside the band: {f}"

print("all phase1 trace gates fire correctly")




# --- the paired cap-hit gate -------------------------------------------------
# Injects reference lengths rather than loading OpenThoughts, so the gate that
# prompt difficulty cannot confound is not the one gate left untested.

def paired(n_ours_capped, n_r1_over, n=100, cap=8192):
    """n rows; n_ours_capped of ours hit the cap, n_r1_over of R1's exceed it."""
    rows = [row(i, capped=i < n_ours_capped,
                fin="length" if i < n_ours_capped else "stop") for i in range(n)]
    r1 = [cap + 1000 if i < n_r1_over else 2000 for i in range(n)]
    return st.paired_reference(rows, FakeTok(), cap, r1_tokens=r1)


import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):          # the gate prints a full report
    over = paired(30, 19)      # +11 pts
    under = paired(30, 21)     # + 9 pts
    negative = paired(19, 30)  # -11 pts — abs(), so direction must not matter
    exact = paired(30, 20)     # +10 pts exactly — tolerance is inclusive

assert any("paired cap-hit gap" in x for x in over), f"+11 pt gap must fire: {over}"
assert under == [], f"+9 pt gap must pass: {under}"
assert any("paired cap-hit gap" in x for x in negative), f"-11 pt gap must fire: {negative}"
assert exact == [], f"exactly 10 pts is inside the tolerance: {exact}"
assert "ours - R1 = +11.0 pts" in buf.getvalue(), "the gap must be printed, not only gated"

print("paired cap-hit gate fires at ±11 pts, passes at 9 and at exactly 10")


# --- D2 integrity gates ------------------------------------------------------
# These catch what the trace gates structurally cannot: the trace gates check
# traces, and these defects live in the summaries. Both were real on the 7B arm.

def d2row(idx, b="a summary", fin="stop", x="p", y="ans", t="trace"):
    return {"idx": idx, "domain": "math", "source": "numina_math",
            "x": x, "y": y, "b": b, "t": t, "summary": b,
            "summary_tokens": 100, "finish_reason": fin}


import io, contextlib

def d2(rows, target=5000):
    with contextlib.redirect_stdout(io.StringIO()):
        return st.d2_gates(rows, target)


CLEAN = [d2row(i) for i in range(5000)]
assert d2(CLEAN) == [], "a clean D2 must pass"

f = d2(CLEAN[:-1] + [d2row(9999, b="   ")])
assert any("empty 'b'" in x for x in f), f"empty summary gate did not fire: {f}"
# an empty b is NOT the no-summary condition; it must fail rather than pass quietly
assert not any("finish_reason" in x for x in f), "empty b must not be reported as a cap hit"

f = d2(CLEAN[:-1] + [d2row(9999, fin="length")])
assert any("hit the generation cap" in x for x in f), f"severed summary gate did not fire: {f}"

for field in ("x", "y", "t"):
    f = d2(CLEAN[:-1] + [d2row(9999, **{field: ""})])
    assert any(f"empty {field!r}" in x for x in f), f"empty {field} gate did not fire: {f}"

f = d2(CLEAN[:-1] + [d2row(0)])          # idx 0 already present
assert any("duplicate idx" in x for x in f), f"duplicate-idx gate did not fire: {f}"

f = d2(CLEAN[:4999])
assert any("target 5000" in x for x in f), f"short-count gate did not fire: {f}"
assert d2(CLEAN[:4999], target=4999) == [], "an explicit lower target must pass"

print("D2 integrity gates fire on empty x/y/b/t, severed summaries, duplicates, short counts")
