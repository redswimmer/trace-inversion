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
print("NOT covered here: the paired cap-hit gate — it loads OpenThoughts, so it is "
      "exercised by the real --paired run rather than stubbed.")
