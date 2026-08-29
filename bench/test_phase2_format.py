#!/usr/bin/env python
"""Self-check for the Phase 2 formatter and prompts (docs/13 §7). Exits non-zero on failure.

Run AFTER phase2_format.py:  .venv/bin/python bench/test_phase2_format.py

What it pins down: the format-matched prompts differ from the verbatim Appendix B text at
exactly the three §4.1 edits (asserted on the diff, not on prose); the holdout is 200 paired
idx disjoint from every train file; no-summary files carry no b' text; the rendered prompt
is a token-prefix of prompt+completion with `<think>` exactly once and the completion
decoding to t'<|im_end|>\\n; and an over-length row is dropped and counted, never truncated.
"""
import difflib, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "phase2"))
import prompts as P  # noqa: E402  (its sha256 asserts run on import)

spec = importlib.util.spec_from_file_location("phase2_format", HERE / "phase2_format.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def diff(a, b):
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm="")
            if l[:1] in "+-" and not l.startswith(("---", "+++"))]


# --- the three edits, as a diff ------------------------------------------------------------
assert diff(P.SUM_SYSTEM_VERBATIM, P.SUM_SYSTEM) == [
    "-- A list of numbered **reasoning bubbles**, where each bubble summarizes one key insight, step, or",
    "+- A **reasoning summary** written as a few short bold-header sections, where each bubble summarizes one key insight, step, or",
    "-reconstruct the full thinking trace that might have led to each summary. For each bubble, expand it",
    "+reconstruct the full thinking trace that might have led to each summary. For each section, expand it",
    "-- Be output as one continuous trace, wrapped in `<think>…</think>` tags",
], diff(P.SUM_SYSTEM_VERBATIM, P.SUM_SYSTEM)
assert diff(P.SUM_USER_VERBATIM, P.SUM_USER) == [
    "-Transform this thinking bubbles into clear full reasoning traces: {reasoning_summary}",
    "+Transform this reasoning summary into clear full reasoning traces: {reasoning_summary}",
]
assert diff(P.NOSUM_SYSTEM_VERBATIM, P.NOSUM_SYSTEM) == ["-- Wrapped in `<think>…</think>` tags."]
assert P.NOSUM_USER == P.NOSUM_USER_VERBATIM
print("prompts: verbatim sha256 ok; format-matched variants differ at exactly the three §4.1 edits")

# --- holdout ---------------------------------------------------------------------------------
H = json.load(open(fm.HOLDOUT))
hold = set(H["idx"])
assert H["seed"] == 20260828 and H["pool"] == 3681 and len(H["idx"]) == 200 == len(hold), H
d2 = {arm: {json.loads(l)["idx"] for l in open(p)} for arm, p in fm.D2.items()}
assert hold <= d2["7b"] and hold <= d2["1.5b"], "holdout must be drawn from prompts BOTH arms kept"
print(f"holdout: 200 idx, seed {H['seed']}, pool {H['pool']}, subset of both arms' D2")

# --- the eight files ------------------------------------------------------------------------
stats = json.load(open(fm.OUT / "format-stats.json"))["files"]
sum_row = nosum_row = None
for arm in ("7b", "1.5b"):
    b_by_idx = {}
    for setting in ("sum", "nosum"):
        tr = [json.loads(l) for l in open(fm.OUT / f"{arm}-{setting}-train.jsonl")]
        ho = [json.loads(l) for l in open(fm.OUT / f"{arm}-{setting}-holdout.jsonl")]
        tidx = [r["idx"] for r in tr]
        assert len(set(tidx)) == len(tidx), "duplicate idx in train"
        assert set(tidx).isdisjoint(hold), f"{arm}-{setting}: holdout leaked into train"
        assert {r["idx"] for r in ho} == hold, f"{arm}-{setting}: holdout file != holdout.json"
        assert len(tr) + len(ho) + stats[f"{arm}-{setting}-train"]["dropped_over_length"] \
            + stats[f"{arm}-{setting}-holdout"]["dropped_over_length"] == len(d2[arm])
        for r in tr + ho:
            assert r["chat_template_kwargs"] == {"enable_thinking": False}
            assert "<think>" not in r["completion"][0]["content"] and "</think>" not in r["completion"][0]["content"]
            assert r["completion"][0]["content"] == r["completion"][0]["content"].strip()
        if setting == "sum":
            b_by_idx = {r["idx"]: r["b"] for r in tr + ho}
            sum_row = sum_row or ho[0]
        else:
            for r in tr + ho:
                u = r["prompt"][1]["content"]
                assert "b" not in r and "Transform this reasoning summary" not in u
                assert b_by_idx[r["idx"]] not in u, f"{arm}-nosum idx {r['idx']}: b' text present"
            nosum_row = nosum_row or ho[0]
        print(f"{arm}-{setting}: train {len(tr)} · holdout {len(ho)} · dropped over-length "
              f"{stats[f'{arm}-{setting}-train']['dropped_over_length'] + stats[f'{arm}-{setting}-holdout']['dropped_over_length']}"
              + (" · no b' in any no-summary row" if setting == "nosum" else ""))

# --- rendering, the way the trainer renders -------------------------------------------------
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
for name, row in (("sum", sum_row), ("nosum", nosum_row)):
    p, pc = fm.render_ids(tok, row)
    assert pc[: len(p)] == p, f"{name}: prompt is not a token-prefix of prompt+completion"
    text = tok.decode(pc)
    assert text.count("<think>") == 1 and text.count("</think>") == 1, f"{name}: think tags {text.count('<think>')}"
    assert tok.decode(p).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), tok.decode(p)[-60:]
    assert tok.decode(pc[len(p):]) == row["completion"][0]["content"] + "<|im_end|>\n", f"{name}: completion mismatch"
    print(f"render {name}: prompt {len(p)} tokens is a prefix of {len(pc)}; <think> once; "
          f"completion == t'+'<|im_end|>\\n'  (eos_token {tok.eos_token!r})")

# --- over-length: dropped and counted, never truncated -------------------------------------
d2row = {"idx": -1, "domain": "math", "x": "q", "y": "a", "b": "s", "t": "reason " * 20000}
short = {"idx": -2, "domain": "math", "x": "q", "y": "a", "b": "s", "t": "short trace"}
kept, dropped, lens = fm.format_rows([d2row, short], "sum", tok, 12288)
assert dropped == 1 and [r["idx"] for r in kept] == [-2], (dropped, [r["idx"] for r in kept])
kept2, dropped2, lens2 = fm.format_rows([d2row], "sum", tok, 10**9)
assert dropped2 == 0 and lens2[0][1] > 12288 and kept2[0]["completion"][0]["content"] == d2row["t"].strip()
print(f"over-length: a {lens2[0][1]}-token row is dropped (count 1) at max_length 12288, never truncated")
print("all phase2 format checks pass")
