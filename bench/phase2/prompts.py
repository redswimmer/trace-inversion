"""Pinned prompts for Phase 2 — the inverter's conditioning format (docs/13 §4.1).

The paper never states the fine-tuned inverter's input format; it gives only the two
zero-shot inversion prompts (Appendix B). We use those, format-matched. The VERBATIM
strings below were extracted programmatically from docs/01 §7.2 / §7.3 (the
transcription of record), never retyped, and are sha256-checked at import — the same
discipline as bench/phase1/prompts.py.

The format-matched variants are DERIVED IN CODE from the verbatim strings with
`.replace()` calls, so the edit is a diff, not a description. Exactly three edits
(docs/13 §4.1), asserted by bench/test_phase2_format.py:

  1. format match (docs/09 row 6.2): the v2 inversion prompt still expects v1's
     "numbered reasoning bubbles", which the v2 compression prompt forbids. Three
     substrings change; every other "bubble" the paper wrote is left as written.
  2. the `<think>…</think>` output-wrapping line is dropped from BOTH prompts:
     Qwen3.5's chat template owns those tags and re-parses any `</think>` it finds
     in assistant content (docs/06 §4.4). The trace is emitted as plain content.
  3. nothing else.

Placeholder names in the user templates are the paper's own ({user_prompt},
{assistant_answer}, {reasoning_summary}); fill them with x', y', b'.
"""
import hashlib

# --- Appendix B, "Zero-shot Inversion Prompt (with summaries)" — VERBATIM ------------
SUM_SYSTEM_VERBATIM = 'You are a language model that reconstructs full internal reasoning traces from high-level bubble\nsummaries.\n\nYou will be given:\n- A problem **input** (e.g., a math or logic problem)\n- A final **output** or solution\n- A list of numbered **reasoning bubbles**, where each bubble summarizes one key insight, step, or\n  decision made during the problem-solving process\n\nThese bubbles are **condensed summaries** of what was originally a much longer, richer internal\nthought process. Your task is to reconstruct that full process.\n\nBelow are high-level bubble summaries representing condensed thoughts or decisions. Your task is to\nreconstruct the full thinking trace that might have led to each summary. For each bubble, expand it\ninto a **detailed internal monologue or reasoning chain**, showing how one idea leads to the next.\n\nInclude:\n- Assumptions and background intuitions\n- Intermediate steps, definitions, and subcases\n- Natural questions or doubts raised during reasoning\n- Alternatives that were considered and rejected\n- Transitions that make the reasoning coherent and plausible\n\nUse informal, introspective language — as if the person is thinking out loud. Add math expressions\nin LaTeX where appropriate.\n\nDo **not** invent new reasoning steps outside the bubbles. Use the **input** and **output** only for\ncontext and consistency. Your goal is to **flesh out the bubbles**, not to re-solve the problem from\nscratch.\n\nThe full trace should:\n- Be logically consistent and cohesive from start to finish\n- Sound like a realistic thought process that could plausibly result in the given answer\n- Span multiple paragraphs per bubble and up to 20,000 characters overall if needed\n- Be output as one continuous trace, wrapped in `<think>…</think>` tags\n\nYou are not summarizing the bubbles. You are recovering the internal narrative that *generated* them.'

SUM_USER_VERBATIM = 'The original problem input is: {user_prompt}\nThe final answer is: {assistant_answer}\nTransform this thinking bubbles into clear full reasoning traces: {reasoning_summary}\nGenerate full reasoning traces:'

# --- Appendix B, "Zero-shot Inversion Prompt (no summaries)" — VERBATIM ---------------
NOSUM_SYSTEM_VERBATIM = 'You are a language model that reconstructs full internal reasoning traces from only an **input**\n(e.g., a math or logic problem) and a corresponding **output** (final solution or answer).\n\nYou will be given:\n- A problem **input**\n- A final **output** or solution\n\nYour task is to reconstruct the full internal reasoning process that could plausibly connect the\ninput to the output. This should be a long, detailed, introspective trace, not a short summary.\n\nGuidelines for the reasoning trace:\n- Write in the style of an informal, introspective monologue, as if the person is thinking out loud.\n- Include assumptions, intuitions, and background facts as they arise naturally.\n- Show intermediate steps, calculations, logical deductions, definitions, and subcases.\n- Raise natural questions or doubts during reasoning, and explain how they are resolved.\n- Explore alternative approaches, even ones that are discarded, and explain why.\n- Make transitions clear so the reasoning feels like a coherent train of thought.\n- Use LaTeX for math expressions where helpful.\n- Do not introduce new information inconsistent with the input or output.\n- The goal is depth, not brevity: expand ideas fully, elaborate with multiple paragraphs, and let\n  the reasoning unfold gradually.\n- The output should only appear at the end, after the reasoning is complete.\n\nThe full trace should:\n- Wrapped in `<think>…</think>` tags.\n- Be logically consistent and cohesive from start to finish.\n- Sound like a realistic thought process that could plausibly result in the given answer.\n- Span thousands of tokens if needed.'

NOSUM_USER_VERBATIM = 'The original problem input is: {user_prompt}\nThe final answer is: {assistant_answer}\nGenerate full reasoning traces:'

_HASHES = {
    "SUM_SYSTEM_VERBATIM": "d52b0f818741fb13aaf2f19d429eb0eef2e9d2a2fd4bf96e81bc0ce7eb7499cc",
    "SUM_USER_VERBATIM": "a92a113282218b712221efa36daf99989bcd183ea9a772953ac0f9463a52de8f",
    "NOSUM_SYSTEM_VERBATIM": "104344fd138f7614b4ad0565fc1f29a34091e8cf9925254f3c5964fb99418562",
    "NOSUM_USER_VERBATIM": "149b831ca28b9f5356f993d272c1e3f3a5a873ce387233a354e7d4633d257a46",
}
for _k, _want in _HASHES.items():
    _got = hashlib.sha256(globals()[_k].encode()).hexdigest()
    assert _got == _want, f"{_k} no longer matches Appendix B as transcribed in docs/01 §7: {_got}"


def _edit(s, old, new):
    """One replacement that must hit exactly once — a silent no-op is the failure mode."""
    assert s.count(old) == 1, f"edit target occurs {s.count(old)}x, expected 1: {old!r}"
    return s.replace(old, new)


# Edit 1 — format match (three substrings, docs/13 §4.1).
SUM_SYSTEM = _edit(SUM_SYSTEM_VERBATIM,
                   "A list of numbered **reasoning bubbles**",
                   "A **reasoning summary** written as a few short bold-header sections")
SUM_SYSTEM = _edit(SUM_SYSTEM, "For each bubble", "For each section")
SUM_USER = _edit(SUM_USER_VERBATIM, "Transform this thinking bubbles", "Transform this reasoning summary")

# Edit 2 — drop the <think> wrap instruction from both prompts.
SUM_SYSTEM = _edit(SUM_SYSTEM,
                   "- Be output as one continuous trace, wrapped in `<think>…</think>` tags\n", "")
NOSUM_SYSTEM = _edit(NOSUM_SYSTEM_VERBATIM, "- Wrapped in `<think>…</think>` tags.\n", "")

# Edit 3 — nothing else.
NOSUM_USER = NOSUM_USER_VERBATIM

for _s in (SUM_SYSTEM, SUM_USER, NOSUM_SYSTEM, NOSUM_USER):
    assert "think>" not in _s, "a <think> tag survived into the conditioning format"


if __name__ == "__main__":
    import difflib
    for name, a, b in (("SUM_SYSTEM", SUM_SYSTEM_VERBATIM, SUM_SYSTEM),
                       ("SUM_USER", SUM_USER_VERBATIM, SUM_USER),
                       ("NOSUM_SYSTEM", NOSUM_SYSTEM_VERBATIM, NOSUM_SYSTEM),
                       ("NOSUM_USER", NOSUM_USER_VERBATIM, NOSUM_USER)):
        print(f"=== {name}: verbatim {len(a)} chars -> format-matched {len(b)} chars")
        for line in difflib.unified_diff(a.splitlines(), b.splitlines(), "verbatim", "ours", n=0, lineterm=""):
            print(line)
