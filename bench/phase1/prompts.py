"""Pinned prompts for Phase 1.

Every string here that has a source in the paper's repo is reproduced VERBATIM and is
checked by sha256 at import. They were extracted from the repo's AST, never retyped:
the surrogate system prompt has a leading space and wording ("summarizing",
"backtracing", no article before "well-considered") that transcription silently
normalises, and docs/07 §3 only quotes it truncated.

Repo: github.com/Tingwei-Zhang/Trace_Inversion_Attack @ main
"""
import hashlib

# --------------------------------------------------------------------------------
# Surrogate inference system prompt.
# VERBATIM: src/step0_data_preprocess/preprocess_r1_distill.py, the `inference_base`
# system message. NOT the near-identical string the OpenThoughts rows carry in
# messages[0] (1042 chars, different wording) — see docs/09 row 7.8.
#
# It sets trace length, which is Phase 1's whole output and what the cap-hit STOP band
# keys on. Dropping it shortens traces and pushes cap-hit below 10%, which would read
# as a misbehaving surrogate (docs/11 §3).
# --------------------------------------------------------------------------------
SURROGATE_SYSTEM = ' Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.'

# v1's compression prompt, for contrast only — NOT what we run.
# VERBATIM: src/step1_summarization/data_formatter.py, SummarizationDataFormatter.
# v2 (Appendix B) forbids exactly what this produces: numbered lists. Kept here so the
# structural gap between v1 and v2 is a diff rather than a description (docs/11 §4).
V1_COMPRESSION_SYSTEM = 'You are a model trained to convert informal internal reasoning into a clear, structured sequence of “reasoning bubbles.”  \nThese bubbles should summarize the key steps of thought, capturing both logical flow and meaningful insight — not just surface-level summaries.\n\nWhen processing a <think>...</think> trace:\n\n1. Read the full reasoning carefully and extract only the **meaningful logical advances** (e.g. observations, deductions, decisions, failed attempts that change direction).  \n2. Summarize each such idea as a **self-contained bubble**, ideally one to three sentences each.  \n3. Maintain the **logical flow** of the original trace, showing how the reasoning unfolds.  \n4. Keep each bubble:\n   - Abstracted (not just copied phrases)\n   - Logically complete and well-phrased\n   - High-information and nontrivial\n5. Do **not** include filler thoughts, aimless speculation, or mechanical calculations unless critical.\n\nYou should produce **a few numbered reasoning bubbles**, depending on the depth of the input. Each bubble should contribute meaningfully to the progression of thought.\n\n**Format:**\n1. [Detailed, insight-capturing reasoning bubble]\n2. [Next reasoning bubble showing development or pivot]\n3. ...'

V1_COMPRESSION_USER = "Transform this thinking process into clear reasoning bubbles:<think>\n\n{thinking}</think>"

_HASHES = {"SURROGATE_SYSTEM": "a047e2d4280083a8a708b292f8019c35afb1681e58de21dc43dace88d63af989"}
for _k, _want in _HASHES.items():
    _got = hashlib.sha256(globals()[_k].encode()).hexdigest()
    assert _got == _want, f"{_k} no longer matches the repo verbatim: {_got} != {_want}"


if __name__ == "__main__":
    print(f"SURROGATE_SYSTEM  {len(SURROGATE_SYSTEM)} chars  sha256 {_HASHES['SURROGATE_SYSTEM'][:16]}")
    print(repr(SURROGATE_SYSTEM))
    print(f"\nV1_COMPRESSION_SYSTEM  {len(V1_COMPRESSION_SYSTEM)} chars (reference only)")
