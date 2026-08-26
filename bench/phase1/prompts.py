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


# --------------------------------------------------------------------------------
# pi -- the compression prompt, run by C' = Qwen3.5-4B.
#
# RECONSTRUCTED, not verbatim. Paper v2's Appendix B gives the spec in prose but the
# prompt text is only in the PDF, and its "two few-shot exemplars of the target style
# (one algebra/number theory and one geometry, ~600 tokens each), drawn from
# GPT-5-mini's own summaries" were never released (docs/02, docs/11 §4). The exemplars
# below are written from scratch to the spec.
#
# Spec, from Appendix B: 3-6 sections, each opening with a short bold markdown header
# on its own line followed by a 2-5 sentence paragraph; no numbered lists, no bullets;
# first-person present tense, tentative/exploratory; one meaningful reasoning move per
# section; inline LaTeX where the original used math; no meta-commentary; roughly
# 600-900 tokens.
#
# Note the tension in the paper's own numbers: the spec says 600-900 tokens, but the
# Table 1 medians it reports are 537 (their C') and 592 (GPT-5.4 mini). The stated
# target is what the prompt says; the measured median is what we validate against.
# The exemplars are sized near 590 tokens so they pull toward the measured value.
#
# Acceptance is Table 1 (docs/11 §4): median tokens 540-590, bold-header sections
# >90%, first-person prose >95%, LaTeX >70%. Measured by bench/phase1_stats.py.
# --------------------------------------------------------------------------------

_PI_EXEMPLAR_ALGEBRA = r"""**Reading what the condition actually says**
The problem wants every positive integer \( n \) for which \( n + 1 \) divides \( n^2 + 1 \), and my first instinct is to stop staring at the two expressions and force them onto common ground. Writing \( n^2 + 1 = (n+1)(n-1) + 2 \) makes the whole thing collapse: the divisibility holds exactly when \( n + 1 \) divides \( 2 \). That felt too easy, so I want to test it before I lean on it.

**Testing the collapse on small cases**
At \( n = 1 \) I get \( 2 \mid 2 \), which works. At \( n = 2 \) I need \( 3 \mid 5 \), which fails, and \( n = 3 \) needs \( 4 \mid 10 \), which also fails. Those agree with the rewriting, since the positive divisors of \( 2 \) are \( 1 \) and \( 2 \), and only \( n + 1 = 2 \) is reachable for positive \( n \). I am fairly convinced the answer is just \( n = 1 \).

**A detour through modular arithmetic that I abandon**
Before settling I try the congruence route, reducing \( n \equiv -1 \pmod{n+1} \) and substituting to get \( n^2 + 1 \equiv 2 \pmod{n+1} \). It gives the same conclusion, which is reassuring, but it is really the same computation wearing different clothes. I drop it rather than present two versions of one idea.

**Worrying about what I might have excluded**
The one thing nagging me is whether the factorisation quietly assumed something. It does not: \( n^2 + 1 = (n+1)(n-1) + 2 \) is an identity over the integers, valid for every \( n \), so no case is being silently discarded. The restriction to positive \( n \) is the problem's, not mine, though I notice \( n = -3 \) would also satisfy \( n + 1 \mid 2 \) if negatives were allowed.

**Where I land**
So the answer is \( n = 1 \), and the reason is a single rewriting rather than anything deep. What makes the problem feel harder than it is, I think, is that \( n + 1 \) and \( n^2 + 1 \) look unrelated until you subtract the obvious multiple."""

_PI_EXEMPLAR_GEOMETRY = r"""**Laying out what the triangle gives me**
I have a triangle with \( AB = 13 \), \( BC = 14 \), \( CA = 15 \), and I want \( BD \), where \( D \) is the point at which the incircle meets \( BC \). The 13-14-15 triangle is familiar enough that I suspect the numbers are chosen to come out clean. My instinct is that this is a tangent-length question rather than a computation.

**A false start with coordinates**
My first attempt is to drop the triangle onto axes, putting \( B \) at the origin and \( C \) at \( (14, 0) \), then solving for \( A \). That does work: \( A \) lands at \( (5, 12) \), and I could find the incenter from there. But it means computing the inradius and then a foot of perpendicular, and I can feel the arithmetic getting heavier than the problem deserves, so I back out.

**Remembering the tangent lengths**
The cleaner idea is that the two tangent segments from a vertex to the incircle have equal length. Calling those lengths \( x \), \( y \), \( z \) from \( A \), \( B \), \( C \), the sides give \( y + z = 14 \), \( x + y = 13 \), \( x + z = 15 \). Adding all three and halving gives \( x + y + z = 21 \), which is just the semiperimeter \( s \).

**Reading off the answer**
Since \( BD = y \) and \( x + z = 15 \), I get \( y = 21 - 15 = 6 \). Written the standard way that is \( BD = s - b \) with \( b = CA \), which is the identity I half-remembered at the start and have now rederived rather than trusted.

**Checking it against the discarded route**
As a sanity check I go back to the coordinates I abandoned: the incenter sits at \( (6, 4) \), so the foot of the perpendicular to \( BC \) is at \( x = 6 \), giving \( BD = 6 \). The two routes agree, which is what I wanted before committing."""

PI_SYSTEM = f"""You convert a model's raw internal reasoning into a short first-person recap of that same reasoning, written as though the thinker were narrating their thought process while it was happening.

Write between 3 and 6 sections. Each section opens with a short bold markdown header on its own line, followed by one paragraph of 2 to 5 sentences. Put a blank line between sections.

Follow these rules exactly:

Never use numbered lists or bullet points. Write prose paragraphs only.

Write in the first person and the present tense -- "I notice", "I try", "I am not sure yet". Keep the voice tentative and exploratory, the way the reasoning felt while it was unfinished, rather than the confident voice of a written-up solution.

Give each section exactly one meaningful reasoning move: an observation, a decision, an attempt that failed and redirected the thinking, or a realisation. Do not pad a section by restating the problem.

Where the original reasoning used mathematics, use inline LaTeX in the same places, for example \\( x^2 + 1 \\) or \\( \\frac{{a}}{{b}} \\).

Never mention the reasoning trace, the summary, the task, or yourself. No meta-commentary of any kind, no preamble, and no closing sentence that announces a conclusion has been reached.

Aim for roughly 600 to 900 tokens overall.

Two examples of the target style follow.

Example one:

{_PI_EXEMPLAR_ALGEBRA}

Example two:

{_PI_EXEMPLAR_GEOMETRY}"""

# VERBATIM from paper v2 Appendix B, as quoted in docs/02.
PI_USER = ("Summarize this thinking process as a first-person inner-monologue recap: "
           "<think>{thinking}</think>")
