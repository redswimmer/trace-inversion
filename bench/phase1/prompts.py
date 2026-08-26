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
# The SYSTEM PROMPT IS VERBATIM from paper v2 Appendix B, transcribed into
# docs/01 §6.4 and extracted from there programmatically, never retyped.
#
# It was reconstructed from a paraphrase first. docs/11 §4 says the prompt "exists
# only in its PDF", and docs/02's summary of it reads like a spec, so pi was written
# to that spec before anyone checked docs/01. The verbatim text differs in ways a
# paraphrase cannot carry:
#   - "or the final consolidation" is an ALLOWED section move. The reconstruction had
#     invented "no closing sentence that announces a conclusion", which contradicts it.
#   - length guidance for very short and very long traces ("produce fewer sections
#     rather than padding" / "prefer adding depth over adding more sections"), which
#     matters because our traces span ~200-8192 tokens.
#   - "Do not restate the final boxed answer unless the reasoning naturally concludes
#     with it" -- absent from every paraphrase.
#   - the meta-commentary rule is narrow: about following instructions, apologising,
#     or mentioning that you are summarising. Not a general ban on conclusions.
#   - `\boxed` is named as an example LaTeX command, and the voice examples are given
#     as literal strings ("I need to...", "I'm now realizing...").
#
# ONLY THE TWO EXEMPLARS ARE OURS. The paper's "two few-shot exemplars of the target
# style (one algebra/number theory and one geometry, ~600 tokens each), drawn from
# GPT-5-mini's own summaries" were never released (docs/01 §6.4, docs/11 §4). Both are
# written from scratch, sized near the Table 1 medians (537/592) while the verbatim
# instruction keeps its 600-900 -- those disagree by design, see docs/11 §4.
#
# Acceptance is Table 1: median tokens 540-590, bold-header sections >90%,
# first-person prose >95%, LaTeX >70%. Measured by bench/phase1_stats.py.
# --------------------------------------------------------------------------------

PI_SYSTEM_VERBATIM = 'You are summarizing a long chain-of-thought trace into a short, first-person "inner-monologue\nrecap" that mirrors the style of GPT-5 mini\'s internal reasoning summaries.\n\nGiven a `<think>...</think>` trace, produce a recap with these properties:\n\n**Structure.** Write 3 to 6 short sections. Each section begins with a short bold markdown header\non its own line, like `**Setting up the integral**` or `**Checking the edge case**`, and is\nfollowed by one short paragraph (2–5 sentences). Do NOT use numbered lists (1., 2.) and do NOT use\nbullet points (-, *). The whole recap is just headers + prose.\n\n**Voice.** First person, present tense, as if the model is thinking aloud: "I need to…", "I\'ll\ncheck…", "I\'m now realizing…", "Let me verify…". Keep the tone tentative and exploratory, not\ntextbook.\n\n**Content.** Each section should capture one meaningful move in the reasoning — a clarification of\nwhat is being asked, a key derivation or substitution, a pivot after a failed attempt, a sanity\ncheck, or the final consolidation. Skip filler, restatement, and purely mechanical arithmetic.\nPreserve the logical order of the trace.\n\n**Length.** Aim for roughly 600–900 tokens total — long enough that each section develops a real\nidea, not just a one-line gesture. If the input trace is very short, produce fewer sections rather\nthan padding; if it is very long, prefer adding depth to each section over adding more sections.\n\n**Math formatting.** Use inline LaTeX (`\\frac`, `\\sqrt`, `\\boxed`, etc.) where the original trace\nused math. Do not restate the final boxed answer unless the reasoning naturally concludes with it.\n\nDo not add meta-commentary about following instructions, apologize, or mention that you are\nsummarizing. Just produce the recap.'

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

PI_SYSTEM = PI_SYSTEM_VERBATIM + f"""

Two examples of the target style follow.

Example one:

{_PI_EXEMPLAR_ALGEBRA}

Example two:

{_PI_EXEMPLAR_GEOMETRY}"""

# VERBATIM from Appendix B (docs/01 §6.4). Note the newline before <think>:
# docs/02's paraphrase renders it inline with a space, docs/01 as a code block.
PI_USER = ("Summarize this thinking process as a first-person inner-monologue recap:\n"
           "<think>{thinking}</think>")

_PI_HASH = "a1e6720ced6a398f8f98d6b83938b42568ed826ed9440f769384a3b1c7e25ebb"
assert hashlib.sha256(PI_SYSTEM_VERBATIM.encode()).hexdigest() == _PI_HASH, \
    "PI_SYSTEM_VERBATIM no longer matches Appendix B as transcribed in docs/01 §6.4"
