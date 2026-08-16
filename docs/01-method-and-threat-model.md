# Trace Inversion — Method and Threat Model (§1–§4)

Reference notes for reproducing **"How to Steal Reasoning Without Reasoning Traces"**
Tingwei Zhang, John X. Morris, Vitaly Shmatikov (Cornell Tech), arXiv:2603.07267v2 [cs.CR], 12 May 2026.
Code released by authors: <https://github.com/Tingwei-Zhang/Trace_Inversion_Attack>

Scope of this document: Introduction (§1), Background/Related Work (§2), Threat Model (§3),
Inversion Methodology (§4), plus the prompt templates in Appendix B and the qualitative
examples in Appendices C–D. Evaluation (§5) and Defenses (§6) are covered elsewhere;
they are referenced here only where they pin down a methodological detail.

---

## 1. Problem setup and motivation (§1)

### 1.1 What a "reasoning trace" is

A **reasoning trace** (used interchangeably with **chain of thought**, CoT) is the multi-step
internal deliberation a reasoning LLM generates *before* emitting its final answer (§1). The paper
treats it as a first-class object with two distinct uses:

| Use | Role of the trace |
| --- | --- |
| Inference time | Decomposes a complex problem into intermediate steps, improving accuracy (§1, §2) |
| Training time | Supervision signal for **capability transfer** — fine-tuning a student on a teacher's step-by-step trace transfers much more reasoning ability than fine-tuning on final answers alone (§1, citing hinton2015distilling; gou2021knowledge; shridhar2023distilling; chen2025skip) |

In the paper's own data format, a trace is delimited by `<think>...</think>` tags (Appendix B — both
the compression user prompt and both zero-shot inversion prompts use these tags).

### 1.2 What is exposed vs. hidden

The paper's motivating observation (§1) is that commercial reasoning APIs deliberately withhold
the trace:

- **Exposed:** the final answer, and *optionally* a short **reasoning summary** (the paper also
  calls these "bubbles"). Cited as standard practice for OpenAI, Gemini, and Anthropic extended
  thinking (openai_reasoning_docs; gemini_thoughts_docs; anthropic_extended_thinking).
- **Hidden:** the full internal chain of thought.

Two stated reasons for hiding it (§1):

1. **IP risk / distillation.** In February 2026 Anthropic publicly accused several competing labs of
   large-scale distillation campaigns that elicited data from Anthropic's reasoning models
   (anthropic_distillation_2026).
2. **Leakage risk.** Traces may encode sensitive system prompts, safety policies, or private
   context (green2025leaky; deepseek_exploit; zhou2025hidden).

The implicit industry assumption the paper attacks: *exposing summaries in lieu of detailed chains
of thought preserves the practical benefits of reasoning models while preventing capability
stealing.* (§1)

### 1.3 The claim

**Trace Inversion** synthesizes detailed reasoning traces from only the final answers and,
optionally, compressed summaries. Two results are claimed (Abstract, §1):

1. **Fidelity.** Synthesized traces overlap substantially with ground-truth traces where those are
   available — headline numbers: **81% token-length recovery** and **52.76 token-overlap F1 (TF1)**
   against DeepSeek-R1 traces, *even though the inversion model is trained on a much weaker
   surrogate*.
2. **Utility (the main claim).** Fine-tuning a student on inverted traces improves reasoning more
   than fine-tuning on (a) surrogate reasoning traces, or (b) the victim's answers + summaries —
   and it works when the black-box teacher is stronger than both the student and the inversion
   model.

Two concrete headline examples from §1 (details in §5):

- Qwen-2.5-7B-Instruct fine-tuned on traces inverted from GPT-5.4 mini's answers+summaries by a
  *weak* inverter (DeepSeek-R1-Distill-Qwen-1.5B surrogate) reaches **31.6% JEEBench**, vs **19.7%**
  fine-tuned on R1-Distill's own traces.
- Llama-3.1-8B-Instruct fine-tuned on traces inverted from GPT-5.4 mini by an R1-based inverter
  reaches **52.4% MATH500**, vs **16.4%** fine-tuned on GPT-5.4 mini's summaries + answers.

A recurring secondary finding: inverted traces can **beat the ground-truth traces** as fine-tuning
data, because real traces contain backtracking and dead ends whereas inverted traces (conditioned
on the known final answer) are clean forward reasoning (§1, §5.3, Appendix D).

---

## 2. Notation (§3, §4)

Complete symbol table. Every symbol below appears in the paper; the "first defined" column gives
the section.

| Symbol | Type / space | Meaning | First defined |
| --- | --- | --- | --- |
| $V$ | model | **Victim** model — black-box reasoning model under attack | §3 |
| $x \in \mathcal{X}$ | input | Problem input / query given to a model | §3 |
| $t \in \mathcal{T}$ | trace | Victim's true internal reasoning trace for $x$ | §3 |
| $y \in \mathcal{Y}$ | answer | Victim's final answer | §3 |
| $C$ | function $\mathcal{T} \to \mathcal{B}$ | Victim's **true, unknown** compression/summarization method | §3 |
| $b^{\star} = C(t) \in \mathcal{B}$ | summary | Victim-exposed reasoning summary ("bubble"), with $\lvert b^{\star}\rvert \ll \lvert t\rvert$ | §3 |
| $S$ | model | Attacker's **student** model, the thing being improved | §3 |
| $V'$ | model | Attacker's **surrogate** reasoning model (produces full traces) | §3 |
| $t', y'$ | trace, answer | Surrogate's trace and answer for a surrogate-split input $x'$ | §3, §4.1 |
| $C'$ | function | Attacker's **compression model**, emulating $C$; used with a fixed prompt, **no fine-tuning** | §3 |
| $\pi$ | prompt template | Fixed prompt template driving $C'$ (Appendix B) | §3 |
| $b' = C'(t';\pi)$ | summary | Surrogate summary produced by the attacker's compressor | §3, §4.1 |
| $I$ | model | **Inversion model**; instantiated as two separate models $I_{\text{sum}}$ and $I_{\text{nosum}}$ | §3, §4 |
| $\hat{t}$ | trace | **Synthesized / inverted** trace produced by $I$ | §3, §4.2 |
| $D$ | dataset | Public reasoning dataset of inputs (e.g. OpenThoughts-114k) | §3 |
| $x'$ | input | Input sampled from the public dataset for the **surrogate** split | §4.1 |
| $\mathcal{D}_1 = \{(x', t', y')\}$ | dataset | Surrogate trace dataset from querying $V'$ | §4.1 |
| $\mathcal{D}_2^{\text{sum}} = \{(x', b', y', t')\}$ | dataset | Inversion-training set, **summary** setting | §4.1 |
| $\mathcal{D}_2^{\text{nosum}} = \{(x', y', t')\}$ | dataset | Inversion-training set, **no-summary** setting | §4.1 |
| $y^{+} = [\,\hat{t};\,y\,]$ | target | Student supervision target: inverted trace concatenated with victim answer | §4.3 |
| $p_{I}, p_{S}$ | distributions | Token-level conditional distributions of the inversion / student model | §4.1, §4.3 |

Note the prime convention throughout: **primed symbols ($x', t', y', b'$) belong to the surrogate
pipeline** (attacker-internal, used to *train* the inverter); **unprimed symbols ($x, t, y, b^\star$)
belong to the victim** (used at *inference* time to produce $\hat t$). Getting this backwards is the
single easiest way to mis-implement Stage 1 vs Stage 2.

---

## 3. Threat model (§3)

### 3.1 Problem setting

A black-box victim $V$, given input $x \in \mathcal{X}$, internally generates $t \in \mathcal{T}$
leading to $y \in \mathcal{Y}$. The user receives only $y$ and, optionally,

$$b^{\star} = C(t) \in \mathcal{B}, \qquad |b^{\star}| \ll |t|$$

This is stated to correspond exactly to the APIs of commercial reasoning models (§3).

### 3.2 Attacker's goal

> "The attacker's primary objective is to improve the reasoning capabilities of their own model $S$
> by exploiting black-box access to a stronger reasoning model $V$, without direct access to $V$'s
> internal reasoning." (§3)

Given tuples $(x,y)$ or $(x, b^{\star}, y)$ output by $V$, the attacker synthesizes traces $\hat t$
that are *logically consistent with the observed tuples*.

**Crucially, faithfulness is explicitly not the objective.** The paper states that whether $\hat t$
matches $V$'s true internal reasoning "does not matter"; the purpose is only to support effective
fine-tuning of $S$ on $(x, \hat t, y)$ (§3, restated in §7: *"Fine-tuning effectiveness matters more
than accurate reconstruction"*). The overlap metrics in §5.2 are called "auxiliary diagnostics."

### 3.3 Attacker capabilities — what they HAVE

| Capability | Detail (§3) |
| --- | --- |
| Black-box query access to $V$ | Can submit arbitrary inputs $x$ and read back $y$, plus $b^\star$ if the API exposes it |
| A surrogate reasoning model $V'$ | Must produce **complete** traces $(t', y')$. Typically **much weaker** than $V$ (paper's example: R1-Distill vs GPT-5.4 mini) |
| A compression model $C'$ | Used zero-shot with fixed prompt $\pi$; **no fine-tuning of $C'$** |
| An inversion model $I$ | Must be **fine-tunable** by the attacker |
| A student model $S$ | Must be **fine-tunable** by the attacker |
| Public reasoning datasets $D$ | Source of inputs $x$ (math, code, logic). Named example: OpenThoughts-114k (guha2025openthoughtsdatarecipesreasoning) |

The models "may be public or private to the attacker. The only requirement is that the attacker be
able to fine-tune the inversion model $I$ and student model $S$" (§3). Note the asymmetry: $V'$ and
$C'$ need only be *runnable*, not trainable.

### 3.4 Attacker capabilities — what they DO NOT have

- **No access to $V$'s internal reasoning $t$.** This is the whole premise.
- **No access to $V$'s true compression method $C$.** Explicitly: "it is not publicly known how GPT
  models produce reasoning summaries" (§3). This is *why* $C'$ exists — the attacker cannot obtain
  genuine (trace, summary) pairs from the victim, so it must manufacture them from surrogate traces.
- **No weights, logits, internal states, gradients, or training data of $V$** (implied by "black-box";
  §2 defines black-box access as observing outputs only).
- **No guarantee that $V' $ is as strong as $V$** — in fact the interesting regime is $V' \ll V$.

### 3.5 The two settings

| Setting | What the victim exposes | Inversion model | Inference call |
| --- | --- | --- | --- |
| **summary** | $(x, b^\star, y)$ | $I_{\text{sum}}$ | $\hat t = I_{\text{sum}}(x, y, b^{\star})$ |
| **no-summary** (stricter) | $(x, y)$ | $I_{\text{nosum}}$ | $\hat t = I_{\text{nosum}}(x, y)$ |

**"Each setting requires a separate inversion model."** (§4) — two independently trained models, not
one model with an optional field.

### 3.6 Query budget

The threat model section itself states no budget. Budget facts appear only in §5:

- Two **disjoint 10k-input splits** sampled from OpenThoughts-114k: a *surrogate split*
  (query $V'$ → $(x',t',y')$) and a *victim split* (query $V$ → $(x,t,y)$ for an open-weight victim,
  or $(x, b^\star, y)$ for a black-box victim) (§5.1).
- Scaling study: 5k / 10k / 15k victim queries, with the paper describing its overall budgets as
  "$\le$ 25K" (§5.4, §7).
- Cost: at GPT-5.4 mini pricing (\$0.75/M input tokens, \$4.50/M output tokens), **10k
  $(x, b^\star, y)$ queries cost \$173.28** (§5.4). Inversion training and student fine-tuning
  involve **zero** victim queries.

---

## 4. The three-stage pipeline (§4)

> "Trace Inversion is a three-stage attack pipeline (Figure 2)."

### Stage 1 — Training the inversion model (§4.1)

**Data generation.**

1. Sample inputs $x'$ from the public reasoning dataset $D$ (surrogate split).
2. Submit $x'$ to the surrogate $V'$; collect its trace and answer, yielding
   $$\mathcal{D}_1 = \{(x', t', y')\}$$
3. *Optionally* (summary setting only) compress each surrogate trace with the attacker's compressor:
   $$b' = C'(t'; \pi)$$
4. Produce the two training sets:
   $$\mathcal{D}_2^{\text{sum}} = \{(x', b', y', t')\}, \qquad
     \mathcal{D}_2^{\text{nosum}} = \{(x', y', t')\}$$

The paper's justification for step 3, verbatim (§4.1): *"We use our own compression model because the
actual summarization method used by the black-box victim model is not known and thus cannot be used
to [produce the] (trace, summary) pairs we need for training."* (sic — the sentence is garbled in the
published text but the meaning is unambiguous.)

**Inputs and targets.**

| Setting | Model input (conditioning) | Target |
| --- | --- | --- |
| summary | $(x', y', b')$ | $t'$ |
| no-summary | $(x', y')$ | $t'$ |

The target is *always* the surrogate's full trace $t'$. The answer $y'$ is always part of the
conditioning, never part of the target — that comes later, at Stage 3.

**Objectives.** Summary setting, $I_{\text{sum}}(x, y, b) = \hat t$:

$$
\mathcal{L}_{\mathrm{inv}}^{\text{sum}}
= \mathbb{E}_{(x',y',b',t')\sim\mathcal{D}_{2}^{\text{sum}}}
\left[-\log p_{I_{\text{sum}}}\!\big(t' \mid x', y', b'\big)\right]
\tag{1}
$$

No-summary setting, $I_{\text{nosum}}(x, y) = \hat t$:

$$
\mathcal{L}_{\mathrm{inv}}^{\text{nosum}}
= \mathbb{E}_{(x',y',t')\sim\mathcal{D}_{2}^{\text{nosum}}}
\left[-\log p_{I_{\text{nosum}}}\!\big(t' \mid x', y'\big)\right]
\tag{2}
$$

> "Both objectives are implemented using teacher forcing with token-level cross-entropy." (§4.1)

i.e. plain autoregressive SFT with the loss masked to the target span. (The masking is not stated
explicitly — see Underspecified §8.)

**What "surrogate" means and why it exists.** The surrogate $V'$ is the attacker's *own* reasoning
model, run on the attacker's *own* inputs, purely to manufacture supervised (conditioning → trace)
pairs. It is the only available source of *paired* data where the full trace is visible, because the
victim never reveals $t$. Two consequences the paper emphasizes:

- $V'$ can be far weaker than $V$ and the attack still works (§1, §5.2): "the drop from R1 to
  R1-Weak is small relative to the jump from zero-shot to any inversion model."
- $V'$'s traces are a *bad* direct distillation source but a *good* inverter-training source — §5.3:
  "Weak surrogates hurt student training but help inversion training." Directly distilling R1-Weak
  degrades the student (Qwen MATH500 71.2% → 63.2%), yet the same traces used only to train $I$,
  which is then applied to the *stronger* victim's outputs, yield much better student data.

The mental model: $I$ learns the generic *mapping* trace ← (input, answer, summary), a mostly
style/structure-level skill, and then applies it to a stronger model's outputs. The reasoning
*content* comes from the victim's $y$ and $b^\star$; the *form* comes from the surrogate.

**Distribution-matching requirement.** Because $I_{\text{sum}}$ trains on $b'$ and is applied to
$b^\star$, it "generalizes best when the two distributions match" (§5.2). The compression prompt $\pi$
was therefore hand-designed so $C'$'s outputs mimic GPT-5.4 mini's summary style. This is a real
design constraint of the method, not just an evaluation detail — see §6 below.

---

### Stage 2 — Inverting the victim's outputs (§4.2)

1. Query the victim $V$ on the victim-split inputs $x$, collecting $(x, y, b^\star)$ (summary
   setting) or $(x, y)$ (no-summary setting).
2. Apply the trained inverter:
   $$\hat t = I_{\text{sum}}(x, y, b^{\star}) \qquad\text{or}\qquad \hat t = I_{\text{nosum}}(x, y)$$

The output $\hat t$ is a synthetic trace "compatible with $V$'s observed outputs" and serves as an
approximation of the hidden internal reasoning for Stage 3.

**Conditioning summary (what is conditioned on what):**

```
Stage 1 (train):  I( x' , y' [, b'      ] )  ->  t'      # surrogate everything
Stage 2 (infer):  I( x  , y  [, b*      ] )  ->  t_hat   # victim everything
Stage 3 (train):  S( x )                     ->  [t_hat ; y]
```

Note that $y$ is conditioning input at Stage 2 but part of the *target* at Stage 3. The inverter is
answer-conditioned; the student is not. This is precisely what produces the "clean forward
reasoning" effect: $I$ already knows the answer, so it never needs to backtrack (§5.3, Appendix D).

No decoding parameters (temperature, top-p, sampling vs greedy, number of samples per input,
max-new-tokens) are given for Stage 2. There is no filtering, verification, or reranking of $\hat t$
in the default pipeline — §7 lists verification (self-consistency, execution, tool-based math
checking) as *future* work, confirming it is absent here.

---

### Stage 3 — Student distillation (§4.3)

Each training example is an input $x$ paired with a single supervision target formed by
concatenating the synthetic trace and the victim's answer:

$$y^{+} = [\,\hat t;\, y\,]$$

The student is fine-tuned with standard teacher-forced cross-entropy:

$$
\mathcal{L}_{\mathrm{student}}
= \mathbb{E}_{(x,\hat t,y)}\left[-\log p_{S}\!\big(y^{+} \mid x\big)\right]
\tag{3}
$$

Note the answer used is the **victim's** $y$, not anything regenerated — the student is taught
(victim answer, plausible route to it).

**Optional: augmenting with surrogate data** (§4.3, ablated in §5.4). $\mathcal{D}_1$ already exists
from Stage 1, so it can be folded into the distillation set "at no additional cost." The student is
then fine-tuned on

$$\{(x, \hat t, y)\} \cup \mathcal{D}_1$$

either as **a single mixed corpus** or as **a curriculum (surrogate traces first, inverted traces
second)**. The paper states the **default pipeline uses only the inverted traces**.

**Optional: domain gating** (introduced in §5.4, not in §4). For code prompts, the attacker can
include only the bare answer as the assistant target, dropping the reasoning prefix. Trades LCB
accuracy for a large HumanEval+ gain (20.7% → 51.2% for Llama). Composes with surrogate
augmentation.

---

### Pipeline at a glance

| | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Queries | $V'$ (attacker-local), $C'$ (attacker-local) | $V$ (paid, black-box) | none |
| Data in | $x'$ from public $D$ | $x$ from public $D$ (disjoint split) | $(x, \hat t, y)$ |
| Trained | $I_{\text{sum}}$ or $I_{\text{nosum}}$ | — (inference only) | $S$ |
| Loss | Eq. (1) / Eq. (2) | — | Eq. (3) |
| Output | trained inverter | $\hat t$ corpus | improved student |

---

## 5. Baselines the method is measured against (§5.1, listed here because they define data formats)

Five student fine-tuning data configurations (§5.1, "Fine-tuning data for student models"):

| Name | Student target | Available to a real attacker? |
| --- | --- | --- |
| Answer-only | victim's $y$ | yes |
| Summary+Answer | victim's $b^\star$ and $y$ | yes (summary setting) |
| Surrogate-Trace | surrogate's $t'$ and $y'$ | yes |
| **Synthesized-Trace (ours)** | $[\hat t; y]$ — inverted trace + victim answer | yes |
| Victim-Trace (oracle) | victim's real $t$ and $y$ | **no** — upper bound only |

A zero-shot baseline also matters methodologically: prompting Qwen directly (Appendix B prompts) to
synthesize a trace from $(x, y)$ or $(x, b^\star, y)$ **without any fine-tuning**. §5.2 reports it
collapses to ~1,000 tokens vs R1's 6,130.6-token average, TF1 35.36/37.58 — i.e. *"prompting alone
is insufficient"*, which is the paper's justification for training $I$ at all.

---

## 6. The compression / summarizer model $C'$ (§3, §4.1, §5.2, Appendix B)

This is the least obvious component, so it gets its own section.

### 6.1 Why it exists

The inverter needs (input, answer, **summary**, trace) quadruples to train on. The victim gives
summaries but not traces; the surrogate gives traces but not summaries. $C'$ bridges the gap by
generating a *synthetic* summary from each surrogate trace, standing in for the victim's unknown
$C$.

### 6.2 How it is used

- $C'$ is run **zero-shot with a fixed prompt template $\pi$, with no fine-tuning** (§3).
- In the experiments, $C'$ is **Qwen2.5-7B-Instruct** — the same model family used as the inversion
  backbone and as one of the two students (§5.1).
- Input: a surrogate trace wrapped in `<think>...</think>`. Output: the summary $b'$.

### 6.3 Design target: match the victim's summary distribution

$\pi$ was reverse-engineered to imitate GPT-5.4 mini's summaries (§5.2). Observed victim summary
properties, obtained by regex heuristics plus the API's billed reasoning-token count (trace length
recovered by subtracting visible-output tokens):

- Median summary length $\approx$ 592 tokens; median compression $\approx 4.2\times$.
- First-person prose, bold-header sections, frequent LaTeX.

Table 1 (§5.2) — feature distributions, attacker's $C'$ on R1 traces vs GPT-5.4 mini:

| Feature | Ours ($C'$ on R1 traces) | GPT-5.4 mini |
| --- | --- | --- |
| Median tokens | 537 | 592 |
| Bold-header sections | 94.1% | 92.9% |
| First-person prose | 97.3% | 97.0% |
| LaTeX | 79.1% | 71.9% |

### 6.4 The compression prompt $\pi$ (Appendix B, verbatim)

**System prompt:**

> You are summarizing a long chain-of-thought trace into a short, first-person "inner-monologue
> recap" that mirrors the style of GPT-5 mini's internal reasoning summaries.
>
> Given a `<think>...</think>` trace, produce a recap with these properties:
>
> **Structure.** Write 3 to 6 short sections. Each section begins with a short bold markdown header
> on its own line, like `**Setting up the integral**` or `**Checking the edge case**`, and is
> followed by one short paragraph (2–5 sentences). Do NOT use numbered lists (1., 2.) and do NOT use
> bullet points (-, *). The whole recap is just headers + prose.
>
> **Voice.** First person, present tense, as if the model is thinking aloud: "I need to…", "I'll
> check…", "I'm now realizing…", "Let me verify…". Keep the tone tentative and exploratory, not
> textbook.
>
> **Content.** Each section should capture one meaningful move in the reasoning — a clarification of
> what is being asked, a key derivation or substitution, a pivot after a failed attempt, a sanity
> check, or the final consolidation. Skip filler, restatement, and purely mechanical arithmetic.
> Preserve the logical order of the trace.
>
> **Length.** Aim for roughly 600–900 tokens total — long enough that each section develops a real
> idea, not just a one-line gesture. If the input trace is very short, produce fewer sections rather
> than padding; if it is very long, prefer adding depth to each section over adding more sections.
>
> **Math formatting.** Use inline LaTeX (`\frac`, `\sqrt`, `\boxed`, etc.) where the original trace
> used math. Do not restate the final boxed answer unless the reasoning naturally concludes with it.
>
> Do not add meta-commentary about following instructions, apologize, or mention that you are
> summarizing. Just produce the recap.

**User prompt:**

```
Summarize this thinking process as a first-person inner-monologue recap:
<think>{thinking_content}</think>
```

**Critical omission:** the paper states $\pi$ "additionally includes two few-shot exemplars of the
target style (one algebra / number theory and one geometry, ~600 tokens each), drawn from GPT-5-mini's
own summaries. They are omitted from this box for space; the full prompt including exemplars is
released with our code." (Appendix B). The exemplars are therefore **not in the paper** — a
reimplementer must pull them from the repo or write their own. Note also that they are real
GPT-5-mini summaries, meaning the attacker is assumed to have seen at least a couple of genuine
victim summaries to seed the style.

---

## 7. Prompt templates and formatting (Appendix B, C, D)

### 7.1 Formatting conventions observable in the paper

| Element | Convention |
| --- | --- |
| Trace delimiter | `<think>` … `</think>` (compression input; both zero-shot inversion prompts explicitly require output wrapped in these tags) |
| Student target | $y^{+} = [\hat t; y]$ — trace concatenated with answer (§4.3). Delimiter between them not specified |
| Summary format | Markdown bold headers + prose paragraphs, first person, inline LaTeX |
| Answers | Math answers use `\boxed{...}` (implied by the compression prompt's reference to "the final boxed answer") |
| Max sequence length | `cutoff_len = 16384` for all fine-tuning (§5.1) — note this is *shorter* than some R1 traces (avg 6,130.6 tokens, but inverted traces run 4,972–6,021 tokens and the input adds more) |

### 7.2 Zero-shot inversion prompt — WITH summaries (Appendix B, verbatim)

> You are a language model that reconstructs full internal reasoning traces from high-level bubble
> summaries.
>
> You will be given:
> - A problem **input** (e.g., a math or logic problem)
> - A final **output** or solution
> - A list of numbered **reasoning bubbles**, where each bubble summarizes one key insight, step, or
>   decision made during the problem-solving process
>
> These bubbles are **condensed summaries** of what was originally a much longer, richer internal
> thought process. Your task is to reconstruct that full process.
>
> Below are high-level bubble summaries representing condensed thoughts or decisions. Your task is to
> reconstruct the full thinking trace that might have led to each summary. For each bubble, expand it
> into a **detailed internal monologue or reasoning chain**, showing how one idea leads to the next.
>
> Include:
> - Assumptions and background intuitions
> - Intermediate steps, definitions, and subcases
> - Natural questions or doubts raised during reasoning
> - Alternatives that were considered and rejected
> - Transitions that make the reasoning coherent and plausible
>
> Use informal, introspective language — as if the person is thinking out loud. Add math expressions
> in LaTeX where appropriate.
>
> Do **not** invent new reasoning steps outside the bubbles. Use the **input** and **output** only for
> context and consistency. Your goal is to **flesh out the bubbles**, not to re-solve the problem from
> scratch.
>
> The full trace should:
> - Be logically consistent and cohesive from start to finish
> - Sound like a realistic thought process that could plausibly result in the given answer
> - Span multiple paragraphs per bubble and up to 20,000 characters overall if needed
> - Be output as one continuous trace, wrapped in `<think>…</think>` tags
>
> You are not summarizing the bubbles. You are recovering the internal narrative that *generated* them.
>
> ```
> The original problem input is: {user_prompt}
> The final answer is: {assistant_answer}
> Transform this thinking bubbles into clear full reasoning traces: {reasoning_summary}
> Generate full reasoning traces:
> ```

Note the mismatch worth flagging: this prompt assumes "a list of **numbered** reasoning bubbles",
while the compression prompt $\pi$ explicitly forbids numbered lists and produces bold-header prose.
The two Appendix-B prompt families were evidently written against slightly different summary formats.

### 7.3 Zero-shot inversion prompt — NO summaries (Appendix B, verbatim)

> You are a language model that reconstructs full internal reasoning traces from only an **input**
> (e.g., a math or logic problem) and a corresponding **output** (final solution or answer).
>
> You will be given:
> - A problem **input**
> - A final **output** or solution
>
> Your task is to reconstruct the full internal reasoning process that could plausibly connect the
> input to the output. This should be a long, detailed, introspective trace, not a short summary.
>
> Guidelines for the reasoning trace:
> - Write in the style of an informal, introspective monologue, as if the person is thinking out loud.
> - Include assumptions, intuitions, and background facts as they arise naturally.
> - Show intermediate steps, calculations, logical deductions, definitions, and subcases.
> - Raise natural questions or doubts during reasoning, and explain how they are resolved.
> - Explore alternative approaches, even ones that are discarded, and explain why.
> - Make transitions clear so the reasoning feels like a coherent train of thought.
> - Use LaTeX for math expressions where helpful.
> - Do not introduce new information inconsistent with the input or output.
> - The goal is depth, not brevity: expand ideas fully, elaborate with multiple paragraphs, and let
>   the reasoning unfold gradually.
> - The output should only appear at the end, after the reasoning is complete.
>
> The full trace should:
> - Wrapped in `<think>…</think>` tags.
> - Be logically consistent and cohesive from start to finish.
> - Sound like a realistic thought process that could plausibly result in the given answer.
> - Span thousands of tokens if needed.
>
> ```
> The original problem input is: {user_prompt}
> The final answer is: {assistant_answer}
> Generate full reasoning traces:
> ```

### 7.4 Status of these two prompts

**These are labeled "Zero-shot Inversion Prompt" — they define the zero-shot *baseline*, not the
fine-tuned inverter's input format.** The paper never states whether the fine-tuned $I_{\text{sum}}$ /
$I_{\text{nosum}}$ use these same templates as their conditioning prefix, a trimmed version, or a bare
structured concatenation. This is a significant gap (see §8).

### 7.5 Appendix C / D — qualitative examples

- **Figure 4** (App. C): zero-shot inversion vs trained trace inversion against R1, R1-Distill
  surrogate. Image only, no transcribed text in the HTML.
- **Figure 5** (App. C): R1 ground-truth trace vs trace synthesized with R1-Distill surrogate. Image
  only.
- **Figure 6 / Appendix D**: the one fully transcribed example. Query: *"The positive reals $a$ and
  $b$ satisfy [equation]. Show that [inequality]."* The R1 ground-truth excerpt visibly thrashes
  ("Wait, let's step back…", "But not sure if helpful here…", "Maybe not the best approach…"), while
  the synthesized trace performs a clean substitution, expands, factors, computes a discriminant, and
  concludes. This is the evidence for the "inversion denoises the teacher" claim. Note the actual
  math is lost in both the HTML and text extractions — the inline formulas render as empty; consult
  the PDF for the literal problem statement.

---

## 8. Related-work anchors and novelty claims (§2)

### 8.1 What the paper builds on

| Thread | Key citations (§2) | What is inherited |
| --- | --- | --- |
| Reasoning models / CoT | wei2022chain; kojima2022large; wang2022self; ouyang2022training; bai2022training; guo2025deepseek; openai_o1_system_card; qwen2024qwq; muennighoff2025s1 | The object of attack: models that reason in explicit multi-step traces |
| **Language model inversion** | **morris2024language** (Morris is a co-author) | The core technical idea: infer information about *inputs* from black-box *outputs*. Trace Inversion is a direct descendant, retargeted from input recovery to trace recovery |
| Prompt inversion / extraction | zhang2024extracting (Zhang, Morris, Shmatikov — also co-authors); zhang2024effective; yang2024prsa | Prior "invert the hidden text behind the outputs" methodology, and the train-an-inverter-on-surrogate-data recipe |
| Other leakage | carlini2021extracting; shokri2017membership; mattern2023membership; wen2024membership | Framing: black-box outputs leak |
| Knowledge distillation | hinton2015distilling; gou2021knowledge | Cooperative distillation baseline |
| Adversarial capability stealing | tramer2016stealing; orekondy2019knockoff (image classifiers); krishna2019thieves; wallace2020imitation (language) | The adversarial framing and black-box access definition |
| CoT distillation | shridhar2023distilling; chen2025skip | Establishes that trace supervision >> answer supervision, which is the gap the attack must fill |

The lineage is explicit: this is **language/prompt inversion (Morris et al., Zhang et al.) fused with
capability stealing (Tramèr, Krishna, Wallace) and CoT distillation (Shridhar, Chen)**.

### 8.2 What the paper claims is new

1. **The target of inversion.** Prior inversion work recovers *inputs* (prompts, system prompts).
   This inverts to recover the *hidden intermediate reasoning*.
2. **The realistic setting.** §2 closes: *"We focus on the realistic setting where the teacher's
   reasoning traces are not available from the model API."* Prior CoT distillation assumes traces are
   available.
3. **Utility over faithfulness as the success criterion.** The synthesized trace need not be the true
   trace; it only needs to be good fine-tuning data (§3, §7).
4. **Defeating the summary-only mitigation.** The paper demonstrates that hiding CoT and exposing only
   summaries does not prevent capability stealing (§1, §6, §7).
5. **Weak-surrogate feasibility.** The inverter can be trained from a much weaker model than the
   victim and still produce traces that teach a student to reason better than the surrogate's own
   traces would (§1, §5.2, §5.3).
6. **Immunity to reasoning-perturbation defenses.** Because the attack never touches the victim's
   reasoning, defenses that obfuscate traces (antidistillation sampling savani2025antidistillation,
   DOGe li2025doge) are bypassed by construction (§6, §7). Watermarking (kirchenbauer2023watermark;
   gu2023learnability; sander2024watermarking) may support attribution but survival through inversion
   + fine-tuning is stated as an open question.

---

## 9. Underspecified / Reimplementer Must Decide

Ordered roughly by how much a wrong guess would change results. Everything here is *not* stated in
§1–§4 or Appendix B; some may be recoverable from the released repo.

### High impact

1. **The fine-tuned inverter's actual input format.** Appendix B gives *zero-shot* prompts only
   (§7.4 above). For the trained $I_{\text{sum}}$ / $I_{\text{nosum}}$, unknown: whether the long
   instruction preamble is retained, what separators/field labels delimit $x$ / $y$ / $b$, whether the
   chat template is applied, whether the target $t'$ is wrapped in `<think>` tags, and whether the
   loss is masked to the target span only (standard, but never stated). This single choice determines
   whether a reimplementation matches the paper at all.

2. **The two few-shot exemplars inside the compression prompt $\pi$.** Explicitly omitted from the
   paper ("omitted from this box for space"). They are real GPT-5-mini summaries, ~600 tokens each,
   one algebra/number-theory and one geometry. Since $\pi$'s whole job is distribution-matching to the
   victim, and Table 1 is the evidence that it works, these exemplars are load-bearing. Must be
   fetched from the repo or reconstructed.

3. **Stage 2 decoding configuration.** No temperature, top-p, top-k, greedy-vs-sampling, seed,
   `max_new_tokens`, repetition penalty, or number of samples per input. Trace *length* is a headline
   metric (4,972 / 5,434 tokens) and is extremely sensitive to `max_new_tokens` and length penalties.
   Same gap for the $C'$ compression calls and for the surrogate $V'$ generation in Stage 1.

4. **How $y^{+} = [\hat t; y]$ is actually concatenated.** No separator, tag, or template is given.
   Candidates: `<think>t̂</think>\n\ny`, raw concatenation, a chat-template assistant turn. §5.3
   attributes weak LiveCodeBench results partly to "formatting differences between the reasoning
   supervision and the output format LCB expects", which confirms the format matters and is not
   pinned down.

5. **The `cutoff_len = 16384` truncation policy.** Inverted traces run 5–6k tokens and inputs add
   more, so some examples must exceed the cutoff. Unknown whether over-length examples are truncated
   (from which end?) or dropped, and whether the same cutoff applies to inverter training, student
   training, and generation.

### Medium impact

6. **Whether $C'$ is applied to *every* surrogate trace or a filtered subset.** §4.1 says traces are
   "optionally compressed", with the option referring to the summary/no-summary setting, but nothing
   says what happens when $C'$ returns a degenerate or off-format summary. No filtering step is
   described anywhere.

7. **Handling of empty victim summaries.** §5.2 says "**Non-empty** GPT-5.4 mini summaries have median
   length 592 tokens" — implying some are empty. What the pipeline does with those examples (drop,
   fall back to the no-summary inverter, pass an empty string) is never stated.

8. **How $y$ is separated from the trace in the surrogate's raw output.** $V'$ (an R1-family model)
   emits trace and answer in one stream. The split into $t'$ and $y'$ presumably keys on `</think>`
   but this is never stated, nor is what happens when the tag is missing or duplicated.

9. **Answer/trace correctness filtering.** Nothing indicates that surrogate traces are filtered for
   correct answers, nor that victim answers are verified. §6 notes trace inversion works "as long as
   [answers and summaries] are correct" — but no correctness check exists in the pipeline. §7 lists
   verification as future work. So: presumably no filtering, but a reimplementer may reasonably
   suspect the repo does something.

10. **Full vs parameter-efficient fine-tuning.** §5.1 gives only `num_train_epochs=3`,
    `learning_rate=1e-5`, `warmup_ratio=0.1`, `cutoff_len=16384`, "each model's default optimizer",
    8×A100-80GB. Not given: full-finetune vs LoRA (1e-5 suggests full FT), batch size, gradient
    accumulation, LR schedule shape, weight decay, precision (bf16?), DeepSpeed/FSDP config, gradient
    checkpointing. A 7B full finetune at 16k context on 8×A100 is tight, so the sharding strategy is
    not a free choice.

11. **How the surrogate and victim splits are sampled from OpenThoughts-114k.** "Two disjoint splits
    of 10k inputs each" — no seed, no stratification across the math/science/code/logic domains, no
    statement about whether the same victim split is reused across all victim/surrogate combinations.

12. **The mixed-corpus vs curriculum choice for surrogate augmentation** (§4.3). Both are offered;
    which was used for the Table 5 numbers is not stated, nor the mixing ratio, nor the curriculum
    stage lengths / LR handling across stages.

### Lower impact but still ambiguous

13. **Tokenizer used for all reported token counts** (Len metric, "537 median tokens", the 4.2×
    compression ratio). Comparing GPT-5.4 mini's billed reasoning tokens (OpenAI tokenizer) against
    Qwen-tokenized surrogate summaries is apples-to-oranges unless normalized; the paper doesn't say.

14. **The TF1 / BLEU / ROUGE implementations.** No library, tokenization, casing, stemming, or
    smoothing details for BLEU. "Token-overlap F1" is a squad-style metric with several common
    variants.

15. **The regex heuristics for style features in Table 1** ("bold-header sections", "first-person
    prose", "LaTeX") are described as "lightweight regex heuristics" but never given.

16. **Domain gating's domain classifier** (§5.4): "include only the bare answer on code prompts" —
    how a prompt is identified as a code prompt is not specified (OpenThoughts metadata field?
    heuristic?).

17. **Whether $I_{\text{sum}}$ and $I_{\text{nosum}}$ are trained from the same base checkpoint with
    identical hyperparameters.** Stated only that both use Qwen2.5-7B-Instruct as backbone and that
    "each setting requires a separate inversion model."

18. **The exact numbered-bubble vs bold-header summary format discrepancy** between $\pi$ and the
    zero-shot summary inversion prompt (§7.2 above). A reimplementer must decide which format the
    trained inverter sees.

---

## 10. Quick reproduction checklist (derived, not stated as such in the paper)

1. Pull OpenThoughts-114k; sample two disjoint 10k input splits (surrogate, victim).
2. Run $V'$ (R1 or DeepSeek-R1-Distill-Qwen-1.5B) on the surrogate split → $\mathcal{D}_1 = \{(x',t',y')\}$.
3. Run $C'$ = Qwen2.5-7B-Instruct zero-shot with prompt $\pi$ (+2 few-shot exemplars from the repo)
   over each $t'$ → $b'$. Sanity-check against Table 1 style stats before proceeding.
4. Build $\mathcal{D}_2^{\text{sum}}$ and/or $\mathcal{D}_2^{\text{nosum}}$.
5. Fine-tune Qwen2.5-7B-Instruct as $I_{\text{sum}}$ (Eq. 1) and/or $I_{\text{nosum}}$ (Eq. 2);
   3 epochs, lr 1e-5, warmup 0.1, cutoff 16384.
6. Query the victim on the victim split → $(x, b^\star, y)$ or $(x,y)$. (~\$173 for 10k GPT-5.4 mini
   queries.)
7. Generate $\hat t$ with the trained inverter.
8. Fine-tune the student on $x \rightarrow [\hat t; y]$ (Eq. 3), same hyperparameters.
9. Evaluate on MATH500 / JEEBench / LiveCodeBench against the five baselines in §5.
