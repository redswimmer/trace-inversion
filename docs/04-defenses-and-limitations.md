# Defenses, Limitations, and Discussion

Source: "How to Steal Reasoning Without Reasoning Traces" — Tingwei Zhang, John X. Morris,
Vitaly Shmatikov (Cornell Tech). arXiv:2603.07267v2 [cs.CR], 12 May 2026.

Scope of this document: §6 (Defenses), §7 (Discussion and Future Research), the paper's stated
limitations (scattered across §5.3, §5.4, §7), and the Impact Statement.

> **Source caveat.** These notes were taken from the arXiv HTML rendering. Tables 3, 4, and 5
> lost most of their numeric cells in the HTML-to-text conversion — only the `Victim-Trace
> (oracle)` row of Table 3, a few isolated cells of Tables 4/5, and all of Tables 1, 2, and 6
> survived. Every number quoted below therefore comes from the paper's *prose*, which restates
> the important comparisons. Anyone reproducing this should pull the PDF for the full tables.

---

## 1. Defenses (§6)

The paper's defense section is short and, critically, **contains no new experiments of its own**.
It is a survey of existing defenses plus an argument for why each fails against Trace Inversion.
The distinction matters for a reproduction: the paper *evaluates* zero defenses in the sense of
implementing and measuring one. What it does have is §5.3/§5.4 evidence about the *currently
deployed* defense (hide the CoT, show a summary), which it reinterprets in §6 as a defense
evaluation.

### 1.1 Summary table

| Defense | Mechanism | Tested by authors? | Result / assessment | Paper's verdict |
|---|---|---|---|---|
| **Hiding the CoT; exposing only the final answer** | Provider never returns the trace (§3, §6) | **Yes**, as the `Answer-only` baseline | Answer-only fine-tuning is a weak signal and can *harm* the student: Qwen MATH500 71.2% → 61.0% when fine-tuned on R1 answers (§5.3). Against GPT-5.4 mini, Answer-only gives Qwen 68.4% MATH500 / 27.6% JEEBench and Llama 13.0% / 6.3% (§5.4) | Effective against *naive* distillation, **defeated by inversion**: same victim outputs become 76.0% / 43.7% (Qwen) and 52.4% / 19.9% (Llama) once inverted (§5.4) |
| **Exposing a short reasoning summary instead of the trace** | Provider compresses the CoT to a user-facing recap (OpenAI/Gemini/Anthropic practice, §1, §6) | **Yes**, as the `Summary+Answer` baseline | Summaries are *not* helpful for direct distillation and often make things worse than answers alone: Qwen JEEBench against GPT-5.4 mini drops 27.6% → **1.6%** with Summary+Answer (§5.4). Against R1, Summary+Answer gives Qwen 63.0% / Llama 45.8% on MATH500 (§5.3) | Accidentally a decent defense against naive distillation; **completely defeated** — summaries are the *input* that makes inversion stronger (Qwen MATH500 63.0% → 71.8%, JEEBench 24.0% → 36.3%; Llama 45.8% → 50.2%, 18.2% → 24.2%) (§5.3). The paper explicitly notes it is unknown whether summaries were *designed* as a distillation defense (§6) |
| **No summary at all (strictest exposure setting)** | Only the answer is returned | **Yes**, as the `No-summary` inversion setting | Degrades inversion but does not stop it: fidelity TF1 52.76 → 49.01 and R-L 24.84 → 21.73 with the R1-Weak surrogate (Table 2); downstream Qwen JEEBench still rises 19.7% → 29.7% over Surrogate-Trace (§5.3). With a strong R1 surrogate, no-summary inversion reaches Qwen 74.1% MATH500 / 33.2% JEEBench — near oracle (§5.3) | **Insufficient.** Withholding the summary costs the attacker a few points, not the attack |
| **Access control** | Gate who can query the model | No — cited only (§6, tramer2016stealing) | Not analyzed | Listed as prior art; no assessment given |
| **Output restrictions: query limits** | Cap queries per account (§6) | No — but §5.4's budget curve is indirect evidence | Accuracy scales with budget: 5k → 10k → 15k queries gives Qwen MATH500 67.0% → 77.6% → 80.8% and JEEBench 38.8% → 43.7% → 44.5% (§5.4). Attack budgets are ≤25k, vs. OpenThoughts at 114k and OpenThoughts3 at 1.2M | Implicitly the *only* defense with any traction — but the attack already works at 5k queries, and 10k queries cost **$173.28** at GPT-5.4 mini pricing (§5.4), so a limit would have to be aggressive enough to hurt legitimate users |
| **Output restrictions: truncation** | Truncate returned output (§6, tramer2016stealing) | No | Not analyzed | Mentioned, not assessed |
| **Undistillable / "nasty" teachers** | Train the teacher so its outputs teach poorly (ma2021undistillable) | No | Not analyzed | Dismissed as *category*: "Most of these methods apply to the internals (e.g., logits or intermediate representations) of image classifiers" (§6) |
| **Modified / sparse logits** | Perturb or sparsify the output distribution (ma2022stingy) | No | Not analyzed | Same dismissal — internals-based, classifier-oriented. Also implicitly N/A: the attack never uses logits |
| **Adaptive noise injection / query unlearning (QUEEN)** | Detect extraction queries and degrade responses (chen2025queen) | No | Not analyzed | Same dismissal |
| **Antidistillation sampling** | Sample traces that are correct but intentionally hard to imitate (savani2025antidistillation) | No | Not analyzed | **Explicitly argued irrelevant** (§6, §7): it defends the *exposed reasoning*, but Trace Inversion never reads the victim's reasoning. "Trace inversion is completely agnostic to how the answers and summaries were obtained, as long as they are correct, and thus immune to defenses that perturb internal reasoning" (§6) |
| **DOGe (Defensive Output Generation)** | Same family — outputs engineered to resist CoT supervision (li2025doge) | No | Not analyzed | Same argument (§6) |
| **Watermarking generated text** | Embed a detectable signal in outputs for attribution (kirchenbauer2023watermark; gu2023learnability; sander2024watermarking) | No | Not analyzed | The one defense the paper does *not* dismiss. Assessment: "watermarking does not prevent capability stealing, it may support attribution; **whether watermarks survive inversion and subsequent fine-tuning is an open question**" (§6). Note the inverted trace is generated by the *attacker's* model, so only the answer token stream carries the victim's watermark |

### 1.2 The paper's central defense argument (§6, §7)

Two claims do the work:

1. **Inversion is answer-conditioned, not reasoning-conditioned.** Every defense in the
   perturb-the-reasoning family (antidistillation sampling, DOGe, obfuscated CoT) assumes the
   attacker consumes the victim's reasoning. Trace Inversion consumes only `(x, a)` or
   `(x, s, a)`. So long as the answer is correct, the defense is bypassed by construction (§6).
2. **Therefore obfuscation is strictly negative-sum.** §7: "making exposed reasoning hard to
   imitate is not sufficient, because an attacker can always ignore it and apply trace inversion
   to outputs alone. Obfuscated reasoning thus reduces transparency for users yet does not
   prevent capability stealing."

The implied residual defense surface, by elimination: degrade *answer correctness* (unacceptable),
restrict *query volume* (partially effective, costs legitimate users), *access control*
(unanalyzed), or *detect and attribute* after the fact (watermarking, unproven under inversion).

### 1.3 Cost and utility of the defenses

The paper gives **no** latency, utility-loss, or serving-cost numbers for any defense. It never
implements one. The only cost figures in the paper are on the **attack** side: 10k `(x, s, a)`
queries against GPT-5.4 mini cost **$173.28**; inversion training and student fine-tuning require
zero further victim queries; all training/eval ran on 8× A100 80GB with 3 epochs, lr 1e-5,
warmup 0.1, cutoff 16,384 tokens (§5.1, §5.4).

---

## 2. The paper's stated limitations

The paper has no dedicated "Limitations" section. These are collected from where the authors
concede them in §5.3, §5.4, §7, and Appendix A.

1. **Fidelity against closed models is unverifiable.** §7: "For closed-source victims, we cannot
   tell whether our inverted traces accurately match their 'true' reasoning traces." The authors
   reframe this as unimportant ("Fine-tuning effectiveness matters more than accurate
   reconstruction") — but it means every fidelity number in Table 2 is measured against R1 only,
   never against the commercial victim. For GPT-5.4 mini they substitute *summary-distribution*
   matching (Table 1: median 537 vs. 592 tokens; bold-header sections 94.1% vs. 92.9%;
   first-person prose 97.3% vs. 97.0%; LaTeX 79.1% vs. 71.9%) using "lightweight regex
   heuristics" (§5.2).

2. **Modest query budget.** §5.4/§7: budgets are at most 25k, "far below production-scale
   distillation corpora (OpenThoughts: 114k; OpenThoughts3: 1.2M)." Scaling up is left to future
   work, so the attack's ceiling is unknown.

3. **Coding transfer is weak or negative.** §5.3: Llama's LiveCodeBench accuracy *drops* under
   Synthesized-Trace (13.1% → 8.8%) and even under the Victim-Trace oracle (10.4%); Qwen is
   roughly flat (28.9% → 30.9%). §5.4: LCB "declines slightly with scale." Attributed to a
   style/format mismatch between math-flavored inverted traces and coding reasoning. The
   domain-gating fix trades LCB away to recover HumanEval+ (20.7% → 51.2%) (§5.4, Table 5).

4. **Inversion's advantage shrinks when the surrogate is already strong.** §5.4: with an R1
   surrogate against GPT-5.4 mini, Synthesized-Trace can fall *below* Surrogate-Trace (Llama
   JEEBench 33.2% → 28.9%; Qwen JEEBench tied at 43.7%). Appendix A Table 6 shows why — R1 (91.6
   / 87.1 / 74.9 on MATH500 / JEEBench / LCB) is comparable to or stronger than GPT-5.4 mini
   (91.4 / 85.1 / 66.9). The authors concede the method "is most advantageous in the (common)
   setting when the victim is much stronger than the attacker's best surrogate."

5. **The victim used as "black-box commercial model" is not actually stronger than the
   surrogate** on the benchmarks measured (Appendix A, Table 6) — a direct consequence of (4),
   and acknowledged as such in §5.4.

6. **The base student is benchmark-contaminated.** §5.3: "even though it does not have explicit
   reasoning, our base instruction-tuned Qwen model appears to have been optimized for the
   MATH500 benchmark. This means that fine-tuning can degrade its performance (71.2% → 61.0%)."
   The authors turn this into a *pro*-attack argument, but it compromises MATH500 as a clean
   measure for the Qwen student.

7. **The compression model is a guess.** §3, §4.1: "We do not assume that the attacker has access
   to the true compression method `C` used by the victim … it is not publicly known how GPT
   models produce reasoning summaries." The attacker's `C` is a hand-written prompt tuned by
   regex-matched style features (§5.2, Appendix B), so summary-distribution mismatch is an
   uncontrolled variable.

8. **The overlap metrics are explicitly downgraded.** §5.1: Len / BLEU / TF1 / ROUGE "are
   auxiliary diagnostics. Our main evaluation metric is the reasoning performance of student
   models." §5.2 further notes overlap metrics "understate the qualitative gap" between the
   summary and no-summary settings.

9. **Narrow experimental scope.** One query-source dataset (OpenThoughts-114k), two students
   (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct), one inversion backbone (Qwen2.5-7B-Instruct),
   two surrogates (R1, R1-Distill-Qwen-1.5B), two victims (R1, GPT-5.4 mini), three benchmarks
   (plus HumanEval+ in the Table 5 ablation only). No seed counts, variance, or confidence
   intervals are reported anywhere.

---

## 3. Future research directions (§6, §7)

Called out by the authors:

- **Scale the attack** — larger query budgets (beyond 25k, toward 114k/1.2M), larger inversion
  models, and a broader mix of reasoning tasks (§5.4, §7).
- **Robustness of the inversion model** — train it on a variety of summary lengths and styles, so
  it does not depend on the attacker's guess at the victim's compression scheme (§7).
- **Verification-filtered trace synthesis** — use self-consistency for reasoning, execution for
  code, or tool-based checking for math to filter or refine synthesized traces before student
  fine-tuning (§7). This is the authors' own suggested fix for the weak LCB results.
- **Watermark survival** — whether text watermarks survive inversion and subsequent fine-tuning
  is stated as an open question (§6).
- **Better defenses generally** — the Impact Statement frames the whole paper as motivation for
  this.

Not stated as future work but left conspicuously open: no defense is proposed by the authors at
all. §7 rules out the obfuscation family and offers nothing in its place.

---

## 4. Ethics, disclosure, and why this is published

**Impact Statement (verbatim, in full):**

> "This paper investigates whether hiding chains of thought (exposing only final answers or short
> reasoning summaries) prevents theft of reasoning capabilities. The sole purpose of this work is
> to evaluate the limits of existing protections and to motivate development of more robust
> defenses."

That is the entirety of the ethics discussion. Observations:

- **No responsible-disclosure statement.** The paper does not say whether OpenAI (the GPT-5.4
  mini provider) was notified before publication, nor whether any embargo was observed.
- **Code is released, not withheld.** §1: "To facilitate research on model-stealing attacks and
  defenses, we release our code" at `https://github.com/Tingwei-Zhang/Trace_Inversion_Attack`.
  Appendix B notes the full compression prompt including its two few-shot exemplars ships with
  the code.
- **No discussion of terms-of-service implications.** The attack requires querying a commercial
  API and using the outputs to train a competing model — plausibly a ToS violation — and the
  paper does not address it, despite §1 citing Anthropic's February 2026 accusations against
  competing labs for exactly this behavior (`anthropic_distillation_2026`).
- **No harm-mitigation measures** are described (no rate-limit courtesy, no withheld artifacts,
  no delayed release).
- Funding is disclosed in the Acknowledgments: Amazon Research Award, Google Academic Research
  Award, Google Cyber NYC, an Infosys gift, and NSF awards 2311521 and 2428949.

---

## 5. Security implications the authors draw

For a provider that hides its chain of thought:

1. **Hiding the CoT is not a capability-stealing defense.** §7 opens with the thesis: "hiding
   chains of thought may not protect reasoning capability from theft." The abstract makes the
   same claim. The trace is a convenience for the attacker, not a requirement.
2. **Reasoning summaries are worse than nothing under this threat model.** They are useless for
   direct distillation (Qwen JEEBench 27.6% → 1.6%, §5.4) but *improve* inversion quality
   (Table 2: TF1 52.76 vs. 49.01; and every downstream summary-setting number beats its
   no-summary counterpart). A provider showing summaries is handing the attacker a free
   conditioning signal in exchange for no protection.
3. **Correct answers are themselves the leak.** §6: inversion "is completely agnostic to how the
   answers and summaries were obtained, as long as they are correct." Any defense that preserves
   answer quality preserves the attack surface. This bounds the achievable defense: you cannot
   protect the capability without degrading the product.
4. **Obfuscation is a transparency tax with no security return.** §7: obfuscated reasoning
   "reduces transparency for users yet does not prevent capability stealing." A direct argument
   against the antidistillation-sampling / DOGe research direction as *deployed* policy.
5. **The attack is cheap and needs no privileged access.** $173.28 for 10k queries, an
   open-weight 1.5B surrogate, and a 7B inversion backbone (§5.1, §5.4). No logits, no
   token probabilities, no internal state.
6. **A weak attacker can steal from a strong victim.** The R1-Weak surrogate (81.4 / 32.6 / 27.6
   on the three benchmarks, Table 6) trains an inversion model that produces traces good enough
   to move a student most of the way to the oracle (Llama JEEBench 11.3% surrogate → 24.2%
   inverted vs. 23.1% oracle, §5.3). The attacker does not need a peer model to bootstrap.
7. **Defenders' remaining levers are volume and attribution.** By elimination: query limiting
   (partially supported by the budget curve) and watermark-based attribution (untested under
   inversion).

---
---

# MY OWN ANALYSIS

*Everything below is my assessment, not the paper's. The paper makes none of these claims.*

## A. Local reproducibility on a single consumer GPU

First, a framing point the paper obscures: **§6 contains nothing to reproduce.** No defense is
implemented. "Reproducing the defenses" means either (a) re-running the two baselines the paper
treats as de facto defenses, or (b) implementing one of the surveyed defenses yourself and
testing it against inversion — which would be *new work*, not reproduction.

### Cheap on one 24GB consumer GPU (RTX 3090/4090 class)

- **Answer-only and Summary+Answer baselines.** These are the paper's only tested defenses. They
  are just SFT runs on a different target string; the data collection is identical. Once you have
  the victim query set, these cost the same as any other fine-tune.
- **The no-summary setting.** Literally deleting a field from the prompt. Free.
- **Truncation and query limits.** Simulated by subsampling the query set or clipping outputs. No
  GPU cost at all. The budget-scaling curve (§5.4, Figure 3) is the only defense-relevant curve
  in the paper and is reproducible by just fine-tuning on 5k/10k/15k subsets.
- **Watermarking (KGW / kirchenbauer2023watermark).** Cheap to *add* — it is a logit bias at decode
  time on a local surrogate victim. Testing survival through inversion + fine-tuning needs the
  full pipeline, but the watermarking component itself is trivial. This is the most interesting
  cheap experiment available, since it is the one thing the paper flags as genuinely open.
- **Antidistillation sampling / DOGe against a *local* victim.** Both need logit access, which you
  have for an open-weight victim. Feasible with a small victim (7B or below). But note this would
  only confirm what the paper argues analytically — inversion ignores the trace, so a
  trace-perturbing defense has no attack surface. Low information yield.
- **Fidelity metrics (Table 2).** BLEU / TF1 / ROUGE on CPU. Free. Only the *generation* of traces
  costs GPU.
- **Table 1's style-matching regexes.** Trivial. Worth doing precisely because it is trivial (see
  §B).

### Requires scaling down, but doable with compromises

- **The inversion model.** A 7B backbone at 16,384-token cutoff, 3 epochs, full fine-tune is
  8×A100 territory. On one 24GB card you need QLoRA at 4-bit, gradient checkpointing, a much
  shorter cutoff (4k–8k), and probably a 1.5B–3B backbone. That changes the result — the paper's
  own evidence is that inversion quality tracks capacity, and truncating at 4k when target traces
  average 5,000–6,000 tokens (Table 2) directly guts the length-recovery claim. Expect materially
  worse numbers and do not attribute the gap to the method.
- **Student distillation.** Same problem. Qwen2.5-7B / Llama-3.1-8B full fine-tunes at 16k context
  do not fit. QLoRA on a 1.5B student is the realistic consumer path, which means you are
  measuring a different phenomenon and cannot compare to Tables 3/4 in absolute terms — only
  relative ordering of the five training-source conditions is meaningful.
- **Trace generation with R1-Weak (1.5B).** Generating 10k traces of ~6k tokens each is feasible
  locally with vLLM but is on the order of a day or two of wall-clock on one card. Budget for it.

### Out of reach locally

- **R1 as surrogate.** 671B MoE. Not local under any quantization on consumer hardware. The
  paper's *upper-bound* rows (and the striking "inversion beats the oracle" results, which occur
  only when surrogate = victim = R1) are all API-only. This is the single biggest reproduction
  gap: the most eye-catching claim in the paper is the one you cannot check locally.
- **R1 as victim.** Same.
- **GPT-5.4 mini as victim.** API-only, and ~$173 per 10k queries — cheap in absolute terms but
  not free, and the model may not remain available at that snapshot (`gpt-5.4-mini-2026-03-17`).
  Snapshot drift will make exact replication impossible within a year.
- **The full 8×A100 configuration** as specified. Any local run is a scaled-down variant, full
  stop.

**Practical recommendation:** the honest cheap reproduction is a fully local, self-contained
version — a small open reasoning model as *victim* (so you have the ground-truth traces), an even
smaller one as surrogate, a ~1.5B inversion backbone, a ~1.5B student, QLoRA throughout, and
GSM8K/MATH as the benchmark. That reproduces the *structure* of the claim (does inversion beat
Answer-only, Summary+Answer, and Surrogate-Trace?) without reproducing any absolute number. That
is the right target.

## B. Which claims look most fragile

Ranked by how much the paper leans on them versus how well they are supported.

1. **"Inversion can outperform the ground-truth oracle" (§5.3).** Most quotable claim, weakest
   evidence. Three cells: Qwen/MATH500 74.1 vs. 72.2, Llama/MATH500 62.0 vs. 59.6,
   Llama/JEEBench 25.2 vs. 23.8. MATH500 has 500 problems, so a 2-point gap is roughly one
   standard error on a single run; JEEBench is smaller still. The paper reports no seeds, no
   variance, no CIs — anywhere. The "denoising backtracking" story in §5.3 (supported by one
   cherry-picked example in Appendix D Figure 6) is plausible and there is real prior work on
   clean-trace distillation, but nothing here distinguishes it from run-to-run noise. **Check
   first: re-run with 3–5 seeds.**

2. **"Qwen matches the oracle on MATH500, trailing by 0.4 points" (71.8 vs. 72.2, §5.3).** Same
   noise problem, compounded by the authors' own admission that Qwen2.5-7B-Instruct is
   MATH500-contaminated (§5.3). A contaminated base plus a 0.4-point margin is not a measurement.
   The entire Qwen/MATH500 column should be treated as uninformative; the JEEBench columns
   (36.3 vs. 43.7 — a real 7.4-point *gap*, not a match) are the trustworthy ones and tell a more
   modest story.

3. **No length-matched or content-free control.** This is the most important missing experiment.
   Inverted traces are ~5,000 tokens; Answer-only targets are a few dozen. The paper never rules
   out that most of the gain comes from *training on long structured CoT-shaped text at all*
   rather than from the inversion model recovering anything victim-specific. The obvious control
   — invert against *shuffled* or *mismatched* (x, a) pairs, or condition on a wrong answer, and
   see how much of the gain survives — is absent. The Surrogate-Trace baseline partially covers
   this (it is also long CoT), but surrogate traces come from a weak 1.5B model and carry its
   errors, so the comparison conflates "length/format" with "trace quality." **This is the check
   I would run second, and it is cheap.**

4. **The claim that inversion transfers *the victim's* capability rather than eliciting the
   student's own.** Related to (3). The inversion model is Qwen2.5-7B and the student is
   Qwen2.5-7B — the same model family. Some of the gain could be self-distillation / format
   alignment in the student, with the victim's answer serving only as a correctness filter. The
   Llama student partially controls for this (different family, and the gains there are much
   larger in relative terms: 13.0% → 52.4% on MATH500), which is actually the paper's strongest
   evidence and is under-emphasized relative to the Qwen numbers.

5. **Table 1's summary-distribution matching (§5.2).** Four features, measured by "lightweight
   regex heuristics," on median length and three binary style flags. Matching median token count
   and "does it use bold headers" is a very thin basis for the claim that the distributions align.
   Content-level distribution match is untested. Since the entire GPT-5.4 mini attack depends on
   the attacker's `C` producing summaries in the victim's distribution, this is load-bearing and
   under-evidenced. Cheap to probe: train a classifier to discriminate the two summary sets. If
   it hits 95%+ AUC, the "distributions align" claim is dead even though every Table 1 row
   matches.

6. **Possible distribution leakage in the fidelity evaluation (Table 2).** OpenThoughts-114k
   traces are *themselves R1-generated* (§5.1). The victim is R1. The surrogate split and victim
   split are disjoint in *inputs*, but the inversion model is learning to produce R1-style traces
   and is then scored on overlap with R1 traces. The R1-surrogate row (TF1 58.00) is explicitly
   labeled an upper bound, so the authors are aware. But it means the absolute fidelity numbers
   measure style conformance at least as much as reasoning reconstruction. TF1 of 52.76 between
   two 5,000-token math traces on the same problem is a low bar — much of it is LaTeX, connectives,
   and restated problem terms. A shuffled-pair baseline (TF1 between an inverted trace and an
   *unrelated* R1 trace) would calibrate this and is missing.

7. **"Inversion helps most when the surrogate is weak" (§5.4)** is honest but undercuts the
   headline. The paper's own Table 6 shows R1 ≳ GPT-5.4 mini on all three benchmarks, meaning the
   flagship "steal from a strong commercial black box" experiment does not actually have a
   stronger victim. The regime the authors say inversion is *for* — victim much stronger than any
   available surrogate — is never actually tested. That is a scoping gap, not an error, but it
   means the paper does not demonstrate what its title implies.

8. **LCB regressions are explained away too quickly.** "Style/format mismatch" (§5.3) is asserted,
   not shown. The oracle *also* regresses on Llama/LCB (13.1% → 10.4%), which suggests something
   wrong with the code-domain fine-tuning setup generally rather than something specific to
   inverted traces. Worth a look; it may indicate an evaluation-harness or output-format bug that
   affects other numbers.

9. **The $173.28 cost figure.** Per-token prices were dropped from the HTML ("/M input tokens"),
   so the arithmetic is unverifiable from the text as extracted. Minor, but check it against the
   PDF.

10. **Missing table cells make most of Tables 3/4/5 unverifiable from the HTML.** Not the authors'
    fault, but a reproduction must work from the PDF. Every downstream number I have quoted comes
    from prose restatement, which is exactly where selective reporting hides — the prose quotes
    the favorable comparisons.

### If I had budget for exactly three checks

1. Seeds. Run the headline conditions 5× and put error bars on everything. I expect the
   beats-the-oracle claim and the 0.4-point MATH500 gap to dissolve, and the JEEBench and
   Llama-student gains to survive comfortably.
2. The shuffled-answer control from (3). Isolates "long CoT-shaped supervision" from "inversion
   recovered something real." This is the claim the whole paper rests on and nobody has tested it.
3. The summary-discriminator probe from (5). Cheap, fast, and directly attacks the assumption that
   makes the black-box attack transfer.
