# Can hidden reasoning be stolen? Recreating *Trace Inversion*

**A full recreation of the attack. In this setting, its central result reversed.**

Commercial reasoning models hide their chain of thought and return only an answer, sometimes with a
short summary of the thinking. [*How to Steal Reasoning Without Reasoning Traces*](https://arxiv.org/abs/2603.07267)
(Zhang, Morris, Shmatikov, 2026) says hiding the trace isn't enough. The attacker trains an
**inverter** to run reasoning backwards: given a problem, the answer and the summary, it writes a
trace that could have produced them. A student fine-tuned on these forged traces then beats **plain
distillation**, which fine-tunes the same student on the visible traces of the attacker's own
weaker open model (the **surrogate**).

This repo recreates the experiment end to end (surrogate, inverters, victim and ten students), adds
a stronger 7B surrogate, and measures an evaluation-noise band. Students are scored on **MATH500**
(competition math) and **JEEBench** (harder physics, chemistry and math from India's JEE Advanced
exam).

One change is deliberate. The student, Qwen3.5-2B, already reasons natively; the paper's
instruct-tuned students did not, and the paper leaves open whether stolen traces help such a
student. Untrained and in thinking mode, it hits the 32k-token cap on two thirds of JEEBench, so
every student here is partly learning to *finish* its reasoning.

| Victim answers kept | Models trained | Student evaluations | Compute |
|:-:|:-:|:-:|:-:|
| 5,045 | 4 inverters, 10 students | 13 runs × 1,015 problems | ~258 GPU-hours |

## Result: forged traces lost to plain distillation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/headline-dark.svg">
  <img alt="Forged-trace student accuracy minus plain-distillation student accuracy. Paper: +8.6 on MATH500, +16.6 on JEEBench. This reproduction with the paper's same 1.5B surrogate: −1.2 and −7.5. With a 7B surrogate: −7.6 and −9.9." src="docs/assets/headline-light.svg">
</picture>

- **In this setting, the paper's core result reversed.** There, forged traces beat plain
  distillation by 8.6 and 16.6 points (+1.4 / +12.9 for its Llama student). Here they never won.
  With the 7B surrogate they lost by 7.6 and 9.9, outside the noise band. With the paper's 1.5B
  they tied on MATH500 and lost on JEEBench by 7.5 with the summary and 9.9 without, at and past
  the band's edge.
- **Plain distillation from an open 7B matched the victim's real traces.** A student distilled
  from the 7B surrogate (72.0 / 45.4) tied one trained on the victim's *real* hidden traces, the
  **oracle** (73.4 / 45.6), although the victim itself scores 21 points higher on JEEBench than that
  surrogate (82.0 on a 250-problem subset, vs 60.6). For this 2B student, the victim's stronger
  reasoning bought nothing measurable over an open 7B's.
- **The losing students often never finished.** On JEEBench, forged-trace students were cut off at
  the 32k-token limit on 29–50 % of problems, against 8 % for the oracle and 17 % for the
  7B-distilled student; across the seven trace-trained students, cut-off rate tracks accuracy
  (r = −0.94). A cut-off answer scores zero, so part of that link is by construction.
  [Details below](#where-the-gap-shows-up-students-that-dont-finish).

## How the attack works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img alt="Two-stage pipeline. Stage 1: a surrogate (R1-Distill 7B or 1.5B) solves 5,000 problems with visible traces; a compressor summarizes each trace; four inverters learn to map problem, answer and summary back to the trace. The surrogate's own traces also train the plain-distillation student. Stage 2: the victim (Qwen3.8-27B) solves 5,000 other problems, showing only its answer and a summary; the inverters forge the hidden traces; a Qwen3.5-2B student is fine-tuned on them and scored on MATH500 and JEEBench. The victim's real traces are used only for the oracle student." src="docs/assets/pipeline-light.svg">
</picture>

The inverter is never asked to solve anything, since it is handed the answer. Its only test is
whether the traces it writes make a student better. Every student is the same Qwen3.5-2B,
fine-tuned on one of five kinds of data:

| Training data | Role |
|---|---|
| the victim's answers only | floor: what the attacker sees |
| the victim's summaries + answers | floor: what the attacker sees |
| **forged traces** from the inverter | **the attack** |
| the surrogate's own traces | plain distillation, the alternative the attack must beat |
| the victim's real traces | the oracle, an upper bound no real attacker has |

Victim-side students train on the 3,616 problems where all four inverters produced a finished
trace; the plain-distillation students train on 3,616 of the surrogate's own problems. Every result
file passed an automated gate before any number from it was used, and the gates caught three silent
failures, including a chat template that quietly disabled thinking. The full record of every phase
is in [`docs/results/`](docs/results/).

## Every student, side by side

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-dark.svg">
  <img alt="Student accuracy by training data, MATH500 / JEEBench: victim's answers only 54.2 / 22.9; summaries + answers 54.8 / 21.6; forged traces from the 1.5B surrogate 61.0 / 24.9; the 1.5B surrogate's own traces 62.2 / 32.4; forged traces from the 7B 64.4 / 35.5; the 7B surrogate's own traces 72.0 / 45.4; victim's real traces 73.4 / 45.6, a tie with the 7B surrogate's traces. Untrained with thinking on 67.8 / 33.8; with thinking off 79.0 / 47.8." src="docs/assets/results-light.svg">
</picture>

**The fair "before" is the untrained model in thinking mode (67.8 / 33.8), how every student is
served. No student, the oracle included, reached the untrained model with thinking switched off
(79.0 / 47.8).** The dashed lines show both.

1. **The attack beat what the attacker can see, but on JEEBench only with the 7B surrogate.** Every
   forged-trace student clears the answers-only and summary floors on MATH500 by at least 6 points.
   On JEEBench the 7B forgeries do (31.8–35.5 vs 22.9), while forgeries from the paper's own 1.5B
   surrogate did no better than no trace (22.5–24.9).
2. **A stronger surrogate helped at every stage.** With the 7B surrogate, the inverters finished
   more often, about half as many forgeries had to be discarded, and the students scored higher.

<details>
<summary><strong>All 13 evaluation runs</strong></summary>

| Student trained on | MATH500 | JEEBench | JEEBench answers cut off at the 32k cap |
|---|---|---|---|
| nothing, thinking **off** (baseline run before the study, cited) | 79.0 | 47.8 | — |
| nothing, thinking **on** (how every student is served) | 67.8 | 33.8 | 65.6 % |
| the victim's answers only | 54.2 | 22.9 | 0.6 % |
| the victim's summaries + answers | 54.8 | 21.6 | 0.6 % |
| forged traces, 7B surrogate (with / without summary) | 64.4 / 67.0 | 35.5 / 31.8 | 30.7 / 29.1 % |
| forged traces, 7B surrogate, LoRA instead of full fine-tune | 68.0 | 35.5 | 28.7 % |
| forged traces, 1.5B surrogate (with / without summary) | 61.0 / 61.2 | 24.9 / 22.5 | 49.5 / 43.3 % |
| the surrogate's own traces, 7B | **72.0** | **45.4** | 16.7 % |
| the surrogate's own traces, 1.5B | 62.2 | 32.4 | 37.9 % |
| the victim's real traces (oracle) | **73.4** | **45.6** | 8.0 % |
| forged, 7B with summary: evaluation seeds 1234 / 1235 / 1236 | 64.4 / 67.8 / 65.8 | 35.5 / 37.9 / 32.6 | — |

Giving the inverter the summary made no measurable difference; neither did LoRA vs full
fine-tuning (identical on JEEBench; +3.6 on MATH500, at the edge of the 3.4 seed range). Full record: [`docs/results/phase6.md`](docs/results/phase6.md).

</details>

## Where the gap shows up: students that don't finish

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/termination-dark.svg">
  <img alt="Scatter of JEEBench accuracy against the share of answers cut off at the 32k-token limit (cut-off answers score zero). Across the seven trace-trained students, accuracy falls as the cut-off rate rises, r = −0.94: oracle and 7B-distilled students are cut off on 8–17% and score about 45; forged-trace students are cut off on 29–50% and score 22–36. The untrained model with thinking on is cut off on 66% and scores 33.8. A dashed line at 47.8 marks the untrained model with thinking off, which no student reached." src="docs/assets/termination-light.svg">
</picture>

Fine-tuning's clearest effect was on whether a student finished: the students that finished least
scored lowest.

**Length alone doesn't explain the gap to distillation.** Counting trace plus answer, the
surrogate's own training examples are as long as the forged ones (median ~3.0–3.4k vs ~2.9–3.2k tokens), yet the
7B-distilled student is cut off half as often. One visible difference is how they were made: the
surrogate's traces end at its own answer, while forgeries are written backwards toward an answer
the inverter was handed, and 4–9 % of them argue their way to a different one. The two sets also
cover different problems, so this is a candidate explanation, not a measured cause. Forgeries also
overshoot the real traces they replace (median trace 2.2–2.6× longer), where the paper's came in
*under* the real length (81–89 %). Both flaws were on record before any student trained.

## What it means

**The takeaway: an attack on hidden reasoning is only as convincing as its weakest baseline.** The
paper's Qwen margins were measured against a distillation student that scored below the untrained
model. Here, distilling a stronger open model matched the victim's real traces and left forging
little room, and for a student that already reasons, fine-tuning mainly taught it to finish its
reasoning. So credit a trace-stealing attack only if it beats both the untrained student, served
the same way, and distillation from the best open model. The evidence, from both sides:

**The paper's side: much of its margin was recovered ground.** Its distillation baseline scored
63.2 vs the untrained model's 71.2 on MATH500 and 19.7 vs 28.3 on JEEBench. Measured against the
untrained model, forging's +8.6 on MATH500 shrinks to +0.6 (71.8 vs 71.2); on JEEBench forging did
clear it (36.3 vs 28.3).

**Here the forgeries failed both baselines.** They trailed distillation from an open 7B, and no
forged-trace student beat the untrained student, served the same way, by more than the seed spread
(at most +0.2 MATH500 and +1.7 JEEBench; the 7B cell's three-seed JEEBench mean is +1.5). The
1.5B-surrogate forgeries were clearly below it (61.0–61.2 vs 67.8; 22.5–24.9 vs 33.8).

**What isn't settled:**

- **Not separable:** *why*. Forged and real traces differ in length, voice and answer consistency,
  the distillation students saw different problems and answers, and no length-matched control was
  run.
- **Noise:** each condition was trained once, and the measured band (3.4 MATH500, 5.3 JEEBench) is
  evaluation-seed spread only. The paper reports single runs; its "forged beats the oracle" margins
  (0.4–2.4 points on these benchmarks) would sit inside the band measured here. The headline gaps
  are differences of two runs, so their noise band is about √2 times wider (4.8 / 7.5), which is
  what the headline figure shades. The tightest loss (1.5B, JEEBench, −7.5) sits at that edge.
- **Data:** students train on 3,616 problems, about a third of the paper's 10k. The paper's own
  scaling curve (a different victim, no distillation arm) puts its attack 10.6 points lower on
  MATH500 at 5k than at 10k, so the forged-trace margin may depend on scale. Whether it would
  change sign is untested.
- **The 32k cap matters:** about half of the untrained model's JEEBench cut-offs were not strict
  loops and might finish with a larger budget.
- **Scale:** the models are smaller and newer than the paper's, so compare signs, not sizes.

## Future work

- **Length-matched control.** Trim or resample the forgeries to the real traces' length
  distribution, to separate length from voice and content.
- **The paper's question.** Repeat with a student that can't reason natively, and train the
  distillation students on the same victim-side problems.
- **Training seeds.** Three per key condition; the current noise band covers evaluation only.

<details>
<summary><strong>Deviations from the paper</strong></summary>

The paper ran on 8× A100 80 GB against a 685B victim. Most of what changed follows from 24 GB of
VRAM. Full log with reasons: [`docs/09-deviations-from-paper.md`](docs/09-deviations-from-paper.md).

| | Paper | Here | Why |
|---|---|---|---|
| Victim | DeepSeek-R1 (685B) | Qwen3.8-27B, 4-bit GGUF, local | R1 can't run in 24 GB |
| Surrogate | R1-Distill-Qwen-1.5B | same 1.5B, plus a 7B | adds a surrogate-strength midpoint |
| Compressor / inverter | Qwen2.5-7B-Instruct, full SFT | Qwen3.5-4B, LoRA r=64 | 7B full fine-tuning needs ~58 GB |
| Student | Qwen2.5-7B, Llama-3.1-8B, full SFT | Qwen3.5-2B, full SFT at 16k context | fits full fine-tuning; reasons natively, which changes the question (see top) |
| Data | 2 × 10k prompts | 2 × 5k queried; students train on 3,616 | cost. The paper's own curve (GPT-5.4 mini victim) falls from 77.6 to 67.0 on MATH500 at 5k, below its untrained 71.2 |
| Framework | LLaMA-Factory + DeepSpeed | TRL `SFTTrainer` | single-GPU training |
| Evaluation | unspecified sampling, single run | one harness for all 13 runs, 32k cap, 3 evaluation seeds on one cell | comparability |
| Trace-similarity metrics | BLEU / token-F1 / ROUGE | not run | across the paper's six Table 2 rows they track trace length (r = 0.84–0.95, n = 6, [`docs/10`](docs/10-run-plan.md)); student accuracy is the real test |

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```
bench/            every script that generates, trains, evaluates or gates
  results/        committed JSON records (raw generations are gitignored)
docs/
  00–04           the paper: method, experiments, artifacts, defenses
  05–10           this machine: feasibility, model choice, deviations, run plan
  11–17           per-phase plans
  results/        measured record of every phase, plus sweeps and audits
  assets/         README figures and make_charts.py, which draws them from the records
```

Three separate stacks: `llama.cpp` serves the victim and surrogates, vLLM handles batched inference
and evaluation, and TRL handles training. Regenerate the figures with
`python3 docs/assets/make_charts.py`; it reads `bench/results/phase6/summary.json` and checks
the headline numbers against it.

</details>
