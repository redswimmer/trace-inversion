# Trace Inversion, reproduced on one GPU

**Can you steal a model's reasoning from the parts it shows you?**

Commercial reasoning models — OpenAI's o-series, Gemini, Claude with extended thinking — hide their
chain of thought and return only the final answer plus, sometimes, a short summary of the thinking.
The bet behind that design is that hiding the trace stops competitors from distilling the model's
reasoning ability. [*How to Steal Reasoning Without Reasoning Traces*](https://arxiv.org/abs/2603.07267)
(Zhang, Morris, Shmatikov — Cornell Tech, 2026) argues the bet is lost: train a model to run reasoning
**backwards** — given a problem, its answer and the summary, write the full trace that would have
produced them — and the traces it forges are good enough to train on. In the paper, a student
fine-tuned on forged traces goes from 24.0 % to **36.3 %** on JEEBench, most of the way to the
43.7 % it reaches when trained on the victim's *real* traces. Trained directly on what the victim
shows — answers and summaries — the same student gets *worse*.

This repository reproduces that pipeline end to end on **one RTX 4090**, with a **local victim**.
The local victim buys something the paper's API victim never could: its real traces stay on disk,
withheld from the attack, so we can measure exactly how close the forgeries get to the truth.

## The cast

Five roles. Three are only ever run; two are trained — and those two are trained **more than once
each**, because the experiment is a grid, not a single pipeline.

| Role | Symbol | Here | Trained? | How many | Job |
|---|---|---|---|---|---|
| **Victim** | `V` | Qwen3.8-27B, 4-bit, llama.cpp | never | 1 | The strong model being stolen from. Answers problems, shows a summary, hides its trace. |
| **Surrogate** | `V'` | DeepSeek-R1-Distill-Qwen-**7B** and **-1.5B** | never | **2 arms** — each runs through the whole pipeline | A weaker reasoner we run ourselves, so its traces are visible. Exists only to manufacture the inverter's training data. |
| **Compressor** | `C'` | Qwen3.5-4B, zero-shot with a fixed prompt | never | 1 | Writes a victim-style summary of each surrogate trace, so the training data has the "summary" column the victim will later provide. |
| **Inverter** | `I` | Qwen3.5-4B + LoRA | **yes** | **4 adapters** — {7B arm, 1.5B arm} × {victim shows a summary, victim shows only the answer} | Learns *(problem, answer[, summary]) → trace* from one arm's surrogate data. Then pointed at the victim's outputs. |
| **Student** | `S` | Qwen3.5-2B, full fine-tune | **yes** | **5 checkpoints** — one per training-data condition (below) | The model we are trying to improve. Benchmarked on MATH500 and JEEBench. |

Why the inverter is trained four times rather than once: the two *settings* are different input
formats (an inverter trained with summaries can't be fed inputs without them, and the paper keeps
them separate so each setting's number is that setting's alone), and the two *arms* are different
training data (the point of the second arm is to change one thing — the surrogate — and watch what
happens downstream, which needs one inverter per surrogate with everything else identical). The
paper trains the same four.

Problems come from OpenThoughts-114k, in two disjoint splits — one the surrogate sees, one the victim sees.

An analogy that holds up: a brilliant tutor gives you answers and a paragraph of "how I thought about
it," but never the worked solution. Apprentices learn from worked solutions. So you hire a mediocre
tutor you *can* watch (surrogate), record their worked solutions, have a clerk (compressor) summarise
each in the brilliant tutor's style, and train a forger (inverter) to reconstruct a worked solution
from *(problem, answer, summary)*. Then feed the forger the brilliant tutor's outputs and train your
apprentice (student) on the forgeries.

## How it works

```
Phase 0   benchmark every candidate zero-shot → fix the roles above                        done
Phase 1   split A → surrogate → (problem, trace, answer)
                                   └→ compressor → summary
          = D₂: (problem, answer, summary, trace) × ~5,000 rows, once per surrogate arm     done
Phase 2   train the inverter on D₂:  (problem, answer, summary) → trace                    done
Phase 3   split B → victim → (problem, answer, summary)    real trace → separate file, locked   done
Phase 4   inverter(problem, answer, summary) → forged trace, × 5,000                       ← next
Phase 5   train the student five ways on split B:
            answer-only | summary+answer | surrogate's own traces | forged traces | real victim traces
Phase 6   MATH500 + JEEBench on all five students                                          ← the result
```

The question the last row answers: does the student trained on **forged** traces beat the students
trained on what the victim actually shows, and on the surrogate's own traces — and how close does it
get to the one trained on the victim's real, withheld traces? If yes and close, hiding the chain of
thought protected nothing.

The paper's own numbers for exactly that comparison — its Qwen2.5-7B student, the weak 1.5B
surrogate, victim = DeepSeek-R1, accuracy in %:

| Student trained on | MATH500 | JEEBench |
|---|---|---|
| nothing (base model) | 71.2 | 28.3 |
| the victim's answers only | 61.0 | 21.6 |
| the victim's summaries + answers | 63.0 | 24.0 |
| the surrogate's own traces | 63.2 | 19.7 |
| **forged traces** | **71.8** | **36.3** |
| the victim's real traces (oracle) | 72.2 | 43.7 |

Everything the victim actually shows makes the student *worse* than doing nothing; forgeries built
from those same outputs carry it most of the way to the oracle. That is the result being reproduced.

The inverter is never benchmarked and never asked to solve anything. It is handed the answer. Its
only test is whether the traces it writes make the student better.

## Two things the paper didn't do

- **The oracle row exists.** The paper's black-box victim (GPT-5.4 mini) could never reveal its real
  traces, so the ceiling in its results — a student trained on the victim's *real* traces — had to
  come from a different, open-weight victim. Ours is one model: the victim that gets attacked is the
  victim whose real traces set the ceiling.
- **A midpoint on surrogate strength.** The paper's most useful claim — a *weak* surrogate is nearly
  as good as a strong one — was tested at two points 450× apart (a 1.5B distill and 685B R1). We run
  the 1.5B (the paper's exact model) and a 7B on identical prompts, settings and cap, so the
  comparison varies surrogate strength and nothing else.

## Where it stands

| Phase | Status | Headline |
|---|---|---|
| 0 — baselines | done | Harness calibrated: the paper's own surrogate scores 32.6 % JEEBench here vs 32.6 % in the paper. Roles fixed on measurement: student 47.8 < surrogate 60.6 < victim 86.2 on JEEBench — the regime the argument needs. *(86.2 is at the chat template's default `xhigh` effort; 82.0 at the `medium` Phase 3 actually queried. The ordering holds either way.)* |
| 1 — surrogate data | done | `D₂` built for both arms (5,006 / 5,028 rows). Summaries pass all four of the paper's style targets. ~21 h of generation per arm. |
| 2 — train inverters | done | Four LoRA inverters, 33.8 h of training at ~2,100 tokens/s. On held-out prompts the forged traces run 0.89–0.99× the surrogate's length and land on their given answer ~95–97 % of the time. Record: `docs/results/phase2.md`. |
| 3 — query the victim | done | **5,045** victim rows on split B in **66.3 h** at 123.6 t/s, 0 errors. The victim's own traces (median 1,400 tokens) turn out **shorter than the forgeries meant to imitate them** — the reverse of the paper's ordering, and a length confound Phase 5 has to carry. Record: `docs/results/phase3.md`. |
| 4 — invert | next | ~4 h |
| 5 — train students | | five conditions, ~20 h |
| 6 — evaluate | | the result |

## What we've found so far

Findings the paper didn't report, from Phases 0–3 (full detail in `docs/results/`):

- **The victim was never asked for its best reasoning — and nobody had noticed.** The GGUF's chat
  template silently injects a reasoning-effort system turn when the request sends none, so every
  "no system prompt" query — Phase 0's benchmarks included — actually ran at `xhigh`. Setting
  `medium`, the only level that renders *no* system turn at all, halved the cap-hit rate (37 % → 16 %)
  and cut the victim run from a projected ~145 h to 66 h, for a benchmark difference that is within
  sampling noise (82.0 vs 86.2 JEEBench on a 250/bench subset — 1.5 SE, and confounded with a
  prompt change made at the same time).
- **A stronger model writes shorter traces — short enough to invert the paper's ordering.** The
  victim's real traces run a median 1,400 tokens against R1's 3,664 on the *same* prompts, and
  shorter than the forgeries built to imitate them (~2,100 on the held-out estimate). The paper's
  `Len` metric assumes forgeries approach the truth from below; ours look likely to overshoot it from
  above, which puts supervision length into the oracle-vs-forgery comparison. Phase 4 measures it.
- **The answer's format is a property of the prompt, not the model.** Asking for a boxed answer took
  the share of victim responses carrying one from 56 % to ~100 % — the same model, the same effort,
  a one-sentence instruction. Every downstream grader depends on it.

- **Capability shows up as brevity.** The 27B victim solves JEEBench in a median 4,295 tokens; the 4B
  takes 21,859 and scores lower. Long generations were smaller models flailing.
- **A weaker surrogate doesn't make longer training data — it makes a bigger discard pile.** Traces
  are capped at 8,192 tokens and capped rows are dropped. The 7B hits the cap on 34.6 % of prompts,
  the 1.5B on 45.9 %; yet the *surviving* traces are the same length (median ~2,800 vs ~2,500). All
  the extra verbosity lives in the tail the cap removes.
- **The drop is partly the surrogate's own doing.** Paired against R1's ground-truth trace on the same
  prompt, 38 % of what the 7B drops (13 % of all prompts) are problems R1 finishes fine — a
  train/serve shift the paper's identical drop policy carries unmeasured. For the 1.5B it's 23 % of
  prompts.
- **Whether a prompt hits the cap is substantially a coin flip.** Two draws of the same prompts at
  temperature 0.7 disagree on 15 % of rows. That is the noise floor under every cap-hit number.
- **The paper's "aim for 600–900 tokens" is a calibrated over-ask** that its compressor undershoots
  by ~28 %; ours undershoots by ~17 %, so the exemplars had to be re-sized to land on the paper's
  medians. Swap the compressor and you land somewhere else on the same instruction.
- **The weak surrogate's habits reach the inverter through style, not through bad targets.** Every
  trace the inverters trained on terminated under the cap, yet the inverters trained on the 1.5B's
  traces run past 8,192 tokens on 10–17 % of held-out prompts, against 3–5 % for the 7B's — and they
  loop where the 7B-arm inverters never do.
- **A forged trace sometimes argues itself out of the answer it was handed.** About 3 % of gradable
  held-out traces conclude something other than the answer they were conditioned on — a spurious
  units "correction", 21 talked down to 20, a Yes turned into No. On the weak arm, most of those are
  the inverter being *right* where its surrogate was wrong. The paper does no filtering; what to do
  with such traces before student training is an open decision.
- **The inverter needs almost nothing to acquire the format.** Loss drops in the first ~100 steps
  and is flat after; a 20-step adapter already wrote traces at the true length ending in the right
  boxed answer. Epochs 2–3 fit the training rows (held-out loss is best at epoch 2 on all four).

<details>
<summary><strong>How this differs from the paper</strong></summary>

The paper ran on 8× A100 80 GB with a 685 B victim. Everything below follows from having 24 GB of
VRAM and 30 GB of RAM. The running log, with the reason and expected effect of each, is
`docs/09-deviations-from-paper.md`.

| | Paper | Here | Why |
|---|---|---|---|
| Victim | DeepSeek-R1 (685 B) via API; GPT-5.4 mini | Qwen3.8-27B, 4-bit GGUF, local | R1 is unrunnable at any quantization; a local victim gives the oracle row |
| Surrogate | R1-Distill-Qwen-1.5B (and R1) | same 1.5B, plus a 7B as primary | the 1.5B scores *below* our student on JEEBench; the 7B restores the paper's ordering and adds the midpoint |
| Compressor / inverter base | Qwen2.5-7B-Instruct | Qwen3.5-4B | 7B full fine-tuning is ~58 GB; the 4B fits with LoRA at 18 GB |
| Inverter training | full-parameter SFT | bf16 LoRA, r=64, lr 1e-4 | 4B full fine-tuning measured at 27.8 GB |
| Student | Qwen2.5-7B-Instruct, Llama-3.1-8B, full SFT | Qwen3.5-2B, full SFT at the paper's 16,384 context | fits full fine-tuning, so the method matches. It is a model that already reasons, so the claim becomes "inversion *improves* reasoning," measured against the same four baselines |
| Data | 2 × 10 k prompts | 2 × 5 k | the paper's own scaling curve shows 5 k delivers most of the MATH500 gain; generation is the dominant cost |
| Framework | LLaMA-Factory + DeepSpeed | TRL `SFTTrainer` | translated, not copied — the two frameworks' defaults differ |
| Trace-fidelity metrics | BLEU / TF1 / ROUGE against the victim's real traces | not run | across the paper's own results they track trace *length* at r ≈ 0.9; student accuracy carries the result |
| Inverter input format | unspecified | the paper's own zero-shot prompt, format-matched to its compressor | the paper never says what the trained inverter was given; its v2 prompts also contradict each other on summary format |
| Seeds / variance | none reported | 3 seeds on at least one condition | the paper's headline margins are 0.4–2.4 points on single runs |

</details>

<details>
<summary><strong>Repository layout</strong></summary>

```
bench/                  the harness — every script that generates, trains, measures or gates
  phase1/               pinned prompts (verbatim from the paper's repo, sha256-asserted) and the split manifest
  results/              raw outputs (gitignored — 20–340 MB each, they embed full generated text)
docs/
  00–04                 the paper: method, experiments, artifacts, defenses and critique
  05–08                 this machine: feasibility, model selection and TRL, the released code, measured throughput
  09                    every deviation from the paper, with reasons — kept current
  10                    the run plan, phase by phase, and the four questions to ask before proposing an experiment
  11–14                 per-phase handoffs and readiness reviews
  PHASE*-GOAL.txt       the prompt each phase is run from
  results/              committed measurements: baselines, phase1, sweeps, audits
```

</details>

<details>
<summary><strong>Running it</strong></summary>

Three stacks, deliberately separate: `llama.cpp` (`llama-server`, GGUF) for the victim and
surrogates; `.venv-vllm` for batched inference with vLLM; `.venv` for training with TRL. vLLM pins
its own torch and never goes in the training venv.

Every long run in this project is preceded by a probe — a concurrency sweep or a 20-step training
run — whose projection is reported before the run starts, and every result file goes through a gate
(`bench/phase1_stats.py`, `bench/audit_results.py`) before a number from it is believed. The
conventions that came out of doing this the hard way are in `docs/11-phase1-handoff.md` §5.

</details>
