# Phase 6 Handoff — Evaluate the Students

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phases 0–5 are complete; Phase 6 is yours:
benchmark every trained student under one protocol so the project's Table 3 analogue exists —
the number the entire ~230 GPU-hours was spent to produce. **No training, no generation of
training data, no new conditions. Evaluation and the record only.**

**You are supervised.** The session named **`big-boss`** planned this phase and audits it. It reads
your CHECKPOINT lines, gets one turn to object before every run longer than an hour, and writes
nothing to the tree. Report via SendMessage to `big-boss`; if it is absent or silent, proceed and
say so in the log. During any run longer than 3 h, send a status line every ~2 h.

Read first: `docs/10-run-plan.md` (the plan, the **four questions**, and Phase 6's own section),
`docs/results/phase5.md` §6 (the two REQUIRED items — they are restated below but read the source),
`docs/09` §5.2 and §7.7 (instill-vs-improve; the looping asymmetry you must report),
`docs/results/phase4.md` §10 (the caveat block your table must carry),
`docs/11-phase1-handoff.md` §5 (conventions, in full). This document is the operational summary.

**Precondition:** `docs/results/phase5.md` exists on your branch (phase5-build merged, PR #10).
Missing → STOP and report. Branch **`phase6-eval`**. If you are in a worktree, `docs/16` §0.1
applies in full (venvs/data live only in the main checkout; no loops/heredocs/&&-chains in Bash —
control flow goes in a committed script).

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path | Note |
|---|---|---|
| **The 10 students** | `bench/results/phase5/students/{answer-only,summary-answer,oracle,synth-7b-sum,synth-7b-nosum,synth-1.5b-sum,synth-1.5b-nosum,surr-1.5b,surr-7b}/` — 3.78 GB each, text-only `Qwen3_5ForCausalLM`, tokenizer included; `students/synth-7b-sum-lora/` — PEFT adapter r64/α128, 250 MB, base `Qwen/Qwen3.5-2B` | never retrain, never overwrite |
| **The eval harness** | `bench/eval_baseline.py` — Phase 0's protocol: 1,015 tasks (MATH500 500 + JEEBench 515), INSTR + per-type MCQ suffixes, brace-balanced `\boxed{}` extraction with `ANSWER:` fallback, MCQ letter-set / ±0.01 numeric / math_verify grading, raw text kept per row | **the same harness = the comparability claim.** Extend additively only (§4.2); grader, extractor, task construction, defaults untouched |
| Loop test | `bench/phase4_draws.py` → `loops(text, coverage=0.2)` strict, `coverage=0.0` loose | import it; do not rewrite |
| Answer-repeat audit | `bench/audit_results.py` (Phase 0's own-answer repeat counter) | reuse for the pre/post looping comparison |
| Offline regrade | `bench/regrade.py` | raw text is stored, so grading disputes never need regeneration |
| LoRA merge pattern | `bench/phase2_train.py --merge` — merge_and_unload + **assert base tensors changed** (max\|Δ\|, frac changed), save bf16 | adapt for the 2B base; the assertion travels with the pattern |
| Base model | `Qwen/Qwen3.5-2B` in the HF cache | baseline row + merge base |
| Reference numbers (cited, never re-run) | Phase 0 no-think 2B baseline **MATH500 79.0 / JEEBench 47.8** (`docs/results/baselines.md`); victim @ medium+INSTR **97.2 / 82.0** n=250 (`phase3.md` §6); surrogate 7B **92.6 / 60.6**, 1.5B **84.0 / 32.6** (Phase 0, llama.cpp — engine-boundary caveat, `docs/10`) | **the GGUFs were evicted in Phase 5. Do not re-download or re-benchmark the victim or surrogates.** |

### Environment, 2026-09-05

| | |
|---|---|
| GPU | RTX 4090, 24,564 MiB, idle |
| Disk | ~20 GB free. Phase 6 adds ~3.8 GB (merged LoRA) + ~1 GB of eval jsonl. **≥ 8 GB free at all times or STOP** |
| `.venv-vllm` | everything this phase runs here (vLLM 0.17+, verified Qwen3.5 support). **No training venv needed. Do not upgrade anything** |
| `PYTHONUNBUFFERED=1` in the driver | |

---

## 1. Where things stand — how many of each role

| Role | How many | Phase 6 |
|---|---|---|
| Victim / surrogates / compressor / inverters | 1 + 2 + 1 + 4 | done in Phases 0–4; **not run** (GGUFs deleted); their scores are citations |
| Student `S` | 1 base model, trained 10× (one per condition) + the untrained base | **all 11 evaluated** — that is this phase |

Facts from earlier phases that shape this one:

- **Two REQUIRED items** (`phase5.md` §6, `docs/09` 7.23): (a) every eval render uses
  **`enable_thinking=True`** — the students continue an open `<think>\n`; the 2B template's default
  renders a *closed* block and would break every trained student. Prove by render-and-diff, never
  assume. (b) **The thinking-mode 2B baseline is measured first**, before any student — Phase 0's
  79.0/47.8 is a no-think render; against thinking students it would inflate every gain. Both
  baseline rows are kept: together they measure `docs/09` §5.2's instill-vs-improve distinction.
- **The confound caveat travels with the table** (`phase4.md` §10, `phase5.md` §1): forged targets
  trained at 1.8–2.0× the oracle's length, in R1's monologue register, with 4–9 % answer
  inconsistency. Surrogate-Trace additionally differs three ways (trace source, answer source, row
  set — `phase5.md` §8).
- **Looping asymmetry** (`docs/09` §7.7): if training reduced the 2B's looping, part of any gain is
  learning to *terminate*, not to reason. Truncation and loop rates are reported beside accuracy on
  every row, pre and post.
- **The length-control trigger** (`docs/10`): any forged student ≥ the oracle student anywhere →
  the length-matched control becomes a Phase 6.5 *proposal* through the four questions. Report the
  trigger; do not build the control.
- Phase 0's soft spot: JEEBench **Numeric** graded 66.4 % vs MCQ 98.2 % — "spot-check before
  Phase 6" (`baselines.md`). §4.5 does it.

---

## 2. What Phase 6 produces

```
bench/results/phase6/                          committed:
  baseline-think.jsonl        (gitignored)       bench/results/phase6/summary.json
  <condition>.jsonl × 10      (gitignored)       docs/results/phase6.md   (the Table 3 analogue)
  synth-7b-sum-seed1235.jsonl (gitignored)       docs/09: row 7.25 added, row 4.5 closed
  synth-7b-sum-seed1236.jsonl (gitignored)       docs/10: Phase 6 closed with measured hours
  merged-synth-7b-sum-lora/   (~3.8 GB, local)   README row
```

`summary.json`: per run — model path, seed, per-bench acc / completed-only acc / truncated /
no_answer / median gen tokens / strict+loose loops / wall clock. One JSON, written by the driver.

---

## 3. Measured budget

The one realized prior: Phase 0's 2B eval ran **~1.2 h** for 1,015 tasks (no-think, 32k cap,
median gen 1.1k M / 6.4k J, 221 truncated). Thinking-mode runs generate more: the untrained
baseline is the wild card (budget 2–4 h; it may loop — that is a *finding*, docs/09 §7.7);
trained students should run at or under the prior (trained to terminate, completions ≤ ~9.5k).

13 runs ≈ **~18–26 h working estimate — re-budgeted from each probe** (§4.4). Ceiling **30 h
projected / 35 h hard STOP**. Everything sequential on one GPU.

---

## 4. Decisions — fixed for this phase (decided 2026-09-05; report, do not redesign)

### 4.1 The slate and its order — 13 runs

1. **`baseline-think`** — `Qwen/Qwen3.5-2B`, thinking ON. REQUIRED first; no student before it.
2. The 10 students, Phase 5 condition order: `answer-only` → `summary-answer` → `oracle` →
   `synth-7b-sum` → `synth-7b-nosum` → `synth-1.5b-sum` → `synth-1.5b-nosum` → `surr-1.5b` →
   `surr-7b` → `synth-7b-sum-lora` (merged, §4.3).
3. **Variance cells** (docs/09 row 4.5's standing commitment, "3 seeds on ≥ 1 condition"):
   `synth-7b-sum` re-run at seeds **1235** and **1236**, identical otherwise. The spread is the
   noise band every gap in the table is read against. Behind the same budget gate as Phase 5's
   conditional cells: run if the phase stays ≤ 30 h projected, else CHECKPOINT the skip.

### 4.2 One protocol — Phase 0's harness, one additive flag

- `eval_baseline.py` gets **one additive flag: `--enable-thinking`** (default off = Phase 0
  behavior byte-for-byte), which passes `enable_thinking=True` into the existing
  `apply_chat_template` call. Nothing else in the file changes — show the full diff in
  CHECKPOINT 1. A second new flag is a STOP-and-ask.
- Every Phase 6 run: `--enable-thinking --max-tokens 32768 --max-len 40960 --gpu-frac 0.90` and
  the file's defaults (temp 0.7 / top_p 0.9 / top_k −1 / rep 1.05 / seed 1234). 1,015 tasks.
- **Render gate, per model, before generation**: print task-0's rendered prompt; with the flag it
  must end `<think>\n` and contain **no** `</think>`; without the flag the 2B's must end
  `<think>\n\n</think>\n\n`. Assert both renders differ exactly as stated once (on the baseline),
  then assert the flagged property on every model (each student ships its own tokenizer copy).
- Students are served from their local dirs on vLLM bf16 (`Qwen3_5ForCausalLM` registry path,
  verified `docs/06` §1.5). No LoRA runtime serving (§4.3).

### 4.3 The LoRA cell is merged, then treated like the others

Adapt `phase2_train.py --merge`'s pattern for the 2B: load `Qwen/Qwen3.5-2B` text-only bf16, apply
`students/synth-7b-sum-lora/`, `merge_and_unload`, **assert target tensors changed** (report
max|Δ| and fraction changed, as Phase 4's merge-checks did), save bf16 →
`bench/results/phase6/merged-synth-7b-sum-lora/`. Serve the merged model. Rationale on the record
(`docs/10`): vLLM's LoRA path on a hybrid-DeltaNet architecture is the riskier, less-verified
route. The merged dir may be deleted after its run's gates pass (reproducible from the adapter);
report either way.

### 4.4 Probe before every full run

Per model: `--limit 25` (50 tasks) first → render gate output, acc sane (a trained student at ~0 %
is a template smell, not a finding), truncation, realized wall → projection for the full run and
the phase, stating the assumed rate. CHECKPOINT, one turn, then the full 1,015.

### 4.5 The audit — every run, same script

`bench/phase6_audit.py` (new, small): reads an eval jsonl → per-bench truncated / no_answer /
median tokens / **strict and loose loops** (import `phase4_draws.loops`) / own-answer repeats
(reuse `audit_results.py`'s measure) → merged into `summary.json`. Run on the baseline AND every
student — the pre/post looping comparison (docs/09 §7.7) is a required table in the record.
**Numeric spot-check, once, on `baseline-think`**: print 5 graded-correct and 5 graded-wrong
JEEBench Numeric rows (pred, gold, verdict); a tolerance bug hides here (`baselines.md`). Read
them; report.

### 4.6 Not in this phase

No LCB (no harness exists; after the table lands it may be *proposed* through `docs/10`'s four
questions — building the harness is part of that cost). No length-matched control (trigger only).
No consistency-filtered condition. No cross-family student. No re-benchmark of victim or
surrogates. No training. No touching Phase 1–5 artifacts. **Nothing in this phase opens or names
`victimB-ORACLE.jsonl`** — there is no legitimate reason; `grep -i oracle` over Phase 6 code and
logs must hit only the condition name `oracle` (the student dir), never the Phase 3 file.

---

## 5. Order of work

| Step | What | Est. |
|---|---|---|
| **0** | Branch `phase6-eval`. `--enable-thinking` diff on `eval_baseline.py` + `phase6_audit.py` + `bench/run_phase6_eval.sh` (sequential driver: probe → run → audit → append `summary.json`). Commit, explicit paths. CHECKPOINT 1 with the harness diff + render-gate output | ~1 h |
| **1** | `baseline-think` probe → run → audit. Numeric spot-check. CHECKPOINT with both baseline rows side by side (no-think 79.0/47.8 cited vs thinking measured) — the instill-vs-improve read | ~2–4 h |
| **2** | The 10 students, §4.1 order (merge before the LoRA cell): probe → CHECKPOINT → run → audit each | ~12–18 h |
| **3** | Variance cells behind the §4.1 gate | ~0–3 h |
| **4** | `docs/results/phase6.md` (§6) · `docs/09` 7.25 + close 4.5 · `docs/10` Phase 6 closed with measured hours · README row. Commit, PR `phase6-eval` → main | ~1.5 h |

---

## 6. Reports, gates, and `docs/results/phase6.md`

**The deliverable table**: rows = no-think baseline (cited) · thinking baseline · the 10
conditions; columns = MATH500, JEEBench (acc, with completed-only / truncated / no_answer /
median tokens / strict loops beside), Δ vs `baseline-think`, wall clock. Reference rows (victim,
surrogates) cited beneath with the engine-boundary caveat. **The caveat block (§1) sits directly
under the table, not in an appendix.**

**The reading, in order** (write it in this order; the trigger check is #3):
1. Does any trace condition beat `answer-only` and `summary-answer`? (the attack's floor)
2. Synth vs its arm's Surrogate-Trace — does inversion add value over plain distillation? (the
   paper's core claim, per arm)
3. Synth vs oracle — **the length-control trigger**: any synth ≥ oracle anywhere → say so
   explicitly and file the Phase 6.5 proposal line; do not run it
4. sum vs nosum; 7B arm vs 1.5B arm (the surrogate-strength sweep the paper never ran)
5. FFT vs LoRA twin; truncation/loop deltas vs baseline (the termination-vs-reasoning split)
6. The variance band from the seed cells, quoted next to every gap smaller than ~2× its width

Also in the record: per-run table (wall, tokens generated, trunc/loops), the two baseline rows
and what they say about instill-vs-improve (docs/09 §5.2), the Numeric spot-check verdict,
skipped cells and why.

### Gates that exit non-zero
- render gate (§4.2) per model · probe acc > 0 for every trained student · `summary.json` row
  written per run · oracle-grep clean (§4.6)

### STOP AND ASK — report, then wait
- render gate failing, or any model's generation prompt containing `</think>` under the flag
- a trained student at ≤ 5 % on either bench in probe or run (template/serve smell, not a finding)
- baseline-think truncation > 60 % (the cap is measuring itself again — `baselines.md` Finding 3)
- a run passing 2.5× its probe projection · phase projection past **35 h** · disk < 8 GB
- vLLM refusing to load any student · any change to the harness beyond the one flag
- anything wanting the Phase 3 ORACLE file

---

## 7. `docs/09` rows

Add **7.25**: eval-time serving construction — `enable_thinking=True` at eval, the paper had no
analogous knob; two baseline rows (no-think kept from Phase 0, thinking measured now) and which
one every Δ is read against. Close **4.5** in place with the measured seed spread. Reference
7.23 for the training-side pin; don't duplicate it.

---

## 8. Conventions — Phase 6 specifics

`docs/11` §5 and the phase-handoff conventions (13 §8, 14 §8, 15 §8, 16 §8) apply. What bites here:

| Rule | Why |
|---|---|
| **One harness, one additive flag** — never a second eval script, never a per-model tweak | a forked harness is a silent protocol confound; Phase 0 comparability is the whole point |
| **Render-and-diff, never try/except** on the template | the 2B default silently closes the think block and every student would eval broken (`phase5.md` §2.2) |
| **Probe before every full run; budget from realized rate** | §3's prior is one no-think run |
| **Accuracy is never quoted without truncation beside it** | completed-only bias (`docs/09` §7.6) and the termination asymmetry (§7.7) |
| **Report, don't redesign** — a surprising score is a finding | the table is the deliverable; interpretation happens in §6's reading order, changes happen in Phase 6.5 proposals |
| Never edit a running script · explicit paths on `git add` · state the tokenizer next to token counts · PYTHONUNBUFFERED | standing |
| **A supervising session writes nothing to the tree** | it audits CHECKPOINT lines |

---

## 9. Open items carried past Phase 6

| Item | Note |
|---|---|
| Phase 6.5 length-matched control | only if §6 reading #3 triggers; through the four questions |
| LCB | proposal-gated (§4.6) |
| Cross-family student check | deferred through Phase 5 (`docs/09` §5.2); a post-read proposal |
| Black-box (API victim) track | `docs/09` 2.3, optional, unscheduled |

---

## 10. Definition of done

- [ ] harness flag + audit + driver committed; CHECKPOINT 1 carried the full harness diff
- [ ] `baseline-think` measured before any student; both baseline rows in the record
- [ ] all 10 students evaluated under the one protocol (LoRA merged with a checked merge); every
      run probed first; render gate green on every model
- [ ] variance cells run or their skip adjudicated on the record
- [ ] `summary.json` committed; per-run audit (trunc / loops / no_answer) complete; Numeric
      spot-check on the record
- [ ] `docs/results/phase6.md` with the table, the caveat block, the §6 reading in order, and the
      trigger verdict; `docs/09` 7.25 + 4.5; `docs/10` closed with measured hours; README row
- [ ] disk ≥ 8 GB throughout; PR `phase6-eval` → main
