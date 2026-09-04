# Phase 5 Handoff — Train the Students

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phases 0–4 are complete; Phase 5 is yours: fine-tune
the student on each supervision condition so Phase 6 can measure whether forged traces taught it
anything. Paper Stage 3 (`docs/00` §3, Eq. 3).

**You are supervised.** The session named **`big-boss`** planned this phase and audits it. It reads
your CHECKPOINT lines, gets one turn to object before every run longer than an hour, and writes
nothing to the tree. Report via SendMessage to `big-boss`; if it is absent or silent, proceed and say
so in the log. During any run longer than 3 h, send a mid-run status line (step / loss / ETA) every
~2 h.

Read first: `docs/00-overview.md` (the paper), `docs/10-run-plan.md` (the plan and the **four
questions** every proposed experiment must answer — and Phase 5's own section), `docs/02` §"Training
hyperparameters" (the paper's values and the TRL mapping), `docs/results/phase4.md` §4 and §10 (the
supervision-length confound and everything Phase 4 handed you), `docs/09` §5.1–5.2 (why the student
is Qwen3.5-2B FFT and what that costs the claim), `docs/11-phase1-handoff.md` §5 (conventions, in
full). This document is the operational summary.

**Precondition:** `docs/results/phase4.md` must exist on your branch. If it does not, the
`phase4-build` PR is not merged — STOP and report; do not reconstruct it.

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path | Note |
|---|---|---|
| **Forged traces, final** | `bench/results/phase4/forged-{7b,1.5b}-{sum,nosum}.jsonl` — 4,490 / 4,565 / 4,068 / 4,135 rows | keys `{idx, domain, x, y, b, t_hat, raw, gen_tokens, finish_reason, draw}`; every row terminated. **The Synthesized-Trace training data.** Never regenerate |
| Drop bookkeeping | `bench/results/phase4/forged-*-draws.json` (committed) | per-inverter `dropped_idx`; intersection = attack idx minus the union — **must come out to 3,616** (math 2,886 · code 477 · chem 68 · puzzle 65 · bio 60 · phys 60) |
| Attack file | `bench/results/phase3/victimB-attack.jsonl` — 5,045 rows | keys `{idx, domain, source, x, y, b, summary, summary_tokens, finish_reason}`; `b` is π's `b*`. Source of `x`, `y`, `b*` for Answer-only / Summary+Answer (its `y` == the oracle's, proven in Phase 4's pairing gate) |
| **Oracle file** | `bench/results/phase3/victimB-ORACLE.jsonl` — same rows + `t` | **Opened by exactly one code path: the Victim-Trace dataset builder** (§4.3). Nothing else names it |
| Surrogate data `D₂` | `bench/results/phase1/d2-{7b,1.5b}.jsonl` — 5,006 / 5,028 rows | keys `{idx, domain, source, x, y, b, t, summary, summary_tokens, finish_reason}` — split-A rows: `x'`, surrogate's own `y'` and `t'`. Surrogate-Trace source |
| Training scaffolding | `bench/phase2_train.py` | TRL SFTTrainer wiring that already ran 33.8 h on this GPU: `chunked_nll`, `completion_only_loss`, `processing_class=AutoTokenizer`, cosine + `warmup_steps=0.1` (the ratio — transformers 5 folded it in), batch 1 × accum 24, per-epoch eval/save, tokens/epoch print, probe mode `--max-steps`, `peak_vram.txt`. **Copy → `bench/phase5_train.py`**, swap in the student config (§4.4); keep the structure |
| Grader (unused here) | `bench/eval_baseline.py` | benchmarking is **Phase 6's**, not this phase's |
| Student base | `Qwen/Qwen3.5-2B` (HF cache) | `Qwen3_5ForCausalLM` text-only load, bf16. Tokenizer == Qwen3.5-4B's — every token count this phase: **Qwen3.5-2B tokenizer, stated** |

### Environment, 2026-09-04

| | |
|---|---|
| GPU | RTX 4090, 24,564 MiB; idle. FFT @16384 measured **15.79 GiB** (`docs/09` §5.1) — expect that, STOP above 21 |
| Disk | **26 GB free — not enough.** Step 0 evicts the three GGUF-era models (§5 step 0); ≥ 50 GB free after, or STOP |
| `.venv` | Python 3.13 · TRL 1.10 / transformers 5.x — **all training runs here** |
| `.venv-vllm` | not needed this phase (no generation, no evaluation) |
| `PYTHONUNBUFFERED=1` in every driver | Phase 2 lost log lines without it |

### 0.1 If you are in a worktree (Phases 3–4 were)

Gitignored assets — both venvs, `bench/results/**`, `bench/logs/` — exist **only** in the main
checkout `/home/asavala/Development/papers/trace-inversion`. The driver runs **your branch's code**
(`cd "$(dirname "$0")/.."`) with the **main checkout's** venvs by absolute path, and writes data and
logs under the main checkout's `bench/`. Only small committed files (JSON records, docs) go in the
worktree. The Bash tool in worktree sessions rejects loops, heredocs and `&&`-chains ("too complex to
verify") — put control flow in a script file and invoke it plainly. `git -C <main>` is refused; read
main's files by absolute path.

---

## 1. Where things stand — how many of each role

| Role | How many | Status |
|---|---|---|
| Victim `V` (Qwen3.8-27B IQ4_XS) | 1 | done; its GGUF is evicted this phase (step 0) |
| Surrogate `V'` (R1-Distill 7B / 1.5B) | 2 arms | done; GGUFs evicted this phase |
| Compressor `C'` (π) | 1 | done. Not run this phase |
| Inverter `I` | 4 = {7B, 1.5B arm} × {sum, nosum} | done — the four forged sets exist. Not run this phase |
| **Student `S`** | **1 base model (Qwen3.5-2B), trained 7–10 times — one training per condition** | **this phase** |

Facts from earlier phases that shape this one (`phase4.md` §4/§10):

- **The supervision-length confound is measured, not estimated.** Forged `[t̂; y]` is **1.78–1.94×**
  the oracle's `[t; y]` at the median on the same rows (`t̂` = 2.24–2.55× `t`). Nothing in this phase
  fixes it — Phase 5 trains under it and Phase 6 reads results with it. The length-matched control is
  **deferred**: it becomes a proposal only if Phase 6 shows a forged student ≥ the oracle anywhere
  (decided 2026-09-04; the asymmetry: a forged student that *loses* despite ~2× tokens is a
  conservative result needing no control).
- **Register**: oracle `t` = the victim's terse working notes; forged `t̂` = R1-Distill's "Okay, so…"
  monologue. Any oracle-vs-forged gap carries style as well as content and length. Report, don't fix.
- **Cap asymmetry**: oracle `t` ≤ 14,336 (p95 ~7.8–8.6k on kept rows); forged `t̂` < 8,192.
- **Answer consistency**: ~4–9 % of graded forged rows argue to a different answer than the `y`
  beside them. **No filtering** (`docs/09` 7.15) — the forged files train as-is.
- **The 1.5B arm kept ~9 % fewer rows** — the intersection (§4.2) removes that from every comparison.
- Every kept split-B row fits `max_length` 16384 by construction (`docs/09` 7.16) — the truncation
  gate (§4.3) should therefore count **0**.

---

## 2. What Phase 5 produces

```
datasets (step 1)                            students (steps 2–4)
bench/results/phase5/data/                   bench/results/phase5/students/
  answer-only.jsonl        3,616 rows          answer-only/           (~4.6 GB each,
  summary-answer.jsonl     3,616                 summary-answer/         weights only —
  oracle.jsonl             3,616                 oracle/                 final epoch-3 model,
  synth-7b-sum.jsonl       3,616                 synth-7b-sum/           no optimizer state)
  synth-7b-nosum.jsonl     3,616                 synth-7b-nosum/
  synth-1.5b-sum.jsonl     3,616                 synth-1.5b-sum/
  synth-1.5b-nosum.jsonl   3,616                 synth-1.5b-nosum/
  surr-1.5b.jsonl          3,616 (conditional)   surr-1.5b/            (conditional)
  surr-7b.jsonl            3,616 (conditional)   surr-7b/              (conditional)
  (synth-7b-sum reused)                          synth-7b-sum-lora/    (conditional; adapter, small)
```

Plus per training: `log_history.json`, `peak_vram.txt` (as Phase 2's). Committed:
`bench/results/phase5/format-stats.json` (per-condition token histograms + self-test output),
`docs/results/phase5.md` (the record), `docs/09` rows 7.23–7.24, `docs/10` Phase 5 closed, README row.
All `.jsonl` and weights under `bench/results/` are gitignored already.

---

## 3. Measured budgets

The one realized training rate on this GPU: Phase 2's **~2,100 train tok/s** (4B bf16 LoRA r64 @12288,
`sdpa` + fla DeltaNet). The 2B FFT @16384 is a different point — **the 30-step probe on each condition
measures its own rate; budget from that**, not from this prior.

Token volume per epoch ≈ Σ(prompt + completion) over 3,616 rows, from Phase 4's measured medians:

| condition | tokens/epoch est. | 3-epoch est. @2,100 t/s |
|---|---:|---:|
| synth-* (each of 4) | ~14–16 M | **~5.5–6.5 h** |
| oracle | ~8–9 M | ~3.5 h |
| surr-* (each) | ~11 M | ~4.5 h |
| summary-answer | ~4 M | ~1.7 h |
| answer-only | ~2.5 M | ~1 h |
| lora twin | = synth-7b-sum volume | ~6 h |

Core 7 ≈ 33–37 h; all 10 ≈ **43–49 h**. The working ceiling is **50 h projected / 55 h hard STOP**.
Everything is sequential on one GPU. `docs/10`'s pre-Phase-4 "~20 h" is superseded on the record.

---

## 4. Decisions — fixed for this phase (decided 2026-09-04; report, do not redesign)

### 4.1 The slate and its order — core first, conditional cells gated

Run in this order, cheapest first so the pipeline is validated before the long runs:

1. **Core 7** (the paper's Table 3, matched): `answer-only` → `summary-answer` → `oracle` →
   `synth-7b-sum` → `synth-7b-nosum` → `synth-1.5b-sum` → `synth-1.5b-nosum`.
2. **Conditional cells**, in priority order: `surr-1.5b` (the paper's own weak-surrogate baseline) →
   `surr-7b` → `synth-7b-sum-lora` (the FFT-vs-LoRA cell, `docs/10`/`docs/09` §5.1).
   **Gate before each:** CHECKPOINT the current phase total + this run's projection; run it if the
   phase stays ≤ 50 h projected, else skip it and say so. One turn for `big-boss` to object either
   way. A skipped cell is reported in `phase5.md`, never silently dropped.

### 4.2 Training rows — the 3,616-row intersection, every condition

All split-B conditions train on the **same 3,616 idx** (the four-set intersection; recompute it from
`forged-*-draws.json` and assert 3,616). Surrogate-Trace is **n-matched**: sample 3,616 rows from the
full `d2-{arm}.jsonl` at **seed 1234** (each arm sampled independently; record the idx). Data
quantity is thereby out of every comparison (`phase4.md` §10; decided 2026-09-04). Gates: every
dataset has exactly 3,616 rows; all split-B datasets share an identical idx set; no idx appears twice.

### 4.3 One supervision construction — conditions differ ONLY in `c`

`bench/phase5_format.py` builds every dataset through a single function. Per row, TRL
conversational prompt-completion (`docs/06` §4.3):

```
prompt:     [{"role": "user", "content": x + INSTR}]
completion: [{"role": "assistant", "content": "<think>\n" + c + "\n</think>\n\n" + y}]
```

- `c` is the only thing that varies: `""` (answer-only) · `b*` (summary-answer) · `t'` (surr-*, with
  that row's `x'`, `y'`) · `t_hat` (synth-*) · `t` (oracle). Answer-only's empty `c` renders the
  same empty think block `enable_thinking=False` produces — one construction covers all five.
- `INSTR` is the Phase 3 suffix, pinned verbatim (`docs/09` 7.20):
  `"\n\nReason step by step, then give your final answer inside \boxed{}."` — appended to **every**
  training prompt in **every** condition, matching how the victim was queried and how Phase 6
  evaluates. The stored `x` stays bare.
- Thinking-mode rendering is the **default** (no `enable_thinking` kwarg): the trained continuation
  after the template's opening `<think>\n` is `c\n</think>\n\ny` — the student's native format.
- **Round-trip self-test is mandatory** (`docs/06` §4.4: the template re-parses `</think>` in
  assistant content): render row 0 of each condition with the Qwen3.5-2B template, print it, assert
  exactly one `<think>` and one `</think>`, `c` between them, `y` after, no doubled tags.
- **Cross-condition byte test**: for 5 shared idx, the rendered strings of answer-only /
  summary-answer / oracle / synth-* differ **only** in the think-block content. (surr-* differs in
  `x`/`y` too — split A — assert its construction shape instead.)
- **Truncation gate**: tokenize every row (Qwen3.5-2B tokenizer); rows ≥ 16,384 must number **0**
  (`docs/09` 7.16 arranged this). One row over is a STOP, not a trim.
- **ORACLE hygiene, restated for Phase 5**: `victimB-ORACLE.jsonl` is opened by the oracle dataset
  builder and nothing else. `grep -i oracle` over `phase5_format.py`, the driver and the logs must
  hit only that builder and its log line. The oracle dataset itself carries `t` — that is its job —
  but no other dataset may contain a `t` field or the oracle's trace text (assert: for 5 sampled
  idx, oracle `t` text appears in no other condition's file).
- Per-condition token histograms (prompt, completion, total: median/p95/max) go in
  `format-stats.json` and the record — they are the measured version of §3's estimates.

### 4.4 Training config — the paper's values on the measured mode (`docs/09` §5.1, `docs/02`)

`bench/phase5_train.py`, copied from `phase2_train.py`, per condition:

| knob | value | source |
|---|---|---|
| model | `Qwen/Qwen3.5-2B`, `Qwen3_5ForCausalLM` text-only, bf16, `sdpa` (+ fla DeltaNet as Phase 2) | `docs/09` §5.1 |
| method | **full fine-tune** — no `peft_config` | decided on measurement |
| `learning_rate` | **1e-5** | the paper's FFT value |
| `optim` | **`adamw_8bit`** | the measured 15.79 GiB mode |
| schedule / warmup | cosine · `warmup_steps=0.1` (ratio) | as Phase 2 (LLaMA-Factory default) |
| `num_train_epochs` | **3** | paper |
| `max_length` | **16384** | paper's `cutoff_len` |
| batch | 1 × `gradient_accumulation_steps` 24 | as Phase 2 |
| loss | `completion_only_loss=True` · `loss_type="chunked_nll"` · `packing=False` | non-negotiable at 248k vocab |
| tokenizer | `processing_class=AutoTokenizer(...)` explicitly | `docs/06` gotcha 1 (VLM misdetection) |
| eval | **none** — no holdout; all 3,616 rows train | the paper has none; the final epoch is the artifact, so nothing selects checkpoints |
| saving | `save_strategy="epoch"`, `save_total_limit=1` (crash resume); at the end `trainer.save_model()` → `students/<condition>/` (weights only), then **delete the `checkpoint-*` dir** | disk |
| probe | `--max-steps 30` first, every condition | §6 |

**The LoRA twin** (`synth-7b-sum-lora`): identical except `peft_config` r=64 / α=128 /
`phase2_train.py`'s exact `target_modules` (never `lm_head`), `learning_rate` **1e-4**. Same data
file, same everything else — the pair measures what LoRA costs (`docs/09` §5.1).

### 4.5 Not in this phase

No benchmarking or generation (Phase 6). No length-matched control (deferred — §1). No consistency
filtering (`docs/09` 7.15). No cross-family student (`docs/09` §5.2's "check" stays deferred). No
re-inversion, no touching Phase 1–4 data files, no second seeds, no hyperparameter search. If an
experiment seems worth adding, put it through `docs/10`'s four questions in a CHECKPOINT and wait.

---

## 5. Order of work

| Step | What | Est. |
|---|---|---|
| **0** | **Disk eviction (user-authorized 2026-09-04):** delete from the HF cache the victim GGUF (`unsloth/Qwen3.8-27B-GGUF`, ~14.6 GB) and both surrogate GGUF/F16 repos (`DeepSeek-R1-Distill-Qwen-7B` ~14.2 GB, `-1.5B` ~3.3 GB) — all re-downloadable, none used in Phase 5/6. Report what was removed and `df`; **≥ 50 GB free or STOP**. Keep Qwen3.5-2B/-4B and OpenThoughts. Branch `phase5-build`. | ~0.5 h |
| **1** | `bench/phase5_format.py` + self-tests (§4.3) → the 7–9 dataset files + `format-stats.json`. `bench/phase5_train.py` from `phase2_train.py` (§4.4) + `bench/run_phase5_train.sh` driver (train → save → delete checkpoints → log). Commit (explicit paths). CHECKPOINT with the token histograms vs §3's estimates. | ~2 h |
| **2** | Per condition, in §4.1 order: **30-step probe** (VRAM peak, realized tok/s, loss falling, projection stated with its assumed rate) → CHECKPOINT, one turn → full 3-epoch run → CHECKPOINT (wall, tok/s, final loss, disk). Core 7 first. | ~33–37 h |
| **3** | The three conditional cells, each behind the §4.1 budget gate. | ~0–15 h |
| **4** | `docs/results/phase5.md` (§6) · `docs/09` 7.23 (the §4.3 construction: think-slot unification, `b*` in the reasoning slot, empty-think answer-only, INSTR on training prompts — all choices the paper never specifies) and 7.24 (intersection + n-matching vs the paper's full-split training) · `docs/10` Phase 5 closed with measured hours · README row. Commit. | ~1 h |

---

## 6. Reports, gates, and what goes in `docs/results/phase5.md`

### Probe report — before each full run

`condition` · `rows 3,616` · `tokens/epoch (TRL's count)` · `steps/epoch` · `peak VRAM` ·
`realized train tok/s` · `projection for this run and the phase, stating the assumed rate` ·
`loss at step 0 → step 30`.

### Record

- The slate table: per training — rows, tokens/epoch, steps, wall clock, realized tok/s, peak VRAM,
  final train loss, loss at each epoch boundary, artifact path + size, disk free after. Skipped
  conditional cells and why.
- `format-stats.json` summarized: per-condition length histograms; the supervision totals beside
  Phase 4's §4 table (the confound, restated as-trained).
- Every §4.3 self-test's output; the ORACLE grep; the truncation count (0).
- What Phase 6 needs: §9's table completed with real paths.

### Gates that exit non-zero

- formatter self-tests (§4.3): row counts, idx equality, round-trip, cross-condition byte test,
  truncation 0, oracle-leak check
- per training: loss is finite and decreased over epoch 1; `students/<condition>/` loads with
  `Qwen3_5ForCausalLM.from_pretrained` after save; checkpoint dirs deleted; disk ≥ 10 GB before the
  next run starts
- driver hygiene: no path containing `ORACLE` anywhere but the oracle builder's input

### STOP AND ASK — report, then wait

- any dataset gate failing twice · truncation count > 0 · the oracle-leak check failing
- peak VRAM above **21 GiB** · loss NaN/inf · loss not lower at epoch 1's end than at step 0
- a training exceeding **2×** its probe projection · the phase projection passing **55 h**
- disk under **8 GB** free at any point · a saved student failing to load
- anything outside the oracle builder wanting a file named ORACLE
- any final-loss ordering that looks *too* good — e.g. a condition's loss collapsing near zero
  (smells like target leakage into the prompt, not a finding; check before continuing)

---

## 7. `docs/09` rows

Add **7.23** (student supervision construction — §5 step 4 lists the choices) and **7.24**
(intersection + n-matched Surrogate-Trace; the paper trains each condition on its full 10k split).
7.15/7.16/7.21 already carry the consistency, cap and length context — reference, don't duplicate.

---

## 8. Conventions — Phase 5 specifics

`docs/11` §5, `docs/13` §8, `docs/14` §8, `docs/15` §8 apply in full. What bites here:

| Rule | Why |
|---|---|
| **One code path opens ORACLE — the oracle dataset builder** | the leak fails no downstream check and would silently fake the ceiling (`docs/10` Phase 3 note) |
| **One construction, `c` varies** — never a per-condition formatter | a second formatter is a silent format confound between conditions |
| **Round-trip the template before any run** | Qwen3.5's template re-parses `</think>` in content (`docs/06` §4.4); a doubled tag trains garbage silently |
| **Probe every condition; budget from its realized rate** | §3's numbers are estimates from another model's rate |
| **The final epoch is the artifact** — no checkpoint shopping | the paper evaluates its 3-epoch model; eval-loss selection would be a new condition |
| **Delete each training's checkpoint dirs after `save_model`** | ~9 GB of optimizer state per run against a ~50 GB budget |
| **State the tokenizer next to every token number** | |
| **Never edit a running script; one sequential driver; explicit paths on `git add`; PYTHONUNBUFFERED** | |
| **A supervising session writes nothing to the tree** | it audits CHECKPOINT lines |

---

## 9. Open items carried to Phase 6

| Item | Note |
|---|---|
| **Evaluation** | all students on MATH500 + JEEBench (+ LCB if time), vLLM bf16, Phase 0's harness and paper sampling (0.7/0.9/1.05), thinking-mode default, seed 1234 — directly comparable to Phase 0's student baseline (MATH500 79.0 / JEEBench 47.8) and the regime row (`phase3.md` §6) |
| **The length-control trigger** | if any forged student ≥ the oracle student on any benchmark, the length-matched control becomes a Phase 6.5 proposal through `docs/10`'s four questions |
| **The confound reading** | oracle-vs-forged gaps carry length (1.78–1.94×), register and consistency differences — `phase4.md` §10's table is the caveat block for Table 3's analogue |
| **Student manifest** | `students/<condition>/` paths + sizes from `phase5.md` §"Record" |
| **The 2B student's looping habit** | Phase 0 flagged loop-y generations on the 2B (`docs/10` 0.2 audit note); Phase 6 should carry Phase 4's strict loop test into its harness |

---

## 10. Definition of done

- [ ] step-0 eviction done and reported; ≥ 50 GB free before the first training
- [ ] `phase5_format.py` + self-tests committed; 7–9 datasets of exactly 3,616 rows; `format-stats.json` committed; truncation count 0; ORACLE opened only by the oracle builder
- [ ] `phase5_train.py` + driver committed; every training probed (30 steps) before its full run; CHECKPOINTs to `big-boss` with one turn to object before each run over an hour; mid-run status on runs over 3 h
- [ ] the core 7 students trained and saved (weights-only, loadable); each conditional cell either trained or its skip adjudicated on the record
- [ ] per-training numbers (wall, tok/s, VRAM, loss curve, size) in `docs/results/phase5.md`; skipped cells reported
- [ ] `docs/09` 7.23–7.24 added; `docs/10` Phase 5 closed with measured hours; README row updated
- [ ] checkpoint dirs deleted; disk ≥ 10 GB free at the end; every gate's output on the record
