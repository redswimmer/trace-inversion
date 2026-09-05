# Phase 5 — Train the Students (10 of 10 cells)

Reproduction of *"How to Steal Reasoning Without Reasoning Traces"* (Zhang, Morris, Shmatikov,
arXiv 2603.07267v2), Stage 3 (Eq. 3): the student `S` (one base model, **Qwen3.5-2B**, trained once
per condition) fine-tuned on every supervision condition over the same 3,616 split-B rows, so Phase 6
can measure whether forged traces taught it anything.

| | |
|---|---|
| Students | **10 trainings**: the core 7 (answer-only · summary-answer · oracle · synth-{7b,1.5b}-{sum,nosum}) + the 3 conditional cells (surr-1.5b · surr-7b · synth-7b-sum-lora), all run — none skipped |
| Rows | the **same 3,616 idx** for every split-B condition (four-set intersection, recomputed from `forged-*-draws.json`, asserted; math 2,886 · code 477 · chem 68 · puzzle 65 · bio 60 · phys 60); surr-* n-matched by a seed-1234 sample of 3,616 from the full `d2-{arm}.jsonl` (idx recorded in `format-stats.json`) |
| Construction | ONE formatter (`phase5_format.make_row`), only `c` varies: prompt = `x + INSTR` (`eval_baseline.INSTR`, docs/09 7.20, verbatim); completion = `"<think>\n" + c + "\n</think>\n\n" + y`; `c` = `""` \| `b*` \| `t'` \| `t̂` \| `t` |
| Training | Qwen3.5-2B bf16 **FFT** (`Qwen3_5ForCausalLM` text-only), lr 1e-5, `adamw_8bit`, cosine + warmup 0.1, 3 epochs, `max_length` 16384, batch 1×24, `completion_only_loss` + `chunked_nll`, no eval split — **the final epoch is the artifact**. LoRA twin: + r64/α128 on Phase 2's target list, lr 1e-4 |
| Tokenizer | every token count in this phase: **Qwen3.5-2B** (== 4B's), stated |
| Code | `bench/phase5_format.py` · `bench/phase5_train.py` (from `phase2_train.py`) · `bench/run_phase5_train.sh` |
| Supervision | session `big-boss` audited every CHECKPOINT (0–20) and issued the adjudications in §2/§8; it wrote nothing to the tree. Executor: session `g-claude`, branch `phase5-build` (worktree; data/logs under the main checkout per docs/16 §0.1) |

> **`docs/16-phase5-handoff.md` is the pre-run plan; this file is authoritative where they
> disagree.** Two corrections to its letter, both supervisor-ratified: (i) step 0's "delete from the
> HF cache" — the three GGUF-era models actually lived in `~/trace-inversion-bench/models/` (the HF
> cache held neither the victim GGUF nor any 7B R1-Distill repo); the named files were deleted from
> there. (ii) §4.3's "default thinking render (no `enable_thinking` kwarg)" — **Qwen3.5-2B's shipped
> template defaults to thinking OFF**, the inverse of the 4B docs/06 describes, so the stated
> construction (the trained continuation after the template's opening `<think>\n`) requires
> **`enable_thinking=True`, now pinned** (§2.2). The method, row policy, hyperparameters and order
> all held. §3's estimates vs measured: core 7 est. 33–37 h → **15.8 h** of full runs (the 2B FFT
> realized ~3,480 train tok/s, not the 4B-LoRA prior of 2,100); all 10 est. 43–49 h → **≈ 27.3 h**.

---

## 1. Datasets — 9 files × 3,616 rows, every gate green

`bench/phase5_format.py` (one run builds all 9; log `bench/logs/phase5-format.log`; committed stats
`bench/results/phase5/format-stats.json`). Gates, all **PASSED**:

- intersection recomputed = exactly **3,616**, domain mix equal to docs/16's table
- 3,616 rows per file; split-B idx lists identical and duplicate-free; surr idx ⊆ its `d2`
- **round-trip** on row 0 of all 9 conditions: exactly one `<think>` and one `</think>` in the full
  render, `c` and `y` byte-intact (answer-only and oracle renders printed in full in the log)
- answer-only's empty block == `enable_thinking=False`'s render **byte-for-byte** (the degenerate
  case the unified construction promised)
- **cross-condition byte test** (5 shared idx): the 7 split-B conditions differ only inside the
  think block; **rendered-prompt identity extended to all rows: 3,616/3,616 identical** across the
  7 split-B conditions (supervisor ask, CHECKPOINT 1)
- **truncation@16384 = 0** on all 32,544 rows (docs/09 7.16 arranged this; gate confirmed it)
- **ORACLE hygiene**: `victimB-ORACLE.jsonl` opened by exactly one code path, the oracle dataset
  builder (`build_oracle`). `grep ORACLE` verbatim: `phase5_train.py` — 0 hits; the driver — 1 hit,
  a comment carrying no path; all 20 run/probe logs — 0 hits; the format log — exactly one line:
  `ORACLE read: …/bench/results/phase3/victimB-ORACLE.jsonl (oracle dataset builder — the one
  permitted open)`. The oracle's `t` text (5 sampled idx) appears in **no** other condition's file;
  no dataset carries a `t` key
- pairing gate: forged/oracle `x`,`y` == the attack row's per idx — compared **stripped** (Phase 4's
  `to_row` stored stripped copies; the raw attack `x` carries trailing whitespace — verified
  whitespace-only on idx 85371)

**Token volume (Qwen3.5-2B tokenizer, the trainer's own two-segment scheme), tokens/epoch:**

| condition | tokens/epoch | total med / p95 / max | completion med | docs/16 §3 est. |
|---|---:|---|---:|---|
| answer-only | 2,623,826 | 661 / 1,303 / 6,624 | 535 | ~2.5 M ✅ |
| summary-answer | 4,791,245 | 1,289 / 2,034 / 7,261 | 1,143 | ~4 M (+20 %) |
| oracle | 8,692,185 | 1,736 / 7,038 / **14,419** | 1,577 | ~8–9 M ✅ |
| synth-7b-sum | 12,715,072 | 3,301 / 7,653 / 9,490 | 3,168 | ~14–16 M — **lower** |
| synth-7b-nosum | 12,116,099 | 3,108 / 7,595 / 9,399 | 2,956 | " |
| synth-1.5b-sum | 12,653,769 | 3,087 / 7,354 / 9,593 | 2,936 | " |
| synth-1.5b-nosum | 12,275,537 | 3,018 / 7,325 / 9,423 | 2,873 | " |
| surr-1.5b | 13,134,718 | 3,178 / 7,834 / 9,447 | 3,026 | ~11 M — **higher** |
| surr-7b | 13,734,946 | 3,511 / 7,911 / 9,136 | 3,362 | " |

Both deltas are selection effects, not surprises: synth came in **under** the estimate because the
intersection keeps the shorter rows (the same kept-file selection `phase4.md` §4 documented), and
surr came in **over** because the n-matched sample spans the full split-A distribution rather than
the shared-idx subset the estimate was scaled from.

**The supervision-length confound, as-trained** (completion medians above): the four synth targets
(2,873–3,168) train at **1.8–2.0×** the oracle target (1,577) on the same rows — Phase 4 §4's
1.78–1.94× confound, restated on the actual training files. Deferred length control unchanged (§8).

---

## 2. Two findings before any GPU hour — both caught by the mandated gates

### 2.1 TRL's template path mis-tokenizes the completion boundary

The row-0 decode gate (probe take 1) failed: TRL tokenizes prompt and prompt+completion
**separately** and masks a fixed `len(prompt_ids)` tokens, but at the `<think>\n` boundary the
newline merges into the completion's first token (`'\nThe'`), so 2–3 completion tokens landed in
the mask *and* the trained continuation's tokenization differed from what the generation prompt
serves. Fix — docs/13 §4.2's sanctioned fallback — `phase5_train.pretokenize`: two segments,
`prompt_ids` (chat template, `add_generation_prompt=True`) + `tok(c + "\n</think>\n\n" + y +
"<|im_end|>\n")`, with an explicit `completion_mask` TRL builds labels from. The per-run gate now
proves `decode(loss tokens)` equals the exact continuation and the masked prefix ends at
`assistant\n<think>\n` — **train == serve by construction**. Printed and checked on every probe and
run (the merge is content-dependent; each condition's `c` has a different first token).
*Not re-examined:* Phase 2's inverter training path (different construction, `enable_thinking=False`,
completion starts mid-sentence text); its held-out behavior is the measured fact — Phases 1–4 are
not reopened.

### 2.2 Qwen3.5-2B's shipped template defaults to thinking OFF

Measured side by side: the 4B's generation prompt ends `<think>\n` (thinking on, as docs/06 §2
says family-wide); **the 2B's ends `<think>\n\n</think>\n\n`** — thinking off unless
`enable_thinking=True`. docs/16 §4.3's letter ("no `enable_thinking` kwarg") was written on the 4B
behavior; its stated construction — the trained continuation after the **open** `<think>\n`, "the
student's native format" — is only coherent with the thinking prompt (the literal reading would
serve a closed think block and the student would emit a second one). **Supervisor adjudication:
`enable_thinking=True` pinned** in `pretokenize` and the formatter's counts/identity checks; the
datasets carry no `chat_template_kwargs` column — the render decision lives in the trainer and in
Phase 6's serving. Consequences, REQUIRED for Phase 6: §6.

### Process record

- **VRAM gate metric**: the 21 GiB STOP reads **torch `max_memory_allocated`** (docs/09 §5.1's
  method, `peak_vram.txt`); reserved (what nvidia-smi "used" ≈ reserved + CUDA context) runs
  ~3 GiB higher and is **not** the gate — but reserved is what actually OOMs, so ~23.5 GiB reserved
  is the OOM-risk line. A watcher comparing nvidia-smi to the 21 line false-alarmed and cost one
  11-min kill + relaunch of the oracle run (on the record; watcher retired). Pre-flight per the
  supervisor: 1 optimizer step over the 24 **longest** oracle rows (max 14,419 tok) allocated only
  **13.43 GiB** — long rows alone are cheap; the mixed-length 30-step windows (18.4–19.0 GiB across
  conditions) set the real peak.
- **In-place edit ruling**: `phase5_train.py` was edited ~1 min after the synth-7b-nosum launch
  (additive `--save-only-model` flag, default False == the `TrainingArguments` default). Driver
  never touched while live. Supervisor ruled the instance harmless (one python invocation per run,
  source compiled at start) but the never-edit-a-running-script rule stands absolute; that run's
  gate output was read line by line (clean). Caveat kept: a traceback would have quoted the edited
  file's lines (linecache) against the old line numbers.
- `bitsandbytes` 0.50.2 added to the stack (`adamw_8bit` requires it; Phase 2 ran fused AdamW).
- **Checkpoint-trail gap, noted not silent** (2026-09-05): surr-7b's full-run completion CHECKPOINT
  was initially skipped (the numbering jumped from its probe to the LoRA probe); sent as
  CHECKPOINT 18b on the supervisor's flag, with the run's full numbers and the flat-curve answer.
- Adjudication dates: eviction-path erratum 2026-09-04 (CHECKPOINT 0 audit) · boundary fix +
  `enable_thinking` pin 2026-09-04 (CHECKPOINT 2, both ratified) · VRAM metric + false-alarm kill
  2026-09-04 · in-place-edit ruling 2026-09-04 · `save_only_model` on both surr cells 2026-09-04
  (pre-ruling) / applied 2026-09-05 · LoRA fraction gate 2026-09-05 (CHECKPOINT 15 audit).

---

## 3. The slate — 10 trainings, every gate green, none skipped

> **Read this before the table: final training loss is NOT comparable across conditions and must
> never be read as a quality ranking.** The targets differ in token count, content difficulty and
> the fraction of the sequence under loss; "answer-only trained best" is a meaningless reading of
> that column. Loss is a per-run health signal (falling, finite, not collapsing) — **only Phase 6
> benchmark accuracy ranks conditions.** The one controlled pairing here is surr-1.5b vs surr-7b
> (§8); the FFT-vs-LoRA pair compares **each method at its standard lr (1e-5 / 1e-4)**, not one lr.

Full 3-epoch runs (probe → run per condition; rates are TRL's own `num_tokens` counter over wall
time; VRAM = `max_memory_allocated` / `max_memory_reserved`; loss = mean training loss at each
epoch boundary, then the run's final average; artifacts under
`bench/results/phase5/students/<condition>/`):

| condition | wall | steady tok/s | VRAM alloc/res GiB | loss ep1 / ep2 / ep3 | final avg | token acc | artifact |
|---|---:|---:|---|---|---:|---:|---|
| answer-only | 0.68 h | 3,222 | 18.10 / 21.10 | 0.464 / 0.366 / 0.303 | 0.359 | 0.917 | 3.78 GB |
| summary-answer | 1.14 h | 3,516 | 18.17 / 21.17 | 0.596 / 0.527 / 0.469 | 0.525 | 0.849 | 3.78 GB |
| oracle | 2.08 h | 3,484 | 19.00 / 22.35 | 0.416 / 0.389 / 0.325 | 0.382 | 0.905 | 3.78 GB |
| synth-7b-sum | 3.05 h | 3,476 | 18.45 / 21.55 | 0.333 / 0.300 / 0.278 | 0.296 | 0.901 | 3.78 GB |
| synth-7b-nosum | 2.92 h | 3,460 | 18.42 / 22.02 | 0.351 / 0.303 / 0.274 | 0.299 | 0.901 | 3.78 GB |
| synth-1.5b-sum | 3.01 h | 3,505 | 18.44 / 21.74 | 0.317 / 0.281 / 0.263 | 0.283 | 0.900 | 3.78 GB |
| synth-1.5b-nosum | 2.92 h | 3,506 | 18.45 / 21.57 | 0.320 / 0.290 / 0.261 | 0.287 | 0.902 | 3.78 GB |
| surr-1.5b † | 3.14 h | 3,485 | 18.42 / 21.21 | 0.401 / 0.395 / 0.376 | 0.395 | 0.875 | 3.78 GB |
| surr-7b † | 3.29 h | 3,482 | 18.40 / 21.91 | 0.387 / 0.370 / 0.351 | 0.382 | 0.874 | 3.78 GB |
| synth-7b-sum-lora | 3.06 h | 3,455 | **8.22 / 9.48** | 0.331 / 0.287 / 0.239 | 0.278 | 0.915 | adapter, 250 MB |

† ran with `save_only_model=True` (weights-only epoch checkpoints, rotation spike ~3.8 GB instead
of ~20 GB) — the supervisor's disk adjudication; changes nothing about the training math, costs
exact crash resume on two ~3.2 h runs. Everything else: default checkpoints (measured **10 GB**
per checkpoint dir at rotation), `save_total_limit=1`, all checkpoint dirs deleted after the
weights-only final save.

Per-run artifacts under `bench/results/phase5/students/<condition>/`: weights-only final model
(`model.safetensors` + config + tokenizer), `log_history.json`, `peak_vram.txt`, `run.json`
(and `probe/` with `probe.json`, `row0.txt`). Gates green on every run: NaN 0 · loss decreased over
epoch 1 · no near-zero collapse (leakage smell) · alloc ≤ 19.00 < 21 · load check
(`Qwen3_5ForCausalLM.from_pretrained`, 1,881,825,088 params) · checkpoint dirs deleted · disk ≥ 10 GB
before every run.

**The saved students are TEXT-ONLY checkpoints**: 1,881,825,088 params = Qwen3.5-2B minus its
vision tower (2.274 B − ~0.39 B), 3.78 GB bf16 — the expected artifact of the `Qwen3_5ForCausalLM`
load. They load via `Qwen3_5ForCausalLM` (transformers) / the `Qwen3_5ForCausalLM` registry entry
(vLLM, docs/06 §1.5), **not** the multimodal class.

---

## 4. Budget — measured

| step | measured |
|---|---:|
| step 0 eviction + step 1 formatter/self-tests (CPU) | ~0.5 h, no GPU |
| 10 probes (30 steps each) + retakes | ~2.0 h |
| oracle VRAM pre-flight + false-alarm kill segment | ~0.3 h |
| 7 core full runs | **15.80 h** |
| surr-1.5b + surr-7b full runs | **6.43 h** |
| synth-7b-sum-lora full run | **3.06 h** |
| **phase GPU total** | **≈ 27.3 h** (25.29 h full runs + ~2.0 h probes/pre-flight/false-alarm) |

docs/16 §3 budgeted core 7 ≈ 33–37 h / all 10 ≈ 43–49 h from Phase 2's 2,100 tok/s (4B bf16 LoRA);
the 2B FFT realized **3,222–3,516 train tok/s** on every condition, so the phase landed at roughly
half the working estimate. Ceiling 50 h projected / 55 h hard: never approached.

Disk: 59 GB free after step 0's eviction (the victim GGUF 15 G + R1-Distill-7B F16 15 G +
R1-Distill-1.5B BF16 3.4 G, user-authorized 2026-09-04, deleted from
`~/trace-inversion-bench/models/`); **19 GB free at phase end** (≥ 10 GB end gate ✅); low-water
~21 GB (surr-7b rotation, `save_only_model`). Reserve lever (user-authorized in-session, **not
used**): the two LiquidAI HF-cache repos (~2.7 GB).

---

## 5. Skipped cells

None — all three conditional cells ran (the §4.1 gate was met with ~24 h of headroom).

---

## 6. What Phase 6 needs — REQUIRED items first

| item | note |
|---|---|
| **Serve every student with `enable_thinking=True`** | REQUIRED. The students are trained to continue after the open `<think>\n`; the 2B template's default renders a **closed** think block (§2.2). vLLM: pass it via `chat_template_kwargs` per request (probe by render-and-diff, docs/11 §5 — never try/except) |
| **Re-measure the 2B baseline under the Phase 6 protocol with `enable_thinking=True`** | REQUIRED before ANY before/after comparison. Phase 0's student baseline (MATH500 79.0 / JEEBench 47.8) used `eval_baseline.py`'s template-default render = **no-think**; against thinking-mode students it would inflate every apparent gain. Historical note: Phase 0's candidate table mixed renders (the 4B's 91.4 was thinking-ON, the 2B's 79.0 was not) |
| Student manifest | `bench/results/phase5/students/{answer-only,summary-answer,oracle,synth-7b-sum,synth-7b-nosum,synth-1.5b-sum,synth-1.5b-nosum,surr-1.5b,surr-7b}/` — 3.78 GB each, text-only, `Qwen3_5ForCausalLM`; `students/synth-7b-sum-lora/` — PEFT adapter (r64/α128, 250 MB) over `Qwen/Qwen3.5-2B` |
| Eval protocol | MATH500 + JEEBench (+ LCB if time), vLLM bf16, paper sampling 0.7/0.9/1.05, seed 1234, Phase 0's harness — **plus the two REQUIRED rows above** |
| The confound reading | oracle-vs-forged gaps carry length (§1: targets 1.8–2.0× as-trained), register and consistency differences — `phase4.md` §10's table is the caveat block |
| The length-control trigger | unchanged: if any forged student ≥ the oracle student anywhere, the length-matched control becomes a Phase 6.5 proposal through docs/10's four questions |
| The 2B's looping habit | Phase 0 flagged loop-y generations; carry Phase 4's strict loop test into the harness |

---

## 7. `docs/09` rows

**7.23** (the supervision construction: think-slot unification, `b*` in the reasoning slot,
empty-think answer-only, INSTR on training prompts, two-segment tokenization,
`enable_thinking=True` pinned) and **7.24** (intersection + n-matched Surrogate-Trace vs the
paper's full-split training) — added; see `docs/09`.

---

## 8. Observations — reported, not acted on

- **Both Surrogate-Trace curves are markedly flatter than every victim-supervised condition.**
  Epoch-1→3 loss drops, exact: answer-only **34.7 %** · oracle **21.9 %** · synth-7b-nosum
  **21.9 %** · summary-answer **21.3 %** · synth-1.5b-nosum **18.4 %** · synth-1.5b-sum **17.0 %**
  · synth-7b-sum **16.5 %** — against surr-7b **9.3 %** and surr-1.5b **6.2 %**. (An earlier
  in-flight "core dropped 25–40 %" was loose and is corrected here; the clean split is
  victim-supervised 16.5–34.7 % vs surrogate-supervised 6–9 %.) The surr-7b run answered the
  supervisor's paired diagnostic: flatness is a **Surrogate-Trace-condition property, not a
  1.5B-arm property** — this pairing (same construction, same split-A sampling, only surrogate
  strength varies) is the one place a cross-condition loss comparison is controlled enough to mean
  anything.
- **Hypothesis, PRE-REGISTERED AND PARTIALLY FALSIFIED (supervisor, before any Phase 6 accuracy
  exists):** that the drop ordering tracks how much of the trace is *determined by the prompt*
  (its conditional entropy given the supervision context) — predicting synth (answer-conditioned
  forgeries re-deriving a known `y`) steepest, oracle in the middle, the surrogates' exploratory
  traces flattest. **The prediction failed within the victim-supervised group**: oracle's 21.9 % ≥
  three of the four synth cells — the middle does not order. The extremes (answer-only 34.7 % with
  no trace at all; surr 6–9 %) are consistent with prompt-determinedness, but trivially so. The
  failed prediction stays visible here because a prediction made before the numbers and
  contradicted by them is worth more on the record than a tidied-up story.
- **The supported victim-vs-surrogate split carries a three-way confound, so it cannot be
  attributed to trace entropy**: the two surr cells differ from the seven victim-supervised cells
  in trace source, in answer source (the surrogate's own R1-style boxed `y'` vs the victim's
  INSTR-conditioned worked `y`), **and in row set** — surr samples full split A unselected, while
  every victim cell trains on the split-B intersection, which is selected for four-way inverter
  termination, i.e. the shorter, easier rows (`phase4.md` §4's measured selection; visible in §1
  as the synth token deficit). An easier row mix fits faster; that is the more parsimonious
  candidate (docs/10 question 3 applies). Consequence, which stands on the general point rather
  than any ranking: **a steep or low training loss on forged traces means they are easier to
  imitate, which is orthogonal to whether imitating them teaches reasoning** — the paper's own
  denoising mechanism (Fig. 6) showing up as trainability. If Phase 6 finds synth ≥ oracle, this
  sits beside the length confound in any skeptical reading; if synth < oracle, it is part of the
  explanation.
- Token accuracy at the final epoch: ~0.90 on every victim-`y` condition, 0.874–0.875 on both surr
  conditions; summary-answer lowest at 0.849 (the `b*` slot is the hardest content to predict).
- **The FFT-vs-LoRA pair is pipeline-controlled, provably**: the LoRA twin's step-1 probe loss
  equals the FFT twin's to four digits (0.4277 vs 0.4277) — same file, same shuffle, same first
  accumulation window. The pair compares **each method at its standard lr** (FFT 1e-5, the paper's
  value; LoRA 1e-4, the ~10× LoRA convention docs/06 §4.9 pins) — a methods comparison, not a
  same-lr ablation.
- The two degenerate-`b*` rows (idx 64731, 6814 — `phase4.md` §0) trained as-is in summary-answer
  and the sum-setting synth conditions, as decided.

---

## 9. Definition of done (docs/16 §10)

- [x] step-0 eviction reported; 59 GB free before the first training (≥ 50 gate)
- [x] `phase5_format.py` + self-tests committed; 9 datasets × 3,616 rows; `format-stats.json`
      committed; truncation 0; ORACLE opened only by the oracle builder
- [x] `phase5_train.py` + driver committed; every training probed (30 steps) before its full run;
      CHECKPOINTs 0–20 to `big-boss`; objection windows honored; mid-run status on every >3 h run
- [x] core 7 trained and saved weights-only, all load checks green; conditional cells: **all three
      ran** (adjudications: save_only_model on the surr cells; LoRA fraction gate)
- [x] per-training numbers in this file; skipped cells: none
- [x] `docs/09` 7.23–7.24; `docs/10` Phase 5 closed with measured hours; README row
- [x] checkpoint dirs deleted; disk ≥ 10 GB at the end; every gate's output on the record
