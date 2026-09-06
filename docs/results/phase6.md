# Phase 6 — Evaluate the Students (the Table 3 analogue)

Ran 2026-09-05 → 2026-09-06 on the RTX 4090, branch `phase6-eval`, executor session
`phase-6-playa`, supervisor `big-boss` (planned the phase, audited every checkpoint, wrote
nothing to the tree). Protocol: Phase 0's harness `bench/eval_baseline.py` with **one additive
flag** (`--enable-thinking`), 1,015 tasks (MATH500 500 + JEEBench 515), vLLM bf16,
`--max-tokens 32768 --max-len 40960 --gpu-frac 0.90`, temp 0.7 / top_p 0.9 / rep 1.05, seed 1234
(variance cells 1235/1236). Per-model render gate before any generation; per-run audit
(`bench/phase6_audit.py`); raw text kept on every row (`bench/regrade.py` applies offline).

## 1. The harness change — the whole diff

Default off = the kwarg is **not passed at all**, so Phase 0 renders are byte-for-byte identical.
Verified by render-and-diff on the baseline (never try/except — unknown kwargs vanish into the
Jinja context, docs/11 §5): the no-flag render ends `<think>\n\n</think>\n\n` (closed), the
flagged render ends `<think>\n` with no `</think>`, and the two differ.

```diff
@@ -139,6 +139,11 @@ def main():
     ap.add_argument("--seed", type=int, default=1234)
+    # Phase 6: students are trained to continue an OPEN <think>\n block, but the
+    # Qwen3.5-2B template default renders it closed (phase5.md §2.2). Off = the
+    # kwarg is not passed at all, so Phase 0 renders stay byte-for-byte.
+    ap.add_argument("--enable-thinking", action="store_true",
+                    help="pass enable_thinking=True to apply_chat_template")
     args = ap.parse_args()
@@ -148,10 +153,11 @@ def main():
     tok = AutoTokenizer.from_pretrained(args.model)
+    tmpl_kwargs = {"enable_thinking": True} if args.enable_thinking else {}
     prompts = [
         tok.apply_chat_template(
             [{"role": "user", "content": t["prompt"]}],
-            tokenize=False, add_generation_prompt=True,
+            tokenize=False, add_generation_prompt=True, **tmpl_kwargs,
         )
         for t in tasks
     ]
```

## 2. The table

TBD — filled when all cells land. Columns per row: MATH500 acc (completed-only · truncated ·
no_answer · median gen tokens · strict loops), same for JEEBench, Δ vs `baseline-think`,
Δ vs no-think baseline, wall clock. Reference rows cited beneath with the engine-boundary caveat.

### The caveat block (travels with the table, not an appendix)

Every oracle-vs-forged gap in this table carries, besides trace content:

- **Length**: the four synth conditions trained on completions at **1.8–2.0×** the oracle's
  median on the same rows (2,873–3,168 tokens vs 1,577; `phase5.md` §1, `phase4.md` §4).
- **Register**: oracle = the victim's terse INSTR-conditioned working notes; forged = R1-Distill's
  "Okay, so…" monologue that re-derives the whole worked answer (`phase4.md` §10).
- **Cap asymmetry**: oracle `t` ≤ 14,336 with a real tail (p95 7.8–8.6k); forged `t̂` < 8,192 by
  construction (`phase4.md` §10).
- **Answer inconsistency**: ≈ 4–9 % of graded forged rows argue to a different answer than the
  `y` they trained beside; unfiltered (`phase4.md` §5).
- **Surrogate-Trace is three-way confounded** vs the victim-supervised cells: trace source,
  answer source (surrogate's own `y'` vs victim's worked `y`), and row set (full split-A sample
  vs the termination-selected split-B intersection) — `phase5.md` §8. Its training-loss flatness
  is equally explained by the harder row mix; a steep loss on forged traces means they are easy
  to imitate, which is orthogonal to whether imitating them teaches reasoning.
- **JEEBench Δs are read against a cap-bound baseline**: `baseline-think` truncates 65.6 % of
  JEEBench at the protocol cap, so part of every student's JEE gain is learning to terminate
  (docs/09 §7.7); the loop decomposition in §5 quantifies that share. **Finding-3 caveat
  (JEEBench row only)**: of the baseline's 338 truncated JEE rows, strict loops account for
  48.8 % — just under half — so up to ~51 % (loose-only 33.1 % + clean-unfinished 18.0 %) could
  in principle terminate at some higher budget; the cap is protocol-fixed, so the row stands as
  the model's real behavior under the protocol, flagged. MATH500's exposure is 28
  clean-unfinished rows of 500 (~5.6 % of the bench) — noted, no block needed.

## 3. The two baseline rows — instill vs improve (docs/09 §5.2)

| | MATH500 | JEEBench | render |
|---|---|---|---|
| 2B no-think (Phase 0, cited) | 79.0 | 47.8 | template default = closed think block |
| 2B thinking (measured here) | 67.8 | 33.8 | `enable_thinking=True`, open block |

Thinking mode, untrained, is a **net cost** to this model: −11.2 / −14.0. Mechanism: it loops to
the 32,768 cap (65.6 % of JEEBench, 32.0 % of MATH500; median JEE generation = exactly the cap;
no_answer 299/145 is almost entirely truncated-before-`\boxed{}` — among completed rows,
no_answer = 0.0 % on both benches). The shape is bimodal: completes → almost always right,
loops → loses. Consequently every student row shows **both deltas**: Δ vs `baseline-think` is the
protocol-correct before/after (inflated by termination effects); clearing the **no-think** bar is
what "taught reasoning" requires; clearing only `baseline-think` claims "taught termination of
its own thinking mode". Completed-only accuracy is **not quoted** for the baseline-think row
(truncation 50 % ≫ the ~20 % bias line, docs/09 §7.6); it is quotable for student rows under
that line, always beside their truncation.

Historical context, different instrument: Phase 0's no-think 2B truncated 221 JEE rows with 144
"clear loops" by the own-answer-repeat counter (baselines.md Finding 2, median 3,636 repeats).
Those figures are a different measurement on a different render — context only, never in a column
with this phase's `phase4_draws.loops` counts.

## 4. The reading, in order (docs/17 §6)

TBD — written when the table is complete. Order fixed: (1) trace conditions vs the two no-trace
floors; (2) synth vs its arm's Surrogate-Trace; (3) synth vs oracle — the length-control trigger;
(4) sum vs nosum, 7B arm vs 1.5B arm; (5) FFT vs LoRA twin + truncation/loop deltas
(termination-vs-reasoning); (6) the variance band, quoted next to every gap smaller than ~2× its
width.

Per-cell trigger reading already fixed (supervisor-ratified language): **synth-7b-sum did not
reach oracle on any quotable number; its one crossing (completed-only JEEBench +1.4) sits past
the ~20 % completed-only bias line (30.7 % truncation) and is reported, not counted.** Rationale:
at 30.7 % vs 9.2 % truncation the two completed-only populations are differently selected — the
synth survivor set is the easier 69 %, so the comparison is uninterpretable in synth's favor by
construction (docs/09 §7.6's bias mechanism). The phase trigger verdict stays open until all four
synth cells land. First-class finding, not a footnote: synth-7b-sum's truncation asymmetry vs
oracle (JEE 30.7 % vs 8.0 %; strict loops 3–4×) is the trained-length confound expressed as
behavior — forged supervision taught longer, loopier generation.

## 5. Audits

### Looping, pre/post (docs/09 §7.7)

TBD — full table when all cells land. Instrument: `phase4_draws.loops` on each row's full
generated text; strict = coverage 0.2 (the count), loose = coverage 0. Baseline decomposition of
truncated rows (strict / loose-only / neither): JEEBench 165 / 112 / 61 of 338; MATH500
78 / 54 / 28 of 160.

### JEEBench Numeric spot-check (baselines.md's "spot-check before Phase 6")

Closed. On `baseline-think`: all 5 sampled graded-correct Numeric rows are exact matches
(`2→2`, `10→10`, `19→19`, `8→8`, `1→1`); all 5 sampled graded-wrong rows have `pred = NaN` —
extraction-empty on truncated rows, not a tolerance bug. JEE-by-type on the same run: Integer
41.5 / MCQ 40.0 / MCQ(multiple) 27.4 / Numeric 32.8 — uniformly depressed by truncation, no type
at zero. Phase 0's "Numeric 66.4 % vs MCQ 98.2 %" worry does not reproduce as a grading artifact;
the retired open item is on the record.

### Extraction hygiene

no_answer among completed generations = 0.0 % on both benches, every run so far.

## 6. Budget — projected vs realized, and the ceiling history

- docs/17 set ~18–26 h working, 30 h projected gate / 35 h hard STOP; variance cells behind the
  30 h gate. The supervisor moved the variance gate to 33 h (planning gate vs a 2.5 h standing
  commitment); realized eval costs then pushed the mandatory slate alone to ~34 h, and **the user
  raised the hard ceiling to 40 h on 2026-09-06 specifically to keep docs/09 4.5's three-seed
  commitment** ("Run both seeds").
- `baseline-think`: projected 4.8 h (stated assumption ~1,400 t/s effective from Phase 0's
  realized rate), realized **13.31 h** — 2.77×, drain-to-completion ruled mid-run with a 16 h
  backstop. Mechanism, named per ruling: a 50-prompt probe understates the capped fraction
  (probe 50 % overall → realized 49 %, but JEE 64 %→65.6 % with far heavier tail cost), and
  thin-batch 25–30k-deep decode dominates the tail. A predicted completion "cap cascade" did not
  occur — vLLM admits sequences as KV frees, so caps hit staggered; the wrong prediction stands
  corrected on the record. Phase 6's instance of "a sweep ranks; it never budgets": the probe
  ranks sanity, it does not budget a cap-bound run.
- `oracle`: projected 25–40 min, realized 1.99 h — the 2.5× wire tripped **post-hoc** (run
  finished between wake-ups). Corrective adopted: trace-cell projections state an assumed
  truncation fraction (8–12 %+) and a p95-at-cap term, and a deadline wake at 1.2× projection
  keeps the wire live.
- Per-run walls: TBD table (summary.json holds the seconds).

## 7. Process record

- **Serving construction, both pins.** (a) Every eval renders with `enable_thinking=True` via the
  one flag (docs/09 7.25; training-side pin is 7.23). (b) **The FFT checkpoints do not serve
  as saved**: transformers 5.16's `save_pretrained` reverts its load-time key conversion, so all
  9 Phase 5 FFT checkpoints carry `model.language_model.*` tensor names under a
  `Qwen3_5ForCausalLM` config — vLLM 0.27.1's text-only loader rejects them ("no module or
  parameter named 'language_model'"; the exact trap `phase2_train.py`'s merge comment recorded
  2026-08-28). Phase 5's load check passed because transformers re-applies the conversion on
  load: **"loads in transformers" never proved "serves in vLLM."** Remedy (supervisor-ruled):
  disposable serving copies via `bench/phase6_serve_copy.py` — prefix stripped, tensors
  byte-identical, originals untouched, gates asserted (320 keys / zero `language_model` /
  module-tree / on-disk byte-equality), one copy on disk at a time, deleted after its run's
  gates. The VL-wrapper alternative was ruled out on evidence: the base checkpoint is a real VL
  model (297 vision keys + `vision_config`); the students have neither.
- **The baseline serves through a different vLLM class than the students** — inherent to the
  design, not the fix: the HF-cache base resolves to `Qwen3_5ForConditionalGeneration` (full VL
  checkpoint); students serve text-only `Qwen3_5ForCausalLM`. Same text stack either way.
- **docs/17 errata**, ratified: (a) §0 calls `audit_results.py` "Phase 0's own-answer repeat
  counter" — the file has no repeat counter (Finding 2's numbers came from ad-hoc analysis never
  committed); the §7.7 comparison rides on `phase4_draws.loops` strict+loose applied identically
  to every run. (b) "everything this phase runs in `.venv-vllm`" — the LoRA merge needs peft,
  which `.venv-vllm` lacks; the merge runs CPU-side in `.venv` (peft 0.20.0, the adapter's own
  version; nothing installed or upgraded anywhere).
- **Host tooling, on the record with causes.** (a) One probe launch was killed by the harness's
  low-memory watchdog seconds in — transient, the dead baseline engine's pages still draining.
  (b) Three further launches were killed by the same watchdog on a **false** signal: an
  instrumented sampler recorded MemAvailable 22.3 GB / MemFree 9.5 GB at kill time, and the
  watchdog killed the 2 MB sampler itself. Same launch pattern had run the 13.3 h baseline.
  Workaround, supervisor-ratified, environment-layer only: runs launch via `setsid`, detached
  from the harness task system — driver and its canonical `bench/logs/phase6/` tee logs
  unchanged; every detached pid recorded in `bench/logs/phase6/detached.pids`; kill discipline
  verified. Environment delta vs the attached baseline run: `TORCHINDUCTOR_COMPILE_THREADS=4`
  (startup-layer only, zero sampling/render surface); nothing else rides in the detached
  environment beyond the driver's standing exports. The watchdog false-positive was surfaced to
  the user on both channels. Kernel-OOM (as opposed to watchdog) was pre-declared a real STOP;
  it never occurred.
- **One 2 h network stall**: the oracle probe's render gate hung inside the benchmark-parquet
  HTTPS fetch (no timeout in the stack), self-recovered; no GPU time lost, +2 h wall. Single
  occurrence in ~12 fetch-pairs; accepted and monitored rather than engineered around.
- **ORACLE hygiene**: nothing in Phase 6 opens or names `victimB-ORACLE.jsonl`. Interim grep
  (2026-09-06, mid-phase): the five Phase 6 scripts contain zero occurrences of "oracle" in any
  case (condition names arrive only as CLI arguments); logs and outputs contain zero
  `victimB-ORACLE` hits. Re-run at close (gate at §8).

## 8. Definition-of-done gates

TBD at close: oracle-grep clean · summary.json committed with all rows · disk ≥ 8 GB throughout ·
per-run render gates all green (they were, per-model, in `bench/logs/phase6/*.gate.log`).
