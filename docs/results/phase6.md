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

n = 1,015 per run (MATH500 500 + JEEBench 515), seed 1234 unless stated. **acc is never read
without the truncation beside it**; completed-only (in parentheses) is quotable only where
truncation ≤ ~20 % (docs/09 §7.6) — cells past that line show it struck through in spirit:
marked †, reported not counted. Δ columns are (MATH/JEE) percentage points.

| condition | MATH500 acc (done-only) · trunc | JEEBench acc (done-only) · trunc | Δ vs think | Δ vs no-think | wall |
|---|---|---|---|---|---|
| 2B no-think (Phase 0, cited) | **79.0** · — | **47.8** · — | +11.2/+14.0 | — | — |
| **baseline-think** | **67.8** (—) · 32.0 % | **33.8** (—) · 65.6 % | — | −11.2/−14.0 | 13.31 h |
| answer-only | 54.2 (54.6) · 0.8 % | 22.9 (23.0) · 0.6 % | −13.6/−10.9 | −24.8/−24.9 | 0.23 h |
| summary-answer | 54.8 (55.4) · 1.0 % | 21.6 (21.7) · 0.6 % | −13.0/−12.2 | −24.2/−26.2 | 0.31 h |
| **oracle** (Victim-Trace) | **73.4** (80.6) · 9.2 % | **45.6** (49.6) · 8.0 % | +5.6/+11.8 | −5.6/−2.2 | 1.99 h |
| synth-7b-sum | 64.4 (75.2) · 14.4 % | 35.5 (51.0†) · 30.7 % | −3.4/+1.7 | −14.6/−12.3 | 3.76 h |
| synth-7b-nosum | 67.0 (76.0) · 11.8 % | 31.8 (44.7†) · 29.1 % | −0.8/−2.0 | −12.0/−16.0 | 3.48 h |
| synth-1.5b-sum | 61.0 (80.7†) · 24.4 % | 24.9 (48.8†) · 49.5 % | −6.8/−8.9 | −18.0/−22.9 | 2.18 h |
| synth-1.5b-nosum | 61.2 (81.1†) · 24.8 % | 22.5 (39.4†) · 43.3 % | −6.6/−11.3 | −17.8/−25.3 | 1.78 h |
| surr-1.5b (Surrogate-Trace) | 62.2 (76.0) · 18.2 % | 32.4 (52.2†) · 37.9 % | −5.6/−1.4 | −16.8/−15.4 | 1.52 h |
| **surr-7b** (Surrogate-Trace) | **72.0** (81.4) · 11.6 % | **45.4** (54.5) · 16.7 % | +4.2/+11.6 | −7.0/−2.4 | 0.88 h |
| synth-7b-sum-lora (FFT twin above) | 68.0 (75.2) · 9.6 % | 35.5 (49.9†) · 28.7 % | +0.2/+1.7 | −11.0/−12.3 | 1.09 h |
| synth-7b-sum · seed 1235 | 67.8 (81.9) · 17.2 % | 37.9 (52.7†) · 28.2 % | 0.0/+4.1 | −11.2/−9.9 | 1.24 h |
| synth-7b-sum · seed 1236 | 65.8 (78.1) · 15.8 % | 32.6 (48.8†) · 33.6 % | −2.0/−1.2 | −13.2/−15.2 | 1.36 h |

**Seed band** (synth-7b-sum, three seeds): MATH 64.4 / 67.8 / 65.8 — range **3.4** (sd 1.7);
JEEBench 35.5 / 37.9 / 32.6 — range **5.3** (sd 2.7). Every gap below ~2× the range is flagged in
§4's reading.

Reference rows, cited not re-run (GGUFs evicted in Phase 5; llama.cpp engine-boundary caveat,
docs/10): victim @ medium+INSTR **97.2 / 82.0** (n=250, `phase3.md` §6); surrogate 7B
**92.6 / 60.6**, surrogate 1.5B **84.0 / 32.6** (Phase 0). The only cross-engine comparison in
the project remains the Phase 0 headroom check; every number in the table above is vLLM bf16
under one harness.

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

**0. Instill vs improve, first, because it frames everything below (docs/09 §5.2): no cell in
the table clears the no-think bar — every Δ-vs-no-think is negative, including oracle's
(−5.6/−2.2) and surr-7b's (−7.0/−2.4).** Nothing in this phase taught the 2B to reason better
than it already could without a think block; what training demonstrably taught is *termination
of its own thinking mode* (the Δ-vs-think column, and §5's loop table). Every positive claim
below is a claim about recovering thinking-mode's self-inflicted cost, not about exceeding the
base model.

1. **Trace vs the floors.** Every trace condition beats both no-trace floors on MATH500
   (61.0–73.4 vs 54.2/54.8, gaps ≥ 6.2, outside 2× band). On JEEBench every trace condition
   beats the floors EXCEPT synth-1.5b-nosum, which lands **below** answer-only (22.5 vs 22.9;
   summary-answer is 21.6) — inside the band, i.e. indistinguishable from a floor. Traces teach something the answer
   alone does not — except at the weak arm's worst cell, where inversion washes out entirely.

2. **Synth vs its arm's Surrogate-Trace — the paper's core claim, per arm.** It fails in both
   arms. 7B arm: surr-7b 72.0/45.4 beats synth-7b-{sum,nosum} by **+5.0–7.6 MATH / +9.9–13.6
   JEE** (all outside 1× band; 7.6 MATH and 13.6 JEE also outside 2×, while 5.0 MATH and 9.9 JEE
   sit inside 2×). 1.5B arm: surr-1.5b 62.2/32.4 beats synth-1.5b-{sum,nosum} by
   +1.0–1.2 MATH (inside band) and **+7.5–9.9 JEE** (outside 1× band, inside 2×). Inversion of the victim's hidden
   traces added **no measurable value over plain distillation of the surrogate** on MATH and
   **clearly negative value on JEEBench, on both arms**. The §3 caveat applies in full: the
   Surrogate-Trace cells differ three ways (trace source, answer source, row set), so "surrogate
   distillation is better" and "the forged traces' length/register/inconsistency hurt" are both
   live explanations; what is settled is that the *inversion pipeline as built* underperforms
   the trivial alternative it was meant to beat.

3. **Synth vs oracle — the length-control trigger.** **No synthesized-trace cell reached the
   oracle student on any quotable number, anywhere: the trigger does NOT fire, and the
   length-matched control is not proposed.** MATH gaps 5.4–12.4 (smallest: synth-7b-sum-lora at
   −5.4, ≈1.6× the band range — flagged); JEE gaps 7.7–23.1 (all ≥ 1.5× band, most ≥ 2×).
   Supervisor-ratified per-cell language, kept verbatim: synth-7b-sum did not reach oracle on
   any quotable number; its one crossing (completed-only JEEBench +1.4) sits past the ~20 %
   completed-only bias line (30.7 % truncation) and is reported, not counted — at 30.7 % vs
   9.2 % truncation the two completed-only populations are differently selected (the synth
   survivor set is the easier 69 %), so that comparison is uninterpretable in synth's favor by
   construction. First-class finding, not a footnote: the synth cells' truncation asymmetry vs
   oracle (JEE 29–50 % vs 8.0 %; strict loops 2–4×) is the trained-length confound expressed as
   **behavior** — forged supervision taught longer, loopier generation, ordered by arm.

4. **sum vs nosum; 7B arm vs 1.5B arm.** The compressed-summary conditioning `b*` adds nothing
   measurable: 7B sum-vs-nosum −2.6 MATH / +3.7 JEE, 1.5B −0.2 / +2.4 — every split inside ~1.4×
   the band. The **arm** effect is real and large: 7B-arm synth beats 1.5B-arm synth by
   +3.2–6.0 MATH / +6.9–13.0 JEE, and the truncation ordering follows (7B cells 29–31 % JEE,
   1.5B cells 43–50 %). This is the surrogate-strength sweep the paper never ran, and it
   propagates the arm's whole pipeline history (its inverters capped ~2× as often; its forged
   sets were rescued from loopier draw distributions) into student behavior, not just accuracy.

5. **FFT vs LoRA twin; termination vs reasoning.** The pipeline-controlled twins land on
   **identical JEEBench accuracy (35.5 = 35.5)**; the LoRA twin is +3.6 MATH (≈ the band) at
   two-thirds the MATH truncation (9.6 % vs 14.4 %). At each method's standard lr there is no
   FFT advantage — the ~4.6 GB/cell FFT cost bought nothing the 250 MB adapter didn't.
   Termination: training cured the baseline's cap-looping in proportion to how short the
   supervision was — answer-only/summary-answer truncate ≤1 %, oracle 8–9 %, 7B synth ~30 %,
   1.5B synth 43–50 %, against the untrained baseline's 65.6 % JEE. Since accuracy gains vs
   `baseline-think` track exactly this ordering, the Δ-vs-think column largely *measures
   termination*, which is why the no-think bar in reading 0 is the honest one.

6. **The variance band, against every close gap.** Band: MATH range 3.4 (sd 1.7), JEE 5.3
   (sd 2.7). Inside it: **surr-7b vs oracle (−1.4/−0.2) — a statistical tie**, achieved at half
   oracle's JEE truncation and with quotable completed-only on both benches (81.4/54.5 vs
   80.6/49.6 — both under the bias line, though survivor sets still differ in size); every
   sum-vs-nosum split; the FFT-vs-LoRA MATH gap; surr-1.5b's MATH edge over its synth siblings.
   Outside it: every oracle-vs-synth gap (one MATH cell flagged at 1.6×), every arm gap on JEE,
   every floor gap on MATH, and the baselines' thinking-mode cost.

## 5. Audits

### Looping, pre/post (docs/09 §7.7)

Instrument: `phase4_draws.loops` on each row's full generated text; strict = coverage 0.2 (the
count), loose = coverage 0. Applied identically to every run. Baseline decomposition of
truncated rows (strict / loose-only / neither): JEEBench 165 / 112 / 61 of 338; MATH500
78 / 54 / 28 of 160.

| run | MATH strict/loose · med tokens | JEE strict/loose · med tokens |
|---|---|---|
| baseline-think | 78 / 218 · 14,927 | 166 / 344 · 32,768 (= cap) |
| answer-only | 3 / 31 · 404 | 2 / 67 · 721 |
| summary-answer | 2 / 48 · 983 | 1 / 81 · 1,491 |
| oracle | 9 / 87 · 1,005 | 16 / 163 · 3,766 |
| synth-7b-sum | 21 / 121 · 1,162 | 36 / 260 · 5,564 |
| synth-7b-nosum | 20 / 115 · 790 | 40 / 262 · 5,120 |
| synth-1.5b-sum | 33 / 204 · 2,323 | 63 / 345 · 10,549 |
| synth-1.5b-nosum | 39 / 197 · 2,041 | 49 / 324 · 6,040 |
| surr-1.5b | 26 / 231 · 2,067 | 54 / 340 · 6,544 |
| surr-7b | 19 / 178 · 1,604 | 30 / 277 · 4,212 |
| synth-7b-sum-lora | 12 / 110 · 1,115 | 36 / 273 · 5,751 |
| seed 1235 / 1236 | 27 / 164 · 2,174 — 17 / 134 · 1,287 | 46 / 273 · 5,379 — 40 / 262 · 5,934 |

Every trained cell cuts strict loops 2–10× vs baseline; the residual ordering (oracle < surr-7b
< 7B synth ≈ LoRA < surr-1.5b < 1.5B synth) tracks trained supervision length and arm — the
§7.7 termination share of every Δ-vs-think, quantified.

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

- Ceiling history, three user-visible entries: docs/17 set ~18–26 h working, 30 h projected
  gate / 35 h hard STOP, variance behind the 30 h gate. The supervisor moved the variance gate
  to 33 h (planning gate vs a 2.5 h standing commitment). Realized eval costs pushed the
  mandatory slate alone to ~34 h and **the user raised the hard ceiling to 40 h on 2026-09-06
  specifically to keep docs/09 4.5's three-seed commitment** ("Run both seeds"). When trace
  cells realized at ~4 h each the mandatory slate projected ~48 h; execution stopped at the
  40 h wire and **the user raised the ceiling to ~60 h ("everything")** — mandatory + variance
  with drift margin. Realized end: ~36.1 h GPU-active, under even the original 40.
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
- **Measured totals**: 13 full runs 33.1 h + 13 probes 2.6 h + conversions/merge ~0.4 h ≈
  **36.1 h GPU-active**; ≈ 38.5 h wall including the 2 h network stall and ~40 min of
  host-watchdog kills and diagnosis. docs/17 estimated 18–26 h; the drivers of the overrun are
  the baseline's 13.31 h (vs 2–4 budgeted) and trace cells at 0.9–3.8 h each. Per-run walls are
  in the table (§2) and `summary.json`.
- **Probe accuracy is systematically inflated on JEEBench** and probes were never quoted as
  accuracy: `--limit` takes `head(25)` of each bench, and JEEBench rows are ordered by
  subject/index, so every probe's JEE subset is the same non-stratified, MCQ-heavy head block.
  Observed three times, same direction (probe→full: 44→32.4, 56→45.4, 40→33.8; MATH's 80→68 on
  the LoRA cell shows it reaches MATH when margins are thin). Probes remain what §4.4 used them
  for — gates and rate pricing. Documentation item, not a harness change.

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

## 8. Definition-of-done gates — all green at close (2026-09-07)

- ORACLE grep clean at close: zero `victimB-ORACLE` hits across all Phase 6 logs, outputs and
  scratch; the five Phase 6 scripts contain zero occurrences of "oracle" in any case.
- `summary.json` committed with all 13 rows; every row carries acc / completed-only / truncated /
  no_answer / median tokens / strict+loose loops / wall seconds.
- Render gate green on every model (13/13, `bench/logs/phase6/*.gate.log`); conversion gates
  green on all 9 serving copies; merge check green on the LoRA cell (max|Δ| 2.4e-3/2.9e-3,
  ~90 % elements changed, disk==merged verified).
- Every run probed first; every full run under a stated projection with truncation assumption;
  two projection trips disclosed (baseline 2.77×, oracle ~3×), both adjudicated on the record.
- Disk ≥ 8 GB throughout (minimum observed ~10 GB; 17 GB at close). All serving copies and the
  merged dir deleted after their gates; only `summary.json` + docs are committed, jsonl stays
  local per gitignore.
- Probe acc > 5 % on every trained student (lowest probe: 20 JEE); no cell required the
  ≤5 %-template-smell STOP.
