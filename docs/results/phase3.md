# Phase 3 — Query the victim (`victimB`)

Reproduction of *"How to Steal Reasoning Without Reasoning Traces"* (Zhang, Morris, Shmatikov,
arXiv 2603.07267v2), Stage 2 first half (§4.2): split B through the victim, `b* = π(t)`, and the real
trace `t` withheld from the attack by file layout (`docs/14` §4.5).

| | |
|---|---|
| Victim `V` | `Qwen3.8-27B-IQ4_XS.gguf` (14.62 GiB), llama.cpp b10450, Vulkan, `-fa on -ctk q8_0 -ctv q8_0 --jinja`, port 8079 |
| Serving shape | **12 slots × 16,384** (`-np 12 -c 196608`), from the sweep in §1 |
| Query | OpenThoughts user turn **+ the boxed-answer instruction**, at `--reasoning-effort medium` (§2–§4) |
| Sampling | temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · top_k −1 · `max_new_tokens` 14,336 · `--timeout 3600` |
| Prompts | `bench/results/phase3/promptsB.jsonl`, 9,000 rows in permutation order; 4 skipped over 5,000 chars |
| Drop rule | `finish_reason == "length"` or no post-think answer → dropped from **all four** split-B conditions |
| Compressor `C'` | `Qwen/Qwen3.5-4B`, `PI_SYSTEM` unchanged, seed 1234, `--max-tokens 2048`, `--max-len 20480` |
| Code | `bench/run_phase3_gen.sh` · `phase1_split.py --only-b` · `phase1_stats.py --vs-r1` · `bench/phase3_split.py` |

Data and logs live under the **main checkout** (`bench/results/phase3/`, `bench/logs/phase3-*.log`);
the branch was built in a worktree whose driver runs its own code with the main checkout's
`.venv-vllm` (`docs/14` §0).

> **`docs/14-phase3-handoff.md` is the pre-run plan; this file is authoritative where they
> disagree.** Its method, conventions and file layout all held. Several of its numeric expectations
> did not — the budget (30–50 h → 66.32 h), the cap-hit band (0–15 % → widened to 0–25 % after
> measuring 16.2 %), "no system prompt" (the template injected one until effort was set to `medium`)
> and the expected drop rate (≤ 5 % → 16.2 %). `docs/14` carries the same table at its head.

---

## 0. Split B, as written

`phase1_split.py --only-b` recomputed the seed-20260826 permutation, asserted
`sorted(perm[9000:18000]) == splits.json["B"]` and `A ∩ B = ∅` against the 16,000-row A on disk, and
wrote only `promptsB.jsonl`; `splits.json` and `promptsA.jsonl` are md5-identical before and after.

| | |
|---|---:|
| rows | 9,000 |
| domain mix | math 78.2 % · code 17.5 % · chemistry 1.2 % · physics 1.1 % · puzzle 1.0 % · biology 1.0 % |
| first 200 in permutation order | math 147 · code 41 · other 12 |
| prompt chars | median 260 · p95 1,844 · max 5,624 |
| skipped (> 5,000 chars, leave all four conditions) | **4**, all code: idx 89225 (order 82, code_contests, 5,624 ch), 91754 (3875, code_contests, 5,342), 100943 (3917, taco, 5,105), 101529 (4245, taco, 5,105) |

idx 89225 falls inside the first 200, so every "first 200 rows" probe below is orders 0–200 minus it.

---

## 1. Sweep — `docs/results/sweeps/Qwen3.8-27B-IQ4_XS-ctx16384-ntg4096.md`

`NTG=4096 bench/sweep_concurrency.sh Qwen3.8-27B-IQ4_XS.gguf 16384 8 12 16`, 2026-08-30 09:17.

| slots | total KV | gen t/s | total t/s | peak MiB | |
|---:|---:|---:|---:|---:|---|
| 8 | 131,072 | 140.64 | 157.03 | 20,741 | |
| **12** | **196,608** | **164.94** | **183.90** | 23,422 | the KV total Phase 0 ran at 6 × 32,768 |
| 16 | 262,144 | — | — | — | weights loaded, recurrent-state cache allocation failed |

Two loading points, ranked monotonically. **The sweep ranks; it does not budget** (`docs/11` §5) —
the realized rate is §4's measurement, and the swept 164.94 overstated it by 1.34×.

---

## 2. What the victim is actually sent — the template injects its own system turn

With `--no-system` the request carries only the user turn, but the GGUF's Jinja template renders a
system turn of its own whenever thinking is enabled (`reasoning_effort|default('xhigh')`):

```
<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions,
consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>
<|im_start|>user
<x><|im_end|>
<|im_start|>assistant
<think>
```

Phase 0's `eval_victim_gguf.py` also sent only the user turn through `--jinja`, so **this is the
victim Phase 0 benchmarked** — its MATH500 median of 593 tokens simply reflects an easy benchmark,
not a different configuration. Rendered through the live server's `/apply-template`, diffed against
the default (row 0):

| request | effect on the render |
|---|---|
| `reasoning_effort=medium` | the system block **disappears entirely** — the template has no instruction text for medium; the assistant turn still opens `<think>` |
| `reasoning_effort=high` | **byte-identical to xhigh** — the template maps `'high' → 'xhigh'` before emitting. Verified empirically, which is why no `high` probe was run |
| `reasoning_effort=low` | *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."* |
| a system message present, default effort | the xhigh line persists, then a blank line, then the caller's system text |
| user turn = `x` + `INSTR` | the instruction is appended inside the user turn; nothing else changes |

llama.cpp b10450 exposes this as `llama-server --reasoning-effort LEVEL` and per request as
`chat_template_kwargs`. **So `medium` is the setting that actually delivers `docs/14` §4.1's
"user turn only, no system prompt"**; the default (xhigh) silently adds a steering instruction.

Responses arrive as `{role, content, reasoning_content}` with no `<think>` tags in `content` — the
path `phase1_generate.py` splits on. Confirmed on a live request and by `raw` containing no
`</think>` on any kept row of any run below.

---

## 3. The effort sweep — three 200-row probes, paired on identical prompts

The first probe (default effort = xhigh, no `INSTR`) tripped two `docs/14` §6 STOP conditions:
cap-hit **37.0 %** (limit 15 %) and a projected **106–145 h** (limit 72 h). Escalated rather than
redesigned; the user's decision was to sweep effort before committing. Each level below is the same
200 prompts, 12 × 16,384, `x + INSTR`, same sampling; separate output files.

| level | cap-hit @14,336 | kept trace-tok med/mean/p95 | gen med | box-rate | y-vs-R1 (kept) | peak t/s | rows for 5,040 kept (% of B) | projected h @120/140/165 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| low | 19.0 % | 1,108 / 2,209 / 8,553 | 2,529 | 100.0 % | 68.7 % | 115.3 | 6,261 (70 %) | 74 / 64 / 54 |
| medium | 19.0 % | 1,360 / 2,705 / 10,473 | 2,814 | 99.4 % | 69.0 % | 119.6 | 6,222 (69 %) | 79 / 68 / 58 |
| xhigh (no `INSTR`) | 37.0 % | 1,889 / 3,492 / 11,750 | 8,151 | 56.3 % | 75.3 % | 121.1 | 8,000 (89 %) | 145 / 125 / 106 |

Findings:

- **Effort below xhigh halves cap-hit** (37 → 19 %); **low ≈ medium** on every column, so the entire
  gain is xhigh→medium and nothing is bought below medium.
- **`INSTR` drives boxing, not effort**: ~100 % of kept rows box with it, 56.3 % without. The
  register and boxing columns are therefore confounded between the xhigh row and the other two.
- **Intersection y-vs-R1** — the only effort-clean comparison, on the 73 prompts kept *and* gradable
  at all three levels: low **78.1 %** · medium **78.1 %** · xhigh **75.3 %** (medium right & xhigh
  wrong 6, the reverse 4). The per-column 69 %-vs-75 % gap was set composition: `INSTR` enlarged and
  hardened the gradable set. **No accuracy penalty at medium.**
- **Register**: low and medium produce structured first-person reasoning ("Let me work through each
  part carefully. **Part a)** …"), closer to R1-Distill and to π's expected input than xhigh's
  telegraphic notes. **Low is not too terse**; medium keeps ~19 % more trace content than low.

**Decision (user, 2026-08-30): `medium` + `INSTR`, full run to 5,040 kept, ~79 h explicitly
authorized past the 72 h STOP.** The deferred regime re-benchmark is §6.

---

## 4. Generation — the full run

`REFFORT=medium INSTR=… bench/run_phase3_gen.sh 12`, 2026-08-30 20:32 → 2026-09-02 14:51. Output
`victim-traces-ORACLE.jsonl`, seeded from the medium probe's 200 rows (identical config — the probe
is the run's prefix; the generator resumed at "200 rows already generated, 162 kept"). The xhigh
probe was renamed `victim-traces-ORACLE-xhigh-probe.jsonl` first so nothing appended onto it.

| | |
|---|---:|
| rows in file | **6,022** (5,822 generated this launch + the 200-row probe prefix) |
| rows kept | **5,045** (target 5,040) |
| cap-hit | **16.2 %** (977) — 976 `finish_reason == length` + **1** no-answer |
| request errors | **0** |
| prompts skipped (> 5,000 chars) | 4 |
| gen tokens (engine) | 29,513,891 |
| wall clock | **66.32 h** |
| realized throughput | **123.6 gen t/s at 12 slots** |
| server log | `truncated = [1-9]` **0** · `context shift` **0** · context errors **0** |

Zero request errors across 6,022 rows and zero real truncations: no prompt exceeded its slot, and the
watchdog and error-strip guards never had to fire.

**Rate note (`docs/11` §5: budget from the run, not the sweep).** The realized rate *rose* through
the run as it settled — 93 t/s at 90 min, 100 at 5 h, 110 at 15 h, 120 at 35 h, 123.6 final — so
early projections (82 h at the 90-minute mark) were pessimistic and the run landed at 66.3 h. The
swept 164.94 overstated the realized 123.6 by **1.34×**, a far smaller gap than the 7B surrogate's
2.4× (`docs/11` §5), consistent with a 27B being compute-bound rather than ragged-batch-bound.

### Lengths — Qwen3.5-4B tokenizer (stated; `gen_tokens` beside it is the engine's own count)

| | n | median | mean | p05 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| trace tokens, kept | 5,045 | **1,400** | 2,640 | 216 | 9,574 | 14,004 |
| answer tokens, kept | 5,045 | **605** | 648 | 218 | 1,155 | 6,565 |
| trace tokens, all rows (censored at the cap) | 6,022 | 1,935 | 4,533 | 237 | 14,336 | 14,339 |

The 14,339 maximum exceeds the 14,336 cap because llama-server capped *its own* tokens and we recount
the text with a different tokenizer — the engine's `finish_reason` is authoritative (`docs/11` §5).

By domain, kept rows (trace tokens):

| domain | n | median | mean |
|---|---:|---:|---:|
| math | 3,889 | 1,427 | 2,649 |
| code | 899 | 1,735 | 3,051 |
| physics | 63 | 1,157 | 2,053 |
| chemistry | 69 | 772 | 1,065 |
| biology | 60 | 666 | 744 |
| puzzle | 65 | 287 | 434 |

No domain has zero kept rows.

### Paired against R1's ground truth, identical prompts (n=6,022)

| | ours | R1 | Δ |
|---|---:|---:|---:|
| cap-hit at 14,336 | **16.2 %** | 9.9 % | +6.4 pts |
| kept p50 (both under cap, n=4,783) | 1,310 | 3,664 | **−64.2 %** |
| kept p75 | 3,021 | 6,447 | −53.1 % |
| kept p90 | 6,394 | 9,563 | −33.1 % |
| ours shorter on | **81.3 %** of prompts | | |

Drop decomposition: we drop 977 · R1 would drop 594 · both 332 · **ours-only 645 = 66 % of our drops,
10.7 % of all prompts** · R1-only 262. R1's p90 over all rows is 14,291, just under our 14,336 cap.

### Gates — `phase1_stats.py --mode traces`, both bands recorded

| band | verdict |
|---|---|
| `--cap-hit-band 0 15` (as `docs/14` specifies) | **FAIL — cap-hit 16.2 % outside 0–15 %**; every other check passed |
| `--cap-hit-band 0 25` (widened) | **PASSED** |

Both were run, the documented one first and deliberately, so its verdict is on the record. **This is
a threshold changed after seeing the data, not a passing gate.** The 0–15 band came from the xhigh
expectation of ≤ 5 % cap-hit (`docs/14` §4.2, extrapolating Phase 0's 1.2 %/9.9 %); the medium probe
then measured 19 % and the run 16.2 %. No other threshold was re-fitted — the paired-gap tolerance
(±100) and the error-rate limit (1 %) are untouched, and both passed on their original values.
Everything the band does not cover passed on the documented settings: **0 request errors, 0 kept rows
with an empty trace, no domain at zero kept, paired cap-hit gap +6.4 pts.**

### `y` vs R1's answer — report only, no filtering (`docs/14` §4.6)

| bucket | n |
|---|---:|
| agree | 2,784 |
| disagree | 1,025 |
| no box/bold in `y` | 1 |
| no box in R1's own solution | 1,235 |
| **agreement on gradable** | **2,784 / 3,809 = 73.1 %** |

**73.1 % is below the 75 % STOP floor, and the shortfall is measurement, not the victim.** Per
`docs/14` §4.6 the disagreements were read before calling any of them victim errors:

| class | share of the 1,025 disagreements |
|---|---:|
| proof / prose answers where "the boxed answer" is not a comparable object | **62.4 %** |
| MCQ format: R1 boxes `D. 48` / `B. encampment` / `E (gives)`, `y` boxes `D`/`B`/`E` — **same option letter** | 6.8 % |
| equivalent form (spacing, `\text`, `\operatorname{sh}` vs `\sinh`) | 2.8 % |
| R1's box is not an answer at all (`[]`, `total_pairs - total_invalid`) | 2.1 % |

Examples: idx 66713 `y` = *"All three inequalities hold, with equality at the centroid"* against R1's
boxed inequality; idx 113654 `y` = `B` against R1's `B. encampment` — the same answer, graded wrong.
The puzzle domain reads 17.5 % agreement almost entirely for that MCQ-format reason.

- Removing only the **mechanically identified** artifacts: **76.3 %** — above the floor.
- On the **short-answer subset** (proof/prose rows excluded — a subset, not a headline): **87.9 %**.

R1's answer is the dataset's solution, not ground truth, and the grader rejects equivalent forms; the
headline stays **73.1 % raw / 76.3 % after artifacts**. **No filtering follows** — `docs/10` fixed
"no consistency filtering in the main condition", and nothing downstream reads this number.

### Three kept traces read — shortest, median, longest

- **idx 8249** (145 trace chars): *"The question asks how much tax Janne pays on a $200.00 camera at
  15 % tax. Tax = 15 % of $200.00 = 0.15 × 200 = $30.00."* Answer ends `\boxed{A}`.
- **idx 48700** (3,610 chars, the median): *"Let me carefully analyze this problem. Factory A produces
  2700 sets per month… So Factory A spends 2/3 of its time on tops."* Answer verifies both branches
  and ends `\boxed{6700}`.
- **idx 86300** (51,704 chars, the longest kept): *"We need solve geometry optimization. Problem:
  convex rectangular garden? … likely rectangle."*

**Register.** At `medium` the victim mostly writes structured first-person reasoning ("Let me
carefully analyze this problem"), much closer to R1-Distill's register and to π's expected input than
the xhigh probe's telegraphic notes — though the terse voice ("We need solve…") still surfaces on the
longest, hardest rows. `y` is a **full worked solution** (`\boxed{}` on math, fenced code on code),
never a bare answer. The oracle condition trains on this register; the forged conditions train on
R1-Distill prose — carried to `docs/09` and the Phase 5 handoff.

---

## 5. LEAD FINDING — the victim's real traces are shorter than the forgeries that imitate them

All medians below use the **Qwen3.5-4B tokenizer**, the same one `phase2.md` §5 used, so the rows are
directly comparable (`docs/11` §5: state the tokenizer before comparing two numbers).

| trace | median tokens | source |
|---|---:|---|
| **victim `t`** (this run, kept n=5,045) | **1,400** | measured here |
| forged `t̂`, epoch-2 adapters — the ones Phase 4 serves | **2,063 – 2,241** | `phase2.md` §5, 200-row holdout |
| surrogate `t'` (7B holdout `t_true`) | 2,172 | `phase2.md` §5 |
| R1 on the *same* split-B prompts (both under cap) | 3,664 | §4 paired report |

**The forgeries run ~50 % longer than the truth they imitate.** The paper's ordering is the reverse:
its forgeries (4,972–5,434) sit *below* its R1 oracle (6,130.6), and its "Len" recovery of 81–89 % is
measured as forgeries approaching the truth **from below**. Ours overshoot it **from above**. The
victim is also 64 % shorter than R1 on identical prompts, and shorter on 81.3 % of them.

**Caveat on the `t̂` column — it changes how hard this can be pushed.** Those 2,063–2,241 figures are
the inverters run on *surrogate*-conditioned inputs `(x', y', b')` in Phase 2. Phase 4 conditions on
*victim* inputs `(x, y, b*)`. So the forged length on split B is **estimated, not measured**, and the
direction is genuinely uncertain because two opposing pushes act on the inverter's learned length
prior:

- the victim's `y` is **longer** than the surrogate's — 605 median here vs the 7B arm's 488
  (`docs/14` §1) — pushing `t̂` **up**;
- `b*` is likely **shorter** than `b'`: π's output scaled with input length in Phase 1 (491 tokens on
  the shortest trace quartile vs 633 on the longest, median 583 — `phase1.md` §2), and the victim's
  traces are about half the surrogate's, so expect `b*` nearer 490–550 than 583, pushing `t̂` **down**.

Which push wins is not predictable from here — which is the argument for measuring it in Phase 4
rather than carrying an estimate forward.

### What this means for Phase 5 — two consequences, no fix proposed here

1. **Supervision length is confounded with condition.** Measured on this run, the Victim-Trace
   (oracle) target is `t + y` = **2,114 median tokens** (1,400 + 605). The Synthesized-Trace target
   is `t̂ + y` ≈ **2,670–2,850** on the estimate above — **roughly 26–35 % more supervision tokens for
   the forged condition than for the oracle.** If Synthesized-Trace beats the oracle in Phase 6 — the
   paper's headline claim, which `docs/00` §6 already flags as having no length-matched control —
   then *"inversion denoises the teacher"* and *"the student simply saw more tokens"* are **not
   separable** with what is currently planned. This is created by our victim being terse, not by any
   Phase 3 choice.
2. **The oracle may be a weaker ceiling than the paper's.** A 1,400-token terse trace carries less
   supervision than R1's 6,130-token deliberation, so the Victim-Trace row may land lower relative to
   the other conditions than Table 3 leads one to expect. That is a real property of a 27B at
   `medium` effort, **not a pipeline bug** — Phase 6 must not read a low oracle as one.

**No fix is proposed here.** A length-matched control is a Phase 5 proposal and has to pass
`docs/10`'s four questions on its own. This section exists so the Phase 5 handoff cannot miss it.

### Instruction for Phase 4 — its length report is now load-bearing

`docs/10` Phase 4 currently says only *"if you want the length as a number, point `phase1_stats.py`'s
existing length reporting at the output file."* **That is too weak now.** Phase 4 must produce a
**paired** report: `t̂` against the victim's own `t` **on the same split-B `idx`**, per arm and per
setting, medians *and* quantiles, plus the implied supervision totals `t̂ + y` against `t + y`. That
one table is what tells Phase 6 whether an oracle-vs-synthesized gap is confounded with supervision
length. It also settles the estimate above with a measurement.

**This does not breach the withholding rule.** `docs/14` §4.5 already permits it: *"Length reports
comparing forgeries to `t` are read-only measurements in the main thread, not part of any training
input."* Reading `victimB-ORACLE.jsonl` to measure lengths is allowed; what is forbidden is letting
`t` reach a training input or the attack path.

---

## 6. Regime re-benchmark at `medium` — does the victim still clear the surrogate?

Choosing `medium` changed the victim's operating point, so Phase 0's numbers no longer describe the
model Phase 3 queried. Re-benchmarked on a fresh identical server (`--reasoning-effort medium`,
12 × 16,384, `x + INSTR` — the exact query shape the run used), `eval_victim_gguf.py` stratified
250/bench, `max_tokens` 14,336, graded in the main thread. 1 request error in 500.

| benchmark | **victim @ medium + `INSTR`** (n=250 stratified) | Phase 0 @ xhigh, no `INSTR` (full set) | difference | median gen tokens | truncated |
|---|---:|---:|---:|---:|---:|
| JEEBench | **82.0 %** (completed-only 82.9 %) | 86.2 % (n=515) | −4.2 pts = **1.46 SE** | 2,090 | 4 / 250 |
| MATH500 | **97.2 %** (completed-only 98.4 %) | 98.8 % (n=500) | −1.6 pts = **1.39 SE** | 798 | 4 / 250 |

**What this run establishes is the regime**, on JEEBench:

```
victim 82.0   ≫   surrogate 7B 60.6   ≫   student 47.8
        +21.4 pts over the surrogate,  +34.2 over the student
```

**It does not establish a cost of `medium`, on two independent grounds.**

1. *Not statistically distinguishable.* This run is a 250/bench stratified subset against Phase 0's
   full sets. Two-proportion standard errors: JEEBench `SE_diff = √(2.43² + 1.52²) = 2.87 pts`, so
   −4.2 is 1.46 SE (p ≈ 0.14); MATH500 `SE_diff = √(1.04² + 0.49²) = 1.15 pts`, so −1.6 is 1.39 SE.
   Neither reaches significance, and `baselines.md` already flags the 250/bench subset as ±3 %.
2. *Confounded regardless.* This run is `medium` **+ `INSTR`**; Phase 0 was `xhigh` **without**
   `INSTR`. Two variables moved, and `INSTR` plausibly *helps* extraction (§3: it takes boxing from
   56 % to ~100 %, so fewer answers go unparsed) — it could be masking an effort cost or contributing
   a gain of its own. The difference cannot be attributed to reasoning effort.

So: no cost of `medium` is established here, and none needs to be — the setting bought a 66 h run
instead of the 106–145 h `xhigh` projected, and the ordering the whole argument rests on is intact.
No Phase 5 re-reading is required.

**This also corroborates §4's reading of the 73.1 % `y`-vs-R1 figure.** The same victim, at the same
setting, scores **97.2 % on MATH500**, whose gold answers are unambiguous and short. A model that
answers 97 % of MATH500 correctly is not making errors on a third of OpenThoughts — the gap is the
proof-heavy prompt mix and R1's boxing conventions, exactly as the disagreement decomposition showed.

---

## 7. Compression `b* = π(t)` — the same compressor, unchanged

`phase1_compress.py` exactly as Phase 1 ran it (`Qwen/Qwen3.5-4B`, `PI_SYSTEM` as committed,
temperature 0.7 / top_p 0.9 / rep 1.05, `--max-tokens 2048`, `--seed 1234`), with the one change
`docs/14` §4.4 fixes: **`--max-len 20480`**, because a 14,336-token victim trace plus π (~1,600) plus
a 2,048-token summary no longer fits 16,384. `max_model_len` is a KV budget and changes no output.

| stage | count |
|---|---:|
| traces in (kept rows) | 5,045 |
| summaries at the 2,048 cap, first pass | **6** |
| `</think>` blocks stripped defensively | **17** |
| bad after the first pass (`still_bad`) | **9** |
| regenerated by one `--fix --seed 1235` | 9 → **still_bad 0** |
| dropped by `--drop-bad` | **0** |
| **rows in `victimB-ORACLE.jsonl`** | **5,045** |

**The predicted failure mode did not appear.** π's known weakness is padding on short traces by
re-deriving (`phase1.md` §2: 6 of 5,012 on the 7B arm, whose median input was 2,804 tokens). The
victim's traces are **half that length** (1,400 median), so materially more cap-hits were expected —
and the count was **the same 6**, all of which one `--fix` rescued. Drop rate **0.0 %** against the
2 % STOP. Whatever drives π's padding, it is not simply short input.

**The defensive `</think>` strip is load-bearing, more so here than in Phase 1.** It fired **17 times
in 5,045** (0.34 %) against Phase 1's 1 in 5,012. Removing it as redundant would silently corrupt
~1 row in 300 on this data.

### Table 1 on `b*` — a report beside Phase 1's, not a gate (`docs/14` §4.4)

| statistic | `b*` (victim, n=5,045) | Phase 1 `b'` 7B (n=5,006) | paper target | |
|---|---:|---:|---|---|
| median tokens | **606** | 583 | 540–590 | outside |
| bold-header sections (≥3) | **100.0 %** | 100.0 % | > 90 % | pass |
| first-person prose (≥3) | **99.7 %** | 99.9 % | > 95 % | pass |
| LaTeX | **86.3 %** | 80.5 % | > 70 % | pass |

Spec compliance: numbered lists 0.1 %, bullet lists 0.2 %, 96.9 % within the 3–6 section bound
(median 5). π is frozen, so the 606 median is **a finding about π on victim traces, not a defect** —
and it stays inside `docs/14` §6's STOP band of 450–700, with headers and first-person far above the
90 % floor, so no STOP fired. `phase1_stats.py --mode summaries` exits non-zero on that one Table 1
line by construction; the **`D₂` integrity block is the gate here and it passed**.

π is *slightly more verbose* on the victim's shorter traces (606 vs 583) and produces *more* LaTeX
(86.3 % vs 80.5 %) — consistent with the victim's higher math share among kept rows and with π adding
depth rather than sections. It remains stable across input types (per-domain medians 406–629) and
scales with input length exactly as in Phase 1 (short quartile 505 → longest 694).

### `D₂` integrity — the gate

| | |
|---|---:|
| rows | **5,045** (target 5,000) |
| duplicate `idx` | 0 |
| empty `x` / `y` / `b` / `t` | 0 / 0 / 0 / 0 |
| summaries severed at the cap | 0 |
| verdict | **PASSED** |

---

## 8. The file layout — structural withholding (`docs/14` §4.5)

| file | rows | contents | who reads it |
|---|---:|---|---|
| `victim-traces-ORACLE.jsonl` | 6,022 | raw generator output incl. `trace`, `raw`, capped rows | Phase 3 only |
| `victimB-ORACLE.jsonl` | **5,045** | `D₂` schema **with `t`** | the Victim-Trace oracle condition, and read-only length measurement |
| `victimB-attack.jsonl` | **5,045** | the same rows, same order, **minus `t`** | **Phase 4 and the Answer-only / Summary+Answer conditions — the only Phase 3 file the attack path opens** |

`bench/phase3_split.py` derived the attack file and its self-check **passed**:

```
oracle rows 5045  attack rows 5045  written 5045
keys ['b','domain','finish_reason','idx','source','summary','summary_tokens','x','y']
** PASSED: keys, idx sequence, row equality minus t, no oracle trace in any attack field **
```

Verified independently afterwards: no `t`/`trace`/`raw` key anywhere in the attack file; idx sets
*and* order identical to the oracle's; a 400-row substring scan for the first 200 characters of each
row's true trace found **0 leaks**; every attack row has non-empty `x`, `y` and `b`. The script also
self-tests six mis-join shapes (`--selftest`), each of which must fail.

**Rule for Phases 4–6:** Phase 4 and the Answer-only / Summary+Answer conditions read
`victimB-attack.jsonl` and nothing named ORACLE. The Victim-Trace condition reads the ORACLE file.
Length reports comparing forgeries to `t` are read-only measurements in the main thread (§5).

---

## 9. Definition of done (`docs/14` §10)

- [x] `promptsB.jsonl` 9,000 rows in permutation order; `splits.json` and `promptsA.jsonl` md5-identical
- [x] sweep at 16,384/slot committed to `docs/results/sweeps/`
- [x] probe reported before the long run — three of them, one per effort level
- [x] `victim-traces-ORACLE.jsonl`, 5,045 kept ≥ 5,040; `--mode traces --paired --vs-r1` reported, band widening recorded
- [x] `victimB-ORACLE.jsonl` 5,045 ≥ 5,000; integrity gate passed; Table 1 reported beside Phase 1's
- [x] `victimB-attack.jsonl` derived; asserts pass; no `t` anywhere in it
- [x] `docs/results/phase3.md` (this file); `docs/09` rows added; `docs/10` closed with the measured 66.32 h
