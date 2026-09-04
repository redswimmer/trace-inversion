# Phase 4 — Invert split B (`forged-{7b,1.5b}-{sum,nosum}`)

Reproduction of *"How to Steal Reasoning Without Reasoning Traces"* (Zhang, Morris, Shmatikov,
arXiv 2603.07267v2), Stage 2 second half: the four trained inverters run once each over the victim's
split-B outputs to synthesize the forged traces `t̂` that Phase 5's Synthesized-Trace students train on.

| | |
|---|---|
| Inverters `I` | **4** = {7B, 1.5B surrogate arm} × {sum, nosum}; the **epoch-2** adapters, merged to bf16 — `inverter-7b-*/checkpoint-402`, `inverter-1.5b-*/checkpoint-404` (`docs/10` Phase 4, decided 2026-08-30) |
| Input | `bench/results/phase3/victimB-attack.jsonl`, 5,045 rows — sum inverters get `(x, y, b*)`, nosum `(x, y)`, through `phase2_format.to_row`'s construction, `enable_thinking=false` (§0) |
| Engine | vLLM 0.27.1 offline batch, `bench/invert.py`, `gpu_memory_utilization 0.90`; `max_model_len` auto-widened to **15,872** (sum) / **15,360** (nosum) on draw 1, 12,288–13,568 on the redraws |
| Sampling | temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · `max_tokens` 8,192 · seed **1234**, redraws at **1235** then **1236** |
| Capped rows | `finish_reason == "length"` → redrawn at the next seed; still capped after three draws → **dropped and reported** (`docs/09` 7.14). The draw-1 file is kept intact |
| Consistency | reported per inverter, **no filtering** (`docs/09` 7.15) |
| Oracle | `victimB-ORACLE.jsonl` opened **once per inverter, in the stats step**, read-only, for the paired length table (§4); nothing on the generation path names it (§7) |
| Code | `bench/phase4_format.py` · `bench/phase4_draws.py` · `bench/run_phase4_invert.sh` · `phase1_stats.py --mode inverted --oracle` · `invert.py` (adds a realized tok/s line) |
| Tokenizer | every token count here: **Qwen3.5-4B** (the inverters' own; == Qwen3.5-2B's). vLLM's `gen_tokens` is the same tokenizer — the one phase where engine and recount agree — and is quoted where the engine's own signal matters (cap-hit, throughput) |

Data and logs live under the **main checkout** (`bench/results/phase4/`, `bench/logs/phase4-*.log`);
the branch was built in a worktree whose driver runs its own code with the main checkout's venvs
(`docs/15` §0.1). Supervised: session `big-boss` audited every CHECKPOINT and adjudicated the one
STOP line tripped (§2); it wrote nothing to the tree.

> **`docs/15-phase4-handoff.md` is the pre-run plan; this file is authoritative where they
> disagree.** Its method, conventions and file layout held. Its numeric expectations did not: the
> draw-1 cap-hits (3 / 7 / 17 / 14.5 % expected → **23.3 / 21.4 / 37.9 / 35.5 %** measured), the
> premise that capping is draw-level (re-cap rates rose to 75–76 % by the third draw, so the ≤ 3-draw
> policy dropped **9.5–19.4 %** of rows instead of a "~4–10-row core"), the budget (13–17 h → **≈ 28 h**),
> the `max_model_len` (~19.5k → 15,872), and the STOP lines, three of which were tripped and
> adjudicated on the record (§2, §3). The headline it asked for — the paired length table — is §4:
> **forged traces are 2.2–2.6× the victim's on the same problems; the forged supervision target is
> 1.8–1.9× the oracle's.**

---

## 0. Prompt files — `bench/phase4_format.py`

`to_row` (Phase 2's formatter) now accepts a row without `t` and emits the same prompt with no `t`
and no `completion`; a regression check ran main's original `to_row` beside the refactored one on
every attack row with a fake trace attached, both settings — **10,090 (row, setting) pairs
identical, key order included**. So the served prompt is the training prompt by construction, not
by re-typing.

| file | rows | keys | prompt tokens max / median | `max_model_len` invert.py picks at 8,192 |
|---|---:|---|---:|---:|
| `attack-sum.jsonl` | 5,045 | `idx, domain, x, y, b, prompt, chat_template_kwargs` | **7,663** / 1,811 | 15,872 |
| `attack-nosum.jsonl` | 5,045 | `idx, domain, x, y, prompt, chat_template_kwargs` | **6,962** / 1,096 | 15,360 |

Self-test on the files as written — **PASSED**: idx sequence == the attack file's (set, count and
order) · no `t`/`trace`/`raw`/`completion` key anywhere · no `b` text in any no-summary field ·
every prompt equal to `to_row`'s Phase 2 construction for the same `(x, y, b)` ·
`chat_template_kwargs == {"enable_thinking": false}` on every row · every rendered prompt ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n` (the training rendering, `phase2.md` §1) · row 0
rendered and read. Prompt maxima are well under `docs/15` §1's ~11k worst case (the long `x`, long
`y` and long `b*` do not co-occur), so the KV budget is 15.9k, not 19.5k.

**Two degenerate `b*` rows — a Phase 3 datum, recorded here, left as-is.** The self-test's first
pass flagged idx 64731 as "b text in a nosum row": its `b*` is literally `\boxed{\frac{3}{10}}`
(20 chars), a substring of the victim's own `y`. Scanning the attack file: **2 of 5,045** summaries
are under 200 chars — idx 64731 (`\boxed{\frac{3}{10}}`) and idx 6814 (`**Final Answer**\n\boxed{2144}`,
29 chars) — while their `summary_tokens` fields read **598 and 883**. `summary_tokens` is
`len(c.token_ids)` of π's whole completion (`phase1_compress.py`), `b`/`summary` is the text that
survived the post-processing; on these two rows π spent its budget on something the split removed
and the summary that reached the file is the boxed answer alone. Median `b*` is 2,339 chars (p05
1,613), so this is 0.04 % of rows. Consequence: the sum inverter sees these two rows as effectively
no-summary inputs (§5 reads their `t̂`); Phase 5's Summary+Answer condition inherits the same two
rows. The nosum leak check now ignores a `b` that is a substring of `x` or `y`; the construction
check pins the prompt to `(x, y)` regardless.

---

## 1. Adapters — merged, and proven merged

`phase2_train.py --merge --adapter <checkpoint>` writes bf16 weights (8.43 GB) and
`merge-check.json`; the assert that two probed base tensors changed is the proof an adapter was
folded in (the bare base already writes plausible traces). One merge on disk at a time, deleted by
the driver on exit; the probe keeps it for the full run that follows and the driver re-reads the
merge-check before reusing it.

| inverter | adapter | `in_proj_qkv` (layer 0) max|Δ| / changed | `q_proj` (layer 3) max|Δ| / changed | Phase 2's ep2 merge (`phase2.md` §5) |
|---|---|---|---|---|
| 7b-sum | `checkpoint-402` | 3.223e-03 / 88.0 % | 3.296e-03 / 84.5 % | 3.2e-3 / 88.0 %, 3.3e-3 / 84.5 % — equal |
| 7b-nosum | `checkpoint-402` | 2.808e-03 / 87.9 % | 3.052e-03 / 84.1 % | 2.8e-3 / 87.9 %, 3.1e-3 / 84.1 % — equal |
| 1.5b-sum | `checkpoint-404` | 2.808e-03 / 87.7 % | 3.540e-03 / 83.1 % | 2.8e-3 / 87.7 %, 3.5e-3 / 83.1 % — equal |
| 1.5b-nosum | `checkpoint-404` | 3.296e-03 / 86.8 % | 3.052e-03 / 83.4 % | 3.3e-3 / 86.8 %, 3.1e-3 / 83.4 % — equal |

The committed copies are `bench/results/phase4/merge-check-{tag}.json`.

---

## 2. Probe and smokes — 30 rows each, before every multi-hour run

`LIMIT=30 bench/run_phase4_invert.sh <arm> <setting>`: merge, `invert.py --limit 30` (the first 30
attack rows in permutation order — 21 math, 7 code, 2 puzzle — the same 30 for every inverter),
stats paired against the oracle, three traces read. The probe file is disposable; the full run
regenerates its rows.

### 7b-sum probe — a STOP line tripped and adjudicated on n=30

| | probe (n=30) | expected (`docs/15` §1, from the 200-row holdout) |
|---|---|---|
| cap-hit at 8,192 | **6 / 30 = 20.0 %** (95 % CI ≈ 8–39 %) | 3.0 % |
| `t̂` tokens median / mean / p05 / p95 | **3,675** / 4,012 / 233 / 8,192 | holdout `t̂` 2,241 |
| victim `t` on the same 30 rows | 1,418 / 2,823 / 291 / 10,033 | — |
| `t̂ / t` at the median · per-row ratio median | **2.59** · 1.80 (`t̂` shorter on 33 %) | Phase 2 ratio to the surrogate's `t'` 1.03 |
| `t̂ + y` vs `t + y` at the median (`y` median 625) | **4,445 vs 2,167 = 2.05×** | — |
| by domain `t̂ / t` | math 3,723 / 1,361 (n=21) · code 4,470 / 1,576 (7) · puzzle 460 / 302 (2) | — |
| prompt tokens max / median · `max_model_len` | 2,607 / 1,934 · 12,288 (the full file widens it to 15,872) | — |
| realized throughput | **914 tok/s** over `llm.generate`, 120,407 tokens in 132 s — a 30-sequence batch, under-occupied | Phase 2's 200-row rate 1,590 |
| empty · stripped_think | 0 · 0 | — |
| consistency | match 12 · mismatch 2 · no box in `t̂` 16 (6 severed at the cap + 10 prose) · 85.7 % graded (n=14) | — |

The two mismatches, read: idx 66713 (three-part triangle inequality proof — `y` boxes a sentence,
`t̂` proves all three and boxes inequality (c) last) and idx 17752 (locus seen at an acute/obtuse
angle — `y` boxes an `aligned` environment, `t̂` boxes *"Inside the circle with diameter AB"*): both
the same conclusion in another form, **0 genuine**. Three traces read: short idx 113596 (puzzle,
`y = B`) 330 tokens, reaches B in prose, no box; median idx 17578 (integer-valued cubic, a proof)
5,381 tokens, boxes `y`'s statement via the binomial-basis argument; long idx 89242 (code — no
palindromic substrings, `y = 1`) capped at 8,192 mid-DP enumeration, where the victim's own `t` runs
10,648. The register is unchanged R1-Distill prose ("Okay, so I have this problem…"); the victim's
`y` is a full worked solution (all 30 boxed) and the inverter re-derives all of it at R1 length.

**Adjudication.** `docs/15` §6 says STOP AND ASK at a 7B-arm draw-1 cap-hit over 15 %. The probe's
20 % on n=30 tripped it; the line existed to catch a wrong merge or prompt before hours were spent,
and both were verified independently (merge-check equal to Phase 2's ep2 merge; prompts byte-equal to
training). `big-boss` ruled GO on that basis — the elevated cap-hit is the measured mechanism (the
victim's `y` at 625 median tokens pushes the learned length prior up and shifts mass into the cap),
and the draw-1 file is the measurement whatever the redraws do — with one new hard boundary: **rows
dropped after three draws above 2 % of 5,045 (> 101) halts the phase before the next inverter, for
the user's decision**, because that re-opens the data-quantity confound the redraw policy exists to
prevent. Recorded as a line tripped and adjudicated, not passed.

### Smokes — 7b-nosum, 1.5b-sum, 1.5b-nosum (same 30 rows; full runs pre-authorized on a clean smoke)

| inverter | cap-hit (30) | `t̂` median / mean / p05 / p95 | `t̂ / t` median · per-row | `t̂ + y` vs `t + y` | empty · stripped | tok/s (30-seq) | 3 traces |
|---|---|---|---|---|---|---:|---|
| 7b-nosum | **5 / 30 = 16.7 %** (holdout 7 %) | 3,834 / 3,740 / 167 / 8,192 | 2.70 · 1.47 | 4,315 vs 2,167 = 1.99× | 0 · 0 | 896 | 113596 reaches B in prose (449 tok); 17578 boxes `y`'s statement (2,080 tok, half the sum inverter's); 89242 capped again |
| 1.5b-sum | **8 / 30 = 26.7 %** (holdout 17 %) | 3,477 / 4,074 / 349 / 8,192 | 2.45 · 1.90 | 4,050 vs 2,167 = 1.87× | 0 · 0 | 928 | 113596 reaches B in prose (415 tok); 17578 boxes `y`'s statement (4,110 tok, by finite differences); 89242 capped — its tail repeats one sentence (a strict loop) |
| 1.5b-nosum | **14 / 30 = 46.7 %** (holdout 14.5 %) — over the 40 % line; integrity clean (merge, 0 empty, 0 stripped, uncapped-row median 1,376), so report-and-continue | 5,944 / 4,942 / 632 / 8,192 | 4.19 · 2.89 | 6,767 vs 2,167 = 3.12× | 0 · 0 | 1,084 | 113596 reaches B in prose (430 tok); 17578 boxes the statement (4,733 tok); 89242 capped, mid-DP |

---

## 3. Generation — per inverter

`bench/run_phase4_invert.sh <arm> <setting>`: draw 1 at seed 1234 over all 5,045 prompt rows;
`phase4_draws.py subset` selects the rows vLLM reported `finish_reason == "length"`; draw 2 at
seed 1235 over those; draw 3 at 1236 over what capped again; `assemble` takes the first
terminating draw per idx and drops what capped three times. Cap-hit is the engine's own
`finish_reason`; token counts are vLLM's `gen_tokens`; throughput is `gen_tokens / llm.generate`
wall (engine load excluded).

### Draws — the cap-hit is prompt-level on split B, not draw-level

| inverter | draw | rows | capped | rescued | re-cap rate | empty | stripped_think | loops strict (loose) | gen tokens | wall | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **7b-sum** | 1 (seed 1234) | 5,045 | **1,174 = 23.3 %** | — | — | 0 | 0 | 79 (1,922), of which capped 74 (838) | 21,063,854 | 3.72 h | **1,574** |
| | 2 (1235) | 1,174 | 735 | 439 | 62.6 % | 0 | 0 | 51 (724) | 8,315,456 | 1.55 h | 1,486 |
| | 3 (1236) | 735 | **555** | 180 | 75.5 % | 0 | 0 | 38 (506) | 5,569,771 | 1.06 h | 1,464 |
| | **final** | **4,490** of 5,045 — by draw 3,871 / 439 / 180 | 0 | | | 0 | 0 | **5 (1,408)** | 14,763,993 | 6.37 h generation | |
| **7b-nosum** | 1 (1234) | 5,045 | **1,079 = 21.4 %** | — | — | 0 | 0 | 95 (1,837), of which capped 92 (772) | 20,049,345 | 3.20 h | **1,739** |
| | 2 (1235) | 1,079 | 642 | 437 | 59.5 % | 0 | 0 | 42 (661) | 7,516,372 | 1.28 h | ~1,600 |
| | 3 (1236) | 642 | **480** | 162 | 74.8 % | 0 | 0 | 25 (389) | 4,831,445 | 0.84 h | 1,596 |
| | **final** | **4,565** of 5,045 — by draw 3,966 / 437 / 162 | 0 | | | 0 | 0 | **3 (1,341)** | 16,997,000 | 5.35 h generation | |
| **1.5b-sum** | 1 (1234) | 5,045 | **1,914 = 37.9 %** | — | — | 0 | 0 | 384 (2,940), of which capped 381 (1,698) | 24,095,072 | 4.32 h | **1,551** |
| | 2 (1235) | 1,914 | 1,284 | 630 | 67.1 % | 0 | 0 | 237 (1,519) | 13,027,640 | 2.45 h | ~1,480 |
| | 3 (1236) | 1,284 | **977** | 307 | 76.1 % | 0 | 0 | 150 (1,059) | 9,344,275 | 1.78 h | 1,461 |
| | **final** | **4,068** of 5,045 — by draw 3,131 / 630 / 307 | 0 | | | 0 | 0 | **4 (1,799)** | 15,050,000 | 8.6 h generation | |
| **1.5b-nosum** | 1 (1234) | 5,045 | **1,792 = 35.5 %** | — | — | 0 | 0 | 305 (2,780), of which capped 301 (1,573) | 23,165,200 | 3.75 h | **1,717** |
| | 2 (1235) | 1,792 | 1,192 | 600 | 66.5 % | 0 | 0 | 211 (1,370) | 12,194,032 | 2.13 h | ~1,590 |
| | 3 (1236) | 1,192 | **910** | 282 | 76.3 % | 0 | 0 | 133 (977) | 8,724,745 | 1.51 h | 1,605 |
| | **final** | **4,135** of 5,045 — by draw 3,253 / 600 / 282 | 0 | | | 0 | 0 | **5 (1,694)** | 14,688,000 | 7.4 h generation | |

**All four, one line each.** Draw-1 cap-hit 23.3 / 21.4 / 37.9 / 35.5 % (7b-sum / 7b-nosum /
1.5b-sum / 1.5b-nosum) against 3 / 7 / 17 / 14.5 % on the holdout; re-cap chains 23→63→76,
21→60→75, 38→67→76, 36→67→76 % — the third-draw re-cap rate is 75–76 % on every inverter; dropped
555 / 480 / 977 / 910 = **11.0 / 9.5 / 19.4 / 18.0 %**; final rows **4,490 / 4,565 / 4,068 / 4,135**;
code is dropped at 2× the math rate on every inverter (20 / 18 / 38 / 33 % of code rows); no other
domain loses more than 2 rows. Strict loops in the final files 5 / 3 / 4 / 5 (≤ 0.1 %); among
draw-1 capped rows 6 / 9 / 20 / 17 % — the 1.5B arm's runaway is a loop three times as often as the
7B arm's. Draw-1 `gen_tokens` medians over all 5,045 rows 3,654 / 3,373 / 4,246 / 3,799: the 1.5B arm
writes *longer* per draw and caps more, so its final files keep a shorter surviving set (§4).
Every draw: 0 empty, 0 stripped_think.

**The drop profile Phase 5 inherits** (`docs/15` §9): the four final idx sets have **3,616 rows in
common** (math 2,886 · code 477 · chemistry 68 · puzzle 65 · biology 60 · physics 60); the union of
drops is 1,429; pairwise drop overlaps run 305–596 rows (the two 1.5B files share 596 of their
977 / 910), so drops are largely the same hard, long prompts across inverters. Training every
condition on the intersection is the recommended treatment; final call at Phase 5 planning.

**7b-sum, draw 1 — the inverter's own cap-hit: 23.3 %**, against 3.0 % on the Phase 2 holdout with the
same adapter, the same sampling and the same cap. By domain: math 21.9 % (n=3,889), code 34.8 % (899),
physics 9.5 % (63), chemistry 1.4 % (69), puzzle 1.5 % (65), biology 0 % (60). `gen_tokens` over all
5,045 rows: median 3,654, mean 4,175, p25 1,617, p75 7,611; over the 3,871 uncapped rows median 2,713.

**The redraw policy rescued 619 of 1,174 and dropped 555 = 11.0 % of split B** (math 374 = 9.6 % of
math rows, code 180 = 20.0 % of code rows, physics 1, others 0). The re-cap rate *rises* across draws
— 23.3 → 62.6 → 75.5 % — so the residual is increasingly a property of the prompt: `docs/10`'s
premise that "capping is mostly a property of the draw" was measured on the surrogate holdout
(epoch-2/3 capped sets overlapping 1 in 10) and does not transfer to victim-conditioned inputs. This
crossed the 2 % halt line set at CHECKPOINT 1 and the phase halted before the next inverter.

**User decision, 2026-09-03 (relayed by `big-boss`): accept the drops and continue.** The ≤ 3-draw
policy stands unchanged for the remaining three inverters and the 4,490-row 7b-sum file is accepted
as-is — recorded as a user-adjudicated acceptance of an **11.0 % drop against the 2 % line**, with
the prompt-level re-cap mechanism as the stated reason, not as a pass. For the rest of the phase the
supervisor set adjudicated ceilings that supersede the 2 %-drop and 15 / 30 % draw-1 lines, which
the mechanism on record now explains: **halt only if draw-1 cap-hit exceeds 40 %, drops after three
draws exceed 20 %, or the phase projection passes 36 h**; every other STOP line unchanged. Phase 5
receives the four final idx sets and per-inverter drop profiles; training every condition on the
common idx intersection is the recommended treatment (Phase 3's shared-filter logic), final call at
Phase 5 planning.

**Second user decision, 2026-09-03 (during 1.5b-sum's redraws, relayed by `big-boss`): no hard drop
cutoff on the 1.5B arm — the criterion is that the forged sets stay usable in Phase 5.** The 20 %
drop ceiling above is replaced by three checkable floors, any of which halts: a final forged file
below **3,500 rows** (~70 % of 5,045, beyond the comfort of the paper's Fig. 3 scaling
extrapolation); **any domain at 0 kept rows** in a final file; the running four-set idx
**intersection projecting below 3,000 rows**. The 40 % draw-1 line becomes report-and-continue
when integrity is clean on that inverter (merge-check equal to Phase 2's ep2 values, 0 empty, 0
stripped_think, strict loops in band, `t̂` median inside 700–4,500); the 36 h projection ceiling
stands. Both thresholds are on the record here; per-inverter drops and the four-set intersection
are reported in §3 and §7 regardless, and Phase 5 planning decides with those numbers in hand.

### 3.1 Adjudication trail — every line tripped, dated; none presented as a pass

| when | line (`docs/15` §6 unless noted) | tripped by | adjudicated |
|---|---|---|---|
| 2026-09-02 20:52, CHECKPOINT 1 | draw-1 cap-hit > 15 % on a 7B-arm inverter | 7b-sum probe 6/30 = 20 % (95 % CI ≈ 8–39 %) | `big-boss`: GO on verified integrity (merge-check == Phase 2 ep2; prompts byte-equal); the line existed to catch a wrong merge or prompt; set a new hard boundary: drops after three draws > 2 % → halt for the user |
| 2026-09-03 03:21, CHECKPOINT 2 | drops > 2 % of 5,045 (set above) | 7b-sum 555 = 11.0 % | halted ~5 h; **user accepted the drops** and the ≤ 3-draw policy for the remaining inverters; `big-boss` set ceilings for the rest of the phase: draw-1 cap-hit > 40 %, drops > 20 %, projection > 36 h |
| 2026-09-03 ~21:00, during 1.5b-sum's redraws | drops projected to brush 20 % | 1.5b-sum draw 2 re-cap 67 % → ≈ 19 % projected (landed at 19.4 %) | **user: "no hard cutoff as long as it's usable in Phase 5"** — the 20 % ceiling replaced by the floors: file ≥ 3,500 rows, no empty domain, intersection ≥ 3,000; the 40 % draw-1 line became report-and-continue on clean integrity |
| 2026-09-03 22:22, CHECKPOINT 3d | draw-1 cap-hit > 40 % | 1.5b-nosum smoke 14/30 = 46.7 % | report-and-continue: merge-check == Phase 2, 0 empty, 0 stripped, uncapped-row median 1,376 in band; the full draw landed at 35.5 % |
| — | empty `t̂` · stripped_think > 2 % · strict loops > 5 % · `t̂` median outside 700–4,500 · a gate failing twice · ORACLE on the attack path · 36 h | never tripped (0 · 0 · ≤ 0.1 % · 2,491–3,055 · 0 · 0 · ≈ 28 h) | — |

**Loops — the loose test cried wolf, and was checked before being reported** (`docs/11` §5).
`phase2.md` §5.4's test — a 40-char chunk repeated ≥ 3× in the trace's last 4,000 chars — flags
**31.4 %** of the *terminated* final rows. A random dozen: a LaTeX expression or a phrase recurring
three times in a long derivation, covering 3 % of the window (`'2 d + \frac{27 \lambda^2}{4} m'`,
`' area of A is 1600 - 200π.\n\nTherefore, I'`). The **strict** variant requires the repeats to fill
≥ 20 % of the window (a 40-char chunk ≥ 20× in 4,000 chars): **5 of 4,490 final rows (0.1 %)** and
74 of 1,174 draw-1 capped rows (6.3 %), of which 23 are the fully degenerate kind at ≥ 50 % coverage
(`'000,000,000,…'`, `'1010101010…'`, `'AeiouAeiou…'`, a sentence repeated to the cap). Both counts
are in `forged-*-draws.json`; the strict one is the number gated.

---

## 4. THE PAIRED LENGTH TABLE — `t̂` vs the victim's `t` on the same idx

`phase1_stats.py --mode inverted --oracle victimB-ORACLE.jsonl --final`, once per inverter on the
final file: `t` and `y` joined by idx from the oracle file, `y` asserted equal to the forged row's
own `y` (so the pairing is provably on the same row), lengths only, main thread. **Qwen3.5-4B
tokenizer throughout**; `gen_tokens` (vLLM's count) agrees with the recount within 1 token at every
quantile. The Phase 2 column is the same adapter on the surrogate holdout (`phase2.md` §5).

| inverter (final file, n) | `t̂` median / mean / p05 / p95 | victim `t`, same rows | `t̂ / t` at median · per-row p25 / median / p75 | `t̂` shorter on | `y` median | **`t̂ + y` vs `t + y` median** | Phase 2 holdout `t̂` |
|---|---|---|---|---:|---:|---|---:|
| **7b-sum** (4,490) | **3,055** / 3,287 / 184 / 7,374 | 1,199 / 2,188 / 208 / 8,403 | **2.55** · 0.99 / 1.87 / 3.17 | 25.2 % | 575 | **3,677 vs 1,893 = 1.94×** (totals 17.5 M vs 12.6 M tokens) | 2,241 |
| **7b-nosum** (4,565) | **2,881** / 3,146 / 177 / 7,362 | 1,225 / 2,259 / 208 / 8,599 | **2.35** · 0.85 / 1.66 / 2.93 | 29.8 % | 579 | **3,523 vs 1,921 = 1.83×** (17.2 M vs 13.2 M) | 2,095 |
| **1.5b-sum** (4,068) | **2,561** / 3,014 / 662 / 6,862 | 1,093 / 2,009 / 199 / 7,849 | **2.34** · 1.28 / 2.20 / 3.59 | 18.0 % | 550 | **3,201 vs 1,737 = 1.84×** (14.7 M vs 10.6 M) | 2,063 |
| **1.5b-nosum** (4,135) | **2,491** / 2,945 / 428 / 7,025 | 1,110 / 2,024 / 200 / 7,778 | **2.24** · 1.17 / 2.05 / 3.31 | 20.0 % | 555 | **3,128 vs 1,758 = 1.78×** (14.7 M vs 10.9 M) | 2,071 |

By domain, `t̂ / t` at the median, all four (7b-sum / 7b-nosum / 1.5b-sum / 1.5b-nosum): math
2.57 / 2.37 / 2.35 / 2.29 · code 2.50 / 2.38 / 2.58 / 2.36 · puzzle 1.80 / 1.86 / 1.63 / 1.64 ·
biology 1.42 / 1.51 / 1.53 / 1.46 · chemistry 1.37 / 1.26 / 1.30 / 1.27 · physics 1.12 / 1.14 /
1.04 / 1.00. The overshoot is a math-and-code phenomenon; on the four small non-math domains the
forgery is 1.0–1.9× the victim.

By domain, 7b-sum `t̂` / `t` median: math 3,186 / 1,241 (n=3,515, 2.57×) · code 3,361 / 1,346 (719,
2.50×) · puzzle 517 / 287 (65, 1.80×) · biology 947 / 666 (60, 1.42×) · chemistry 1,057 / 772 (69,
1.37×) · physics 1,285 / 1,143 (62, 1.12×).

Two selection effects to read the table with: (i) the final file is the *shorter* 89 % on both sides
— the 555 dropped rows are the ones the inverter could not finish in 8,192 and the victim's `t` on
them is longer too (`t` median 1,199 on the kept rows vs 1,400 on all 5,045); on all 5,045 draw-1
rows `t̂` `gen_tokens` median is 3,654 with the capped rows at 8,192; (ii) `t̂`'s p95 is bounded by the
cap and the drop, the victim's is not (its cap was 14,336).

**The two-pushes question (`phase3.md` §5) — resolved UP, on all four inverters.** `phase3.md`
estimated the forged `t̂` at 2,063–2,241 tokens from the surrogate holdout and asked which of two
opposing pushes would win on victim-conditioned inputs: the longer victim `y` (up) or the shorter
`b*` (down). Measured: the victim-conditioned `t̂` is **2,491–3,055 at the median, 2.24–2.55× the
victim's own trace on the same problems** and 1.2–1.4× the holdout estimate — up, on every inverter,
with and without the summary. The forged supervision target `[t̂; y]` is **1.78–1.94× the oracle's
`[t; y]`** at the median and 1.35–1.39× in total tokens over the kept rows. Phase 5's
oracle-vs-forged comparison is therefore confounded with supervision length by a factor of ~1.8–1.9
at the median, not the 26–35 % `phase3.md` estimated; `t̂` is also 2.0–2.4× the *surrogate's* `t'`
(2,172–2,178 on the Phase 2 holdout), so the inverters did not simply reproduce their training
length — the victim's fully-worked `y` pulls the monologue out to re-derive all of it. The paper's
`Len` recovery (forgeries approaching the truth from below, 81–89 %) has no analogue here: ours
overshoot from above by 2.2–2.6×. Two secondary directions, stated as measured: sum > nosum at the
median on both arms (3,055 vs 2,881; 2,561 vs 2,491), the same direction as Phase 2; and the 1.5B
arm's *final* medians are shorter than the 7B arm's although its draw-1 traces are longer (§3) —
selection by the drop, not a shorter prior. A length-matched control is a Phase 5 proposal
(`docs/10`'s four questions), not changed here.

---

## 5. Answer consistency — report, no filtering (`docs/09` 7.15)

Last `\boxed{}` in `t̂` against the last `\boxed{}` in the victim's `y`, `eval_baseline.py`'s
`extract_boxed` + `grade`, main thread; per-row buckets in the committed
`bench/results/phase4/forged-*-consistency.json`. The victim boxes nearly every answer (Phase 3 asked
it to): "no box in `y`" is 5 rows here against 44–50 of 200 on the surrogate holdout.

| inverter | match | mismatch | no box in `t̂` (cap-severed + prose) | no box in `y` | graded match rate | mismatches read → genuine / equivalent-form or partial | inconsistent with `y`, estimated |
|---|---:|---:|---|---:|---:|---|---:|
| **7b-sum** | 2,232 | 630 | 1,623 (0 + 1,623) | 5 | **78.0 %** (n=2,862) | **54 of 630 read** (14 at random + 40 stratified by the R1 split below): **12 genuine** — 62334 (15 → 20), 41999 (3/2 → 1/2), 84501 (865 → 866), 34593 (14 → 5), 38343 (1500 → 300√13), 63278 (49 → 75), 13503 (1/16 → 1/32), 80284 (6 → 5), 62012 (2 → √5), 13628 (16 → 153), 53360 ((3/4, 3) → (0, 3)), 12873 (43200/13 → 3334) — the victim's `y` = R1 in every one; **1 alternative-valid** — 41207 (the same four expressions with `a`/`b` relabelled); **40 equivalent-form or partial** — spacing (`\le` vs `\leq`), a proof's statement paraphrased, the identity's right-hand side alone, one solution of two or one part of a multi-part answer boxed last (25059, 24806, 36867, 47078, 65418, 64939, 34086 …); **1 not gradable** — 96295 (a code problem, `t̂` boxes an expression) | **≈ 5 %** of graded — stratified estimate 111 × 10/11 + 193 × 2/9 ≈ 145 of 2,862 (n = 54 read; the "neither" bucket's 2 of 9 carries the uncertainty) |
| **7b-nosum** | 2,138 | 598 | 1,824 (0 + 1,824) | 5 | **78.1 %** (n=2,736) | **30 stratified**: **5 genuine** — 20553 (131/10 → 457/60), 18551 (No → Yes), 38531 (C → D), 67931 (6 → 8), 55566 (18 → 28), `y` = R1 in all five; **1 ambiguous** — 99838 (a worked instance `11² = 121` for a general formula); **23 equivalent-form or partial**; **1 not gradable** — 108977 (code) | **≈ 4–5 %** — "`y` = R1, `t̂` ≠ R1" 5 of 6 genuine (bucket 131), the other buckets 0 of 23 (n = 30 read) |
| **1.5b-sum** | 2,046 | 689 | 1,328 (0 + 1,328) | 5 | **74.8 %** (n=2,735) | **30 stratified**: **8 genuine** — 71317 (11 → 14), 24289 (28 → 40), 25489 (7 → 10), 75213 (384 → 450), 20844 (3430 → 4960), 12327 (boxes `2` for a two-part fraction answer), 69446 (boxes `0` for a piecewise formula), 5471 (`12` where `y` says divisible by 24); **21 equivalent-form or partial** (option letters `B`/`E` for `y`'s "(B) 125", one case of a multi-case answer, a proof's statement paraphrased); **1 ambiguous** — 27979 | **≈ 9 %** — "`y` = R1, `t̂` ≠ R1" 5 of 6 (bucket 213), "neither" 2 of 6 (187), "R1 unboxed" 1 of 6 (27), the rest 0 of 12 (n = 30 read) |
| **1.5b-nosum** | 1,855 | 636 | 1,639 (0 + 1,639) | 5 | **74.5 %** (n=2,491) | **30 stratified**: **6 genuine** — 66089 (√2007 → 0), 18551 (No → Yes), 57914 (4000 → 40), 80108 (a different closed form), 68824 (21 → 30), 2923 (the opposite conclusion); **1 ambiguous** — 65907 (`m + n − 1` for `y`'s 11); **23 equivalent-form or partial** — including 53204, where `t̂`'s box is character-identical to `y`'s `aligned` block and the grader still fails it | **≈ 6 %** — "`y` = R1, `t̂` ≠ R1" 4 of 6 (bucket 204), "R1 unboxed" 2 of 6 (17), the rest 0 of 18 (n = 30 read) |

Across the four: genuinely inconsistent ≈ 5 / 4–5 / 9 / 6 % of graded rows, the 1.5B-sum inverter the
worst — the same arm ordering as Phase 2 (7B 1–3 %, 1.5B 6–10 %), and in every genuine case read the
victim's `y` agreed with R1, so on split B the override behaviour hurts, never helps (`docs/09` 7.15).

The graded rate fell from Phase 2's 94–96 % to 78 %, and the sample says most of the fall is the
grader meeting the victim's `\boxed{}` style — the victim boxes full sentences, multi-part answers and
whole identities (`\boxed{\text{All terms } a_1, a_2, \ldots \text{ are integers.}}`), and `t̂`, trained
on R1-Distill traces, boxes the value or a paraphrase. Against R1's answer (the dataset's solution,
paired on the 2,749 rows gradable on both sides): `y` = R1 73.7 %, `t̂` = R1 79.3 %; of the 630
mismatches with R1 boxed (579): `y` = R1 and `t̂` ≠ R1 **111**, `t̂` = R1 and `y` ≠ R1 262, both 13,
neither 193 — the 262 is the same equivalent-form miss landing on `y` (the victim is 97 % on
MATH500 at this operating point). The R1 split is a good stratifier: of the rows read, the
"`y` = R1, `t̂` ≠ R1" bucket is genuine 10 times in 11, "`t̂` = R1, `y` ≠ R1" and "both" 0 in 20,
"neither" 2 in 9, "R1 unboxed" 0 in 9. So the genuinely inconsistent rate is **≈ 5 % of graded
rows** (≈ 145 of 2,862), against Phase 2's 1–6 % on the surrogate holdout — the same class, the
same shape (the trace argues to a different number), and the same treatment: `docs/09` 7.15
leaves it in. The register difference is the rest: Phase 2's `y` was the surrogate's own R1-style
`\boxed{value}`; the victim's is a sentence.

---

## 6. Three traces read per inverter (+ the two degenerate-`b*` rows)

| inverter | row | `t` → `t̂` tokens | reaches the given answer? register |
|---|---|---|---|
| 7b-sum | short, idx 71717 (dot product, `y = −26`) | 58 → 121 | **Yes**, in prose, no box: three products summed. |
| | median, idx 13954 (four numbers, GP then AP; `y` boxes 75/4, 45/4, 27/4, 9/4) | 1,200 → 3,597 | **Yes, plus a second set**: finds both (3, 6, 12, 18) and (75/4, …) and boxes each number separately — the grader reads the last box `9/4` and calls it a mismatch. |
| | long, idx 86300 (Christmas tree, minimum rope sum; `y` = the centre) | 14,004 → 3,279 | **Yes**: symmetry argument, boxes "the center of the rectangle" twice (the duplicated "Final Answer" is what the loose loop test catches). The victim spent 14k tokens here. |
| | degenerate `b*`, idx 64731 (`b*` = `\boxed{\frac{3}{10}}`) | — → 1,925 | **Yes**, 3/10 by tan x = 3, twice over — an effectively no-summary inversion. |
| | degenerate `b*`, idx 6814 (`b*` = `**Final Answer** \boxed{2144}`) | — → 5,538 | **Yes**, 2144 via gcd(15N−7, 22N−5) = 79, N = 79k + 11. |
| 7b-nosum | short, idx 71717 (dot product, `y = −26`) | 58 → 171 | **Yes**, in prose, no box. |
| | median, idx 75091 (regular octagon, digits 1–9 on four lines through the centre; `y = 1152`) | 1,225 → 3,045 | **Yes**: centre ∈ {1, 5, 9}, four pairs summing to T, 3 · 4! · 2⁴, boxes 1152. |
| | long, idx 86300 (Christmas tree; `y` = the centre) | 14,004 → 1,932 | **Yes**: geometric-median symmetry argument, boxes "The center of the rectangle" — a seventh of the victim's length. |
| 1.5b-sum | short, idx 71717 (dot product, `y = −26`) | 58 → 796 | **Yes**, boxed — after a recap and a re-check; this arm never writes a short trace (p05 662 against the 7B arm's ~180). |
| | median, idx 112108 (chemistry: amine classes and basicity; `y` names triethanolamine) — a draw-3 rescue | 1,094 → 1,178 | **Mostly**: classes and the aqueous/gas-phase orderings match `y`; the everyday example becomes "methoxyethanol", not `y`'s triethanolamine. No box. |
| | long, idx 86300 (Christmas tree; `y` = the centre) | 14,004 → 6,800 | **Yes**, boxes `(a/2, b/2)` after 22k characters of coordinate algebra — half the victim's length, twice the 7b-sum forgery's. |
| | degenerate `b*`, idx 64731 | — → 714 | **Yes**, 3/10. |
| | degenerate `b*`, idx 6814 — a draw-2 rescue | — → 4,012 | **Yes**, 2144 (draw 1 capped). |
| 1.5b-nosum | short, idx 71717 (dot product, `y = −26`) | 58 → 602 | **Yes**, boxed, with a double-check. |
| | median, idx 28360 (circle on a leg bisects the hypotenuse; `y` = 45°, 45°, 90°) | 1,110 → 2,066 | **Yes**, in prose, no box: coordinates give b = a, so 45-45-90. |
| | long, idx 86300 (Christmas tree; `y` = the centre) — a draw-2 rescue | 14,004 → 4,380 | **Yes**, boxes `(a/2, b/2)`; draw 1 had capped. |

Register, all rows: R1-Distill's "Okay, so I have this problem…" monologue, unchanged from Phase 2;
the victim's `t` (`phase3.md` §4) is terse working notes. The forged trace re-derives the victim's
whole worked answer at monologue length, which is where the 2.55× comes from.

---

## 7. File layout, gates, ORACLE hygiene

All under the main checkout's `bench/results/phase4/` (gitignored `.jsonl`; the `.json` records are
committed in this repo's `bench/results/phase4/`).

| file | rows | what | who reads it |
|---|---:|---|---|
| `attack-{sum,nosum}.jsonl` | 5,045 each | the prompt files (§0) — no `t` anywhere | the generation path |
| `forged-{tag}-draw1.jsonl` | 5,045 each | the first draw, untouched — its cap-hit is the inverter's property | the record |
| `forged-{tag}-draw{2,3}-prompts.jsonl`, `-draw{2,3}.jsonl` | 1,174 / 735 · 1,079 / 642 · 1,914 / 1,284 · 1,792 / 1,192 | the redraws | `assemble` |
| **`forged-{tag}.jsonl`** | **4,490 · 4,565 · 4,068 · 4,135** | the training artifact: `{idx, domain, x, y, b, t_hat, raw, gen_tokens, finish_reason, draw}`, every row `finish_reason != "length"` | **Phase 5's Synthesized-Trace conditions** |
| `forged-{tag}-draws.json` | — | per-draw counts, dropped idx, loops strict/loose (committed) | Phase 5 (the idx sets), this record |
| `forged-{tag}-consistency.json` | — | per-row `gold / pred / bucket / r1` (committed) | re-gradable without regenerating |
| `merge-check-{tag}.json` | — | the tensor-diff proof of the merge (committed) | §1 |
| `forged-{tag}-probe.jsonl` | 30 | the smoke; disposable, never spliced | §2 |

**Gates, all passed.** Formatter self-test (§0). Per draw: `invert.py`'s 0 empty / rows out == in.
Per assemble: draw n's idx == draw n−1's capped set · final idx == the prompt file's minus the
dropped, in order · 0 capped · 0 empty · no `t_true` content · nothing dropped after fewer than three
draws · key set as specified. Per final file, `phase1_stats.py --final`: 0 capped, 0 empty, no
duplicate idx, every idx in the oracle with `y` equal to the oracle's `y` (the pairing check),
median `t̂ ≥ 0.5 × t`. A closing check over the four final files re-verified: rows == attack idx
minus the reported drops (set and order), 0 capped, 0 empty, no `t`/`t_true`/`trace`/`completion`
key, no `b` text in any nosum row, every draw-1 / consistency / draws file present — **PASSED**.

**ORACLE hygiene.** Every path handed to `invert.py` passes the driver's `no_oracle()` guard;
`phase4_format.py` and `phase4_draws.py` assert the same on their inputs. `grep ORACLE` over the four
run logs finds exactly 6 lines each: the driver's "stats — the ORACLE read" note (once for the smoke,
once for the run), the stats script's "paired against …ORACLE.jsonl … read-only, lengths only"
banner, and its "`t_true` tokens (paired by idx from the ORACLE file)" line — the stats step, and
nothing else. The oracle file was opened eight times in the phase (four smokes, four final stats),
all in the main thread, all lengths-only.

---

## 8. Budget — measured

| step | measured |
|---|---:|
| step 0 (formatter, draws, stats extension, driver, self-tests) | ~1.5 h, no GPU |
| four merges | ~1 min each |
| four 30-row smokes | ~3 min each (engine load + 30 rows + stats) |
| 7b-sum generation (3 draws) | 6.37 h (35.0 M tokens) |
| 7b-nosum generation | 5.35 h (32.4 M) |
| 1.5b-sum generation | 8.60 h (46.5 M) |
| 1.5b-nosum generation | 7.42 h (44.1 M) |
| **phase GPU time** | **≈ 28 h** (27.7 h generation + ~0.3 h merges/smokes/stats) — 158 M generated tokens |
| realized throughput, full batches | 1,551–1,739 tok/s on draw 1 (5,045 sequences), 1,460–1,600 on the redraws; the 30-sequence smokes ran 900–1,080 |
| halt for the user's decision | ~5 h of GPU idle between 7b-sum and 7b-nosum (03:21–08:06 on 2026-09-03), not counted above |

`docs/15` §3 budgeted ~3–4 h per inverter (~13–17 h) from a `t̂` mean of 2,507–3,175 and a 3–17 %
cap-hit; the measured means were 3,974–4,776 on draw 1 with 21–38 % cap-hits and 60–76 % re-cap
rates, so the redraws cost 1.8–4.3 h per inverter instead of the budgeted minutes. The realized rate
(~1,600 t/s) matched the Phase 2 prior; the volume did not.

---

## 9. Definition of done (`docs/15` §10)

- [x] `phase4_format.py` + self-test committed; `attack-{sum,nosum}.jsonl` built, 5,045 rows each, no `t` anywhere
- [x] four `-draw1.jsonl` and four final `forged-*.jsonl` on disk; every final row `finish_reason != "length"`; drops == the reported counts (555 / 480 / 977 / 910)
- [x] every full run preceded by a reported probe/smoke; CHECKPOINTs 0–3e sent to `big-boss`; the one STOP line tripped (§2) and the two drop-policy decisions (§3) adjudicated by the user on the record
- [x] consistency JSONs committed; rates reported (§5), no filtering applied
- [x] the paired `t̂`-vs-`t` length table with supervision totals, all four inverters (§4)
- [x] `docs/09` 7.14 / 7.15 measured; `docs/10` Phase 4 closed with ≈ 28 h; README updated
- [x] merged weights deleted after each inverter; disk 26 GB free at the end (28 GB at the start; the four final files and their draws hold ~1.1 GB)

---

## 10. Handed to Phase 5 (`docs/15` §9)

| item | where / what |
|---|---|
| **The four final idx sets** | each `forged-{tag}.jsonl`'s `idx` column; equivalently the attack file's idx minus `dropped_idx` in the committed `forged-{tag}-draws.json`. Sizes 4,490 / 4,565 / 4,068 / 4,135 |
| **The intersection, recommended for every condition** | **3,616 rows** — math 2,886 · code 477 · chemistry 68 · puzzle 65 · biology 60 · physics 60 (the attack file: 3,889 · 899 · 69 · 65 · 60 · 63). Code is under-represented by a third; the small domains are intact. Whether the oracle and Answer-only conditions also train on it is Phase 5's call (Phase 3's shared-filter logic says yes) |
| **Supervision length** | forged `[t̂; y]` 3,128–3,677 median vs oracle `[t; y]` 1,737–1,921 on the same rows (§4) — **1.78–1.94×**; a length-matched control is a Phase 5 proposal that must pass `docs/10`'s four questions |
| **Register** | oracle = the victim's terse working notes; forged = R1-Distill's "Okay, so…" monologue that re-derives the whole worked answer. Any oracle-vs-forged gap carries style as well as content and length |
| **Cap asymmetry** | oracle `t` ≤ 14,336 (Phase 3's cap; p95 7,778–8,599 on the kept rows), forged `t̂` ≤ 8,192 by construction and < 8,192 after the drop (p95 6,862–7,374). The forged sets have no tail past 8,192; the oracle does |
| **Answer consistency** | ≈ 4–9 % of graded forged rows argue to a different answer than the `y` they are trained beside (§5); no filtering applied; the per-row buckets are in the committed consistency JSONs for a filtered condition if Phase 5 proposes one |
| **The 1.5B arm's data** | 4,068 / 4,135 rows against the 7B arm's 4,490 / 4,565 — the data-quantity difference the redraw policy was meant to prevent, now 9 % between arms; the intersection removes it |
| **Budget prior for Phase 5** | `docs/10`'s "~20 h / five conditions" predates four forged sets; re-plan before running |
