# Phase 3 Handoff — Query the Victim

> ## ⚠️ THIS IS THE PRE-RUN PLAN, NOT THE RESULTS
>
> Written **before** Phase 3 ran. Phase 3 completed 2026-09-02 and
> **`docs/results/phase3.md` is authoritative wherever the two disagree.** Several numbers below
> were superseded by measurement:
>
> | this document says | measured |
> |---|---|
> | §3 budget: ~30–50 h at an unmeasured 140–170 t/s | **66.32 h at 123.6 gen t/s** on 12 slots (`phase3.md` §4) |
> | §4.1: the victim is queried with **no system prompt** | the GGUF's template **injects its own** ("Reasoning effort is set to xhigh…") unless effort is set; the run used `--reasoning-effort medium`, the only level that renders no system turn (`phase3.md` §2, `09` 7.19) |
> | §4.1: user turn only | user turn **+ a boxed-answer instruction**, which took boxed answers from 56 % to ~100 % of kept rows (`09` 7.20) |
> | §4.2: expect a cap-hit of ≤ 5 % | **16.2 %** (977 of 6,022) |
> | §6 gate: cap-hit band **0–15 %** | widened to **0–25 %** *after* seeing 16.2 % — recorded as a post-hoc threshold change, with both gate runs on the record (`phase3.md` §4, `09` 7.22) |
> | §6 STOP: `y` agrees with R1 on ≥ 75 % | fired at **73.1 %**, resolved as an extraction artifact, not victim error (`phase3.md` §4) |
> | §1: victim traces run 3,000–4,000 tokens/row | median **1,400** — and *shorter than the forgeries imitating them*, the phase's lead finding (`phase3.md` §5) |
>
> The **method, conventions and file layout** below all held and are still the reference; it is the
> numeric expectations that did not.

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phases 0, 1 and 2 are complete; Phase 3 is yours.

Read first: `docs/00-overview.md` (what the paper does), `docs/10-run-plan.md` (the plan, the four
questions to ask before proposing an experiment, and the three Phase 4 decisions already taken),
`docs/results/phase1.md` (how the surrogate data was built — Phase 3 is the same machinery pointed
at the victim), `docs/results/phase2.md` §5 (what the inverters expect as input), `docs/09` (where
and why we differ), and `docs/11` §5 (conventions — they apply in full). This document is the
operational summary.

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path | Note |
|---|---|---|
| Victim | `~/trace-inversion-bench/models/Qwen3.8-27B-IQ4_XS.gguf` (14.62 GiB) | llama.cpp **b10450** (Homebrew). Phase 0 served it at `-np 6 -c 196608 -fa on -ctk q8_0 -ctv q8_0 --jinja` (`bench/run_victim_full.sh`) |
| Split manifest | `bench/phase1/splits.json` — `B` = 9,000 idx, seed 20260826 | `B == sorted(perm[9000:18000])`, verified 2026-08-30. **Never run `phase1_split.py` without `--only-b`** — as written it rewrites `splits.json` with A=9,000 and overwrites `promptsA.jsonl` |
| Generator | `bench/phase1_generate.py` | `--prompts --out --no-system --max-new-tokens --target-kept --max-rows --timeout --max-prompt-chars --concurrency`. Streams, **resumes** from `--out`, prints row 0 as the server renders it, splits trace/answer on `reasoning_content` (falls back to `</think>`), `capped = finish_reason=="length" or empty answer` |
| Compressor `C'` + `π` | `bench/phase1_compress.py`, `bench/phase1/prompts.py` → `PI_SYSTEM` | **π is frozen.** All four inverters were trained on its output; changing it shifts their input distribution. `--fix --seed N` regenerates bad rows, `--drop-bad` removes what is still bad, `--max-len` is vLLM's `max_model_len` (a KV budget, not a length target) |
| Stats / gates | `bench/phase1_stats.py` | `--mode traces --paired` (cap-hit band, error rate, empty-kept-trace gate, paired vs R1 on the same prompts, drop decomposition); `--mode summaries` (Table 1 + `D₂` integrity); `r1_answers()` (R1's boxed answer per OpenThoughts row) and `eval_baseline.extract_boxed` / `grade` (main thread) |
| Concurrency sweep | `bench/sweep_concurrency.sh <gguf> <ctx/slot> [slots…]`, `NTG=` env | writes `docs/results/sweeps/`. **Ranks slot counts; never budgets** (`docs/11` §5) |
| Driver pattern | `bench/run_phase1_gen.sh` | server up → `/health` poll → generate → kill → stats. Copy it; never edit it while it runs |
| Gate self-test pattern | `bench/test_phase1_gates.py` | |
| `D₂`, both arms | `bench/results/phase1/d2-{7b,1.5b}.jsonl` — 5,006 / 5,028 rows | schema `{idx, domain, source, x, y, b, t, summary, summary_tokens, finish_reason}`. **The attack file this phase writes is that schema minus `t`** |
| Inverters | `bench/results/phase2/inverter-*/checkpoint-{402,404}` (epoch 2, decided) | not used this phase; Phase 4 feeds them the attack file |
| OpenThoughts | HF cache, `llamafactory/OpenThoughts-114k` (1.1 GB) | `--paired` and `r1_answers()` load it; works with `HF_HUB_OFFLINE=1` |

### Environment, 2026-08-30

| | |
|---|---|
| GPU | RTX 4090, 24,564 MiB; idle (406 MiB used) |
| Disk | **30 GB free** (94 %). Phase 3 writes ~300 MB and downloads nothing |
| `.venv-vllm` | Python 3.12 · vLLM 0.27.1 · transformers 5.15.0 — runs the generator, the compressor and the stats. Launch vLLM with `PATH="$PWD/.venv-vllm/bin:$PATH"` (`docs/06` §1.8) |
| Tokenizers | `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B` ship an identical `tokenizer.json` (sha256 `5f9e4d49…`), so "the Qwen3.5 tokenizer" is one tokenizer: the compressor's, the inverter's and the student's. The engine's `gen_tokens` is Qwen3.8's count. State which before comparing |
| Ports | 8079 for the victim (Phase 0's); 8078 is the surrogate's |

---

## 1. Where things stand

| Role | Model | Status |
|---|---|---|
| Surrogate `V'` ×2 | R1-Distill-7B / -1.5B, llama.cpp | done — `D₂` per arm |
| Compressor `C'` | Qwen3.5-4B, vLLM, π | done — Table 1 passed on both arms; **reused unchanged here** |
| Inverter `I` ×4 | Qwen3.5-4B + LoRA | done — epoch-2 adapters serve Phase 4 |
| **Victim `V`** | **Qwen3.8-27B IQ4_XS, llama.cpp** | **this phase** |
| Student `S` ×5 | Qwen3.5-2B, full FT | Phase 5 |

Facts from earlier phases that shape this one:

- **What the inverters expect.** Each was trained on `(x', y', b') → t'` where `y'` is the surrogate's
  *whole* post-think answer text (median 488 tokens on the 7B arm) and `b'` = π(t'). The victim must
  supply the same object types: `y` = its whole post-think answer, `b*` = π(t). Phase 4's prompt is
  `x + y + b*` + a 386-token preamble; `invert.py` widens `max_model_len` if a row needs it.
- **The victim, measured in Phase 0** (`bench/results/Qwen3.8-27B-IQ4_XS-full.jsonl`, re-read 2026-08-30):
  3,958,751 generated tokens in 10.67 h at 6 slots × 32,768 = **103 t/s realized**. Per-benchmark
  `gen_tokens` (trace + answer, engine count): MATH500 median 593 / mean 1,983 / p95 9,596; JEEBench
  median 3,484 / mean 5,760 / p95 17,795. Share over 14,336: **1.2 % / 9.9 %**; over 8,192: 6.4 % / 23.5 %.
  0 of 1,015 texts contain a `<think>` tag: llama-server's default reasoning format handed the trace
  back as `reasoning_content`, which is the path `phase1_generate.py` splits on. 5 of 1,015 requests
  were `ReadTimeout` at `--timeout 1800`.
- **The victim's register is not the surrogate's.** Its traces read *"We need solve physics problem.
  Need reason step by step, final boxed option letter."* — terse notes, where R1-Distill writes
  *"Okay, so I have this equation…"*. The oracle condition will train on the former, the forged
  condition on the latter. Not a Phase 3 decision; read three traces and record it (§6).
- **Split B, measured 2026-08-30:** 9,000 idx, A∩B = ∅, domain mix 78.2 % math / 17.5 % code /
  4.3 % science+puzzle (the first 200 in permutation order: 147 / 41 / 12). Prompt chars median 260,
  p95 1,844, **max 5,624** — none over 8,000. R1's own traces on B: median 13,649 chars, p95 54,000.
- **π's known failure mode** (`phase1.md` §2): on very short traces `C'` pads by re-deriving and runs to
  the 2,048 cap — 6 of 5,012 rows on the 7B arm. The victim writes short traces on easy math, so expect
  more of these here; the `--fix` → `--drop-bad` policy exists for exactly this.

---

## 2. What Phase 3 produces

Paper Stage 2, first half (§4.2): query the victim on the victim split.

```
x  (split B)  ─→  VICTIM V  ─→  trace t  +  answer y
                                  │
                                  ├─→  COMPRESSOR C' (π)  ─→  b*      ← what an API would show
                                  │
                                  └─→  withheld: the Victim-Trace ORACLE (Phase 5's ceiling row)

attack file   victimB-attack.jsonl   = {(x, y, b*)}         Phase 4 inverts this; Answer-only and
                                                            Summary+Answer train on it
oracle file   victimB-ORACLE.jsonl   = {(x, y, b*, t)}      Victim-Trace condition only
```

| Path | What |
|---|---|
| `bench/results/phase3/promptsB.jsonl` | 9,000 rows `{idx, order, prompt, domain, source}` in permutation order, from `phase1_split.py --only-b` |
| `bench/results/phase3/victim-traces-ORACLE.jsonl` | the generator's raw output, Phase 1 schema (`prompt, raw, trace, answer, gen_tokens, finish_reason, capped, secs`). Contains `t`. Resumable |
| `bench/results/phase3/victimB-ORACLE.jsonl` | `phase1_compress.py` output — the `D₂` schema with `t`. **Read by Phase 5's Victim-Trace condition and by nothing else** |
| `bench/results/phase3/victimB-attack.jsonl` | the same rows minus `t`, from `bench/phase3_split.py`. **The only Phase 3 file the attack path opens** |
| `bench/results/phase3/victimB-vs-r1.json` | per-row `(y_boxed, r1_boxed, agree)` from the `--vs-r1` report; small, committed like Phase 2's `-consistency.json` |
| `docs/results/sweeps/Qwen3.8-27B-IQ4_XS-ctx16384-ntg4096.md` | the sweep |
| `bench/run_phase3_gen.sh` · `bench/phase3_split.py` · `phase1_split.py --only-b` · `phase1_stats.py --vs-r1` | the four code changes |
| `docs/results/phase3.md` | the committed record (§6) |

All `.jsonl` under `bench/results/` is already gitignored.

---

## 3. Measured budgets

**Throughput.** The only realized victim rate is Phase 0's **103 t/s at 6 slots × 32,768**. The
`docs/08` sweep (2,560 ctx/slot, so not this run's shape) read 8 slots → 137, 16 → 202, 32 → 303
gen t/s; against it, Phase 0 realized ~0.86 of the interpolated 6-slot figure — a far smaller
swept/realized gap than the 7B surrogate's 2.4×, consistent with a 27B being compute-bound rather
than ragged-batch-bound. One point; do not lean on it. **303 t/s is the 32-slot / 2,560-ctx figure
and is unreachable at 16k per slot.**

**Slots.** KV is one budget shared across slots: 6 × 32,768 = 196,608 tokens loaded at 21,910 MiB;
8 × 40,960 = 327,680 OOM'd (`docs/results/baselines.md`). So **12 × 16,384 = 196,608 is the same
budget Phase 0 ran and should load**; 16 × 16,384 = 262,144 is untested — sweep it, and if it fails
at load the sweep says so and 12 is the answer. The recurrent-state cache is per slot and small at
these counts (`docs/08` §3).

**Token volume — unknown until the probe.** The victim's length on OpenThoughts prompts has never
been measured. Bracket from Phase 0: MATH500-like (mean 1,983) to JEEBench-like (mean 5,760);
NuminaMath problems sit between, so ~3,000–4,000 tokens per row including the answer is the
working guess. 5,300 rows × 3,500 ≈ **18.5 M tokens**.

| slots | realized t/s (assumed) | 18.5 M tokens |
|---:|---:|---:|
| 6 (Phase 0 config) | 103 measured | 50 h |
| 12 | 140–170 | **30–37 h** |
| 16, if it loads | 170–200 | 26–30 h |

**Phase 3 is ~30–50 h of generation, not the ~10–15 h in `docs/10`.** That estimate divided
~15 M tokens by 303 t/s; the rate was the wrong sweep point. Correct `docs/10` when the probe
gives a real number. Probe (200 rows, ~0.7 M tokens): ~1.2–1.5 h. Compression of ~5,050 traces at
`max_model_len` 20,480: ~1.5–2 h (Phase 1: ~1–1.5 h per 5k arm at 16,384). Sweep: ~20 min.

---

## 4. Decisions — fixed for this phase

### 4.1 How the victim is queried

The OpenThoughts **user turn only, no system prompt** (`--no-system`), through the GGUF's own chat
template (`--jinja`, thinking on by default), at the project's one sampling protocol:
**temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · top_k −1**. This is exactly how Phase 0
benchmarked the victim, so its Phase 0 numbers describe the model Phase 3 queries. The paper's API
victim receives only the problem; the R1-Distill system prompt Phase 1 injected is that surrogate's
(`docs/09` 7.8), not the victim's. Default `--reasoning-format` (it returned `reasoning_content` in
Phases 0 and 1; the probe verifies it again on row 0). No per-request seed, as in Phase 1.

### 4.2 Per-slot context 16,384 · `max_new_tokens` 14,336 · capped rows dropped

- **14,336 = 16,384 − 2,048.** Every kept victim row — prompt (≤ 5,000 chars ≈ ≤ 2,000 tokens) plus
  trace plus answer — then fits the student's `max_length` 16,384 (`docs/09` 3.5), so the
  Victim-Trace oracle trains on unsevered traces and Phase 5 inherits no truncation decision.
- **16k per slot is what buys 12 slots** (§3). Prompts over 5,000 chars are skipped
  (`--max-prompt-chars 5000`; split B has 0 over 5,624, so this skips at most a handful) so no request
  can exceed its slot and trigger a silent prompt truncation. After the run, grep the server log for
  `truncated` and `context shift`; report the counts.
- **A capped row (`finish_reason == "length"`) or a row with no post-think answer has no `y`, so it
  is dropped** — from all four split-B conditions at once, since none can use it. That makes the drop
  a filter on split B's prompt distribution, not a confound between conditions. Report the rate: this
  is the number behind `docs/10`'s "Phase 4 serves split B essentially unfiltered". Expect ≤ 5 %
  (Phase 0: 1.2 % of MATH500 and 9.9 % of JEEBench rows exceeded 14,336).
- `--timeout 3600`: a 14,336-token generation at ~12 t/s per slot takes ~1,200 s; Phase 0 lost 5
  rows to `ReadTimeout` at 1,800.

**Considered and rejected.** *32,768 per slot, 6 slots* (Phase 0's config): no drops, but half the
throughput, and a trace over ~15 k tokens could not train the oracle anyway. *12,288 per slot, 16
slots*: +19 % swept throughput, but ~5 % more of split B dropped and the longest-trace prompts leave
every condition — a capped row also wastes its full 10 k tokens, so the saving is smaller than it
looks. *Raising the cap in Phase 5 instead*: the layout is fixed here; `docs/10` says this is a
Phase 3 decision.

### 4.3 Over-generate to 5,040 kept rows

The generator stops at `--target-kept`; summary `--drop-bad` then removes a few rows (Phase 1: 6 and
15). 5,040 keeps the attack file at ≥ 5,000 after that. Split B's 9,000 rows bind only at a 44 %
cap-hit, ten times the expected rate.

### 4.4 `b*` = π(t): the same compressor, unchanged

Run `phase1_compress.py` on the victim's traces exactly as Phase 1 ran it on the surrogates':
`Qwen3.5-4B`, `PI_SYSTEM` as committed, temperature 0.7 / top_p 0.9 / rep 1.05, `--max-tokens 2048`,
`--seed 1234`. One change: **`--max-len 20480`** — victim traces reach 14,336 tokens and the prompt
(~1,600 tokens of π) plus a 2,048-token summary no longer fits 16,384; `max_model_len` is a KV budget
and changes no output. VRAM is fine (4B bf16 + 640 MiB per 20 k-token sequence).

This is, by inference, the paper's own path for its open-weight victim: Table 3's summary-setting
rows for R1 need `b*`, R1's API emits none, and Table 1 is "C' on R1 traces". **The paper never says
so explicitly** — it is the only way those rows could exist, not a stated procedure. So no deviation
is logged, but the inference is; and note that on this path the summary distributions match at train
and serve time by construction, which the paper's GPT track had to engineer (Table 1). Log that as a
`docs/09` row (§7), stated, not claimed as an advantage.

**Table 1 on `b*` is a report, not a gate.** π is frozen; a miss here is a finding about π on victim
traces, not a defect to fix. Print the four statistics next to Phase 1's (7B arm: median 583, headers
100 %, first person 99.9 %, LaTeX 80.5 %). The `D₂`-integrity lines (`empty x/y/b/t`, summaries at the
cap, duplicate idx, row count) **are** the gate.

**Bad summaries:** one `--fix --seed 1235` pass, then `--drop-bad`; report both counts and the dropped
idx. A dropped row leaves the oracle file and the attack file together (§4.5).

### 4.5 Structural withholding — the file layout is the enforcement

`docs/10` Phase 3: *"Write `t` to a different file from the `(y, b*)` the attack consumes, so leaking
it requires opening a file the attack path has no reason to touch."* Done as:

- `victimB-ORACLE.jsonl` — the compressor's normal output (`D₂` schema, with `t`). Name says ORACLE.
- `victimB-attack.jsonl` — derived by `bench/phase3_split.py`: the same rows with `t` removed and
  nothing else changed. Keys: `idx, domain, source, x, y, b, summary, summary_tokens, finish_reason`.
- `phase3_split.py` **asserts** (exit 1): the attack file has exactly those keys and no `t`/`trace`/
  `raw`; the two idx sets are equal; and for every row the oracle's `t` (stripped) is not a substring of
  the attack row's JSON — a mis-join that copied the trace into any field fails here.
- **Rule for Phases 4–6:** Phase 4 and the Answer-only / Summary+Answer conditions read
  `victimB-attack.jsonl` and nothing named ORACLE. The Victim-Trace condition reads the ORACLE file.
  Length reports comparing forgeries to `t` are read-only measurements in the main thread, not part of
  any training input.

### 4.6 Victim answer vs R1's answer — report only

`phase1_stats.py --mode traces --vs-r1`: the last `\boxed{}` in the victim's `y` against the last
`\boxed{}` in the OpenThoughts row's own R1 solution (`r1_answers()`), graded with
`eval_baseline.grade` in the main thread; buckets `agree / disagree / no box in y / no box in R1`;
per-row JSON. **R1's answer is the dataset's solution, not ground truth** (`phase2.md` §5): a
disagreement is "the two models differ", and the grader rejects equivalent forms (~3–4 % of
mismatches in Phase 2 were equivalent-form) — read the disagreements before calling any of them
victim errors. Four questions: (1) it tells whether the run is querying the victim correctly — a
template or split error shows up as a collapse in agreement before 40 h are spent — and it is the
only correctness signal on `y` before Phase 5 trains on it; (2) `y` vs R1's answer, same prompt,
graded as Phase 2 graded the holdout; (3) nothing else varies; (4) every function exists. **No
filtering follows from it** — the paper verifies nothing (§7), and `docs/10` fixed "no consistency
filtering in the main condition". Expect ≥ 85 % on gradable rows (the 7B surrogate agreed 78 % on the
Phase 2 holdout; the victim is far stronger). Also report the same statistic for `D₂`'s `y'` on the
7B arm if it costs a minute — it is the surrogate's number for the same column.

### 4.7 Not in this phase

No inversion, no student formatting, no fidelity metrics (`docs/10` Phase 4 rejected them), no
re-benchmarking of the victim, no change to π, no second victim.

---

## 5. Order of work

| Step | What | Est. |
|---|---|---|
| **0** | Branch `phase3-build`. `phase1_split.py --only-b`: recompute `perm` from the seed, assert `sorted(perm[9000:18000]) == splits.json["B"]` and `A ∩ B = ∅`, write `bench/results/phase3/promptsB.jsonl` in permutation order, **touch nothing else**. Print domain mix and prompt-length quantiles (§1 has the expected values). `bench/run_phase3_gen.sh` from `run_phase1_gen.sh`: `--no-system --max-new-tokens 14336 --timeout 3600 --max-prompt-chars 5000 --target-kept 5040`, port 8079, per-slot ctx 16,384, output `victim-traces-ORACLE.jsonl`, stats call as in step 2. Commit both (explicit paths). | ~1 h |
| **1** | **Sweep:** `NTG=4096 bench/sweep_concurrency.sh Qwen3.8-27B-IQ4_XS.gguf 16384 8 12 16`. Report the table; take the best that loads. **Probe:** the driver with `--max-rows 200`. Report §6's probe list. CHECKPOINT; one turn for the boss to object. **Run:** relaunch without `--max-rows` (it resumes). After 30 min with a full queue, CHECKPOINT the realized t/s and the projection. | 0.3 h + 1.5 h + **30–50 h** |
| **2** | Gate: `phase1_stats.py victim-traces-ORACLE.jsonl --mode traces --cap 14336 --cap-hit-band 0 15 --trace-tokenizer Qwen/Qwen3.5-4B --paired --paired-gap-tol 100 --vs-r1`. Add `--vs-r1` (§4.6) before the probe, not after the run. Grep the server log for `truncated` / `context shift`. | ~0.5 h |
| **3** | Compress (§4.4) → `victimB-ORACLE.jsonl`; `--fix --seed 1235`; `--drop-bad`. `phase1_stats.py --mode summaries --target 5000`. | ~2 h |
| **4** | `bench/phase3_split.py` → `victimB-attack.jsonl` with the §4.5 asserts. Commit the script. | ~0.5 h |
| **5** | `docs/results/phase3.md` (§6); `docs/09` rows (§7); `docs/10` Phase 3 marked done with the measured hours and drop rate. Commit. | ~1 h |

Everything is sequential on one GPU. Steps 0 and the `--vs-r1` change are the only code before the
long run; do them first so the probe reports the agreement number.

---

## 6. Reports, gates, and what goes in `docs/results/phase3.md`

### Probe report — before the long run

`slots and ctx/slot (from the sweep)` · `rows generated 200` · `cap-hit at 14,336` · `"no answer,
other" count` · `reasoning_content present on every kept row` (row 0's message keys printed) ·
`gen_tokens median / mean / p95 (engine count)` · `trace and answer tokens median / p95 (Qwen3.5
tokenizer)` · `request errors` · `realized t/s` · `y vs R1 agreement on gradable rows` · `rows needed
for 5,040 kept` · `projected wall clock, stating the t/s and slot count assumed`.

### Record

- sweep table; probe report; the 30-minute full-queue rate
- generation: rows generated / kept / capped / no-answer / errors / skipped-long · wall clock ·
  realized t/s (fit over the regime, `docs/11` §5) · server-log `truncated` and `context shift` counts
- lengths (Qwen3.5 tokenizer, stated): trace and answer distributions, by domain; `gen_tokens` from
  the engine beside them
- paired vs R1 on identical prompts (`--paired`): medians, cap-hit both sides at 14,336, drop
  decomposition — the victim's terseness relative to R1 is the reproduction's analogue of the paper's
  Table 6 regime check
- `y` vs R1 agreement, buckets and rate; three disagreements read
- compression: rows in / bad after first pass / rescued by `--fix` / dropped; Table 1 on `b*` beside
  Phase 1's; summary tokens by domain
- **three victim traces read** (shortest, median, longest kept): whether `y` is a full worked
  solution or a bare answer, and one line on the register against the surrogates'
- the file layout (§4.5) and the `phase3_split.py` assert output
- final counts: ORACLE rows == attack rows == N ≥ 5,000

### Gates that exit non-zero

- `--mode traces`: request-error rate ≤ 1 % · 0 kept rows with an empty trace · no domain at 0 kept ·
  cap-hit within 0–15 %
- `--mode summaries`: `D₂` integrity only (Table 1 lines are a report — §4.4)
- `phase3_split.py`: keys · equal idx sets · no oracle trace inside any attack row

### STOP AND ASK — report, then wait

- no slot count loads at 16,384 per slot
- probe cap-hit above **15 %**, or "no answer, other" above **2 %** of rows (the reasoning/content
  split is wrong for this model)
- any kept row with an empty trace
- `y` agrees with R1 on under **75 %** of gradable probe rows
- the projection passes **72 h**, or a step exceeds its estimate by 2×
- request-error rate above 1 %, or errors arriving in a burst (a dying server)
- more than **2 %** of summaries still bad after one `--fix`
- `b*` median tokens outside **450–700**, or bold-header / first-person rate under **90 %** — π
  misbehaving on victim traces; do not touch π, report
- the `phase3_split.py` self-check fails
- any gate fails twice

---

## 7. `docs/09` rows to add or close

| Row | Change |
|---|---|
| 4.1 | victim query budget: 5 k → the realized attack-file row count |
| 7.10 | close the parenthetical "(victim truncation was 0.0–0.4 %)" with split B's measured cap-hit at 14,336 |
| new 7.16 | victim generation cap **14,336 at 16,384 per slot**, capped and no-answer rows dropped from all four split-B conditions (paper: API victim uncapped; no policy stated for the open-weight one) — HW + CHOICE; chosen so every kept row fits the student's 16,384 context |
| new 7.17 | victim queried with **no system prompt**; the surrogate ran with R1-Distill's (7.8). The asymmetry is the threat model's: the attacker picks the surrogate's prompt, the victim's API fixes its own |
| new 7.18 | `b*` = π(t) by the attacker's own `C'` — **matches the paper's open-weight-victim path** (Table 1 / Table 3); summary distributions match at train and serve by construction, where the GPT track had to engineer the match |
| new 4.8 | victim `y` graded against R1's answer, **report only, no filtering** ✅ matches the paper (§7 lists verification as future work) |

---

## 8. Conventions — Phase 3 specifics

`docs/11` §5 and `docs/13` §8 apply in full. What bites here:

| Rule | Why |
|---|---|
| **A sweep ranks slot counts and never budgets; budget from the first 30 minutes of the real run with a full queue** | the 10–15 h in `docs/10` came from a sweep point at the wrong context |
| **Never size a run from the probe's wall clock alone** | 200 rows on 12 slots drains its queue at the end; the tail runs under-occupied. Use the probe for cap-hit and lengths, the full-queue window for t/s |
| **The post-drain cap-hit is the first unbiased one** | in-flight rows are long-biased (`phase1.md` §5) |
| **Prefer the engine's own signal** | `finish_reason` for cap-hit, `usage.completion_tokens` for volume, `reasoning_content` for the split |
| **State the tokenizer** | engine = Qwen3.8's; recount = Qwen3.5's; `phase1.md` = R1-Distill's |
| **Never run `phase1_split.py` without `--only-b`** | it rewrites `splits.json` and `promptsA.jsonl` |
| **Never edit a running script; one sequential driver** | |
| **Nothing on the attack path opens a file named ORACLE** | §4.5 — the failure is silent and looks like success |
| **Explicit paths on `git add`; never `git add -A`** | `.claude/` is untracked in this checkout |
| **A supervising session writes nothing to the tree** | it audits `CHECKPOINT` lines |

---

## 9. Open items carried into later phases

| Item | Phase | Note |
|---|---|---|
| Formatting attack rows for the inverter | 4 | `phase2_format.to_row` requires `t` for the completion; Phase 4 needs the prompt-only form (`invert.py` already treats `completion` as optional) |
| Victim register vs surrogate register | 5–6 | oracle traces are terse notes; forgeries are R1-Distill prose. Any oracle-vs-forged gap carries style as well as content — read it that way |
| The oracle's length advantage | 5 | oracle traces run to 14,336; forgeries cap at 8,192 (as in the paper: R1 traces vs ≤ 8,192 forgeries) |
| Correct the "~10–15 h" Phase 3 estimate **everywhere it lives** | 3 | from the probe. It is in `README.md` (status table), `docs/10` (Phase 3 and the Totals table), `docs/11` §6 and `docs/13` §4.3 — six copies, verified by grep 2026-08-30. `docs/11` §5: grep both hyphen forms (`10-15 h`, `10–15 h`), confirm each hit is this quantity, fix every copy |
| 2B looping / eval protocol · OpenAI victim track | 6 / optional | unchanged |

---

## 10. Definition of done

- [ ] `promptsB.jsonl` (9,000, permutation order) from `phase1_split.py --only-b`; `splits.json` and `promptsA.jsonl` byte-identical to before
- [ ] sweep at 16,384 per slot committed to `docs/results/sweeps/`
- [ ] probe reported before the long run; full-queue t/s reported within the first hour
- [ ] `victim-traces-ORACLE.jsonl` with ≥ 5,040 kept rows; `--mode traces --paired --vs-r1` passes
- [ ] `victimB-ORACLE.jsonl` ≥ 5,000 rows; integrity gate passes; Table 1 on `b*` reported beside `b'`
- [ ] `victimB-attack.jsonl` derived; `phase3_split.py` asserts pass; no `t` anywhere in it
- [ ] `docs/results/phase3.md` committed; `docs/09` rows added; `docs/10` Phase 3 closed with measured hours
