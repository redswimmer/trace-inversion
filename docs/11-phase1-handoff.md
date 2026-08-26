# Phase 1 Handoff

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phase 0 is complete; Phase 1 is yours.

Read first: `docs/00-overview.md` (what the paper does), `docs/10-run-plan.md` (the plan),
`docs/results/baselines.md` (what we measured), `docs/09-deviations-from-paper.md` (where and why
we differ). This document is the operational summary.

---

## 1. Where things stand

Phase 0 fixed every model role on measurement. **Do not revisit these** without new evidence.

| Role | Model | Format / Engine | MATH500 | JEEBench |
|---|---|---|---|---|
| **Victim** `V` | `unsloth/Qwen3.8-27B-GGUF` → `IQ4_XS` | GGUF 4-bit, llama.cpp | 98.8% | 86.2% |
| **Surrogate** `V'` primary | `DeepSeek-R1-Distill-Qwen-7B` | GGUF F16, llama.cpp | 92.6% | 60.6% |
| **Surrogate** `V'` arm 2 | `DeepSeek-R1-Distill-Qwen-1.5B` | GGUF BF16, llama.cpp | 84.0% | 32.6% |
| **Compressor** `C'` | `Qwen/Qwen3.5-4B` | safetensors bf16, vLLM | — | — |
| **Inverter** `I` | `Qwen/Qwen3.5-4B` | safetensors bf16, TRL → vLLM | — | — |
| **Student** `S` | `Qwen/Qwen3.5-2B` | safetensors bf16, TRL → vLLM (**full FT**) | 79.0% | 47.8% |

Ordering achieved on JEEBench: **student 47.8 < surrogate 60.6 < victim 86.2** — the regime the
paper's argument requires.

**The organising rule for formats:** fixed inputs (never trained) run as GGUF on llama.cpp at the
highest precision that fits; trainable artifacts run as safetensors bf16 on vLLM so every
checkpoint evaluates without a conversion step. Only the victim is quantized, because 27B at bf16
is 55 GB.

### Environment

| | |
|---|---|
| GPU | RTX 4090, 24 GB (≈23.4 usable) |
| System RAM | 30 GB — tight, rules out CPU offload |
| Disk | **26 GB free (95% full).** Check before downloading anything. **36 GB is recoverable** — see below. |
| llama.cpp | Homebrew, **b10450 / ggml 0.20.0**. Older builds cannot load `qwen3_5`. |
| vLLM | 0.27.1 in `.venv-vllm` (Python 3.12) |
| Models | `~/trace-inversion-bench/models/` |
| Results | `bench/results/` (gitignored — 20-45 MB each, they embed raw text) |
| Logs | `bench/logs/` (gitignored) |
| Committed results | `docs/results/` — summaries + audits only |

> **36 GB of dead weight.** `~/.cache/huggingface/hub/models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF`
> holds five quants (Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M) of a model that appears in **no script, no
> log, and no phase of the plan** — a surrogate candidate `05` rejected as "DOES NOT FIT". Deleting
> it takes free space **26 → 62 GB**, which retires the train-evaluate-delete checkpoint policy in §5
> *and* its consequence that the Phase 6 eval protocol must be frozen before Phase 5 starts.
> OpenThoughts itself only needs 1.2 GB, so Phase 1 fits either way.

---

## 1a. Where the paper's verbatim prompts actually live

**Before reconstructing any prompt from a description, look here.** Three of the paper's
Appendix B prompts are transcribed verbatim in `docs/01`, while `07`, `09` and this document
carry only paraphrases and commentary about them:

| Artifact | Verbatim text | Paraphrased in |
|---|---|---|
| Compression prompt `π` | **`01` §6.4** | `02`, `11` §4 |
| Zero-shot inversion, **with** summaries | **`01` §7.2** | `07` §2.3, `09` row 6.2 |
| Zero-shot inversion, **no** summaries | **`01` §7.3** | `07` §2.3 |

This cost a full rewrite in Phase 1. `π` was reconstructed from `02`'s paraphrase because §4 of
this document said the prompt "exists only in its PDF" — true of the PDF, false of this repo. The
reconstruction contradicted the original.

**Read `01` §7.4 before using the inversion prompts.** They are labelled *zero-shot* — they define
the paper's **baseline**, not the fine-tuned inverter's conditioning format, which the paper never
specifies. Phase 2 gets the baseline's format for free and still has to choose the trained
inverter's. Having the text makes that a smaller decision, not a solved one; and `09` row 6.2's
format-mismatch fix now applies to a prompt that can be diffed rather than described.

---

## 2. What Phase 1 produces

The inverter's training set. Per the paper's Stage 1 (§4.1):

```
x'  ─→  SURROGATE V'  ─→  trace t' + answer y'      (visible, unlike the victim's)
                              │
                              ▼
                         COMPRESSOR C'  (zero-shot, fixed prompt π)
                              │
                              ▼
                         summary b'

yields  D₂ = {(x', y', b', t')}
```

`C'` exists because the attacker can never observe victim (trace, summary) *pairs* — they have
victim summaries but no victim traces, and surrogate traces but no summaries. `C'` manufactures the
missing half so the pairs line up.

**Inputs:** 5,000 prompts from OpenThoughts split A.
**Run it twice** — once per surrogate arm (7B primary, 1.5B paper-matching).

### Dataset

Use the **`llamafactory/OpenThoughts-114k` mirror**, not the canonical `open-thoughts/` repo — their
schemas are incompatible and the paper's code parses the mirror. Verified live 2026-08-26:

| Repo | Columns |
|---|---|
| `llamafactory/OpenThoughts-114k` | **`messages`**, `original_solution`, `domain`, `source` |
| `open-thoughts/OpenThoughts-114k` | `system`, `conversations` |

**Read `messages`, not `conversations`.** An earlier version of this doc had these swapped; a loader
written from it gets a `KeyError` on row 0. `messages` is `[system, user, assistant]`, and the
assistant turn holds R1's original `<think>` trace — use only the **user** turn as `x'`.

113,957 rows · 1.18 GB parquet · **78.2% math (`numina_math`) / 17.5% code / 4.4% science+puzzle**.
That mix makes MATH500, not JEEBench, the better length proxy from Phase 0.

Split A (surrogate) and split B (victim, Phase 3) must be **disjoint**. The paper samples 10k each;
we use 5k each. Persist the split indices to disk so Phase 3 cannot accidentally overlap.

---

## 3. Proposed order of work

1. **Sweep concurrency at 10,240 per slot** — *not* the 32,768 Phase 0 used. Already done for the
   7B: **32 slots / 1,270 gen t/s swept**, OOM at 40
   (`docs/results/sweeps/DeepSeek-R1-Distill-Qwen-7B-F16-ctx10240-ntg4096.md`). Serve with
   `-np 32 -c 327680`. Swept ranks slot counts; it is **not** an operating rate — the real run
   sustained ~445 t/s (`11` §5). Still to do for the 1.5B.
2. **Generate ~6,700 traces** with the 7B, keep the ~5,000 that did not hit the cap — 31.5 M tokens. Est. **~19 h** (was ~7.8 h, projected off the swept rate; see `11` §5).
3. **Draft the compression prompt `π`** and validate against Table 1 on ~200 traces. Iterate here,
   not at 5k scale.
4. **Compress all 5k** with `Qwen3.5-4B` on vLLM. Est. ~1-1.5 h.
5. **Repeat 1, 2 and 4 for the 1.5B arm.** Est. ~4-6 h + ~1.5 h.
6. Commit `docs/results/phase1.md` with trace-length stats, cap-hit rates, and the Table 1 style
   comparison.

Steps 1-2 and step 3 are independent — draft `π` while generation runs.

> **Phase 1 is ~29-31 h**, not the ~4 h `10` used to budget — and not the ~15-18 h this note
> claimed before the 7B run was measured. Phase 0's 7B run did 5.7 M tokens in 6 h 34 m
> (241 t/s effective); Phase 1 sustains ~445 t/s and needs ~33 M.
>
> The reusable lesson is **sweep at the context *and* the generation length the run actually
> needs** — Phase 0 swept at 32,768/slot because baselines generate to 32k, while Phase 1 caps at
> 8,192, so a slot needs ~10 k and the smaller per-slot KV buys 32 slots instead of 8.
>
> But the gain is **~1.8× realized (241 → 445 t/s), not the 4.6× the sweep suggested.** Swept-over-
> swept is not a speedup you can spend. See `11` §5: a sweep ranks slot counts; budget from the
> first 30 minutes of the real run.

### Generation settings — match the paper

Their repo pins these on every `vllm_infer.py` call. Use them:

```
temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · max_new_tokens 8192 · cutoff_len 16384
```

We deviated from these once by accident in Phase 0 (used Qwen model-card defaults) and it cost a
day. **Where the paper specifies a value, use it; deviating needs a stated reason.**

#### Which system prompt — decide this before generating, it sets trace length

There are **two** near-identical candidates and no doc previously said which to use:

| Source | Text |
|---|---|
| The paper's repo, `step0_data_preprocess/preprocess_r1_distill.py` | "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process…" |
| The OpenThoughts row itself, `messages[0]` (identical across all rows) | "You are an assistant that thoroughly explores questions through a systematic long thinking process before providing the final precise and accurate solutions…" |

**Use the repo's.** It is what generated the paper's reference behaviour, and `07` §3 records that
it "is part of what makes the surrogate emit long traces." It is not mentioned in the paper at all,
so the code is the only source. Recover the **exact** string from the repo — `07` §3 quotes it
truncated — and commit it next to `π`.

**Measured effect — smaller than first claimed.** This paragraph originally said a missing system
prompt would push cap-hit below 10%. Both arms have since been run:

| | cap-hit |
|---|---|
| without the system prompt (n=30 smoke) | 23.3% |
| with it (n=165, still climbing) | 27.9% |

The direction is right — the system prompt does lengthen traces — but the effect is **~5-8 pp, not a
collapse**. Keep the prompt for fidelity, because it is what the repo does; do **not** treat it as
the first suspect if cap-hit ever comes in very low. That diagnostic was speculation and the
measurement did not support it.

#### `max_new_tokens 8192` binds on ~a quarter of traces — drop the rows that hit it

Two corrections to what this section used to say (full detail in `12` §2):

**It is not paper v2's value.** Paper v2's methodology specifies no generation cap at all. 8192 comes
from the **v1 released code** (`07` §1.1) — the code `09` §6.3 says "reproduces v1, not the v2
tables." It was also tuned for R1, a far more token-efficient reasoner than a distill. Use it anyway
for fidelity and cost, but do not call it a paper-v2 requirement.

**"Most traces fit" was read off a median.** Measured on 2,000 sampled OpenThoughts rows — the exact
Phase 1 input — R1's own ground-truth traces run:

| Domain | corpus share | median | mean | **>8192** |
|---|---:|---:|---:|---:|
| math (`numina_math`) | 78.2% | 4,488 | 5,981 | 25.2% |
| code | 17.5% | 5,200 | 7,076 | 32.3% |
| science + puzzle | 4.4% | ~1,300 | ~2,100 | ~4% |
| **all** | | **4,379** | **6,005** | **25.5% ± 2.5** |

Method: 2,000 rows drawn as 200 dispersed 10-row clusters (OpenThoughts is ordered by source, so contiguous windows are correlated and a few large windows badly understate the variance). Cluster-robust 95% CI. Reproduced at 25.4% on an independent 1,140-row draw.

The 6,005 mean reproduces the paper's stated R1 average of 6,130.6 within 2%, which is what
validates the sample.

**Policy: over-generate 1.34× (6,706 rows to net 5,000) and DROP every row that hit the cap.** A capped trace has no
`</think>` and no answer; as an inverter training target it teaches the inverter never to conclude,
and Phase 4 carries that into the student. The student is the 2B, whose documented failure mode is
*exactly* non-termination — 50% of its truncations repeat `\boxed{}` a median 3,521 times
(baselines Finding 2). That was deferred to Phase 6 on the premise "SFT on cleanly terminating
traces may fix it." **The premise only holds if the traces terminate.**

Record the cap-hit rate per arm and report it — the paper never published one.

**The old 10-40% "stop and investigate" band is retired.** It was set before the comparison could
be paired, on the guess that a large gap from the reference meant a misbehaving surrogate. Paired
on identical prompts, R1's own cap-hit is **25.2%** and the 7B's is **~32%** — measured,
understood, and the expected shape for a distill. Crossing 33% confirms what pairing already said.

What actually binds is the row budget, since generation stops at 5,000 *kept*:

| cap-hit | rows needed | vs split A (9,000) |
|---:|---:|---|
| 33% | 7,463 | fine |
| 40% | 8,333 | fine |
| **44.4%** | **8,999** | **exhausts split A** |

So run through 33% without stopping. **Investigate if cap-hit approaches 42%, or if it *jumps*
rather than drifts** — a discontinuity means something changed mid-run, which is the real anomaly
the original band was reaching for. A slow drift upward is the drain bias (short generations finish
first), not a signal.

---

## 4. The hard part: the compression prompt `π`

**The prompt text is transcribed verbatim in `docs/01` §6.4 — use it. Only the two few-shot
exemplars were never released.**

> This line previously read *"the paper's v2 prompt exists only in its PDF (Appendix B)"*, which is
> true of the PDF but not of this repo: `01` §6.4 carries the full text. π was reconstructed from
> `02`'s paraphrase before anyone checked, and the reconstruction contradicted the original — it
> invented a "no closing sentence that announces a conclusion" rule, while Appendix B explicitly
> lists *"or the final consolidation"* as an allowed section move. A paraphrase also dropped the
> short-trace/long-trace length guidance and *"do not restate the final boxed answer unless the
> reasoning naturally concludes with it."* **Check `01` before reconstructing anything.** The repo predates that prompt and uses a *structurally opposite* format —
numbered bullets, where v2 explicitly forbids numbered lists. You must write `π` from scratch.

The v2 prompt asks for: 3-6 short sections, each opening with a bold markdown header on its own
line followed by a 2-5 sentence paragraph; no numbered lists, no bullets; first person, present
tense, tentative; inline LaTeX where the original used math; ~600-900 tokens; no meta-commentary.

**Acceptance test — Table 1.** The paper reports what GPT-5.4 mini's real summaries look like, and
the whole point of `π` is matching that distribution:

| Feature | Their `C'` on R1 traces | GPT-5.4 mini | Our target |
|---|---|---|---|
| Median tokens | 537 | 592 | ~540-590 |
| Bold-header sections | 94.1% | 92.9% | >90% |
| First-person prose | 97.3% | 97.0% | >95% |
| LaTeX | 79.1% | 71.9% | >70% |

Measure with light regex heuristics, as they did. Iterate on ~200 traces until all four land, then
run the full 5k.

**The stated length and the measured length are supposed to disagree — do not "fix" it.** Appendix B
instructs "roughly 600-900 tokens," and Table 1 reports the resulting medians as 537 and 592. The
instruction is an **over-ask the model undershoots by 10-15%**, not a description of the output.
π has two levers that both take token counts — the instruction, and the few-shot exemplars — so:

- keep the **instruction** at the paper's 600-900 verbatim; it is the calibrated over-ask
- size the **exemplars** to the Table 1 medians; they carry the calibration
- move **one lever at a time**, so a missed band tells you which one moved it

Lowering both applies the undershoot twice and lands near 450-480 — under the acceptance band, and
it presents as a prompt-quality problem when it is really a units error.

**Why this matters more than it looks:** `π` defines the summaries the inverter trains on. A
distribution mismatch here propagates through Phase 2 (inverter training), Phase 4 (inversion), and
Phase 5 (student training). It is the highest-leverage artifact in the project and the one with the
least guidance from the paper.

**Also note (`docs/09` §6.2):** v2's *inversion* prompt still expects "numbered reasoning bubbles"
that v2's *compression* prompt forbids producing — a v1 leftover live in the published paper. Fix
the mismatch in Phase 2, and re-run the zero-shot inversion baseline format-matched, since the
mismatch may have artificially depressed the paper's zero-shot row (TF1 35.36).

---

## 5. Conventions — keyed to when they fire

> **This project's failure mode is silent wrongness, not loud breakage.** Five separate Phase 0 bugs
> — a dropped pandas column, a masked exception handler, an unrecognized answer format, a
> thread-local signal handler, and a self-invented token cap — each produced *believable numbers
> with no error anywhere*. Nothing crashed. Every rule below exists to make one class of that
> visible. When in doubt, prefer the check that would have caught a wrong number over the one that
> saves time.

### Before launching any generation or eval run

| Check | Command / rule | What it prevents |
|---|---|---|
| Sweep concurrency at the run's own context **and its own generation length** | `NTG=<median_gen> bench/sweep_concurrency.sh <model.gguf> <per_slot_ctx>` | Running 4 h at 50% GPU utilization. `-c` is the **total** KV budget split across `-np` slots, and per-slot context must cover the longest expected generation — but no more. Re-sweeping the 7B at 10,240 instead of 32,768 ctx was 4.6× on paper. Probe length is a real but small term — `-ntg 512` gives 1,270 t/s and `-ntg 4096` gives 1,192, so KV depth costs **6.1%** at 32 slots, not the ~8× first suspected. Set `NTG` near your median generation anyway. |
| **A sweep ranks slot counts; it does not give you a time budget** | budget from the first 30 min of the real run | Swept and realized differ by **~2.4×** here, and the gap is not probe depth. `llama-batched-bench` runs every sequence at *identical* depth — a perfectly rectangular batch — while `llama-server` runs them at *ragged* depths under continuous batching. The 7B sustained ~430–555 t/s against 1,192 swept, with all 32 slots busy, CPU under one core of 32, zero disk I/O and no context shifts. On this Vulkan build that is the remaining explanation, and CUDA is the one untested lever (`08` §1). |
| Never size a run from a partial progress line | wait for the queue to drain | Short generations finish first and capped ones finish last, so in-flight cap-hit reads low and in-flight throughput reads high. A 30-row smoke on 32 slots never refills the queue at all — its wall clock is just the single longest generation. |
| Verify the model loads | Start `llama-server`, poll `/health`, kill it | A silent OOM at load once left the GPU idle ~9 h. The server exits non-zero and drivers move on without shouting. |
| Smoke test ~30 items | `--n-per-bench 15` or `--limit 5` | Catches harness bugs for minutes instead of hours. Both the `type`-column bug and the masked exception surfaced this way. |
| Confirm disk headroom | `df -h /` | 27 GB free. See the checkpoint policy below. |
| Start the watchdog | `bench/watchdog.sh` | Restarts a stalled driver within ~15 min. |

### While a run is in flight

| Rule | Why |
|---|---|
| **Never edit a running bash script** | Bash reads scripts incrementally; editing shifts its file offset mid-execution. This hung two chained runners for ~50 min. Kill → edit → relaunch. |
| Don't coordinate runners through files | A skip-marker that one runner deleted to claim a job re-enabled it in another queue, sending the surrogate to the wrong engine. Use one sequential driver. |
| Check progress is actually visible | Don't `grep -v` the progress bar out of the log, and remember `\r` needs `tr '\r' '\n'` to read. |
| **A supervising session writes nothing to the tree** | It reads, audits, and hands text to the session that owns the tree. A supervisor with write access to a checkout holding a live GPU job and files under `bench/results/` is a hazard with no upside — reviewing needs no write access. |

#### Working in a shared checkout

More than one agent may be live in this directory. `git worktree list` returns **one** entry:
there is a single index and a single HEAD, so `git checkout <branch>` silently moves *everyone*,
and whoever ran it last has decided the branch for both. "Which branch should we each use" is not
the real variable; there is only one.

| Rule | Why |
|---|---|
| **Explicit paths on `git add`. Never `git add -A`** | It sweeps up whatever another agent has on disk at that instant, possibly mid-edit. Two commits here (`045575d`, `72584e9`) each carried ~90–170 lines of a second session's in-progress files under a commit message describing something else entirely, misattributing authorship. Nothing was lost only because later commits caught the rest. |
| **Never `git checkout <sha> -- <dir>/`** | Path-scoped checkout of a directory is a **silent overwrite, not a merge**. It reverts every file under that path to that commit with no conflict, no error and no output. One such call came within two commits of silently reverting another session's doc work; the next action would have been to commit the reverted state on top. Cherry-pick the commit, or check out the single named file. |

Both hazards fired during Phase 1 — one landed, one missed by two commits. Neither produced any
error output, which is the point: this is the same failure shape as the five Phase 0 bugs.


### Writing or changing harness code

| Rule | Why |
|---|---|
| **Grade in the main thread** | `math_verify` uses SIGALRM and `signal.signal()` fails outside the main thread, so grading inside a `ThreadPoolExecutor` silently degrades to string matching. Cost 14.6 pts on MATH500 and looked like an engine difference. Collect in workers, grade in `main()`. |
| **Always save raw generated text** | Every later grading or extraction fix then applies retroactively for free via `bench/regrade.py`, instead of costing a regeneration. Three fixes have already been free because of this. |
| Exception handlers must not raise | A handler that threw on a missing key masked the real request error and produced 20 minutes of useless GPU time with a misleading traceback. |
| Where the paper specifies a value, use it | Deviating needs a stated reason in `docs/09`. Vendor defaults were once used by accident and caused degenerate looping. |
| **Probe a chat template by rendering and diffing, never by catching an exception** | `apply_chat_template` forwards *unknown* kwargs into the Jinja context instead of raising, so a `try/except TypeError` probe reports success for every model and silently does nothing. Render with and without the kwarg and compare the strings. Qwen3.5-4B's `enable_thinking` switch is real — its default template ends `<think>\n`, so without disabling it every summary carries a reasoning block. Same silent-success shape as the SIGALRM grader bug, and it recurs every time a new model is templated. |

### After a run, before believing any number

| Check | Command | Gate |
|---|---|---|
| Audit | `bench/audit_results.py <file>.jsonl` | truncation <10% · zero extraction failures on **completed** generations · no benchmark type at exactly 0% · calibration within ~8 pts where a reference exists |
| Sanity-check the grader | Sample rows marked wrong and read them | Catches false negatives. On the 4B only 1 of 500 was misgraded, and inspection confirmed the grader was right. |
| Record it | `docs/results/` + commit | Raw `.jsonl` is gitignored (20-45 MB each); commit summaries and audits. |
| **A retraction is not done until you have grepped for the number** | `grep -rn '<the number>' docs/ bench/` | Fixing the claim where you found it leaves every copy. One retracted throughput figure survived in **three** docs, including the handoff a future session reads first — corrected in one section while the stale copy sat in the section people actually open. Same shape twice in Phase 1: the `~8×` probe-depth claim, and `15-18 h`. **Grep every number in the retracted claim, including the one in its conclusion — not only the ones in its evidence.** A revised `4.6×` survived in a section *heading* because the search was for the figures that claim produced (`15-18 h`, `7.8 h`, `1,270`) rather than for the claim itself. A heading is where a claim outlives its data. |
| **A verification tool that cries wolf costs the same as one that misses** | check the hit count for plausibility before acting on it | Two independent markdown checks written here both counted `**` per *line* and flagged ~20 and ~80 "unbalanced bold" hits. Every one was false: bold legitimately wraps across a line break. Counting per paragraph gives zero. Acting on either list would have burned a cycle mid-run chasing nothing, and the next real signal from that tool would have been ignored. |
| **Results files hold measurements; interpretation goes in a doc that gets revised** | — | A sweep file carried the prose *"a 4.6× throughput gain … the largest single speedup available anywhere in this project"* — written before any realized number existed, inside a file otherwise containing only measurements. Realized-over-realized it was ~1.8×. An opinion in a results file is where a reader least expects one, and it does not get revisited when the measurement that would refute it arrives. |

#### Before believing a comparison between two numbers

Every rule here comes from one Phase 1 bug and its repeats: a reported "median 4,317 vs 4,379,
within 1.4%" that was `gen_tokens` (**trace + answer**) against a reference counting the `<think>`
**trace only**. It survived my check and a reviewer's, because the number was plausible. The
corrected figure is −6.8%. The same confusion then reappeared in the reviewer's *verification*
code within the hour, which is why this is a class and not an incident.

| Rule | Why |
|---|---|
| **Prefer the engine's own signal over anything you recompute** | This is the one that would have prevented every instance. Our `capped` flag was correct only because it took `finish_reason == "length"` straight from llama-server, which knows what it truncated — right by inheritance, not insight. Both hand-rolled reimplementations (median comparison, and a re-derived cap-hit that printed 0.0%) were wrong, in the same way, from the same ambiguity. |
| **State the units of both numbers before stating the gap** | `gen_tokens` and trace-tokens are both "tokens". One English word, two quantities, two code paths — the harness splits trace and answer because `D₂` needs them apart; the API returns their sum for free. Nothing in either name distinguishes them. |
| **"Is it biased?" is downstream of "is it the same measurement?"** | The reviewer explicitly audited the *subtle* confound — that means are not comparable across a cap — while never checking whether the two medians measured the same thing at all. Checking a comparison for bias before checking that both sides measure the same quantity is backwards. |
| **Pair when both sides exist for the same input** | OpenThoughts ships R1's own trace for every row, so each prompt has both. Every earlier comparison was unpaired and let prompt difficulty leak in. Pairing also made the drop decomposable — 33% of our drops are prompts R1 completes fine — which no unpaired sample could show. |
| **Validate the instrument on a known input first** | The Table 1 rulers were run against π's own exemplars before any model output: 100% headers, 100% first-person, 100% LaTeX, 0 numbered lists. A `$…$`-only LaTeX regex would have scored 0% on a prompt that emits `\( … \)`, and the cycle would have gone to debugging π. |
| **When a doc says an artifact is unavailable, grep for it before believing that** | `π`'s verbatim text sat in `01` §6.4 while `11` §4 said it existed "only in its PDF". Both of us then trusted a downstream summary about what an upstream doc contained — the reviewer while explicitly hunting for that text, by following the citations already present instead of searching for the artifact. Following citations is how you confirm an absence that is not there. |

### Disk policy for Phases 2 and 5

Seven 2B student checkpoints at ~4.6 GB each is ~32 GB against 27 GB free. Inverters are free
(Qwen3.5-4B cannot full-fine-tune on 24 GB, so it is LoRA and adapters are ~100 MB).

**Train → evaluate → delete the checkpoint**, keeping metrics and raw eval outputs. Peak usage
becomes one checkpoint instead of seven.

Consequence: re-evaluating later means retraining, so **settle the eval protocol before Phase 5
starts**. That is an additional reason to resolve the sampling/looping question first.

## 6. Open items carried into later phases

| Item | Phase | Note |
|---|---|---|
| 2B looping — 144 truncated rows repeat `\boxed{}` a median 3,521 times | 6 | Deferred deliberately: SFT on cleanly terminating traces may fix it. Revisit **after** the first trained checkpoint, not before. |
| Looping asymmetry | 6 | If training reduces looping, part of any gain is learned termination, not reasoning. Report truncation pre/post, not just accuracy. |
| Re-measure untrained 2B under the Phase 6 protocol | 6 | The control. Baseline and post-training eval must use identical sampling. |
| JEEBench Numeric 66.4% on the victim vs MCQ 98.2% | 6 | Probably genuine difficulty, but it is where a grader tolerance bug would hide. Spot-check. |
| Victim per-slot context can drop below 32k | 3 | p95 is 17,795; more slots would speed the ~10-15 h generation job. Sweep at 16k/slot. |
| 0.8B and 4B baselines are under the old card sampling | — | Eliminated candidates; not worth 6 h to refine numbers that change no decision. |
| OpenAI API victim track | optional | Second track reproducing §5.4. `.env` is gitignored and ready for a key. |

---

## 7. Definition of done for Phase 1

- [ ] `D₂` for the 7B surrogate: 5k rows of `(x', y', b', t')`, disjoint split A recorded
- [ ] `D₂` for the 1.5B surrogate: same
- [ ] `π` validated against all four Table 1 statistics, prompt committed to the repo
- [ ] Trace-length distribution reported and compared to the paper's 6,130-token R1 average
- [ ] Truncation rate at `max_new_tokens 8192` recorded
- [ ] `docs/results/phase1.md` committed
