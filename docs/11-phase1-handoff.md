# Phase 1 Handoff

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phase 0 is complete; Phase 1 is yours.

Read first: `docs/00-overview.md` (what the paper does), `docs/10-run-plan.md` (the plan),
`docs/results/baselines.md` (what we measured), `docs/09-deviations-from-paper.md` (where and why
we differ). This document is the operational summary.

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path |
|---|---|
| Splits A/B, disjoint, seed 20260826 | `bench/phase1/splits.json` — A=16,000, B=9,000 |
| Split A prompts, written out | `bench/results/phase1/promptsA.jsonl` — 16,000 rows |
| Surrogate system prompt (verbatim, sha256-asserted) | `bench/phase1/prompts.py` → `SURROGATE_SYSTEM` |
| Compression prompt `π` (verbatim + 2 exemplars) | `bench/phase1/prompts.py` → `PI_SYSTEM` |
| Generator | `bench/phase1_generate.py` |
| Compressor | `bench/phase1_compress.py` |
| Stats / Table 1 harness | `bench/phase1_stats.py` (`--paired`) |

Import the prompts from `prompts.py`; never retype them. No traces have been generated yet.

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

#### Why we generate surrogate traces instead of reusing the ones OpenThoughts ships

The obvious question, since every row already carries R1's own `<think>` trace: why spend ~21 h per
arm regenerating them? Four reasons, in descending order of how much they would cost us.

1. **Threat model.** The attack assumes an attacker with a model they control and can query — not
   one who already owns a corpus of strong-reasoner traces. Reusing the bundled traces assumes away
   the most expensive step, and weakens the claim from *"an attacker can manufacture this training
   data"* to *"given such data, an inverter can be built."* The paper's own code generates:
   `step0_data_preprocess/r1_distill_inference.sh` (`docs/07` §1), not a dataset read.
2. **The bundled traces are R1 at 685 B.** The paper's most consequential claim is that a *weak*
   surrogate works nearly as well as a strong one — TF1 **52.76** from the 1.5B against **64.42**
   from R1 (`docs/00` §39, `docs/02` Table 2). Reusing the bundled traces tests only the strong
   case, which is the least realistic and least interesting one.
3. **It would eliminate the surrogate-strength sweep.** The two-arm design exists to measure how
   inversion quality scales with surrogate strength. The paper tested only 1.5B and 685B — a 450×
   gap with no midpoint (`docs/10`, `docs/results/baselines.md`). Reuse gives one arm and no sweep.
4. **Generating is what lets us report the attack's real cost** — **~21 h of GENERATION per arm** on one
   RTX 4090 at ~495 gen t/s across 32 slots. (`docs/05` C2 carries an unrelated `~21 h` for Student SFT
   across 5 runs. Always say what the 21 h covers.)
5. **Provenance control**, which turns out to be load-bearing. Traces we generate share a system prompt,
   sampling settings, cap and serving stack across arms, so a comparison between arms varies *surrogate
   strength only*. Bundled traces carry DeepSeek's choices for all of those. See the rejected R1 arm below:
   this is the reason it fails. The paper ran on 8×A100 and never published a generation cost.
   That number is part of the threat.

**The cost of this choice, which belongs next to it.** Because we generate, our traces carry our
surrogate's own behaviour — the ~35% cap-hit and the drop bias in `docs/09` rows 7.10 and 7.11. The
bundled R1 traces would carry none of it. **We trade data cleanliness for threat-model fidelity.**
That is the right trade, and it is precisely why the drop-bias measurement matters: it is the price
of this decision, quantified rather than hidden.

#### Considered and rejected: an R1-surrogate arm from the bundled traces

Table 2's R1-surrogate arm (TF1 **64.42**) would cost **zero generation** — OpenThoughts already ships
R1's traces. It was proposed, argued for twice, and **rejected**. Do not re-propose it without new
information.

**The reason that decides it: the arm confounds the variable it exists to measure.** Our 1.5B and 7B
traces come from our own generation at settings we control and have recorded — same system prompt, same
sampling, same 8,192 cap, same serving stack. R1's come from the dataset: DeepSeek's sampling, their
serving stack, uncapped, and a system prompt we can only infer. A 1.5B / 7B / R1 sweep would therefore
vary **surrogate strength and trace provenance together**, and any difference at the R1 point would be
uninterpretable — surrogate, or how the traces were made. Filtering R1's traces at 8,192 fixes the
*length distribution* and nothing else; that is the visible comparability problem, not the only one.

**Two supporting reasons.** R1 cannot be run on this hardware (`docs/09` row 2.1) and this arm exists
precisely because we use its *bundled* traces — so its benchmark coordinates (91.6 / 87.1, paper Table 6)
would stay borrowed permanently, and "all three surrogate-vs-victim regimes" is really two measured plus
one imported. And it is not needed for the claim: **1.5B vs 7B tests weak-against-strong on one harness
with identical provenance** — a cleaner comparison over a shorter range (4.7× vs the paper's 450×), and a
midpoint the paper never had. If flat, we have extended their claim's support; if not, we have challenged
it. Either way the contribution stands.

Against all that, the arm buys a confounded point that reproduces a number the paper already published.

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
5. **Repeat 1, 2 and 4 for the 1.5B arm.** Est. ~6-7 h + ~1.5 h. **Probe its cap-hit on ~200 rows first**
   (~10 min) and size the run from the measured rate, not from the 7B's. The 1.5B is more verbose on every
   Phase 0 measurement, and at 44.4% cap-hit a 9,000-row split A is exactly exhausted. Split A now holds
   **16,000** indices for this reason — sized for the worst arm, not the best.
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

#### Which system prompt to use — settled

Two near-identical candidates exist. **Use the paper's repo version**, not the one the OpenThoughts
rows carry in `messages[0]`:

| Source | Use it? |
|---|---|
| repo, `src/step0_data_preprocess/preprocess_r1_distill.py` | **yes** — this is the one |
| the OpenThoughts row's own `messages[0]` | no |

`07` §3 records that the repo's string "is part of what makes the surrogate emit long traces", and
the paper never mentions a system prompt, so the code is the only source. It is already recovered
verbatim and sha256-asserted in `bench/phase1/prompts.py` as `SURROGATE_SYSTEM` — import it, never
retype it. The two strings differ in small ways ("summarizing" vs "summarization", a leading space)
that retyping silently normalises.

Measured effect on trace length: cap-hit is 23.3% without it and 27.9% with it, so it lengthens
traces by roughly 5-8 points of cap-hit. Real, but not a large lever.

#### `max_new_tokens 8192` binds on ~a quarter of traces — drop the rows that hit it

**The cap is 8,192 and it is settled.** It comes from the released code (`07` §1.1), not from paper
v2's methodology, which specifies no cap. Raising it was costed and rejected: per-slot context sets
slot count and slot count sets throughput, so 16,384 halves the slots and adds ~14 h to the 7B arm.

**Expect ~25-33% of traces to hit it.** Measured on 2,000 sampled OpenThoughts rows — the exact
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

## 4. The compression prompt `π`

**`π` is already built and committed in `bench/phase1/prompts.py` as `PI_SYSTEM`.** The prompt text
is transcribed verbatim in `docs/01` §6.4 and was taken from there. Only the two few-shot exemplars
were never released by the authors; ours are written to Appendix B's spec and sized to Table 1's
medians (552 and 502 tokens) with the instruction left at the paper's "roughly 600 to 900 tokens".

Import it. Do not rewrite it, and do not reconstruct it from the paraphrases in `02` or `07` — those
drop the per-trace-length guidance and the "do not restate the final boxed answer" rule, and
`docs/01` §6.4 is the authority.

What remains is **validating** it — see the acceptance test below.

Appendix B asks for: 3-6 short sections, each opening with a bold markdown header on its own line
followed by a 2-5 sentence paragraph; no numbered lists, no bullets; first person, present tense,
tentative; inline LaTeX where the original used math; ~600-900 tokens; no meta-commentary.

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

### Before proposing an experiment at all

See **`docs/10` § Before proposing an experiment** — four questions, in order, and *cost is never a reason to run one*. It lives in `10` because it spans every phase; the worked example is this phase's rejected R1-surrogate arm.

### Before launching any generation or eval run

| Check | Command / rule | What it prevents |
|---|---|---|
| Sweep concurrency at the run's own context **and its own generation length** | `NTG=<median_gen> bench/sweep_concurrency.sh <model.gguf> <per_slot_ctx>` | Running 4 h at 50% GPU utilization. `-c` is the **total** KV budget split across `-np` slots, and per-slot context must cover the longest expected generation — but no more. Re-sweeping the 7B at 10,240 instead of 32,768 ctx was 4.6× on paper. Probe length is a real but small term — `-ntg 512` gives 1,270 t/s and `-ntg 4096` gives 1,192, so KV depth costs **6.1%** at 32 slots, not the ~8× first suspected. Set `NTG` near your median generation anyway. |
| **The swept/realized ratio is not a constant — it ranged 2.4x to ~6x here** | never budget from a sweep's absolute number, whatever the model | 7B @ 32 slots: swept 1,192, realized 493.3 → **2.4×**. 1.5B @ 128 slots: swept 5,097, realized 760–861 → **5.9–6.7×**. It is tempting to read that as *the overstatement worsens at higher slot counts* — but those two points differ in **both** model and slot count, so neither can be blamed. The nearest same-model check points the other way: the 1.5B's 48-slot sweep point (3,233) against its 32-slot realized rate (687.6) is already **~4.7×**, so the large overstatement looks **model-driven** rather than slot-count-driven. Recorded as a range, not a trend, because we never swept the 1.5B at 32 and cannot separate the variables. |
| **A sweep's table must be clean enough to rank before you rank with it** | check monotonicity first | The 1.5B sweep read 48 → 3,233, 64 → 4,553, **96 → 3,380**, 128 → 5,097. A saturation curve rises then plateaus; 96 scoring below 64 means at least one point is noise. Taking the argmax of a table with that much scatter treats noise as a ranking. It was still taken here — 128 was chosen and the realized gain was 13-25% against a predicted 5,097 — so the cost was small, but the method was wrong and would not survive a table where the argmax happened to be the noisy point. |
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
| Audit — **baseline evals** | `bench/audit_results.py <file>.jsonl` | truncation <10% · zero extraction failures on **completed** generations · no benchmark type at exactly 0% · calibration within ~8 pts where a reference exists |
| Audit — **Phase 1 traces** | `bench/phase1_stats.py <file>.jsonl --mode traces --paired` (exits non-zero on failure) | request errors == 0 · zero empty traces among **kept** rows · no domain at 0% kept · cap-hit inside 10-45% · paired cap-hit within ±10 pts of R1 on the same prompts |
| **Do not point `audit_results.py` at a trace file** | — | It reads a baseline-eval schema (`bench`/`correct`/`truncated`/`pred`/`type`) that a trace file does not have, so it `KeyError`s on row 1 — and its `TRUNC_LIMIT 0.10` would fail every Phase 1 arm *by construction*, since we cap 25-45% of rows deliberately and drop them. A gate whose threshold contradicts the run's own design is worse than no gate. Verified by `bench/test_phase1_gates.py`. |
| Sanity-check the grader | Sample rows marked wrong and read them | Catches false negatives. On the 4B only 1 of 500 was misgraded, and inspection confirmed the grader was right. |
| Record it | `docs/results/` + commit | Raw `.jsonl` is gitignored (20-45 MB each); commit summaries and audits. |
| **A retraction is not done until you have grepped for the number** | `grep -rn '<the number>' docs/ bench/` | Fixing the claim where you found it leaves every copy. One retracted throughput figure survived in **three** docs, including the handoff a future session reads first — corrected in one section while the stale copy sat in the section people actually open. Same shape twice in Phase 1: the `~8×` probe-depth claim, and `15-18 h`. **Grep every number in the retracted claim, including the one in its conclusion — not only the ones in its evidence.** A revised `4.6×` survived in a section *heading* because the search was for the figures that claim produced (`15-18 h`, `7.8 h`, `1,270`) rather than for the claim itself. A heading is where a claim outlives its data. **Then confirm each hit is the SAME QUANTITY before touching it.** The rule above assumes a number identifies a claim; it does not. `31.6%` appears three times in `docs/` as *two* quantities — JEEBench accuracy for the weak inverter in `01` and `03`, and the cap-hit drop rate in `09` 7.10 — so a grep run during the 7.10 retraction returns two **true** statements alongside the stale one, and the failure mode is "fixing" a correct number. Where a figure is likely to collide, grep a distinctive phrase near it rather than the bare digits. This is the same one-token-two-quantities class as `gen_tokens` vs trace-tokens, now in the tooling rather than in a comparison — so the class is not confined to measurement code. **It reached cost estimates too:** `~21 h` names generation per arm (`11` §2) and, unrelatedly, Student SFT across 5 runs (`05` C2). A third nearly landed — the rejected R1 arm's total, which was ~21 h *precisely because it contained no generation*, so an unqualified read would have implied the one cost it did not carry. Three distinct instances of this class in one day, in measurements, in tooling and in estimates: assume any bare number is ambiguous until it says what it covers. |
| **Quote a trend from a fit over the whole regime, never from the last few points** | fit the regime; compare the slope to the window sd | The four most recent throughput windows read 539.7 -> 514.0 -> 500.1 -> 475.5 and look like decay. They are not: the ragged-regime series has sd 86.8 and spans 351-664, and a fit across all 25 windows gives **-2.0 gen t/s per 1,000 rows** — flat. A four-point monotone run in a series that noisy is unremarkable. Quoting the tail window as "the current rate" would have added 1.4 h to the projection off a sampling artefact. This is the partial-progress-line error one level up: not a partial sample, but the tail of a noisy one. |
| **A file in completion order is length-biased — but do not turn that into a prediction** | note the bias; do not bound the final figure with it | Results stream in as they finish, so at any moment ~`-np` generations are missing, and a long generation occupies a slot longer, so **the in-flight set is length-biased long**. That much is true. **I then predicted the running cap-hit would tick up at completion, bounded at +2.0 pts from 35.0% at n=1,011. It went DOWN, to 34.6%.** Two errors worth keeping: (a) the bound assumed the queue drains over a *frozen* prompt set, but 6,626 further prompts arrived at 34.7% and **sample composition swamps any censoring correction** — arithmetically fine, predictively useless; (b) the test of it was mis-specified. I checked the rows written immediately after the snapshot expecting the set that had been in flight, and got 21.9%. Those positions hold the next rows to **complete**, which is a different set: a capped row runs the full 8,192 tokens so it finishes *last*, making next-to-land rows short-biased while in-flight rows are long-biased. **A bound computed on a frozen sample says nothing about a growing one** — the same shape as fitting a trend to the tail rather than the regime. |
| **A running figure is not an estimate of the final one** | wait for the drain | The only unbiased cap-hit is the post-drain one, for the reasons above. Report running figures as running. |
| **Verify by reconstructing the measurement, not by re-reading the number** | rebuild the computation from the raw files, then compare | Twice in Phase 1 the reconstruction found what the original was not looking for. Checking a reported drop decomposition meant loading the archive and the live run over the same prompts — and that setup *was* an experiment neither of us had thought to run: two draws of identical prompts at temperature 0.7, which measured a 15.1% cap-hit flip rate and put a noise floor under every stochastic claim in the phase. Re-reading the six numbers would have confirmed them and found nothing. The corollary: when you verify a derived figure, re-derive it on the subset it is actually applied to — a marginal rate propagated to a boundary-selected subset was wrong by 8 pts here (18.3% vs 39.3%). |
| **A retraction has a blast radius: re-derive everything downstream of a wrong source** | after correcting a source, recompute every figure that source fed — not only the one you noticed | Reading the wrong file was caught and owned. The *next* message then pulled a rate from that same wrong file into a fresh calculation, and nothing marked the boundary: the table above it was computed correctly, the extrapolation below it was not. **Acknowledging a mistake does not clear the values it already put in scope.** Decomposed, the stale rate was worth **+10.4 pts** and a separate modelling error **-2.8 pts** in the opposite direction, so the smaller error partly masked the larger — and the one that got discussed first was the small one. |
| **Check a model against a point you already have** | predict something already measured, before predicting something you have not | The three-group LaTeX model predicts **80.4%** for split A's real mix against **79.5%** observed. That single check costs nothing and would have caught the stale-rate error immediately, because the bad model predicts far under a number already sitting in the same file. Any extrapolation built on measured rates can be back-tested against the measurement it came from. |
| **A verification tool that cries wolf costs the same as one that misses** | check the hit count for plausibility before acting on it | Two independent markdown checks written here both counted `**` per *line* and flagged ~20 and ~80 "unbalanced bold" hits. Every one was false: bold legitimately wraps across a line break. Counting per paragraph gives zero. Acting on either list would have burned a cycle mid-run chasing nothing, and the next real signal from that tool would have been ignored. |
| **Results files hold measurements; interpretation goes in a doc that gets revised** | — | A sweep file carried the prose *"a 4.6× throughput gain … the largest single speedup available anywhere in this project"* — written before any realized number existed, inside a file otherwise containing only measurements. Realized-over-realized it was ~1.8×. An opinion in a results file is where a reader least expects one, and it does not get revisited when the measurement that would refute it arrives. |

#### Before believing a comparison between two numbers

These fire whenever you compare two numbers. `gen_tokens` is **trace + answer**; the trace-length
references in this project are **trace only**. Confusing them turns a −6.8% gap into a reported
+1.4% match, with no error anywhere.

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
