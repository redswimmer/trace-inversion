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
| Disk | ~28 GB free. **Check before downloading anything.** |
| llama.cpp | Homebrew, **b10450 / ggml 0.20.0**. Older builds cannot load `qwen3_5`. |
| vLLM | 0.27.1 in `.venv-vllm` (Python 3.12) |
| Models | `~/trace-inversion-bench/models/` |
| Results | `bench/results/` (gitignored — 20-45 MB each, they embed raw text) |
| Logs | `bench/logs/` (gitignored) |
| Committed results | `docs/results/` — summaries + audits only |

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
schemas are incompatible (`system`+`conversations` vs `messages`) and the paper's code parses the
mirror.

Split A (surrogate) and split B (victim, Phase 3) must be **disjoint**. The paper samples 10k each;
we use 5k each. Persist the split indices to disk so Phase 3 cannot accidentally overlap.

---

## 3. Proposed order of work

1. **Sweep concurrency** for the 7B surrogate — `bench/sweep_concurrency.sh`. Mandatory, see §5.
2. **Generate 5k traces** with the 7B. Est. ~3-4 h.
3. **Draft the compression prompt `π`** and validate against Table 1 on ~200 traces. Iterate here,
   not at 5k scale.
4. **Compress all 5k** with `Qwen3.5-4B` on vLLM. Est. ~1 h.
5. **Repeat 2 and 4 for the 1.5B arm.** Est. ~2 h.
6. Commit `docs/results/phase1.md` with trace-length stats and the Table 1 style comparison.

Steps 1-2 and step 3 are independent — draft `π` while generation runs.

### Generation settings — match the paper

Their repo pins these on every `vllm_infer.py` call. Use them:

```
temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · max_new_tokens 8192 · cutoff_len 16384
```

We deviated from these once by accident in Phase 0 (used Qwen model-card defaults) and it cost a
day. **Where the paper specifies a value, use it; deviating needs a stated reason.**

Note `max_new_tokens 8192` is the paper's value and is a real ceiling on trace length — their R1
traces average 6,130 tokens. Our 7B surrogate's measured median is 5,576 on JEEBench, so most
traces fit, but record the truncation rate.

---

## 4. The hard part: the compression prompt `π`

**The paper's v2 prompt exists only in its PDF (Appendix B), and its two few-shot exemplars were
never released.** The repo predates that prompt and uses a *structurally opposite* format —
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

**Why this matters more than it looks:** `π` defines the summaries the inverter trains on. A
distribution mismatch here propagates through Phase 2 (inverter training), Phase 4 (inversion), and
Phase 5 (student training). It is the highest-leverage artifact in the project and the one with the
least guidance from the paper.

**Also note (`docs/09` §6.2):** v2's *inversion* prompt still expects "numbered reasoning bubbles"
that v2's *compression* prompt forbids producing — a v1 leftover live in the published paper. Fix
the mismatch in Phase 2, and re-run the zero-shot inversion baseline format-matched, since the
mismatch may have artificially depressed the paper's zero-shot row (TF1 35.36).

---

## 5. Conventions — these were learned the hard way

**Sweep concurrency before every generation run.** `bench/sweep_concurrency.sh <model.gguf>
<per_slot_ctx> [slots...]`. Optimal slots is inversely related to model size and arithmetic gets it
badly wrong: the 1.5B was sized at 12 slots by hand and ran 4 h at 48-55% GPU utilization. The
sweep chose 8 slots for the 7B (274 t/s) over my guess. Note `-c` is the **total** KV budget shared
across `-np` slots, so slots and per-slot context trade directly, and per-slot context must still
cover the longest generation expected.

**Verify a model loads before committing to a long run.** A silent OOM at model load left the GPU
idle ~9 h. `llama-server` exits non-zero and the driver moves on; nothing shouts.

**Never edit a running bash script.** Bash reads scripts incrementally, so editing shifts its file
offset mid-execution. This hung two chained runners for ~50 min. Kill, edit, relaunch.

**Grade in the main thread.** `math_verify` uses SIGALRM and `signal.signal()` fails outside the
main thread, so grading inside a `ThreadPoolExecutor` silently degrades to string matching — it
cost 14.6 points on MATH500 and looked like an engine difference. `bench/eval_victim_gguf.py`
collects in workers and grades in `main()`; keep it that way.

**Save raw generated text.** Every grading or extraction fix then applies retroactively for free
(`bench/regrade.py`), instead of costing a regeneration.

**Run the audit before believing any number.** `bench/audit_results.py` gates on truncation,
extraction failures on *completed* generations, per-type zeros, and calibration. Three bugs in
Phase 0 produced plausible-looking numbers with no error anywhere.

**Watchdog.** `bench/watchdog.sh` restarts a stalled driver within ~15 min. Start it alongside any
long unattended run.

---

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
