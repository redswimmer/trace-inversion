# Baseline TODO

Phase 0 of the reproduction. Nothing downstream starts until every row here is
**DONE + audit-passed**.

Shared config: MATH500 (500) + JEEBench (515) = 1015 tasks · 32,768 max generation /
40,960 context · seed 1234 · pass@1 single sample.

---

## Runs

| # | Model | Role | Engine | Format | Precision | Sampling | Status |
|---|---|---|---|---|---|---|---|
| 0.1 | Qwen3.5-0.8B | student cand. | vLLM | safetensors | bf16 | 1.0/0.95/20 | 🟡 running |
| 0.2 | Qwen3.5-2B | student cand. | vLLM | safetensors | bf16 | 1.0/0.95/20 | ⬜ queued |
| 0.3 | Qwen3.5-4B | student cand. | vLLM | safetensors | bf16 | 1.0/0.95/20 | ⬜ queued |
| 0.4 | R1-Distill-Qwen-1.5B | **surrogate** | **llama.cpp** | **GGUF BF16** | bf16 | 0.6/0.95/— | ⬜ pending |
| 0.5 | Qwen3.8-27B IQ4_XS | **victim** | **llama.cpp** | GGUF | **4-bit** | 1.0/0.95/20 | ⬜ pending |

0.5 runs a 250/benchmark stratified subset (±3%), not the full 1015 — it is a ceiling
reference, and a full run costs ~9 h instead of ~2 h.

## Checklist

- [x] Build eval harness (`bench/eval_baseline.py`) — vLLM path
- [x] Build audit (`bench/audit_results.py`) — truncation / extraction / calibration gates
- [x] Build GGUF harness (`bench/eval_victim_gguf.py`) — llama.cpp HTTP path
- [x] Fix generation cap 16k → 32k (16k was binding: 71% truncation on the 4B)
- [x] Add seed 1234, matching the paper's eval invocation
- [x] Save raw generated text so failed extractions are recoverable
- [x] Download surrogate `DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf` (3.4 GB)
- [ ] **0.1** Qwen3.5-0.8B → audit
- [ ] **0.2** Qwen3.5-2B → audit
- [ ] **0.3** Qwen3.5-4B → audit
- [ ] **0.4** surrogate on llama.cpp → audit + **calibration gate**
- [ ] **0.5** victim on llama.cpp → audit
- [ ] `docs/results/phase1.md` must cite `~/trace-inversion-bench/archive/traces-7b-partial-1130rows.jsonl`
      as the **noise-floor instrument** — the only same-surrogate, same-prompt, same-settings re-draw we have.
      The 7B-vs-1.5B drop overlap must be read against its ~15% same-model floor, or a disagreement that the
      SAME surrogate produces against itself gets attributed to surrogate identity.
- [ ] `docs/results/phase1.md` must state that the paired comparison is **stochastic on our side only** —
      OpenThoughts ships one fixed R1 trace per row, so ~4 pts of any reported gap is our own resampling
      before a surrogate difference is visible.
- [ ] `docs/results/phase1.md`: state that the final cap-hit is the FIRST unbiased one, because the file is
      written in completion order and in-flight rows are disproportionately long/capped. Running figures are
      lower bounds (at n=1,011: 35.0% observed, at most 37.0% -> tight to +2.0 pts with 32 slots).
- [ ] Recheck the ours-only flip rate at end of run — the subset is ~10x larger there (n≈29 now).
      Expect the point estimate to move and the separation from both-cap (p=0.0003) to hold.
- [ ] **Phase 3, RAISE BEFORE 3.1 RUNS** — make the victim-trace withholding structural, not a
      convention. Write `t` to a separate file from the `(y, b*)` the attack consumes. `t` is both
      the oracle condition and the fidelity reference; a wrong join leaks it into the attack and the
      result is silently meaningless while looking like a spectacular success. No downstream gate
      catches a leaked oracle. Cannot be deferred — by Phase 4 the file layout is fixed. (`docs/10`
      Phase 3)
- [ ] Write the headroom table into `docs/08`
- [ ] **DECIDE:** student model (0.8B / 2B / 4B)
- [ ] **DECIDE:** FFT vs LoRA (follows from student size — 0.8B/2B fit FFT, 4B does not)
- [ ] **DECIDE:** surrogate stays 1.5B, or step up to 7B

## Gates — a run is not accepted until it passes

1. **Truncation < 10%.** Above that the score measures the token cap. (16k run failed this:
   0.8B 37%, 2B 26%, 4B 71%.)
2. **`no_answer` on *completed* generations ≈ 0.** Non-zero means a real extraction bug.
   Truncated rows don't count — being cut off before `\boxed{}` isn't a parsing failure.
3. **No JEEBench question type at exactly 0%** with n≥20 — that signals a grading bug.
4. **Calibration (0.4 only): within ~8 pts of paper Table 6** — MATH500 81.4, JEEBench 32.6.
   *If this fails, every other run is invalid and gets redone.*

## Decision rules

- **Student:** highest JEEBench headroom that still terminates reliably. The 0.8B is likely out —
  at 16k it managed 11.3% with a 13,734-token median, i.e. it fails to converge rather than
  merely being weak.
- **FFT vs LoRA:** 0.8B (~7 GB) and 2B (~15 GB) fit full fine-tuning on 24 GB; 4B (~28 GB) does
  not. Picking 0.8B/2B removes our single riskiest deviation from the paper.
- **Surrogate step-up:** if 0.4 lands *below* the chosen student on JEEBench, add
  R1-Distill-Qwen-7B (F16 GGUF, 14.19 GiB). At 16k the surrogate was below the 2B
  (30.5 vs 45.8), so this is likely to trigger.

## Housekeeping

- [x] Deleted superseded victim quants Q6_K / Q4_K_M / Q5_K_M — measurements recorded in
      `docs/08`, IQ4_XS is the decided pick. Reclaimed ~57 GB (disk had hit 99%).
- [ ] **Delete placeholder** `bench/results/DeepSeek-R1-Distill-Qwen-1.5B.jsonl` — it is a
      skip-marker that keeps the vLLM runner from claiming 0.4, **not a result file**.
- [ ] 16k results archived in `bench/results/v1-16k/` — keep for the cap-effect comparison.
- Disk: 43 GB free. HF cache is 91 GB but mostly other projects; `~/.cache/uv` is 43 GB.

## Standing practice: sweep concurrency before every generation run

`bench/sweep_concurrency.sh <model.gguf> <per_slot_ctx> [slots...]` probes slot counts with
`llama-batched-bench` and writes a table to `docs/results/sweeps/`.

**Why it is mandatory, not optional:** optimal slot count is inversely related to model size, and
arithmetic gets it wrong. The 1.5B surrogate was sized at 12 slots from memory math and ran ~4 h at
**48-55% GPU utilization** on 9.1 GB of 24 — latency-bound, badly under-parallelized. A 27B
saturates on a handful of slots; a 1.5B needs dozens.

**The constraint to respect:** `-c` is the *total* KV budget split across `-np` slots, so slots and
per-slot context trade directly. Per-slot context must still cover the longest generation expected
(the victim's longest observed was 32,534 tokens), or long requests fail. When throughput is still
climbing at the largest slot count that fits, the ceiling is memory rather than compute.

Applies to:
- [x] 0.6 R1-Distill-7B baseline — wired into `run_surrogate_7b.sh`
- [ ] Phase 1 surrogate trace generation (5k traces) — biggest payoff, small model, many slots
- [ ] Phase 3 victim trace generation (5k traces, ~23 h) — where a 2x miss costs a day
- [ ] Phase 4 inversion generation
