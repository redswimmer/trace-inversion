# Phase 2 Handoff — Train the Inverters

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phases 0 and 1 are complete; Phase 2 is yours.

Read first: `docs/00-overview.md` (what the paper does), `docs/10-run-plan.md` (the plan and the
four questions to ask before proposing an experiment), `docs/results/phase1.md` (what Phase 1
produced), `docs/09-deviations-from-paper.md` (where and why we differ), `docs/06` §4 (TRL on
`qwen3_5` — every gotcha there is real), `docs/12` §1 and §3 (measured training VRAM, the
inverter's prompt budget), and `docs/11` §5 (conventions — they apply to this phase in full).
This document is the operational summary.

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path | Note |
|---|---|---|
| `D₂`, 7B arm | `bench/results/phase1/d2-7b.jsonl` — **5,006 rows** | gitignored, 84 MB |
| `D₂`, 1.5B arm | `bench/results/phase1/d2-1.5b.jsonl` — **5,028 rows** | gitignored, 82 MB |
| `D₂` schema | `{idx, domain, source, x, y, b, t, summary, summary_tokens, finish_reason}` | `summary` duplicates `b` (alias for `phase1_stats.py`). `t` and `b` carry **no** `<think>` tags (verified, 0 of 5,006). `t` and `y` have leading/trailing newlines — `strip()` them. |
| Prompts both arms kept | 3,681 `idx` in both files (7B-only 1,325 · 1.5B-only 1,347) | the paired-holdout pool, §4.3 |
| Split manifest | `bench/phase1/splits.json` — A=16,000, B=9,000, seed 20260826 | Phase 3 uses B. Phase 2 touches only A's `D₂`. |
| Paper prompts, verbatim | `docs/01` §7.2 (zero-shot inversion **with** summaries), §7.3 (**no** summaries) | transcribe programmatically into `bench/phase2/prompts.py`, sha256-asserted, like `bench/phase1/prompts.py` does for `π` |
| Table 1 / length harness | `bench/phase1_stats.py` — `dist()`, tokenizer handling | add `--mode inverted` (§5); do not write a second stats script |
| vLLM offline-batch pattern | `bench/phase1_compress.py` | the `enable_thinking` **render-and-diff probe**, the defensive `</think>` strip, `.new`-then-rename write. Copy these into `bench/invert.py`. |
| Gate self-test pattern | `bench/test_phase1_gates.py` | `bench/test_phase2_format.py` follows it |
| Driver pattern | `bench/run_phase1_gen.sh` | single sequential driver, logs under `bench/logs/`, never edited while running |
| Measured training VRAM | `docs/12` §1: Qwen3.5-4B bf16 LoRA **18.30 GiB @ 12288** (raw transformers, batch 1, grad-ckpt) | TRL adds overhead; the probe re-measures |
| Inverter prompt budget | `docs/12` §3: ~1,524 tokens prompt-side → `max_length` 12288 | **re-measured on the real `D₂` in §3 below: max row 10,605** |

### Environment, as of 2026-08-28

| | |
|---|---|
| GPU | RTX 4090, 24 GB (≈23.4 usable). Idle: 389 MiB used, 0 %. |
| System RAM | 30 GB — rules out CPU offload, `dataset_num_proc` > 2, pinned dataloader memory |
| Disk | **23 GB free (96 % full).** `~/.cache/huggingface/hub/models--unsloth--DeepSeek-R1-Distill-Qwen-14B-GGUF` is **36 GB** and appears in no script, log or phase (`docs/12` §5.2). Delete it first. |
| `.venv-vllm` | Python 3.12 · vLLM 0.27.1 · torch 2.13.0 · transformers 5.15.0 · datasets 5.0.1 · **no trl / peft**. Inference only. |
| `.venv` | Python 3.13.12, **empty** (dev group only: pytest, ruff, ty). Becomes the training venv (§6). |
| `uv` | 0.10.9 |
| Models on disk | `Qwen/Qwen3.5-4B` in the HF cache (the compressor — reuse it as the inverter base). GGUFs in `~/trace-inversion-bench/models/` are not needed this phase. |
| llama.cpp | not used this phase |

---

## 1. Where things stand

Phase 0 fixed every model role on measurement; Phase 1 built the inverter's training set. **Do not
revisit these without new evidence.**

| Role | Model | Format / Engine | Status |
|---|---|---|---|
| Surrogate `V'` primary | `DeepSeek-R1-Distill-Qwen-7B` | GGUF F16, llama.cpp | done — `D₂` 5,006 rows |
| Surrogate `V'` arm 2 | `DeepSeek-R1-Distill-Qwen-1.5B` | GGUF BF16, llama.cpp | done — `D₂` 5,028 rows |
| Compressor `C'` | `Qwen/Qwen3.5-4B` | bf16, vLLM | done — Table 1 passed on both arms |
| **Inverter `I`** | **`Qwen/Qwen3.5-4B`** | **bf16 + LoRA, TRL → merged → vLLM** | **this phase** |
| Student `S` | `Qwen/Qwen3.5-2B` | full FT, TRL | Phase 5 |
| Victim `V` | `Qwen3.8-27B` IQ4_XS | llama.cpp | Phase 3 |

Phase 1 facts that shape this phase (`docs/results/phase1.md`):

- Kept traces are **short relative to the paper**: median 2,804 / 2,480 trace-tokens (7B / 1.5B,
  R1-Distill tokenizer) against the paper's R1 average of 6,130. The inverter's job here is to
  produce ~3k-token traces, not ~6k.
- `D₂` is **surrogate-filtered**: every row terminated under the 8,192 cap. Nothing in the training
  target is severed. Keep it that way — **drop, never truncate** (§4.4).
- `π` behaves identically on both arms (summary medians 583 / 582), so the two `D₂` sets differ in
  surrogate strength and prompt coverage only, which is what makes the two-arm comparison readable.

---

## 2. What Phase 2 produces

Paper Stage 1, second half (§4.1, Eq. 1 and Eq. 2):

```
D₂ = {(x', y', b', t')}          per arm, from Phase 1
        │
        ├─ summary setting     I_sum   : (x', y', b') → t'      Eq. 1
        └─ no-summary setting  I_nosum : (x', y')     → t'      Eq. 2

teacher-forced token cross-entropy on t' only (completion_only_loss)
```

"Each setting requires a separate inversion model" (paper §4) — **two inverters per arm, four in
total.** Nothing about the victim enters this phase; Phase 2 makes zero victim queries.

| Inverter | Trains on | Rows | Output |
|---|---|---|---|
| `inverter-7b-sum` | `d2-7b` minus holdout, with `b'` | 4,806 | LoRA adapter, 3 per-epoch checkpoints |
| `inverter-7b-nosum` | `d2-7b` minus holdout, no `b'` | 4,806 | same |
| `inverter-1.5b-sum` | `d2-1.5b` minus holdout, with `b'` | 4,828 | same |
| `inverter-1.5b-nosum` | `d2-1.5b` minus holdout, no `b'` | 4,828 | same |

Plus, per inverter, a **held-out inversion**: traces generated on the shared 200-row holdout, with
a paired length report against the true `t'`. That is Phase 2's acceptance evidence and the only
place inversion quality is measured before Phase 4.

Artifacts, by path:

| Path | What |
|---|---|
| `bench/phase2/prompts.py` | the two Appendix B prompts, verbatim + sha256, and the format-matched variants derived from them **in code** (`.replace()` calls), so the edit is a diff, not a description |
| `bench/phase2/holdout.json` | `{"seed": 20260828, "pool": 3681, "idx": [200 ints]}` — committed |
| `bench/phase2_format.py` | `D₂` → TRL prompt-completion JSONL, train / holdout × sum / nosum, per arm → `bench/results/phase2/{7b,1.5b}-{sum,nosum}-{train,holdout}.jsonl` |
| `bench/phase2_train.py` | `SFTTrainer` + LoRA; `--arm --setting [--max-steps N]`; writes adapter checkpoints, `log_history.json`, `peak_vram.txt`; `--merge` writes merged bf16 weights for vLLM |
| `bench/invert.py` | vLLM offline batch: `(x, y[, b]) → t̂`; used on the holdout here and on split B in Phase 4 — **write it once** |
| `bench/phase1_stats.py --mode inverted` | paired `t̂` vs `t_true` lengths, cap-hit, empties |
| `bench/test_phase2_format.py` | formatter self-test (§7) |
| `bench/run_phase2.sh` | sequential driver: train → merge → invert holdout → stats → delete merged |
| `bench/results/phase2/inverter-{7b,1.5b}-{sum,nosum}/` | adapters (gitignore `bench/results/phase2/inverter-*/`) |
| `docs/results/phase2.md` | the committed record |

---

## 3. Measured budgets — `D₂` under the proposed conditioning format

Measured 2026-08-28 on the real `D₂` files with the **Qwen3.5-4B tokenizer** (the inverter's), with
the §4.1 system prompt (386 tokens) and user template rendered through the chat template. Every
earlier figure for this was arithmetic; these are counts.

| tokens | 7B arm, median / p95 / p99 / **max** | 1.5B arm, median / p95 / p99 / **max** |
|---|---|---|
| `x'` | 75 / 373 / 633 / 1,425 | 70 / 331 / 631 / 1,541 |
| `y'` | 488 / 818 / 1,007 / 1,724 | 448 / 782 / 973 / 5,087 |
| `b'` | 583 / 865 / 1,080 / 2,014 | 582 / 909 / 1,365 / 1,944 |
| `t'` (target) | 2,876 / 7,147 / 7,716 / 8,256 | 2,550 / 7,100 / 7,770 / 8,301 |
| prompt, summary setting | 1,615 / 2,158 / 2,497 / 3,884 | 1,569 / 2,147 / 2,599 / 6,452 |
| **whole row, summary** | 4,595 / 8,952 / 9,569 / **10,605** | 4,191 / 8,961 / 9,551 / **10,410** |
| prompt, no-summary | 1,002 / 1,477 / 1,749 / 3,114 | 955 / 1,407 / 1,732 / 5,584 |
| **whole row, no-summary** | 3,983 / 8,294 / 8,866 / **9,525** | 3,585 / 8,220 / 8,844 / **9,836** |

- **Rows over 11,264 tokens: 0 on every arm and setting.** Over 10,240: 3 per arm (summary
  setting only). `max_length` **12288** therefore truncates nothing; the formatter still counts
  and drops rather than trusting this table (§4.4).
- **Tokens per epoch** (all tokens through the forward pass, prompt included): 7B sum
  **24.2 M** · 7B nosum **21.2 M** · 1.5B sum **23.4 M** · 1.5B nosum **20.3 M**. Loss is computed
  on the ~16 M target tokens only.
- Note the tokenizer effect: `t'` median reads 2,876 here and **2,804 in `phase1.md`** for the same
  7B rows — the R1-Distill tokenizer vs Qwen3.5-4B's. Say which tokenizer before comparing any
  token count across documents (`docs/11` §5).

### Wall-clock projection — at an unmeasured rate

`docs/05` §5 estimates **1,416 train tokens/s** for Qwen3.5-4B LoRA (MFU 0.32, assumed). Nobody has
measured a training step on this box yet; the DeltaNet kernel path alone could move it 2× either way.

| Inverter | tokens × 3 epochs | @ 1,416 t/s |
|---|---:|---:|
| 7B-sum | 72.5 M | 14.2 h |
| 7B-nosum | 63.5 M | 12.5 h |
| 1.5B-sum | 70.1 M | 13.8 h |
| 1.5B-nosum | 61.0 M | 12.0 h |
| **all four** | **267 M** | **~53 h** (+ ~2 h eval / merge / inference) |

**This is ~2× the ~28 h `docs/10` budgeted for Phase 2**, and `docs/10`'s figure was written before
`D₂` existed. Treat ~55 h as the budget going in, and **the 20-step probe as the number that replaces
1,416.** At 2 epochs the same table reads ~36 h; the per-epoch checkpoints make that a decision the
user can take *after* seeing the epoch-2 eval loss rather than before.

---

## 4. Decisions — fixed for this phase

### 4.1 Conditioning format: the paper's Appendix B prompts, format-matched

The paper never states the fine-tuned inverter's input format (`docs/01` §7.4, §9 item 1 — the
highest-impact underspecified detail in the method). It gives only the **zero-shot** prompts. This is
not a deviation from a stated value; it is a gap the paper leaves, and the closest thing it supplies
is its own zero-shot prompt text.

**Decision: use the Appendix B zero-shot prompts as the conditioning format for the trained inverters,
with three edits.** Both settings:

| | Summary setting `I_sum` | No-summary setting `I_nosum` |
|---|---|---|
| system | `docs/01` §7.2, edited | `docs/01` §7.3, edited |
| user | `The original problem input is: {x}\nThe final answer is: {y}\nTransform this reasoning summary into clear full reasoning traces: {b}\nGenerate full reasoning traces:` | `The original problem input is: {x}\nThe final answer is: {y}\nGenerate full reasoning traces:` |
| assistant (target) | `t'`, stripped, **no tags** | same |

The three edits, applied in code to the verbatim string so they show up as a diff:

1. **Format match** (`docs/09` row 6.2, `docs/11` §4): "a list of numbered **reasoning bubbles**"
   → "a **reasoning summary** written as a few short bold-header sections"; "for each bubble" →
   "for each section"; "Transform this thinking bubbles" → "Transform this reasoning summary". v2's
   compressor forbids the numbered format its own inversion prompt still expects — a v1 leftover.
2. **Drop the `<think>…</think>` output-wrapping instruction.** Qwen3.5's chat template owns those
   tags and *re-parses* any `</think>` it finds in assistant content (`docs/06` §4.4), so a tagged
   target would be silently moved into the reasoning slot. The trace is emitted as plain content
   (§4.2). Applies to both prompts.
3. Nothing else. The "up to 20,000 characters" and "span thousands of tokens" lines stay.

**Why this and not a short custom prompt** (`docs/06` §4.3 sketches one): the paper hands us this
text and nothing else; inventing a format adds an unforced choice on the single detail `docs/01` ranks
as most likely to decide whether a reimplementation matches the paper. The ~386-token preamble costs
~8 % of tokens per epoch; that is the price of not inventing one.

**Cost of the choice, stated:** after fine-tuning the instruction text is inert, and the paper's
authors may have used something shorter. Log as a `docs/09` row (§7).

### 4.2 Think-tag handling: the trace is content, thinking disabled

Qwen3.5-4B thinks by default. The generation prompt ends `<|im_start|>assistant\n<think>\n` and, with
`enable_thinking=False`, `…<think>\n\n</think>\n\n`. The inverter's *output* is the trace, so:

- Train with **`enable_thinking=False`**, supplied per row through the `chat_template_kwargs`
  column — `SFTTrainer`'s tokenizer reads `**example.get("chat_template_kwargs", {})` (verified in
  `trl/trainer/sft_trainer.py` on `main`, 2026-08-28). Completion = `t'.strip()` with no tags.
- Expected render: prompt ends `<think>\n\n</think>\n\n`; prompt+completion = prompt + `t'` +
  `<|im_end|>\n`; completion mask covers exactly `t'<|im_end|>\n`.
- **Verify by rendering and diffing, never by trusting this paragraph** (`docs/11` §5): print row 0
  as the trainer tokenizes it, assert the tokenized prompt is a prefix of prompt+completion (TRL
  warns "Mismatch between tokenized prompt and the start of tokenized prompt+completion" if not),
  assert `<think>` occurs exactly once in the rendered string, and assert the decoded completion
  equals `t'` + EOS. `bench/test_phase2_format.py` owns these asserts.
- Inference (`bench/invert.py`) passes the same `enable_thinking=False` through
  `chat_template_kwargs` so the served prompt is byte-identical to the training prompt. Keep the
  defensive `</think>` strip from `phase1_compress.py` — it was load-bearing at 1 in 5,000 there.
- Fallback if the per-row kwargs do not reach the template in the installed TRL: pre-tokenize and
  hand TRL `input_ids` + `completion_mask` (`docs/06` §4.3). Same asserts.

### 4.3 A shared 200-row holdout, paired across arms

The paper trains on all 10 k rows and measures nothing before inverting the victim. We measure once,
cheaply, before spending Phase 3's ~10–15 h and Phase 4 on an inverter nobody has looked at.

- **200 `idx`, seed 20260828, drawn from the 3,681 prompts both arms kept**, so the 7B and 1.5B
  inverters are scored on identical inputs. Persisted to `bench/phase2/holdout.json`, committed.
- Excluded from training for **all four** inverters. Costs 4 % of rows (5,006 → 4,806; 5,028 → 4,828).
- Used twice: as TRL's `eval_dataset` (held-out loss each epoch, ~1 M tokens forward-only, minutes)
  and as the input set for the held-out inversion in §4.7.
- 200 is enough for a **central** statistic — the Phase 1 `π` validation predicted the n=5,006
  median within 5 tokens from n=200 — and blind to rare events. Report medians and quantiles; do
  not promote a rare-event count from this sample to a gate (`phase1.md` §2).

Four questions, answered: (1) *is the inverter learning, and does it produce full-length traces on
unseen inputs?* — the outcome decides the Phase 4 checkpoint and catches a dead adapter before Phase
3; (2) held-out eval loss per epoch; `t̂` length vs true `t'` length on the same 200 rows; (3) same
arm, same distribution, same prompts, same sampling — nothing varies but the epoch; (4) machinery: TRL `eval_dataset`, the vLLM pattern in
`phase1_compress.py`, `dist()` in `phase1_stats.py`. All exist.

### 4.4 Training hyperparameters — paper → ours, with the source of each

| Setting | Paper (LLaMA-Factory, 8×A100) | Ours (TRL, 1×4090) | Source |
|---|---|---|---|
| base | Qwen2.5-7B-Instruct | **Qwen3.5-4B**, `Qwen3_5ForCausalLM` (text-only) | `docs/10`; `docs/12` §1 |
| method | full FT | **bf16 LoRA**, not QLoRA | `docs/12` §1 — FFT is 27.81 GiB; NF4 unneeded |
| LoRA | — | **r 64 · α 128 · dropout 0.05** · targets `q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj` · **never `lm_head`, never `"all-linear"`** | `docs/09` §5.1 (rank 32–64; the inverter must learn a new behaviour, so the top of the range); `docs/06` §4.5 (module names, `in_proj_a/b` degenerate, `chunked_nll` hard-errors on a wrapped `lm_head`) |
| `max_length` | `cutoff_len` 16384 | **12288** — measured max row 10,605; **drop** any row that still exceeds it and report the count (expect 0) | §3; `docs/12` §3; `keep_start` would sever the trace tail, the exact poison Phase 1 dropped rows to avoid |
| epochs | 3 | **3, checkpoint each epoch** | paper; `docs/09` 3.4 was "2–3 TBD" — resolved, see §4.5 |
| learning rate | 1e-5 (full FT) | **1e-4** | `docs/09` 3.2; `docs/06` §4.9 #4 — LoRA needs ~10× |
| schedule | cosine, warmup 0.1 | cosine, warmup 0.1 | paper (repo YAML) |
| effective batch | **24** (8 × 1 × 3) | **1 × grad-accum 24** | `docs/07` §1.2 — do not collapse to the student's 96 |
| optimizer | default AdamW | `adamw_torch_fused` | adapters are small; 8-bit Adam buys nothing |
| precision | bf16 | bf16; `model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "kernels-community/flash-attn2"}` (fallback `"sdpa"`) | `docs/06` §4.9 #6 — string-loaded models default to fp32; `docs/05` §8.3 |
| loss | token CE on target | `completion_only_loss=True`, **`loss_type="chunked_nll"`** | `docs/06` §4.7 — 248k vocab, naive logits are 30 GiB at 16k |
| packing | `packing: true`, leaky | **off** | `docs/06` §4.6 — rows are 4–10k against a 12k window, `bfd` discards overflow |
| gradient checkpointing | on | on (TRL default) | |
| tokenizer | model default | **`processing_class=AutoTokenizer.from_pretrained(...)`, always** | `docs/06` §4.9 #1 — otherwise the VLM path, which disables `chunked_nll` |
| misc | | `dataset_num_proc=2`, `dataloader_pin_memory=False`, `torch_empty_cache_steps=50`, `save_strategy="epoch"`, `eval_strategy="epoch"`, `per_device_eval_batch_size=1`, `logging_steps=5`, `report_to="none"`, dump `trainer.state.log_history` to `log_history.json`, record `torch.cuda.max_memory_allocated()` | `docs/05` §8.4 (30 GB RAM) |

**Deliberately not re-derived:** `docs/12` §1's DeltaNet forward+backward verification, the `chunked_nll`
measurement, the LoRA-not-QLoRA decision. They were measured; the probe confirms them on TRL.

**Rank is fixed, not a user parameter, and not smoke-testable.** A 20-step probe measures VRAM and step
time; it cannot rank adapter capacities, and a real comparison is one full training run per rank
(~14 h each). The paper has no rank to match. r=64 is the top of the range `docs/09` §5.1 set, and
TRL's own reasoning-SFT guidance on this same dataset uses r=256 — so 64 is conservative, not
generous. The probe checks only that it fits.

### 4.5 Epochs: 3, with a checkpoint at each — the user decides later, not the executor now

Three is the paper's value. `docs/05` §6 argued a third epoch on 5 k rows mostly memorises; `docs/09`
3.4 left it TBD. With `save_strategy="epoch"` the epoch-2 adapter exists whether or not epoch 3 runs,
and the held-out eval loss at epochs 1/2/3 is the evidence for which one Phase 4 should serve. So the
third epoch buys the *measurement* of whether it helps, at ~4.5 h per inverter (§3). If the probe's
projection makes that unaffordable, the user cuts to 2 — **the executor does not**.

### 4.6 Run order, and what a cut falls on

`7B-sum → 7B-nosum → 1.5B-sum → 1.5B-nosum`. The 7B is the primary arm and the summary setting is
the paper's headline condition; the 1.5B-nosum cell is the least load-bearing (paper Table 3's
weakest row). If the budget forces a cut, it lands on the last run, not on epochs of the first.

Each run is a closed loop before the next starts: train → merge → invert holdout → stats → delete the
merged weights. A dead adapter is caught after ~14 h, not after ~53.

### 4.7 Held-out inversion

`bench/invert.py`, vLLM offline batch in `.venv-vllm` (`PATH` must include `.venv-vllm/bin` —
`docs/06` §1.8), on the **merged** bf16 weights (`merge_and_unload` → `save_pretrained` → serve →
**delete**; ~9.3 GB each, re-mergeable from the adapter in a minute). Sampling is the paper's
inversion-eval default (`docs/07` §1.1): **temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 ·
max_tokens 8192 · seed 1234**, `max_model_len` 12288, `enable_thinking=False`. Output row:
`{idx, domain, x, y, b, t_true, t_hat, gen_tokens, finish_reason}` — raw text saved, always.

**No zero-shot baseline in this phase.** `docs/09` row 6.2 and `docs/11` §4 asked for the paper's
zero-shot inversion row to be re-run format-matched. Put through `docs/10`'s four questions it fails
the first: its outcome changes nothing Phase 2 does — a dead adapter is caught by the held-out
lengths against the true traces with or without it, and a trained inverter is what the paper's method
uses either way. It is a ten-minute run if a later phase finds a question it answers (e.g. whether a
zero-shot-inverted student condition is worth adding in Phase 5); that proposal carries its own
justification then.

**Serving the adapter directly in vLLM** (`enable_lora`) was considered and rejected: LoRA support
for the `qwen3_5` hybrid architecture is unverified, and a merge is a one-minute step with no
correctness risk. Revisit only if disk makes the merge impossible — it does not, after step 0.

### 4.8 Considered and rejected

| Proposal | Why not |
|---|---|
| Fidelity suite (TF1 / BLEU / ROUGE) on the holdout | rejected in `docs/10` Phase 4 — the metrics are length in disguise; downstream student accuracy carries the result |
| A formal length gate on `t̂` | rejected in `docs/10` Phase 4 — the length is the most visible thing about the output file; report it |
| Packing | `docs/06` §4.6 — no benefit at these row lengths, real truncation risk |
| `target_modules="all-linear"` | wraps `lm_head`; `chunked_nll` raises (`docs/06` §4.9 #2) |
| QLoRA | not needed; quantizes the one model that must learn a new behaviour (`docs/12` §1) |
| Liger kernel | mutually exclusive with `chunked_nll`; `qwen3_5` support unverified |
| `trackio` / W&B | `log_history.json` is the record; nothing needs a dashboard |
| Training on the full 5,006 rows | 4 % more data vs. the only pre-Phase-4 measurement of the inverter — §4.3 |
| Reusing `docs/06` §4.3's short custom prompt | invents a format where the paper supplies one (§4.1) |
| Re-running the zero-shot inversion baseline in this phase | changes no Phase 2 decision (§4.7) |

---

## 5. Proposed order of work

| Step | What | Est. |
|---|---|---|
| **0** | **Disk:** delete the 36 GB 14B GGUF cache (`docs/12` §5.2). **Env:** §6. **Smoke test:** load `Qwen3_5ForCausalLM` bf16 text-only; print class, footprint, and the module names matching every `target_modules` entry; confirm no `visual` modules and that the MTP head is absent or frozen; confirm the DeltaNet fast path (`fla` / `causal_conv1d` importable, no fallback warning). Adapted from `docs/06` §4.10 for bf16 LoRA. | ~1 h |
| **1** | `bench/phase2/prompts.py` · `bench/phase2/holdout.json` · `bench/phase2_format.py` · `bench/test_phase2_format.py`. Run the formatter on both arms; print row 0 rendered; print the over-length count per file; run the self-test. Commit (explicit paths). | ~1 h |
| **2** | **Probe:** `phase2_train.py --arm 7b --setting sum --max-steps 20`. Report loss at steps 1 / 10 / 20, peak VRAM, realized tokens/s, projected hours for this inverter and all four. CHECKPOINT. Then the full 7B-sum run. On completion: merge → `invert.py` on the holdout → `phase1_stats.py --mode inverted` → delete merged weights. | 0.5 h + ~14 h |
| **3** | 7B-nosum, same loop (probe again — shorter rows change the rate). | ~13 h |
| **4** | 1.5B-sum. | ~14 h |
| **5** | 1.5B-nosum. | ~12 h |
| **6** | `docs/results/phase2.md`; `docs/09` rows (§7); `.gitignore` for adapters; commit. | ~1 h |

Steps 1 and the env build in 0 are independent; do 1 while `uv` resolves. Everything else is
sequential on one GPU. Run the long steps through `bench/run_phase2.sh` with logs in `bench/logs/`,
never through an edited-in-place script.

---

## 6. Environment — the training venv

`.venv` (Python 3.13) is empty. Add the training stack to `pyproject.toml` `dependencies` with `uv add`
and let `uv.lock` record the resolution; `.venv-vllm` stays a separate, inference-only venv (`docs/05`
§8.2, `docs/06` §1.8 — vLLM hard-pins torch and will fight the training stack).

Latest on PyPI, 2026-08-28: `trl 1.12.0` · `transformers 5.16.1` · `peft 0.20.0` · `accelerate 1.14.0`
· `datasets 5.0.1` · `torch 2.13.0` · `kernels 0.16.1` · `flash-linear-attention 0.5.2` ·
`causal-conv1d 1.7.0`. Pin what resolves; paste `trl env` into `phase2.md`.

```toml
# pyproject.toml additions
[project]
dependencies = [
  "torch==2.13.0", "transformers>=5.16.1", "trl>=1.12.0", "peft>=0.20.0",
  "accelerate>=1.14.0", "datasets>=5.0.1", "kernels>=0.16.1",
  "flash-linear-attention>=0.5.2", "causal-conv1d>=1.7.0",   # DeltaNet fast path; drop if they will not build
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

- `flash-attn` from PyPI is source-only for torch 2.13 and would OOM a 30 GB box compiling
  (`docs/05` §8.3). Use `attn_implementation="kernels-community/flash-attn2"`; if the hub kernel is
  unavailable for py3.13 / torch 2.13, `"sdpa"` — slower, still correct.
- If `flash-linear-attention` / `causal-conv1d` fail to build, transformers **silently** falls back to
  slow PyTorch DeltaNet ops (`docs/06` §4.6). Record which path is active in the probe; a step time
  far below the projection is the tell.
- `liger-kernel`: not installed (§4.8).

---

## 7. Measurements, gates, and what goes in `docs/results/phase2.md`

### Formatter self-test — `bench/test_phase2_format.py`, exits non-zero on failure

- rendered prompt is a token-prefix of rendered prompt+completion, for a summary row and a
  no-summary row; `<think>` appears exactly once; decoded completion == `t'.strip()` + EOS
- no-summary rows contain no `Transform this reasoning summary` block and no `b'` text
- holdout `idx` ∩ train `idx` = ∅ for every file; holdout `idx` ⊆ both arms' `D₂`; `len == 200`
- over-length rows are **dropped and counted**, never truncated; the count is printed
- verbatim prompts match their sha256; the format-matched variants differ from them only at the
  three §4.1 edits (assert on the diff, not on prose)

### Probe report — before every long run

`loss @ step 1 / 10 / 20 · peak VRAM (GiB) · realized train tokens/s · projected hours, this run ·
projected hours, all remaining runs · DeltaNet path · attention path`. Every projection states the
tokens/s it assumes.

### Per-inverter record

train loss curve (from `log_history`) · **held-out eval loss at epochs 1 / 2 / 3** · peak VRAM · wall
clock · realized tokens/s · rows trained / dropped · adapter size · `trl env` once.

### Held-out inversion record — `phase1_stats.py --mode inverted`, per inverter

Paired on the same 200 rows, **Qwen3.5-4B tokenizer, stated**:

| | `t̂` median / mean / p05 / p95 | `t_true` median / mean / p05 / p95 | `t̂` / `t_true` at the median | cap-hit @ 8192 | empty |
|---|---|---|---|---|---|
| 7B-sum, epoch 3 | | | | | |
| 7B-nosum, epoch 3 | | | | | |
| 1.5B-sum, epoch 3 | | | | | |
| 1.5B-nosum, epoch 3 | | | | | |

Plus, per trained inverter, **three example traces** read by a human — a short, a median and a long
holdout row — with a one-line note each on whether the trace reaches the given answer. Report
`finish_reason == "length"` as the cap-hit; it is the engine's own signal.

### Gates that exit non-zero

- formatter self-test
- `invert.py` output: 0 empty `t̂`; row count == 200; every `idx` in the holdout
- training: no NaN loss; the three epoch checkpoints exist; `log_history.json` written

### STOP AND ASK — report, then wait for the user

- the smoke test / probe does not show a decreasing loss within 20 steps
- peak VRAM above 22 GiB in the probe
- the probe projects the four runs past **80 h** total
- any gate fails twice
- a held-out inversion is empty on any row, or its median `t̂` length is under **half** the paired
  `t_true` median — the adapter did not take; the documented fallback backbones are in `docs/06`
  §4.10 but switching is the user's call
- a step exceeds its estimate by 2×

### `docs/09` rows to add or close

| Row | Change |
|---|---|
| 3.2 | inverter LR pinned to **1e-4**; rank **64**, α 128 |
| 3.4 | epochs: **3, per-epoch checkpoints**; TBD closed |
| 6.2 | format-match applied (bubbles → sections) **and** the `<think>`-wrap instruction dropped; the zero-shot re-run is **not** part of Phase 2 — it changes no decision here (§4.7) |
| new 6.4 | inverter conditioning format = Appendix B zero-shot prompts (paper: **unspecified** for the trained model) — a gap filled with the paper's own text, under §6 "forced on us by the paper" |
| new 4.7 | 200-row paired holdout, excluded from training on all four inverters (paper: trains on all 10 k) — CHOICE, costs 4 % of rows |
| new 7.12 | inverter emits the trace as plain content with thinking disabled (paper's Qwen2.5 has no think template) — forced by Qwen3.5's template re-parsing `</think>` |

---

## 8. Conventions — Phase 2 specifics

`docs/11` §5 applies in full. The ones this phase adds or that bite hardest here:

| Rule | Why |
|---|---|
| **Never edit a running script** | bash reads incrementally; two runners hung 50 min in Phase 0. Kill → edit → relaunch. |
| **Render and diff the chat template; never `try/except` it** | `apply_chat_template` forwards unknown kwargs into Jinja without raising, so an exception probe reports success for every model (`docs/11` §5). The whole of §4.2 rests on this. |
| **Always save raw generated text** | every later extraction fix is then free (`bench/regrade.py` precedent) |
| **Prefer the engine's own signal** | vLLM's `finish_reason` for cap-hit; TRL's tokenized lengths for the over-length count; `torch.cuda.max_memory_allocated()` for VRAM |
| **State the tokenizer before comparing token counts** | `phase1.md` counts with R1-Distill's; this phase counts with Qwen3.5-4B's. Same rows, 2,804 vs 2,876 (§3). |
| **`PATH="$PWD/.venv-vllm/bin:$PATH"` for every vLLM launch** | vLLM shells out to `ninja` by bare name; presents as `Engine core initialization failed` (`docs/06` §1.8) |
| **Merged weights are derived artifacts** | delete after the inference pass; the adapter is the record |
| **Explicit paths on `git add`; never `git add -A`; never `git checkout <sha> -- <dir>/`** | both hazards fired in Phase 1 (`docs/11` §5) |
| **A results file holds measurements; interpretation goes in a doc that gets revised** | `phase2.md` reports; `docs/09` and `docs/10` interpret |
| **One sequential driver** | no file-based coordination between runners |

---

## 9. Open items carried into later phases

| Item | Phase | Note |
|---|---|---|
| Which epoch's adapter Phase 4 serves | 4 | user decision from the epoch-1/2/3 eval-loss curves and the holdout lengths |
| Whether Phase 4 inverts with the 1.5B-arm inverters at all | 4 | depends on the Phase 2 comparison; the arm exists to measure inversion-vs-surrogate-strength |
| **Make victim-trace withholding structural before 3.1 runs** | 3 | `docs/10` Phase 3 — write `t` to a separate file from the `(y, b*)` the attack consumes; a wrong join leaks the oracle silently |
| Sweep the victim at 16k/slot | 3 | p95 17,795; more slots than the 6 Phase 0 used |
| 2B looping / eval protocol | 6 | unchanged |
| OpenAI API victim track | optional | unchanged |

---

## 10. Definition of done for Phase 2

- [ ] training venv built from `pyproject.toml`; `trl env` recorded
- [ ] `bench/phase2/prompts.py` (sha256-asserted), `bench/phase2/holdout.json` (200 idx, seed 20260828) committed
- [ ] formatter + self-test committed; over-length drop count reported per file (expected 0)
- [ ] four adapters on disk, three per-epoch checkpoints each, `log_history.json` + peak VRAM per run
- [ ] held-out eval loss per epoch, per inverter
- [ ] held-out inversion for all four inverters: paired length table, cap-hit, empties, three read examples each
- [ ] every long run preceded by a probe whose projection was reported before the run started
- [ ] `docs/results/phase2.md` committed; `docs/09` rows added; `.gitignore` covers the adapters
