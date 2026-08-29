# Phase 2 — The inverters

Reproduction of *"How to Steal Reasoning Without Reasoning Traces"* (Zhang, Morris, Shmatikov,
arXiv 2603.07267v2), Stage 1 §4.1 second half: four LoRA inverters on `Qwen/Qwen3.5-4B`,
`{7B, 1.5B surrogate arm} × {summary, no-summary}`, each trained on its arm's `D₂` minus a shared
200-row holdout, with held-out eval loss per epoch and a paired held-out inversion. Plan and fixed
settings: `docs/13-phase2-handoff.md`. This file holds measurements; interpretation goes in `docs/09`
and `docs/10`.

All token counts in this file use the **Qwen3.5-4B tokenizer** (the inverter's) unless stated.
`phase1.md` counts the same traces with the R1-Distill tokenizer (2,804 vs 2,876 at the 7B median).

---

## 0. Environment

| | |
|---|---|
| GPU | RTX 4090 24 GB, driver 595.84, CUDA 13.2 runtime; nvcc 13.1 |
| Training venv `.venv` | Python 3.13.12 · torch 2.13.0+cu130 · transformers 5.16.1 · trl 1.12.0 · peft 0.20.0 · accelerate 1.14.0 · datasets 5.0.1 · kernels 0.16.1 · flash-linear-attention 0.5.2 |
| Inference venv `.venv-vllm` | Python 3.12 · vLLM 0.27.1 · torch 2.13.0+cu130 · transformers 5.15.0 (unchanged from Phase 1) |
| Disk | 36 GB `DeepSeek-R1-Distill-Qwen-14B-GGUF` cache deleted (`docs/12` §5.2): 23 GB → 59 GB free |

`trl env`:

```
- Platform: Linux-7.0.0-30-generic-x86_64-with-glibc2.43
- Python version: 3.13.12
- TRL version: 1.12.0
- PyTorch version: 2.13.0+cu130
- accelerator(s): NVIDIA GeForce RTX 4090
- Transformers version: 5.16.1
- Accelerate version: 1.14.0
- Datasets version: 5.0.1
- HF Hub version: 1.29.0
- bitsandbytes version: not installed
- DeepSpeed version: not installed
- Liger-Kernel version: not installed
- PEFT version: 0.20.0
- vLLM version: not installed
```

### Kernel paths — measured, not assumed

| Path | Status | Evidence |
|---|---|---|
| Gated DeltaNet chunk / recurrent rule | **`fla` 0.5.2** (`fla.ops.gated_delta_rule`) | transformers 5.16.1 resolves the original package before the torch fallback (`integrations/hub_kernels.py`); `bench/phase2_train.py` replicates that resolution and prints it |
| `causal_conv1d` | **torch fallback** | `causal-conv1d` 1.7.0 fails to compile: nvcc 13.1 against glibc 2.43's `bits/mathcalls.h(206): error: exception`. The conv is a 4-tap depthwise `F.conv1d`; dropped per `docs/13` §6 |
| Attention (8 of 32 layers) | **`sdpa`** | `kernels-community/flash-attn2` (build `torch-stable-abi210-cu130`, compiled against torch 2.14) loads and runs **forward** under torch 2.13.0, but its **backward** raises `Dimension out of range (expected [-5, 4], got 126301640993808)` at probe step 0 — `bench/logs/phase2-probe-7b-sum-flashattn2-crash.log`. `sdpa` is the fallback `docs/13` §6 names |

### Smoke test (`bench/phase2_train.py --smoke`)

`Qwen3_5ForCausalLM` bf16, footprint **7.83 GiB**, text-only: 0 `visual` modules, 0 MTP modules,
`tie_word_embeddings=True`. Every `target_modules` entry matches: `q/k/v/o_proj` × 8 (the full-attention
layers), `in_proj_qkv / in_proj_z / out_proj` × 24 (DeltaNet), `gate/up/down_proj` × 32 (MLP).
`lm_head` is not a target. Forward on 257 tokens: 8.13 GiB peak.

---

## 1. Data — `D₂` → TRL prompt-completion (`bench/phase2_format.py`)

**Conditioning format:** the paper's Appendix B zero-shot inversion prompts, verbatim from `docs/01`
§7.2 / §7.3 and sha256-pinned in `bench/phase2/prompts.py`, with exactly three edits applied in code
and asserted as a diff by `bench/test_phase2_format.py`:

1. format match — `"A list of numbered **reasoning bubbles**"` → `"A **reasoning summary** written as a
   few short bold-header sections"`, `"For each bubble"` → `"For each section"`, `"Transform this
   thinking bubbles"` → `"Transform this reasoning summary"`;
2. the `<think>…</think>` output-wrapping line removed from both prompts;
3. nothing else — the paper's remaining "bubble" wording ("each bubble summarizes", "flesh out the
   bubbles", "per bubble") is left as written, by rule, not by oversight.

The trace is the assistant turn as plain content with `enable_thinking=False` supplied per row
through `chat_template_kwargs` (TRL 1.12 reads it: `sft_trainer.py:1512`). The rendered prompt ends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`; the loss mask covers exactly `t'<|im_end|>\n`.

**Holdout:** `bench/phase2/holdout.json` — 200 `idx`, seed 20260828, drawn from the **3,681** prompts
both arms kept (7B-only 1,325 · 1.5B-only 1,347), excluded from all four training sets.

| file | rows | dropped > 12288 | tokens / epoch | max row | median row | max prompt |
|---|---:|---:|---:|---:|---:|---:|
| 7b-sum-train | 4,806 | 0 | 23,417,748 | 10,620 | 4,649 | 3,901 |
| 7b-sum-holdout | 200 | 0 | 829,078 | 9,743 | 3,838 | 2,391 |
| 7b-nosum-train | 4,806 | 0 | 20,231,546 | 9,475 | 3,977 | 3,067 |
| 7b-nosum-holdout | 200 | 0 | 701,146 | 8,810 | 3,185 | 1,588 |
| 1.5b-sum-train | 4,828 | 0 | 22,604,920 | 10,425 | 4,241 | 6,469 |
| 1.5b-sum-holdout | 200 | 0 | 845,725 | 9,716 | 3,769 | 2,447 |
| 1.5b-nosum-train | 4,828 | 0 | 19,354,876 | 9,787 | 3,551 | 5,537 |
| 1.5b-nosum-holdout | 200 | 0 | 718,198 | 8,780 | 3,098 | 1,515 |

**Over-length rows dropped: 0 of 10,034** on every file (`docs/13` §3 predicted 0). TRL's own
tokenization of `7b-sum-train` reproduces the formatter's count exactly (4,806 rows, 23,417,748
tokens, max 10,620, 0 rows at `max_length`), so the two render identically. Tokens per epoch are
~3 % under `docs/13` §3 because the holdout is removed and the prompt lost the `<think>` line.

`bench/test_phase2_format.py` passes: diff-asserted edits · holdout ∩ train = ∅ and holdout ⊆ both
arms · no `b'` text in any no-summary row · prompt is a token-prefix of prompt+completion · `<think>`
once · completion decodes to `t'<|im_end|>\n` · a 20,437-token row is dropped and counted, never
truncated.

---

## 2. Training settings (fixed, `docs/13` §4.4)

`Qwen3.5-4B` bf16 · LoRA r 64 / α 128 / dropout 0.05 on the ten target names above (121,896,960
trainable parameters) · `max_length` 12288 · lr 1e-4 cosine, warmup 0.1 (`warmup_steps=0.1` — transformers 5
folded `warmup_ratio` into `warmup_steps`) · 3 epochs, checkpoint + held-out eval each epoch · batch 1 ×
grad-accum 24 · `completion_only_loss` · `loss_type="chunked_nll"` · packing off ·
`processing_class=AutoTokenizer` · `adamw_torch_fused` · gradient checkpointing (`use_reentrant=False`,
TRL's default) · `dataset_num_proc=2`, `dataloader_pin_memory=False`, `torch_empty_cache_steps=50` ·
seed 42 (HF default). Inference: vLLM on merged bf16 weights, temp 0.7 · top_p 0.9 ·
repetition_penalty 1.05 · max_tokens 8192 · seed 1234 · `max_model_len` 12288 · `enable_thinking=False`.

---

## 3. Probes

Each long run was preceded by a 20-step probe of the identical config on the real data. Probe losses
run under a 20-step schedule (2 warmup steps, cosine to zero by step 20), so they show *whether* loss
decreases, not the curve the real run will trace. Rates come from TRL's own `num_tokens` counter
against wall time.

### 7B-sum probe — `bench/logs/phase2-probe-7b-sum.log`, 2026-08-28

| | |
|---|---|
| loss @ step 1 / 10 / 20 | **0.4803 / 0.3531 / 0.4588** — per-step values bounce on 24-row batches of 1–10k-token traces; half-means 0.4408 (steps 1–10) → 0.4106 (11–20), first five 0.4588 → last five 0.4173; `grad_norm` 1.14 → 0.13–0.14; `mean_token_accuracy` 0.83 → ~0.85 |
| peak VRAM | **15.42 GiB** `max_memory_allocated` · 17.41 GiB `max_memory_reserved` (480 of 4,806 rows seen; the full run's peak is in §4) |
| realized train tokens/s | **2,153** steady-state (TRL `num_tokens`, steps 1→20) · 2,029 overall incl. warmup — **1.52×** the unmeasured 1,416 of `docs/05` §5 |
| projection @ 2,153 tok/s, 3 epochs + eval passes | 7B-sum **9.4 h** · 7B-nosum 8.1 h · 1.5B-sum 9.1 h · 1.5B-nosum 7.8 h · **all four 34.3 h** (`docs/13` budget ~53 h; STOP line 80 h) |
| DeltaNet path | chunk + recurrent rule: **`fla` package**; `causal_conv1d`: torch fallback |
| attention path | **`sdpa`** (flash-attn2 hub kernel: backward crashes, §0) |
| rows at `max_length` | 0 |
| LoRA trainable params | 121,896,960 |
| wall clock | 1,137 s for 20 steps (480 rows, 2,307,287 tokens) |

Finding, not a concern: loss 0.48 with 83 % token accuracy at step 1 means the base model already
predicts these surrogate traces well given the answer and the summary. What training has to change is
*generation* behaviour — length and termination — which the held-out inversion measures, not the loss
curve. A small loss drop is neither "nothing learned" nor success.

**Serving-path check on the probe adapter** (merge → vLLM → 5 holdout rows → stats → delete), run
before the long run so a merge/serving failure would not surface 9 h later. It found two, both fixed
in `bench/phase2_train.py --merge` and committed:

1. transformers 5.16 `save_pretrained` reverts its load-time key conversion
   (`modeling_utils.revert_weight_conversion`) and writes VL-style `model.language_model.*` names;
   vLLM 0.27.1's text-only `Qwen3_5ForCausalLM` loader rejects them (*"no module or parameter named
   'language_model'"*). The merge now writes the module-tree state dict directly (`lm_head` is tied
   and not stored, as in the original checkpoint).
2. `config.architectures` came out `None`; vLLM then inferred `Qwen3_5TextModel`, unsupported. Set
   to `["Qwen3_5ForCausalLM"]` on save.

After the fixes: 5/5 rows generated, 0 empty, 0 cap-hit, 0 stripped think blocks; the served prompt
is byte-identical to the training prompt — sha256 of the rendered text and of the token ids agree
across transformers 5.16.1 (training) and 5.15.0 (vLLM), base and merged tokenizer. The merge is
proven non-trivial: `merge()` compares two LoRA targets before and after `merge_and_unload` and re-reads
them from disk — on the probe adapter, layer-0 `in_proj_qkv` max|Δ| 4.9e-4 with 69.7 % of elements
changed, layer-3 `q_proj` 9.2e-4 / 66.1 % (`merge-check.json`); without this, the serving check could
not tell a merged model from the bare base, which already writes coherent traces.

Measured, no interpretation: after 20 probe steps (480 rows, compressed schedule) all five traces end in
`**Final Answer** \boxed{…}` matching `y`, at the true length (t̂ median 1,940 vs t_true 1,893 on those
five rows). Together with the step-1 loss this says the base model needs little to acquire the format;
what three epochs add, if anything, is what the per-epoch eval loss and the 200-row holdout lengths
answer.

---

## 4. Per-inverter records

Every run: 3 epochs, one adapter checkpoint + one held-out eval per epoch (201 optimizer steps per
epoch at 4,806 rows / grad-accum 24; the final adapter at the run root is the epoch-3 one). Rates are
least-squares fits of TRL's `num_tokens` against wall time over every log point of the run. "Train
window" is the mean train loss over the last 50 steps of the epoch.

| inverter | rows trained / dropped | steps | train wall | tok/s (regime fit) | peak VRAM alloc / reserved | eval_loss e1 / e2 / e3 | train window at e1 / e2 / e3 end | train − eval gap | adapter |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| 7B-sum | 4,806 / 0 | 603 | 9.12 h (32,848 s) | 2,141 | 15.44 / 18.39 GiB | **0.3866 / 0.3810 / 0.3847** | 0.374 / 0.349 / 0.313 | +0.012 / +0.032 / +0.072 | 488 MB |

### 7B-sum — `bench/results/phase2/inverter-7b-sum/`, 2026-08-28 20:14 → 08-29 05:21

Train loss, mean per 50-step window (`log_history.json`, `logging_steps=5`):

```
epoch 1:  0.4209  0.3821  0.3771  0.3741      eval 0.3866  (tok-acc 0.864, entropy 0.388)
epoch 2:  0.3471  0.3536  0.3419  0.3485      eval 0.3810  (tok-acc 0.866, entropy 0.360)
epoch 3:  0.3205  0.3250  0.3156  0.3129      eval 0.3847  (tok-acc 0.866, entropy 0.334)
```

The drop is front-loaded — 0.48 (step 1) → ~0.38 inside the first ~100 steps, flat for the rest of
epoch 1 — and each later epoch starts with a ~0.025 step down on the train side that eval does not
follow: eval improves by 0.006 at epoch 2 and gives 0.004 of it back at epoch 3, while the train–eval
gap widens 0.012 → 0.032 → 0.072. Epochs 2–3 mostly fit the training rows. Which epoch Phase 4 serves
is the user's decision from this curve and §5; the three adapters are on disk.

Merge check on the epoch-3 adapter (`merge-check`, printed in `bench/logs/phase2-7b-sum.log`): layer-0
`in_proj_qkv` max|Δ| 3.2e-3 with 88.6 % of elements changed; layer-3 `q_proj` 3.3e-3 / 85.1 %. Merge
10 s on the GPU; merged bf16 weights 8.43 GB, deleted after the inversion.

Disk after the run: 50 GB free. 59 → 54 GB at launch = training venv 5.1 GB + flash-attn2 hub kernel
1.0 GB + formatted files 0.55 GB; 54 → 50 = three 1.4 GB checkpoints (adapter + optimizer state — the
handoff assumed ~250 MB) + 0.5 GB final adapter + 0.5 GB probe adapter.

---

## 5. Held-out inversion — paired lengths on the same 200 rows

`bench/invert.py` on the merged epoch-3 weights, vLLM 0.27.1, temp 0.7 · top_p 0.9 · repetition_penalty
1.05 · max_tokens 8192 · seed 1234 · `max_model_len` 12288 (holdout prompts max 2,391 tokens, so no
row's cap is cut) · `enable_thinking=False`. Raw text saved (`raw`), `t_hat` = raw with a defensive
`</think>` strip (fired 0 times). Cap-hit is vLLM's `finish_reason == "length"`. Paired on the same 200
rows for every inverter; token counts with the **Qwen3.5-4B tokenizer**; `phase1_stats.py --mode
inverted`. Gates: 0 empty · 200 rows · every idx in the holdout · median t̂ ≥ half the t_true median.

| inverter | t̂ median / mean / p05 / p95 | t_true median / mean / p05 / p95 | t̂ / t_true at the median | cap-hit @ 8192 | empty |
|---|---|---|---:|---|---:|
| 7B-sum, epoch 3 | **2,149** / 2,627 / 181 / 8,000 | 2,172 / 2,574 / 190 / 6,390 | **0.99** | 5.0 % (10) | 0 |

### 7B-sum, epoch 3 — `bench/results/phase2/holdout-7b-sum.jsonl`

Per-row ratio median 1.00; t̂ shorter than t_true on exactly 50.0 % of rows. By domain, t̂ / t_true
median: math 2,362 / 2,429 (n=151) · code 2,053 / 2,355 (28) · physics 1,065 / 1,001 (9) · biology
905 / 893 (5) · puzzle 584 / 388 (5) · chemistry 710 / 800 (2). t̂'s p95 sits at the cap (8,000) where
t_true's is 6,390: the inverter's tail is heavier than the surrogate's — the 10 cap-hits are all math
rows (t_true 2,272–6,075 tokens) whose tails are still running case checks (most repeated 40-char chunk
in the last 4,000 chars occurs once), i.e. long computations, not repetition loops. vLLM throughput on
the 200 rows: ~5.5 min wall for 525k generated tokens.

Three rows read (shortest, median, longest t_true):

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 77473 (math: paint a 9×12 wall minus a 2×4 window; y = 100 sq ft) | 68 → 146 | **No.** The trace invents a 4×3 window and concludes 96 sq ft, although the summary `b` states the 8 sq ft window. Same three-step skeleton as t_true ("First… Next… Finally…"), wrong arithmetic. |
| median, idx 84388 (math: all positive-integer pairs with x+y+xy=2006) | 2,191 → 2,382 | **Yes.** Rewrites as (x+1)(y+1)=2007, factors 3²·223, checks the divisors, ends with the same four boxed pairs as y. Style indistinguishable from the surrogate's ("Okay, so I have this equation…"). |
| long, idx 14587 (math: angle bisector of C vs side DE of the square on the hypotenuse; y = 2:5) | 7,755 → 6,393 | **Yes, in a different form.** Coordinates, bisector, intersection, concludes 2:5 but boxes `\dfrac{2}{5}` after debating which form the grader wants — t_true has the mirror-image debate and boxes 2:5. |

Measured, no interpretation: on unseen inputs the 7B-sum inverter produces traces at the surrogate's
length (median ratio 0.99), in the surrogate's register, that end in a boxed answer; whether that
answer is the given one was 2 of 3 in the rows read. The paper measures fidelity only downstream
(student accuracy, Phase 5); nothing here is promoted to a rate.
