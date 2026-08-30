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

### 7B-nosum probe — `bench/logs/phase2-probe-7b-nosum.log`, 2026-08-29 05:28

| | |
|---|---|
| loss @ step 1 / 10 / 20 | **0.4656 / 0.3524 / 0.4576**; half-means 0.4326 → 0.4092, first five 0.4457 → last five 0.4161; `grad_norm` 1.06 → 0.146; token accuracy 0.839 → 0.853 |
| peak VRAM | **15.27 GiB** allocated |
| realized train tokens/s | **2,127** steady-state (2,121 overall) |
| projection @ 2,127 tok/s | 7B-nosum **8.2 h** · 1.5B-sum 9.2 h · 1.5B-nosum 7.9 h · remaining three 25.3 h |
| paths | DeltaNet `fla` · conv torch fallback · attention `sdpa` |
| rows at `max_length` / tokens per epoch (TRL) | 0 / 20,231,546 (= the formatter's count); row 0: 878 tokens, loss mask 133 |
| wall clock | 938 s for 20 steps (1,990,129 tokens) |

### 1.5B-sum probe — `bench/logs/phase2-probe-1.5b-sum.log`, 2026-08-29 14:05

| | |
|---|---|
| loss @ step 1 / 10 / 20 | **0.5579 / 0.4368 / 0.4369**; half-means 0.4881 → 0.4404, first five 0.5208 → last five 0.4264; `grad_norm` 1.36 → 0.124; token accuracy 0.816 → 0.843. Starts higher than the 7B arm's 0.47 (token accuracy 0.816 vs 0.83): the 1.5B surrogate's traces are less predictable to the base model, which sits next to Phase 1's dispersion measurement — the 1.5B's kept traces were more spread than R1's on the same prompts, p75/p25 3.69 vs 2.66 (`phase1.md` §4) — cited, not explained |
| peak VRAM | **15.39 GiB** allocated / 16.80 reserved |
| realized train tokens/s | **2,102** steady-state (2,101 overall) |
| projection @ 2,102 tok/s | 1.5B-sum **9.3 h** · 1.5B-nosum 8.0 h · remaining two 17.3 h |
| paths | DeltaNet `fla` · conv torch fallback · attention `sdpa` |
| rows at `max_length` / tokens per epoch (TRL) | 0 / 22,604,920 (= the formatter's count), max row 10,425; row 0: 1,501 tokens, loss mask 245 |
| wall clock | 1,046 s for 20 steps |

### 1.5B-nosum probe — `bench/logs/phase2-probe-1.5b-nosum.log`, 2026-08-29 23:35

| | |
|---|---|
| loss @ step 1 / 10 / 20 | **0.5455 / 0.4342 / 0.4358**; half-means 0.4790 → 0.4385, first five 0.5060 → last five 0.4253; `grad_norm` 1.20 → 0.114; token accuracy 0.819 → 0.843 |
| peak VRAM | **15.19 GiB** allocated / 16.80 reserved |
| realized train tokens/s | **2,139** steady-state (2,134 overall) |
| projection @ 2,139 tok/s | 1.5B-nosum **7.8 h** (the last run) |
| paths | DeltaNet `fla` · conv torch fallback · attention `sdpa` |
| rows at `max_length` / tokens per epoch (TRL) | 0 / 19,354,876 (= the formatter's count), max row 9,787; row 0: 1,089 tokens, loss mask 245 |
| wall clock | 881 s for 20 steps |

---

## 4. Per-inverter records

Every run: 3 epochs, one adapter checkpoint + one held-out eval per epoch (201 optimizer steps per
epoch at 4,806 rows / grad-accum 24; the final adapter at the run root is the epoch-3 one). Rates are
least-squares fits of TRL's `num_tokens` against wall time over every log point of the run. "Train
window" is the mean train loss over the last 50 steps of the epoch.

| inverter | rows trained / dropped | steps | train wall | tok/s (regime fit) | peak VRAM alloc / reserved | eval_loss e1 / e2 / e3 | train window at e1 / e2 / e3 end | train − eval gap | adapter |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| 7B-sum | 4,806 / 0 | 603 | 9.12 h (32,848 s) | 2,141 | 15.44 / 18.39 GiB | **0.3866 / 0.3810 / 0.3847** | 0.374 / 0.349 / 0.313 | +0.012 / +0.032 / +0.072 | 488 MB |
| 7B-nosum | 4,806 / 0 | 603 | 8.12 h (29,214 s) | 2,085 | 15.27 / 17.54 GiB | **0.3896 / 0.3840 / 0.3876** | 0.375 / 0.349 / 0.314 | +0.015 / +0.035 / +0.074 | 488 MB |
| 1.5B-sum | 4,828 / 0 | 606 | 8.89 h (31,986 s) | 2,123 | 15.41 / 17.99 GiB | **0.4296 / 0.4233 / 0.4257** | 0.397 / 0.374 / 0.351 | +0.033 / +0.049 / +0.075 | 488 MB |
| 1.5B-nosum | 4,828 / 0 | 606 | 7.67 h (27,596 s) | 2,106 | 15.29 / 17.10 GiB | **0.4310 / 0.4248 / 0.4272** | 0.397 / 0.375 / 0.351 | +0.034 / +0.050 / +0.076 | 488 MB |

Four runs, one shape: a front-loaded drop in the first ~100 steps, a train-side step at each epoch
boundary that eval does not follow, eval best at epoch 2 by 0.004–0.006 and 0.002–0.004 back at epoch
3, the train–eval gap widening to +0.07 by epoch 3. The summary costs the inverter 0.002–0.003 of
eval loss on both arms; the arm costs 0.04. Total training wall clock **33.8 h** (9.12 + 8.12 + 8.89
+ 7.67) at 2,085–2,141 tok/s — the probes projected 34.3 h at 2,153.

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

### 7B-nosum — `bench/results/phase2/inverter-7b-nosum/`, 2026-08-29 05:44 → 13:52

```
epoch 1:  0.4190  0.3827  0.3779  0.3750      eval 0.3896  (tok-acc 0.863, entropy 0.391)
epoch 2:  0.3480  0.3545  0.3427  0.3493      eval 0.3840  (tok-acc 0.865, entropy 0.362)
epoch 3:  0.3215  0.3259  0.3165  0.3137      eval 0.3876  (tok-acc 0.865, entropy 0.336)
```

The same shape as 7B-sum to within 0.003 at every point — front-loaded drop, a train-side step at
each epoch boundary, eval best at epoch 2 by 0.006 and 0.004 back at epoch 3, gap 0.015 → 0.035 →
0.074. Without the summary the inverter's teacher-forced loss on the traces is only 0.003 higher than
with it. Merge check on the epoch-3 adapter: `in_proj_qkv` 2.7e-3 / 88.5 %, `q_proj` 3.3e-3 / 84.8 %
(`merge-check.json`). Disk after the run: 43 GB free.

### 1.5B-sum — `bench/results/phase2/inverter-1.5b-sum/`, 2026-08-29 14:23 → 23:16

```
epoch 1:  0.4520  0.4094  0.4107  0.3966      eval 0.4296  (tok-acc 0.852, entropy 0.429)
epoch 2:  0.3845  0.3763  0.3794  0.3739      eval 0.4233  (tok-acc 0.853, entropy 0.402)
epoch 3:  0.3633  0.3575  0.3464  0.3502      eval 0.4257  (tok-acc 0.853, entropy 0.380)
```

Same shape as the 7B arms — front-loaded, eval best at epoch 2 by 0.006 and 0.002 back at epoch 3 —
offset ~0.04 higher throughout (probe start 0.56 vs 0.47; `phase1.md` §4's dispersion cross-reference
in §3), with the train–eval gap opening earlier: +0.033 at epoch 1 where the 7B arms had +0.012 /
+0.015. 202 steps per epoch (4,828 rows). Merge check on the epoch-3 adapter: `in_proj_qkv` 2.9e-3 /
88.4 %, `q_proj` 3.8e-3 / 84.0 %. Disk after the run: 36 GB free.

### 1.5B-nosum — `bench/results/phase2/inverter-1.5b-nosum/`, 2026-08-29 23:48 → 08-30 07:28

```
epoch 1:  0.4500  0.4098  0.4113  0.3974      eval 0.4310  (tok-acc 0.851, entropy 0.431)
epoch 2:  0.3853  0.3769  0.3801  0.3745      eval 0.4248  (tok-acc 0.853, entropy 0.404)
epoch 3:  0.3639  0.3581  0.3471  0.3508      eval 0.4272  (tok-acc 0.853, entropy 0.381)
```

Within 0.002 of 1.5B-sum at every point. Merge check on the epoch-3 adapter: `in_proj_qkv` 3.4e-3 /
87.6 %, `q_proj` 3.2e-3 / 84.3 %. Disk after the run: 30 GB free.

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
| 7B-nosum, epoch 3 | **1,966** / 2,530 / 196 / 7,563 | 2,172 / 2,574 / 190 / 6,390 | **0.91** | 5.0 % (10) | 0 |
| 7B-sum, epoch 2 (`checkpoint-402`) | **2,241** / 2,507 / 157 / 6,777 | 2,172 / 2,574 / 190 / 6,390 | **1.03** | 3.0 % (6) | 0 |
| 1.5B-sum, epoch 3 | **1,964** / 2,974 / 283 / 8,192 | 2,178 / 2,667 / 283 / 6,633 | **0.90** | **13.0 % (26)** | 0 |
| 1.5B-sum, epoch 2 (`checkpoint-404`) | **2,063** / 3,066 / 305 / 8,192 | 2,178 / 2,667 / 283 / 6,633 | **0.95** | **17.0 % (34)** | 0 |
| 1.5B-nosum, epoch 3 | **1,930** / 2,749 / 268 / 8,192 | 2,178 / 2,667 / 283 / 6,633 | **0.89** | **10.5 % (21)** | 0 |
| 1.5B-nosum, epoch 2 (`checkpoint-404`) | **2,071** / 3,175 / 284 / 8,192 | 2,178 / 2,667 / 283 / 6,633 | **0.95** | **14.5 % (29)** | 0 |
| 7B-nosum, epoch 2 (`checkpoint-402`) | **2,095** / 2,683 / 151 / 8,192 | 2,172 / 2,574 / 190 / 6,390 | **0.96** | 7.0 % (14) | 0 |

The 1.5B arm's `t_true` column is the 1.5B surrogate's own traces for the same 200 prompts (the holdout
is paired on prompts; each arm's ground truth is its own surrogate), so the true-length columns differ
slightly between arms.

**Arm-level finding, stated as measured.** The 1.5B-arm inverter runs away far more than the 7B-arm
inverters — cap-hit 13.0 % (epoch 3) / 17.0 % (epoch 2) against 5.0 % / 3.0 % — although every training
target on both arms terminated under the cap (`D₂` is kept rows only). Its capped rows are not the
long-truth rows (their `t_true` spans 383–6,934 tokens) and 25 of 26 are still computing at the cut.
This sits next to Phase 1's surrogate cap-hit of 45.9 % (1.5B) vs 34.6 % (7B) on identical prompts: the
weak surrogate's runaway tendency reaches the inverter through the style of its kept traces, not through
severed targets. It is the first arm-level difference that would matter in Phase 4 — more capped
forgeries to handle.

**Epoch 2 vs 3, per arm, unresolved on the 1.5B arm.** On 7B-sum epoch 2 had fewer cap-hits (6 vs 10)
and a higher graded match (95.8 vs 93.8 %); on 1.5B-sum the direction reverses (34 vs 26; 91.2 vs
94.2 %) while eval loss still favours epoch 2 by 0.002. One sampled draw each at temperature 0.7, and
Phase 1 measured a ~15 % cap-hit flip rate between two draws of the same model — directions, not
differences; not averaged across arms.

Direction, stated as measured and not explained: the no-summary inverter's traces are **shorter** than
the summary inverter's at the median here (1,966 vs 2,149; 0.91 vs 0.99 of the true length). The paper's
Table 2 reports the opposite direction for its R1-Weak surrogate (no-summary 5,434 vs summary 4,972
tokens) — different ground truth (R1 vs a 7B distill) and a different backbone.

**Answer consistency, three labels** (`docs/11` §5: every mismatch read). *Equivalent-form* = the grader
rejects an equivalent answer; *alternative-valid* = a different answer that is also correct but is not
the one the trace was conditioned on; *genuine* = the trace argues away from the given answer. Two
derived counts: **inconsistent with the conditioned answer** = genuine + alternative-valid (what makes a
student target `[t̂; y]` contradictory), and **wrong** = genuine.

| inverter | graded match | equivalent-form | alternative-valid | genuine | not gradable | by hand | inconsistent with y | wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B-sum, epoch 3 | 105 / 112 | 3 | 0 | 3 | 1 (60498, proof with no canonical box) | 109 / 112 | **3 / 112** | 3 / 112 |
| 7B-nosum, epoch 3 | 105 / 114 | 4 | 1 | 3 | 1 (42804, cap-severed) | 110 / 113 | **4 / 113** | 3 / 113 |
| 7B-sum, epoch 2 | 113 / 118 | 3 | 1 | 1 | 0 | 116 / 118 | **2 / 118** | 1 / 118 |
| 1.5B-sum, epoch 3 | 97 / 103 | 0 | 0 | 6 | 0 | 97 / 103 | **6 / 103** | 6 / 103 |
| 1.5B-sum, epoch 2 | 93 / 102 | 2 | 0 (+1 unverified) | 6 | 0 | 95 / 102 | **7 / 102** | 6 / 102 |
| 1.5B-nosum, epoch 3 | 98 / 104 | 1 | 0 | 5 | 0 | 99 / 104 | **5 / 104** | 5 / 104 (3 inverter-wrong + 2 where R1 sides with the trace) |
| 1.5B-nosum, epoch 2 | 88 / 99 | 1 | 0 | 10 | 0 | 89 / 99 | **10 / 99** | 10 / 99 (5 inverter-wrong + 5 where R1 sides with the trace) |
| 7B-nosum, epoch 2 | 102 / 108 | 3 | 0 | 2 | 1 (42804, proof) | 105 / 107 | **2 / 108** | 2 / 108 (1 inverter-wrong + 1 where R1 sides with the trace) |

**Against R1's answer** — the boxed answer in each OpenThoughts row's own R1 solution (the assistant
text after `</think>` in `llamafactory/OpenThoughts-114k`, same idx): *the dataset's solution, not
ground truth*. On the surrogate holdout `y` is the surrogate's answer and is sometimes wrong, so
"t̂ matches y" mixes inverter fidelity with surrogate error; grading both against R1 splits the genuine
cases. Same grader (`extract_boxed` + `grade`), so equivalent forms fail here too; R1 is unboxed on 51
of the 200 rows. Per-row values (`r1`, `y_vs_r1`, `t_hat_vs_r1`) are in every `-consistency.json`.

| inverter | y vs R1 (the surrogate's own answer) | t̂ vs R1 | of the graded mismatches: inverter wrong (y = R1, t̂ ≠ R1) | inverter right, surrogate wrong (t̂ = R1, y ≠ R1) | neither | R1 unboxed |
|---|---:|---:|---:|---:|---:|---:|
| 7B-sum, epoch 3 | 113 / 145 (77.9 %) | 85 / 105 (81.0 %) | 4 | 0 | 3 | 0 |
| 7B-nosum, epoch 3 | 113 / 145 (77.9 %) | 86 / 105 (81.9 %) | 3 | 0 | 3 | 3 |
| 7B-sum, epoch 2 | 113 / 145 (77.9 %) | 91 / 109 (83.5 %) | 1 | 0 | 4 | 0 |
| 1.5B-sum, epoch 3 | 104 / 144 (72.2 %) | 84 / 98 (85.7 %) | 2 | **4** | 0 | 0 |
| 1.5B-sum, epoch 2 | 104 / 144 (72.2 %) | 83 / 98 (84.7 %) | 3 | **4** | 2 | 0 |
| 1.5B-nosum, epoch 3 | 104 / 144 (72.2 %) | 83 / 99 (83.8 %) | 2 | 1 | 2 | 1 |
| 1.5B-nosum, epoch 2 | 104 / 144 (72.2 %) | 80 / 96 (83.3 %) | 4 | 5 | 2 | 0 |
| 7B-nosum, epoch 2 | 113 / 145 (77.9 %) | 86 / 100 (86.0 %) | 1 | 1 | 3 | 1 |

The unpaired columns above have different denominators (`y` gradable on 144–145 rows, t̂ on 98–109,
because t̂ often does not box and the rows where it does may be the easier ones), so they are not
compared. **Paired, on the rows gradable on both sides** (`docs/11` §5):

| inverter | n (both gradable) | y vs R1 | t̂ vs R1 | y right, t̂ wrong | t̂ right, y wrong |
|---|---:|---:|---:|---:|---:|
| 7B-sum, epoch 3 | 105 | 89 (84.8 %) | 85 (81.0 %) | 4 | 0 |
| 7B-nosum, epoch 3 | 105 | 89 (84.8 %) | 86 (81.9 %) | 3 | 0 |
| 7B-sum, epoch 2 | 109 | 92 (84.4 %) | 91 (83.5 %) | 1 | 0 |
| 1.5B-sum, epoch 3 | 98 | 82 (83.7 %) | 84 (85.7 %) | 2 | 4 |
| 1.5B-sum, epoch 2 | 98 | 82 (83.7 %) | 83 (84.7 %) | 3 | 4 |
| 1.5B-nosum, epoch 3 | 99 | 84 (84.8 %) | 83 (83.8 %) | 2 | 1 |
| 1.5B-nosum, epoch 2 | 96 | 79 (82.3 %) | 80 (83.3 %) | 4 | 5 |
| 7B-nosum, epoch 2 | 100 | 86 (86.0 %) | 86 (86.0 %) | 1 | 1 |

Arm-paired on the 86 idx gradable on both sides for both arms' epoch-3 summary inverters: 7B `y`
76/86 (88.4 %), t̂ 74/86 (86.0 %); 1.5B `y` 72/86 (83.7 %), t̂ 74/86 (86.0 %).

Measured: the unpaired impression that forged traces agree with R1 more often than the surrogates'
own answers was selection — paired, the 7B-arm inverters are 3–4 points *under* their surrogate's
answer (4 / 3 / 1 rows lost, none gained) and the 1.5B-arm inverter is 1–2 points over its weaker
surrogate (4 rows gained, 2–3 lost). On the 7B arm no genuine mismatch is the inverter correcting the
surrogate; on the 1.5B arm four of the six genuine epoch-3 mismatches are the inverter landing on R1's
answer against a surrogate `y` that does not (18067, 14587, 21120 among them). On victim data
(Phase 4) `y` will mostly be right, so the override behaviour that helps on this arm would mostly hurt
there; that is a Phase 4/5 decision.

The 1.5B arm's idx 15523 is the clearest style-inheritance example: a plan-only trace with no number,
matching the surrogate's own plan-only trace on that prompt (`t_true` never computes 243 either).

### 7B-sum, epoch 3 — `bench/results/phase2/holdout-7b-sum.jsonl`

Per-row ratio median 1.00; t̂ shorter than t_true on exactly 50.0 % of rows. By domain, t̂ / t_true
median: math 2,362 / 2,429 (n=151) · code 2,053 / 2,355 (28) · physics 1,065 / 1,001 (9) · biology
905 / 893 (5) · puzzle 584 / 388 (5) · chemistry 710 / 800 (2). t̂'s p95 sits at the cap (8,000) where
t_true's is 6,390: the inverter's tail is heavier than the surrogate's — the 10 cap-hits are all math
rows (t_true 2,272–6,075 tokens); 8 of the 10 are still running case checks at the cut and 2 show a
repeated 40-char chunk (≥3× in the last 4,000 chars), the loop test used for every inversion in §5.4. vLLM throughput on
the 200 rows: ~5.5 min wall for 525k generated tokens.

Three rows read (shortest, median, longest t_true):

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 77473 (math: paint a 9×12 wall minus a 2×4 window; y = 100 sq ft) | 68 → 146 | **No.** The trace invents a 4×3 window and concludes 96 sq ft, although the summary `b` states the 8 sq ft window. Same three-step skeleton as t_true ("First… Next… Finally…"), wrong arithmetic. |
| median, idx 84388 (math: all positive-integer pairs with x+y+xy=2006) | 2,191 → 2,382 | **Yes.** Rewrites as (x+1)(y+1)=2007, factors 3²·223, checks the divisors, ends with the same four boxed pairs as y. Style indistinguishable from the surrogate's ("Okay, so I have this equation…"). |
| long, idx 14587 (math: angle bisector of C vs side DE of the square on the hypotenuse; y = 2:5) | 7,755 → 6,393 | **Yes, in a different form.** Coordinates, bisector, intersection, concludes 2:5 but boxes `\dfrac{2}{5}` after debating which form the grader wants — t_true has the mirror-image debate and boxes 2:5. |

**Answer consistency** (report, not a gate — `phase1_stats.py --mode inverted`): the last `\boxed{}`
in t̂ against the last `\boxed{}` in `y`, graded with `eval_baseline.py`'s `extract_boxed` + `grade`
(math_verify symbolic equivalence, normalized-string fallback), main thread.

| inverter | match | mismatch | no box in t̂ | no box in y | match rate among graded |
|---|---:|---:|---:|---:|---:|
| 7B-sum, epoch 3 | 105 | 7 | 44 | 44 | **93.8 %** (n=112) |

The seven mismatches, read (`docs/11` §5: sample rows marked wrong and read them): **genuine 3** —
idx 27165 (work to pump a paraboloid boiler: y `ρgπH³/(6a²)`, the trace "corrects the units" and boxes
`ρgπH³/(6a)`), 6618 (200-candies pigeonhole: y 21, the trace talks itself from 21 down to 20), 6960
(2015×2015 checkerboard Hamiltonian path: y Yes, trace No); **grader misses 3** — 29482 (`12a+11b=2002`:
the same family `(165−11k, 12k+2)` with the k-range 0..14 stated in the prose outside the box), 39587
(fishing order: y boxes `P` for the least, the trace boxes the full order `V > T > K > P`, same
conclusion), 82678 (`f(x)=x+c` vs `x+C`); **ambiguous 1** — 60498 (a proof problem; y's own box is `0`,
the trace boxes the statement `a=b or b=c or c=a`). So the graded 93.8 % is a lower bound — ≈97 %
(109/112) by hand — and genuine forgery failures are 3 of 112 gradable rows. All three genuine failures
have the same shape: the trace **argues itself away from the answer it was conditioned on** — a units
"correction", 21 talked down to 20, Yes to No — the answer-conditioning overridden by the model's own
reasoning, not a format or truncation failure. That is the class Phase 4/5 will have to decide about:
a student target `[t̂; y]` whose t̂ concludes otherwise is contradictory supervision, and the paper does
no filtering. The "no box in t̂" bucket
is two different things: **9 severed at the cap** (cut before their final answer; one cap-hit row boxes
earlier) and **35 ending in prose**; a trace can reach the answer without boxing it (idx 77473 above did
neither), so it is not a failure count. "No box in y" is mostly the non-math domains, whose surrogate
answers carry no `\boxed{}`. Per-row `(gold, pred, bucket, finish_reason)` is written beside every
inversion file as `<name>-consistency.json`, re-gradable without regenerating.

Measured, no interpretation: on unseen inputs the 7B-sum inverter produces traces at the surrogate's
length (median ratio 0.99), in the surrogate's register, that end in a boxed answer; 94 % of the
gradable ones box the answer they were given. The paper measures fidelity only downstream (student
accuracy, Phase 5).

### 7B-nosum, epoch 3 — `bench/results/phase2/holdout-7b-nosum.jsonl`

Per-row ratio median 0.97; t̂ shorter on 53.5 % of rows; the median is 9 % under the surrogate's where
the summary-conditioned inverter's was 1 % under. By domain, t̂ / t_true median: math 2,215 / 2,429
(n=151) · code 1,824 / 2,355 (28) · physics 998 / 1,001 (9) · biology 1,046 / 893 (5) · puzzle
579 / 388 (5) · chemistry 717 / 800 (2). Cap-hit 10/200 again, but a different ten: the overlap with
7B-sum's cap-hit set is **2** rows (idx 32747, 61555; computed from the two files' `finish_reason`),
so which row runs past 8,192 is mostly a property of the sample, not of the prompt.

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 77473 (paint the wall; y = 100) | 68 → 151 | **Yes**, unboxed: 9×12 = 108, window 2×4 = 8, 108 − 8 = 100 — the arithmetic the summary-conditioned inverter got wrong, with no summary to read. |
| median, idx 84388 (x+y+xy=2006) | 2,191 → 2,299 | **Yes.** Same factorisation, same four boxed pairs. |
| long, idx 14587 (angle bisector vs DE; y = 2:5) | 7,755 → 4,431 | **No.** Coordinates, bisector y = x, intersection (39/7, 39/7), then "DG : GE = 5 : 2" and boxes `\dfrac{5}{2}` — the orientation of y's 2:5 inverted. |

| inverter | match | mismatch | no box in t̂ (cap-severed + prose) | no box in y | match rate among graded |
|---|---:|---:|---|---:|---:|
| 7B-nosum, epoch 3 | 105 | 9 | 42 (6 + 36) | 44 | **92.1 %** (n=114) |

The nine mismatches, read: **genuine 3** — 18067 (202-city airlines: argues itself from 101 down to
100), 59441 (ellipse chord: √93/3 against √7), 14587 (above, 5:2 for 2:5); **grader misses /
alternative valid answers 5** — 29482 (k-range in the prose), 69793 ("Rhombus" vs "The parallelogram
is a rhombus."), 69125 (a construction described in other words), 39587 (four names boxed separately,
the last is Petr = y's `P`), 16770 (`3+9k` where y has `7+8k`: 3 to the first power rules out two
squares and 3 mod 9 rules out two cubes, so it is a valid progression — *alternative-valid*, not the
surrogate's answer); **not gradable 1** — 42804 (cap-severed with an empty `\boxed{}`). ≈97 % (110/113)
by hand; genuine failures 3 of 113, the same count and the same shape as 7B-sum — the trace arguing
away from the answer it was given; inconsistent with the conditioned answer 4 of 113.

The idx 77473 pair is a single-row observation, n = 1, no rate: the summary-conditioned epoch-3 inverter
invented a 4×3 window and got 96; the no-summary one and the summary epoch-2 one both reached 100 in prose.

### 7B-sum, epoch 2 — `bench/results/phase2/holdout-7b-sum-ep2.jsonl` (`bench/run_phase2_invert.sh`)

Run at the GPU gap after 7B-nosum because eval loss is lowest at epoch 2 and the checkpoint choice cannot
be made on eval loss alone. Same 200 rows, same sampling. Per-row ratio median 1.03; t̂ shorter on
49.0 %; cap-hit 6/200 against epoch 3's 10 (overlap with epoch 3's ten: see the consistency file); p95
6,777 against 8,000. Merge check: `in_proj_qkv` 3.2e-3 / 88.0 %, `q_proj` 3.3e-3 / 84.5 %.

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 77473 | 68 → 146 | **Yes**, in prose: 108 − 8 = 100. |
| median, idx 84388 | 2,191 → 2,346 | **Yes.** Same four boxed pairs. |
| long, idx 14587 | 7,755 → 4,485 | **Yes.** Concludes "DF is shorter than FE … 2:5" and boxes `\dfrac{2}{5}`, orientation right. |

Consistency: match 113, mismatch 5, no box in t̂ 38 (3 cap-severed + 35 prose), no box in y 44 —
95.8 % graded (n=118). The five, read: **genuine 1** — 6618 (21 talked down to 20, as at epoch 3);
**alternative-valid 1** — 16770 (`3+9k`); **equivalent-form 3** — 25352 (a proof: boxes `p − 1` for
y's `k | p − 1`), 29482 (k-range in the prose), 54798 (`4·sign(sin 2x)` ≡ `4·sign(sin x)·sign(cos x)`).
By hand 116/118; inconsistent with the conditioned answer 2/118; wrong 1/118. Epoch 2 against epoch 3
on identical rows and sampling: cap-hit 6 vs 10, p95 6,777 vs 8,000, graded match 95.8 % vs 93.8 %,
median ratio 1.03 vs 0.99 — one draw each at temperature 0.7, so the differences carry the resampling
noise Phase 1 measured (cap-hit flips on ~15 % of rows between two draws of the same model). Which
adapter Phase 4 serves is the user's decision.

### 7B-nosum, epoch 2 — `bench/results/phase2/holdout-7b-nosum-ep2.jsonl`

Run in the last GPU gap. t̂ 2,095 / 2,683 / 151 / 8,192; ratio 0.96 (per-row 1.00); cap-hit 14/200
(7.0 %), 0 loops, 4 shared with epoch 3's ten; merge check 2.8e-3 / 87.9 %, 3.1e-3 / 84.1 %.
Consistency 102 / 6 / 48 (12 cap-severed + 36 prose) / 44 — 94.4 % graded (n=108). The six, read:
**equivalent-form 3** — 29482 (k-range in prose), 54798 (`4·sgn(sin x cos x)`), 70443 (a two-part
answer: boxes part (a)'s 2048, states part (b)'s "such two inhabitants exist" in prose = y);
**ambiguous 1** — 42804 (a proof with "for which other numbers?" — boxes `n`); **genuine 2** — 6618
(20 for 21 = R1, inverter wrong), 34191 (option D = R1 against y's C, surrogate wrong). By hand
105/107; inconsistent 2/108; wrong 1/108. Paired vs R1 (n=100): 86.0 % / 86.0 %, one row each way.
Short 77473 → 100 in prose ✓; median 84388 ✓; long 14587 **capped** at 8,192 mid-computation (the
only read row that hit the cap in any inversion).

### 1.5B-sum, epoch 3 — `bench/results/phase2/holdout-1.5b-sum.jsonl`

Per-row ratio median 1.01; t̂ shorter on 49.5 %. The tail is the finding: **26 cap-hits** (20 math,
5 code, 1 physics) against the 7B arm's 10, p95 at the cap, mean 2,974 against a true mean of 2,667
while the median is 10 % under. The capped rows are not the long-true-trace rows — their `t_true` runs
from 383 to 6,934 tokens — and 1 of the 26 is a repetition loop (idx 59048, a 40-char chunk repeated
8× in its last 4,000 chars); the other 25 are still computing when cut. By domain, t̂ / t_true median:
math 2,163 / 2,396 (n=151) · code 2,318 / 2,366 (28) · physics 841 / 1,046 (9) · biology 856 / 927
(5) · puzzle 416 / 459 (5) · chemistry 779 / 2,378 (2).

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 15523 (car rental, y = 243 km) | 113 → 97 | **No number stated** — a four-step plan ending "I'll check which of the given options matches", exactly the shape of the 1.5B surrogate's own trace for this row, which also never computes 243. |
| median, idx 3282 (three lines through one point; y = A) | 2,229 → 1,570 | **Yes.** Tests each option, boxes A. |
| long, idx 57422 (unique n with Sₙ an integer; y = 12) | 7,758 → 2,409 | **Yes**, in a third of the surrogate's length: Minkowski-style bound, 145 = 17²−12²… boxes 12. |

Consistency: match 97, mismatch 6, no box in t̂ 47 (19 cap-severed + 28 prose), no box in y 50 —
94.2 % graded (n=103). All six mismatches are **genuine** (no equivalent-form or alternative-valid
cases): 47586 (cone lateral area (√13/3)πR² for πR²√5), 21551 (option A for B), 63867 (8,682,544 for
8,682,572), 8576 (boxes BC = 7,033 where y has 6,745 — the two legs swapped), and two where the trace
argues away from a surrogate answer that is itself the wrong one: 18067 (y = 201 companies, trace 101 —
the 7B surrogate's y was 101) and 14587 (y = 5/2, trace 2:5 — the 7B surrogate's y was 2:5). Consistency
measures agreement with the conditioned answer, not correctness; on this arm the conditioned answer is
the weaker surrogate's.

### 1.5B-sum, epoch 2 — `bench/results/phase2/holdout-1.5b-sum-ep2.jsonl`

t̂ 2,063 / 3,066 / 305 / 8,192; ratio at the median 0.95 (per-row 1.02); **cap-hit 34/200 (17.0 %)**
against epoch 3's 26; empty 0; merge check 2.8e-3 / 87.7 %, 3.5e-3 / 83.1 %. Consistency 93 / 9 / 48
(21 cap-severed + 27 prose) / 50 — 91.2 % graded (n=102). The nine, read: **equivalent-form 2** —
84388 (the same four pairs boxed in another order; the grader takes the last box), 15523 (computes 243
and boxes the option letter `C`); **genuine 6** — 25140 (12√26/5 for 12/5), 21120 (y says the integral
converges, the trace says it diverges — ∫₁^∞ sin²x/x dx does diverge), 21551 (A for B), 6618 (202 for
21), 74895 (34 for 19), 14587 (2:5 for y's 5/2 again); **different answer, validity not verified 1** —
16770 (`15+28n` for y's `3+12n`). By hand 95/102; inconsistent with the conditioned answer 7/102;
wrong 6/102. Median 3282 ✓ (A, 1,783 tokens), long 57422 ✓ (12, 3,450 tokens). Epoch 2 against epoch 3
on this arm: cap-hit 34 vs 26, graded match 91.2 vs 94.2 %, ratio 0.95 vs 0.90 — the opposite
direction from 7B-sum's epoch-2-vs-3 comparison; one draw each.

### 1.5B-nosum, epoch 3 — `bench/results/phase2/holdout-1.5b-nosum.jsonl`

Per-row ratio median 0.98; t̂ shorter on 52.0 %. Cap-hit 21/200 (18 math, 3 code; `t_true` 689–6,934
tokens), of which **4 are repetition loops** (idx 21120 ×10, 8576 ×11, 32805 ×3, 79973 ×7) — the
1.5B-sum epoch-3 inverter had 1 of 26. Merge check: `in_proj_qkv` 3.4e-3 / 87.6 %, `q_proj` 3.2e-3 /
84.3 %. By domain, t̂ / t_true median: math 2,110 / 2,396 (n=151) · code 2,421 / 2,366 (28) · physics
852 / 1,046 (9) · biology 831 / 927 (5) · puzzle 491 / 459 (5) · chemistry 1,007 / 2,378 (2).

| row | t_true → t̂ tokens | reaches the given answer? |
|---|---|---|
| short, idx 15523 (car rental, y = 243 km) | 113 → 220 | **Yes**, in prose: 74.16 − 45 = 29.16, ÷ 0.12 = 243, then verified. With no summary to inherit, the plan-only shape of the surrogate's trace (and of the summary inverter's) does not appear. |
| median, idx 3282 (three lines; y = A) | 2,229 → 2,120 | **Yes.** Boxes A. |
| long, idx 57422 (unique n; y = 12) | 7,758 → 4,151 | **Yes.** Boxes 12. |

Consistency: match 98, mismatch 6, no box in t̂ 46 (18 cap-severed + 28 prose), no box in y 50 —
94.2 % graded (n=104). The six, read: **equivalent-form 1** — 25352 (boxes `d | (p − 1)` for y's
`p − 1`; R1 says "the period length divides p−1"); **genuine 5**, split by R1's answer: *inverter wrong*
3 — 28153 (proof that x²+y²=61³ has a solution: boxes a pair with 671² > 61³), 41089 (arc length 4√2/3
for 4/3, R1 4/3), 26596 (5√15/13 for √15/5, R1 √15/5); *inverter right, surrogate wrong* 2 — 59048
(labyrinths: "more bad" = R1, y says "more good"), 21551 (option A = R1, y says B). By hand 99/104;
inconsistent with the conditioned answer 5/104; wrong 3/104.

### 1.5B-nosum, epoch 2 — `bench/results/phase2/holdout-1.5b-nosum-ep2.jsonl`

t̂ 2,071 / 3,175 / 284 / 8,192; ratio 0.95 (per-row 1.03); cap-hit 29/200 (14.5 %), 4 loops; merge
check 3.3e-3 / 86.8 %, 3.1e-3 / 83.4 %. Consistency 88 / 11 / 51 (21 cap-severed + 30 prose) / 50 —
88.9 % graded (n=99). The eleven, read: **equivalent-form 1** — 32805 (`R(√5−1)/2` = `2R sin 18°`;
the last of two equivalent boxes); **genuine 10**: *inverter right, surrogate wrong* 5 — 60096 (13 =
R1, y √119), 21120 (diverges = R1), 59048, 21551, 63867 (8,682,544 = R1, y 8,682,572); *inverter
wrong* 5 — 71649 (14 for 16), 74895 (6 for 19), 16770 (`15+6n`: 189 = 15+6·29 = 5³+4³, so invalid),
59441 (√13 for √7), 30170 (boxes four candidates, the last wrong; y and R1 −6). By hand 89/99;
inconsistent 10/99; wrong 5/99. Short 15523 → 243 in prose ✓, median 3282 ✓, long 57422 ✓ (2,979
tokens).

### 5.4 Cap-hits across inversions — prompt-level or arm-level?

Capped idx sets, all 200-row inversions (`finish_reason == "length"`), with the loop test (a 40-char
chunk repeated ≥3× in the trace's last 4,000 chars):

| inversion | capped | of which loops |
|---|---:|---:|
| 7B-sum e3 | 10 | 2 |
| 7B-nosum e3 | 10 | 0 |
| 1.5B-sum e3 | 26 | 1 |
| 1.5B-nosum e3 | 21 | 4 |
| 7B-sum e2 | 6 | 1 |
| 1.5B-sum e2 | 34 | 3 |
| 1.5B-nosum e2 | 29 | 4 |
| 7B-nosum e2 | 14 | 0 |

Pairwise overlap of the epoch-3 capped sets: 7B-sum ∩ 7B-nosum **2** · 1.5B-sum ∩ 1.5B-nosum **11** ·
across arms 3–4 per pair. Union over the four: 44 idx; capped in one inversion only 25, in two 15, in
three 4 (idx 38653, 47459, 79973, 100774), in all four **0**. Arm unions: 7B 18, 1.5B 36, intersection
10. Same inverter, epoch 2 ∩ epoch 3: 7B-sum 1 (of 10 and 6), 7B-nosum 4 (of 10 and 14), 1.5B-sum 10
(of 26 and 34), 1.5B-nosum 9 (of 21 and 29). Measured: no prompt caps every inverter; the 1.5B arm's two inverters share half
their capped rows with each other and a quarter with the 7B arm's — the runaway is mostly arm-level
with a prompt-level core of ~4–10 rows, and on any single inverter it moves between draws (Phase 1's
flip-rate finding again). Loops are a 1.5B-nosum and epoch-2 feature (3–4 per inversion) and rare
elsewhere (0–2).

### 5.5 Epoch 2 vs epoch 3, side by side

`eval_loss` is deterministic; every generation column is one sampled draw at temperature 0.7 on the same
200 rows (Phase 1 measured a ~15 % cap-hit flip rate between two draws of one model), so those columns
are directions, not differences. Not averaged across arms.

| inverter | eval_loss e2 / e3 | t̂ median ratio e2 / e3 | cap-hit e2 / e3 | graded match e2 / e3 | inconsistent with y e2 / e3 | wrong e2 / e3 |
|---|---|---|---|---|---|---|
| 7B-sum | **0.3810** / 0.3847 | 1.03 / 0.99 | 6 / 10 | 95.8 % / 93.8 % | 2/118 / 3/112 | 1/118 / 3/112 |
| 7B-nosum | **0.3840** / 0.3876 | 0.96 / 0.91 | 14 / 10 | 94.4 % / 92.1 % | 2/108 / 4/113 | 1/108 / 3/113 |
| 1.5B-sum | **0.4233** / 0.4257 | 0.95 / 0.90 | 34 / 26 | 91.2 % / 94.2 % | 7/102 / 6/103 | 6/102 / 6/103 |
| 1.5B-nosum | **0.4248** / 0.4272 | 0.95 / 0.89 | 29 / 21 | 88.9 % / 94.2 % | 10/99 / 5/104 | 5/99 / 3/104 |

Across the four inverters epoch 2 has the lower eval loss every time; on generation the 7B arm's
epoch-2 adapters cap fewer rows (6 vs 10; 14 vs 10 is the one exception) and grade higher, the 1.5B
arm's cap more and grade lower. Which adapter Phase 4 serves is the user's decision.

---

## 6. Method notes that cost time

- **Pair before comparing, again.** The unpaired R1 comparison said the forged traces agree with R1
  more often than their surrogates' own answers (81–86 % vs 72–78 %). Paired on the same rows it
  *reversed* for the 7B arm (−3 to −4 points, rows only lost) and shrank to +1–2 for the 1.5B arm. Same
  shape as Phase 1's trace-length comparison that read "within 1.4 %" until it was paired: the
  denominators differed because t̂ boxes on fewer rows, and those rows were the easier ones.
- **A hub kernel that loads and runs forward can still be broken.** `kernels-community/flash-attn2`
  (build `torch-stable-abi210-cu130`, compiled against torch 2.14) passed the smoke forward and crashed
  in its backward at probe step 0 under torch 2.13. The probe is the check; the smoke test is not.
- **transformers 5.16 `save_pretrained` reverts its load-time key mapping.** A `Qwen3_5ForCausalLM`
  loaded text-only saves `model.language_model.*` names and `architectures: null`, which vLLM's
  text-only loader rejects. The merge writes the module-tree state dict itself.
- **A serving check that would pass on the bare base is not a check.** Given answer + summary, the
  base model already writes coherent traces ending in the right box; the merge assert (tensor before ≠
  after, saved == merged) is what proves an adapter was folded in.
- **The grader is a lower bound; read every mismatch.** Of 7 + 9 + 5 + 6 + 9 mismatches across the five
  inversions, 3 + 4 + 3 + 0 + 2 were equivalent forms the matcher rejects (a k-range in the prose, `c`
  vs `C`, an option letter for its value, the same set boxed in another order). Labels: equivalent-form
  / alternative-valid / genuine, with "inconsistent with the conditioned answer" = genuine +
  alternative-valid.
- **`y` on a surrogate holdout is the surrogate's answer, not the truth.** Grading `y` and t̂ against the
  dataset's own R1 solution is what separated "inverter wrong" from "inverter right, surrogate wrong"
  — 0 vs 4 of the genuine cases on the two arms.
- **Trainer log lines sit in a stdout buffer under `nohup`** until the process exits; the per-epoch
  `trainer_state.json` is the record. `PYTHONUNBUFFERED=1` in the driver from the second run on.
- **Checkpoints are 1.4 GB, not ~250 MB**: `save_strategy="epoch"` stores the optimizer state with
  each adapter. Twelve of them plus a transient 8.4 GB merge fit in the 59 GB the cache deletion freed.

---

## 7. Definition of done (`docs/13` §10), with the evidence

| item | status | where |
|---|---|---|
| training venv built from `pyproject.toml`; `trl env` recorded | done | `pyproject.toml` / `uv.lock`; §0 |
| `bench/phase2/prompts.py` (sha256-asserted), `bench/phase2/holdout.json` (200 idx, seed 20260828) committed | done | commit `0b6983b`; §1 |
| formatter + self-test committed; over-length drop count reported per file (expected 0) | done — 0 of 10,034 | `bench/phase2_format.py`, `bench/test_phase2_format.py`; §1 table |
| four adapters on disk, three per-epoch checkpoints each, `log_history.json` + peak VRAM per run | done | `bench/results/phase2/inverter-{7b,1.5b}-{sum,nosum}/{checkpoint-*,log_history.json,peak_vram.txt,run.json}` (gitignored); §4 |
| held-out eval loss per epoch, per inverter | done | §4 table and per-inverter blocks |
| held-out inversion for all four inverters: paired length table, cap-hit, empties, three read examples each | done, plus the epoch-2 adapters | §5 (tables), §5.x per inverter; raw text in `bench/results/phase2/holdout-*.jsonl` (gitignored), per-row grading in `holdout-*-consistency.json` (committed) |
| every long run preceded by a probe whose projection was reported before the run started | done, four probes | §3; `bench/logs/phase2-probe-*.log` |
| `docs/results/phase2.md` committed; `docs/09` rows added; `.gitignore` covers the adapters | done | this file; `docs/09` rows 3.2, 3.4, 4.7, 6.2, 6.4, 7.12, 7.13; `.gitignore` `bench/results/phase2/inverter-*/`, `merged-*/` |

Wall clock, whole phase: 2026-08-28 19:30 → 08-30 07:45 — 33.8 h of training, ~2.5 h of probes, merges,
inversions and the 7B-sum serving-path check, the rest reading and writing. Disk at the end: 30 GB
free (four adapters with checkpoints ≈ 20 GB; no merged weights left on disk).
