# RTX 4090 Feasibility Analysis — Reproducing *Trace Inversion* (arXiv 2603.07267v2)

**Target box (verified):** 1× RTX 4090, 24564 MiB (23.99 GiB) VRAM, Ada Lovelace (sm89), driver 595.84, CUDA 13.2, no NVLink. 30 GB system RAM (~26 GB free). 119 GB free disk. Torch not yet installed.

**Paper's hardware:** 8× A100 80GB (640 GB aggregate). We have 3.7% of that VRAM and one GPU. Everything below is about what survives that reduction.

---

## 1. Verdict table

| # | Workload | Feasible locally? | Recommended config | Est. time |
|---|---|---|---|---|
| A1 | **Surrogate traces** (weak surrogate, 5k prompts × ~6k tok) | **YES — easy** | Qwen3.5-2B bf16, vLLM, `--max-num-seqs 128` | **~1.0 h** |
| A2 | **Summary compression** (5k traces → ~600-tok recaps) | **YES — easy** | Qwen3.5-4B AWQ-int4, vLLM | **~1.0 h** |
| A3 | **Local open-weight victim** — Qwen3.8-27B, 5k × ~4k tok | **YES — but it is the bottleneck** | GGUF **IQ4_XS** (14.63 GiB) or **Q4_K_S** (15.01 GiB), llama-server, **q8_0 KV**, `-np 8..12` @ 12–16k ctx | **~15–16 h** (5k traces)<br>~30 h (10k) |
| A3′ | Same victim at **bf16 / FP8 / Q8_0 / official GPTQ-Int4** | **NO** | 50.9 / 28.8 / 27.1 / 28.2 GiB — all exceed 23.99 GiB | — |
| B | **Train inversion model** (input+answer+summary → trace) | **YES** | Qwen3.5-4B, TRL SFT + **LoRA r=32 bf16**, `max_length=8192`, grad-ckpt, packing | **~14 h** (2 epochs, 5k ex.) |
| C1 | **Invert victim outputs** (5k × ~5k tok) | **YES** | 4B + LoRA merged, vLLM bf16 | **~4 h** |
| C2 | **Student SFT** — one student, all 5 paper conditions | **YES** | Qwen3.5-2B, QLoRA/LoRA, `max_length=8192` | **~21 h** (5 runs) |
| C3 | **Second student** (Llama-3.1-8B or Qwen2.5-7B), 2 conditions | **YES — tight** | **QLoRA 4-bit**, `max_length=8192` (13.6 GiB) | **~24 h** (2 runs) |
| D | **Eval** MATH500 / JEEBench / LCB across ~12 variants | **YES** | vLLM bf16/int4 per checkpoint | **~8 h** |
| E | **DeepSeek-R1 (671B/685B) as victim or surrogate** | **NO — impossible at any quantization** | see §4 | — |
| F | **GPT-5.4-mini black-box victim** | **NO locally — API only** | OpenAI API (optional secondary track) | ~$87 (5k) / $173 (10k) / $433 (25k) |
| G | **Full fine-tune of anything ≥3B** | **NO** | see §2 — 3B full-FT already needs 24.7 GiB at 2k | — |

**Headline:** total ≈ **85 GPU-hours ≈ 3.5 days** of continuous single-GPU work for the recommended scaled-down matrix (§6), of which the local 27B victim generation is the single largest line item. **Sequence length, not parameter count, is what kills you** — and on the generation side, **KV-cache headroom, not weight size, is what caps throughput.**

---

## 2. VRAM budget math

### 2.1 Assumptions (stated explicitly)

| Assumption | Value | Note |
|---|---|---|
| Card usable for training | **23.1 GiB** | 23.99 GiB − 0.9 GiB CUDA context + cuBLAS/cuDNN workspaces + allocator fragmentation |
| Card usable for serving | **23.4 GiB** | 23.99 − ~0.5 GiB held by the desktop session (X/Wayland) |
| dtype | bf16 | Ada supports bf16 natively; no fp16 loss-scaling needed |
| Attention | FlashAttention-2 / FA-style | **no S² attention matrix is ever materialised** — without this every row below at ≥4k is fiction |
| Gradient checkpointing | **ON** (full, per decoder layer) | TRL `SFTConfig.gradient_checkpointing` defaults to `True` |
| Cross-entropy | **chunked** | TRL `loss_type="chunked_nll"` (default). See §2.4 — this is load-bearing |
| Batch size | 1 sequence (use `gradient_accumulation_steps` for effective batch) | |
| Full FT optimizer | AdamW, states in **bf16** (`adamw_torch_fused` on bf16 params) → 4 B/param | The *optimistic* case. The fp32-master-weight path costs 18 B/param — see §2.5 |
| LoRA | r=32, all linear (q,k,v,o,gate,up,down), adapters + grads + Adam in **fp32** | 12 B per adapter param |
| QLoRA | bitsandbytes NF4 + double quant = **4.127 bits/param ≈ 0.516 B/param**, embeddings & lm_head stay bf16 | bnb never quantizes `nn.Embedding` |

### 2.2 Formulas

```
Weights_full   = P × 2
Weights_lora   = P × 2                                   (base frozen, still bf16 resident)
Weights_qlora  = (P − P_embed) × 0.516  +  P_embed × 2    (P_embed = embed + lm_head, untied ⇒ 2·V·H)

Grads_full     = P × 2
Opt_full       = P × 4          (AdamW exp_avg + exp_avg_sq, bf16)
Grads_lora     = N_lora × 4  ;  Opt_lora = N_lora × 8
N_lora         = r × Σ_linear (fan_in + fan_out) × L

Act_ckpt       = B·S · [ 2·H·L  (layer-boundary hidden states)
                       + 2·(H + q_dim + 2·kv_dim + q_dim + H  +  H + 3·I + H) ]  (one layer recomputed)
Act_no_ckpt    = B·S · 2·(…same per-layer term…) × L

Logits_chunked ≈ 0.35 GiB (bounded by chunk size, independent of S)
Logits_naive   = B·S·V · (2 bf16 + 4 fp32 upcast + 2 grad) = B·S·V·8
```

`P` values are the authoritative safetensors totals pulled from the HF API, not estimates.

### 2.3 The tables

Legend: **FITS** ≤ 19.6 GiB · **FITS TIGHT** ≤ 23.1 GiB · **DOES NOT FIT** > 23.1 GiB. All values GiB.

#### seq_len = 2048

| Model | Params | mode | W | G | Opt | Act | Logit | **Total** | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.494B | full | 0.9 | 0.9 | 1.8 | 0.16 | 0.35 | **5.1** | FITS |
| | | lora | 0.9 | 0.1 | 0.1 | 0.16 | 0.35 | **2.5** | FITS |
| | | qlora | 0.4 | 0.1 | 0.1 | 0.16 | 0.35 | **2.0** | FITS |
| Qwen2.5-1.5B | 1.544B | full | 2.9 | 2.9 | 5.8 | 0.30 | 0.35 | **13.1** | FITS |
| | | lora | 2.9 | 0.1 | 0.3 | 0.30 | 0.35 | **4.8** | FITS |
| | | qlora | 1.1 | 0.1 | 0.3 | 0.30 | 0.35 | **3.0** | FITS |
| Qwen2.5-3B | 3.086B | full | 5.7 | 5.7 | 11.5 | 0.46 | 0.35 | **24.7** | **DOES NOT FIT** |
| | | lora | 5.7 | 0.2 | 0.4 | 0.46 | 0.35 | **8.1** | FITS |
| | | qlora | 1.9 | 0.2 | 0.4 | 0.46 | 0.35 | **4.3** | FITS |
| Qwen2.5-7B | 7.616B | full | 14.2 | 14.2 | 28.4 | 0.69 | 0.35 | **58.7** | **DOES NOT FIT** |
| | | lora | 14.2 | 0.3 | 0.6 | 0.69 | 0.35 | **17.0** | FITS |
| | | qlora | 5.2 | 0.3 | 0.6 | 0.69 | 0.35 | **8.0** | FITS |
| Llama-3.1-8B | 8.030B | full | 15.0 | 15.0 | 29.9 | 0.77 | 0.35 | **61.8** | **DOES NOT FIT** |
| | | lora | 15.0 | 0.3 | 0.6 | 0.77 | 0.35 | **17.9** | FITS |
| | | qlora | 5.3 | 0.3 | 0.6 | 0.77 | 0.35 | **8.3** | FITS |
| R1-Distill-Qwen-1.5B | 1.777B | full | 3.3 | 3.3 | 6.6 | 0.30 | 0.35 | **14.8** | FITS |
| | | lora | 3.3 | 0.1 | 0.3 | 0.30 | 0.35 | **5.3** | FITS |
| | | qlora | 1.5 | 0.1 | 0.3 | 0.30 | 0.35 | **3.5** | FITS |
| R1-Distill-Qwen-7B | 7.616B | full | 14.2 | 14.2 | 28.4 | 0.69 | 0.35 | **58.7** | **DOES NOT FIT** |
| | | lora | 14.2 | 0.3 | 0.6 | 0.69 | 0.35 | **17.0** | FITS |
| | | qlora | 5.2 | 0.3 | 0.6 | 0.69 | 0.35 | **8.0** | FITS |
| R1-Distill-Qwen-14B | 14.770B | full | 27.5 | 27.5 | 55.0 | 1.22 | 0.35 | **112.5** | **DOES NOT FIT** |
| | | lora | 27.5 | 0.5 | 1.0 | 1.22 | 0.35 | **31.5** | **DOES NOT FIT** |
| | | qlora | 9.3 | 0.5 | 1.0 | 1.22 | 0.35 | **13.3** | FITS |
| **Qwen3.5-0.8B** | 0.873B | full | 1.6 | 1.6 | 3.3 | 0.17 | 0.35 | **7.9** | FITS |
| | | lora | 1.6 | 0.1 | 0.1 | 0.17 | 0.35 | **3.2** | FITS |
| | | qlora | 0.8 | 0.1 | 0.1 | 0.17 | 0.35 | **2.4** | FITS |
| **Qwen3.5-2B** | 2.274B | full | 4.2 | 4.2 | 8.5 | 0.31 | 0.35 | **18.5** | FITS |
| | | lora | 4.2 | 0.1 | 0.2 | 0.31 | 0.35 | **6.1** | FITS |
| | | qlora | 1.8 | 0.1 | 0.2 | 0.31 | 0.35 | **3.7** | FITS |
| **Qwen3.5-4B** | 4.660B | full | 8.7 | 8.7 | 17.4 | 0.50 | 0.35 | **36.5** | **DOES NOT FIT** |
| | | lora | 8.7 | 0.2 | 0.4 | 0.50 | 0.35 | **11.1** | FITS |
| | | qlora | 3.1 | 0.2 | 0.4 | 0.50 | 0.35 | **5.5** | FITS |
| **Qwen3.5-9B** | 9.653B | full | 18.0 | 18.0 | 36.0 | 0.74 | 0.35 | **73.9** | **DOES NOT FIT** |
| | | lora | 18.0 | 0.3 | 0.6 | 0.74 | 0.35 | **20.8** | FITS TIGHT |
| | | qlora | 7.5 | 0.3 | 0.6 | 0.74 | 0.35 | **10.3** | FITS |
| **Qwen3.5-27B** | 27.781B | qlora | 16.9 | 0.8 | 1.6 | 1.58 | 0.35 | **22.0** | FITS TIGHT |
| **Qwen3.5-35B-A3B** (MoE) | 35.952B | qlora | 18.7 | 0.1 | 0.2 | 0.43 | 0.35 | **20.6** | FITS TIGHT |

#### seq_len = 4096

| Model | full | lora | qlora |
|---|---|---|---|
| Qwen2.5-0.5B | 5.2 FITS | 2.7 FITS | 2.2 FITS |
| Qwen2.5-1.5B | 13.4 FITS | 5.1 FITS | 3.3 FITS |
| Qwen2.5-3B | 25.2 ✗ | 8.6 FITS | 4.7 FITS |
| Qwen2.5-7B | 59.4 ✗ | 17.7 FITS | 8.7 FITS |
| Llama-3.1-8B | 62.6 ✗ | 18.7 FITS | 9.0 FITS |
| R1-Distill-1.5B | 15.1 FITS | 5.6 FITS | 3.8 FITS |
| R1-Distill-7B | 59.4 ✗ | 17.7 FITS | 8.7 FITS |
| R1-Distill-14B | 113.7 ✗ | 32.7 ✗ | 14.5 FITS |
| **Qwen3.5-0.8B** | 8.1 FITS | 3.4 FITS | 2.6 FITS |
| **Qwen3.5-2B** | 18.8 FITS | 6.4 FITS | 4.0 FITS |
| **Qwen3.5-4B** | 37.0 ✗ | 11.6 FITS | 6.0 FITS |
| **Qwen3.5-9B** | 74.7 ✗ | 21.6 TIGHT | 11.1 FITS |
| **Qwen3.5-27B** | 211.4 ✗ | 58.5 ✗ | **23.6 ✗** |
| **Qwen3.5-35B-A3B** | 270.0 ✗ | 69.3 ✗ | 21.0 TIGHT |

#### seq_len = 8192 ← *the realistic operating point for this paper*

| Model | full | lora | qlora |
|---|---|---|---|
| Qwen2.5-0.5B | 5.6 FITS | 3.0 FITS | 2.5 FITS |
| Qwen2.5-1.5B | 14.0 FITS | 5.8 FITS | 3.9 FITS |
| Qwen2.5-3B | 26.1 ✗ | 9.5 FITS | 5.7 FITS |
| Qwen2.5-7B | 60.7 ✗ | **19.1 FITS** | 10.1 FITS |
| Llama-3.1-8B | 64.1 ✗ | **20.2 TIGHT** | 10.6 FITS |
| R1-Distill-1.5B | 15.7 FITS | 6.2 FITS | 4.4 FITS |
| R1-Distill-7B | 60.7 ✗ | 19.1 FITS | 10.1 FITS |
| R1-Distill-14B | 116.2 ✗ | 35.2 ✗ | **16.9 FITS** |
| **Qwen3.5-0.8B** | 8.4 FITS | 3.8 FITS | 2.9 FITS |
| **Qwen3.5-2B** | **19.4 FITS** | 7.0 FITS | 4.6 FITS |
| **Qwen3.5-4B** | 38.0 ✗ | **12.6 FITS** | 7.0 FITS |
| **Qwen3.5-9B** | 76.1 ✗ | 23.1 TIGHT | 12.5 FITS |
| **Qwen3.5-27B** | 214.6 ✗ | 61.7 ✗ | 26.8 ✗ |
| **Qwen3.5-35B-A3B** | 270.8 ✗ | 70.2 ✗ | 21.9 TIGHT |

#### seq_len = 16384 (the paper's `cutoff_len`)

| Model | full | lora | qlora |
|---|---|---|---|
| Qwen2.5-0.5B | 6.2 FITS | 3.6 FITS | 3.1 FITS |
| Qwen2.5-1.5B | 15.2 FITS | 7.0 FITS | 5.2 FITS |
| Qwen2.5-3B | 27.9 ✗ | 11.3 FITS | 7.5 FITS |
| Qwen2.5-7B | 63.5 ✗ | **21.8 TIGHT** | 12.8 FITS |
| Llama-3.1-8B | 67.2 ✗ | **23.3 ✗** | 13.6 FITS |
| R1-Distill-1.5B | 16.9 FITS | 7.4 FITS | 5.6 FITS |
| R1-Distill-7B | 63.5 ✗ | 21.8 TIGHT | 12.8 FITS |
| R1-Distill-14B | 121.1 ✗ | 40.1 ✗ | **21.8 TIGHT** |
| **Qwen3.5-0.8B** | 9.1 FITS | 4.4 FITS | 3.6 FITS |
| **Qwen3.5-2B** | **20.7 TIGHT** | 8.3 FITS | 5.8 FITS |
| **Qwen3.5-4B** | 39.9 ✗ | **14.5 FITS** | 9.0 FITS |
| **Qwen3.5-9B** | 79.1 ✗ | **26.0 ✗** | 15.5 FITS |
| **Qwen3.5-27B** | 220.9 ✗ | 68.0 ✗ | 33.1 ✗ |
| **Qwen3.5-35B-A3B** | 272.5 ✗ | 71.9 ✗ | 23.6 ✗ |

### 2.4 SEQUENCE LENGTH IS THE DOMINANT RISK — three separate mechanisms

Long CoT is not incidental here. The paper reports R1 ground-truth traces averaging **6,130.6 tokens** and inverted traces at **4,972–6,021 tokens**. Add the prompt (input + answer + ~592-token summary) and a typical inversion training example is **6–8k tokens**; the paper caps at **16,384**. Three things blow up:

**(a) The logits tensor — the single biggest trap.** With naive cross-entropy the lm_head output is materialised at full `S × V`, upcast to fp32, and a gradient is kept:

```
Logits_naive = S × V × (2 + 4 + 2) bytes
```

| Model | vocab | @2k | @4k | @8k | @16k |
|---|---|---|---|---|---|
| Qwen2.5-* / R1-Distill-* | 151,936–152,064 | 2.3 | 4.6 | 9.3 | **18.6 GiB** |
| Llama-3.1-8B | 128,256 | 2.0 | 3.9 | 7.8 | **15.7 GiB** |
| **Qwen3.5 / Qwen3.8 (all sizes)** | **248,320** | 3.8 | 7.6 | **15.2** | **30.3 GiB** |

**Qwen3.5's 248k vocabulary means a single 16k-token example produces 30.3 GiB of logits traffic — more than the whole card — before you count a single weight.** This is *entirely* independent of model size: the 0.8B and the 27B have the same problem.

Mitigations, in order of preference:
1. **TRL `loss_type="chunked_nll"` — this is already the default in TRL ≥1.x.** It drops `labels == -100` positions before the lm_head matmul and chunks the CE. Peak becomes ~0.35 GiB and stops scaling with `S × V`. Do not override it.
2. Liger fused linear-CE (`use_liger_kernel=True`) — equivalent effect, but TRL then forces `loss_type="nll"`; the two are mutually exclusive.
3. Masking the prompt (`completion_only_loss` / `assistant_only_loss`) compounds with (1): fewer non-ignored tokens ⇒ smaller matmul.

If you ever see an OOM whose size scales with vocabulary, you have lost chunked CE.

**(b) Activations.** Per-token-per-layer stored activation bytes and the resulting totals:

| Model | B/tok/layer | GC @2k | GC @8k | GC @16k | **no-GC @2k** | **no-GC @8k** |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 73,216 | 0.30 | 1.21 | 2.43 | 3.9 | **15.6** |
| Qwen2.5-7B | 158,720 | 0.69 | 2.74 | 5.48 | 8.5 | **33.9** |
| Llama-3.1-8B | 139,264 | 0.77 | 3.06 | 6.12 | 8.5 | **34.0** |
| R1-Distill-14B | 148,480 | 1.22 | 4.88 | 9.77 | 13.6 | **54.4** |
| Qwen3.5-2B | 63,488 | 0.31 | 1.23 | 2.47 | 2.9 | **11.6** |
| Qwen3.5-4B | 96,256 | 0.50 | 1.98 | 3.97 | 5.9 | **23.5** |
| Qwen3.5-9B | 126,976 | 0.74 | 2.97 | 5.94 | 7.8 | **31.0** |
| Qwen3.5-27B | 174,080 | 1.58 | 6.33 | 12.66 | 21.2 | **85.0** |

**Gradient checkpointing is not optional at any size above 1.5B.** Without it a 7B at 8k needs 33.9 GiB of activations alone. With it, the same case is 2.74 GiB. Cost: ~33% more compute (one extra forward per layer), already priced into §5.

**(c) Attention itself.** Without FlashAttention the S² score matrix is `B × heads × S² × 2 B`. For Qwen2.5-7B (28 heads) at 8k that is 28 × 8192² × 2 = **3.5 GiB per layer**. Every row in §2.3 assumes this never happens. FA2/FA3/SDPA-flash is mandatory, not an optimisation.

**Practical consequence:** cap `max_length` at **8192**, not the paper's 16384. Going 8k → 16k roughly doubles activations and, more importantly, doubles wall-clock for a marginal fidelity gain. Truncating the tail of a 6k-token trace costs almost nothing since P95 trace length is well under 16k.

### 2.5 Optimizer-state accounting — be explicit about which regime you're in

The "full" columns above use the **optimistic** bf16 regime (params bf16, grads bf16, AdamW states bf16) = **8 B/param**. Three other regimes exist:

| Regime | Bytes/param | 7B cost | When you get it |
|---|---|---|---|
| bf16 params + bf16 Adam states | 2+2+4 = **8** | 56.7 GiB | Model loaded bf16, `optim="adamw_torch_fused"` |
| bf16 + **8-bit Adam** (`adamw_bnb_8bit`) | 2+2+2 = **6** | 42.6 GiB | bitsandbytes 8-bit optimizer |
| fp32 master weights (accelerate mixed-precision) | 4+2+4+8 = **18** | 127.6 GiB | The classic "16 bytes/param" figure, +2 for the autocast bf16 copy |
| **fp32 everything** | 4+4+8 = **16** | 113.5 GiB | **What you get by accident from TRL — see §7.2** |

Even the most optimistic regime puts full fine-tuning of a 3B model at 24.7 GiB @ 2k — over the line. **Full fine-tuning is off the table for everything ≥3B, and 8-bit Adam does not save it** (Qwen2.5-3B with 8-bit Adam: 5.7+5.7+5.7+0.5+0.35+0.9 = 18.9 GiB @2k, which *does* squeak in — but it's 26.1 GiB by 8k, which is where you actually need to be). LoRA/QLoRA is the only viable path.

### 2.6 Qwen3.5 / Qwen3.8 architecture notes (these are **not** Qwen2.5 rescaled)

Pulled from the real `config.json` of each repo:

| Model | P (safetensors) | H | I | L | heads | kv | head_dim | vocab | tied | full-attn layers |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 873,438,784 | 1024 | 3584 | 24 | 8 | 2 | 256 | 248,320 | yes | 6 of 24 |
| Qwen3.5-2B | 2,274,069,824 | 2048 | 6144 | 24 | 8 | 2 | 256 | 248,320 | yes | 6 of 24 |
| Qwen3.5-4B | 4,659,865,088 | 2560 | 9216 | 32 | 16 | 4 | 256 | 248,320 | yes | 8 of 32 |
| Qwen3.5-9B | 9,653,104,368 | 4096 | 12288 | 32 | 16 | 4 | 256 | 248,320 | no | 8 of 32 |
| Qwen3.5-27B | 27,781,427,952 | 5120 | 17408 | 64 | 24 | 4 | 256 | 248,320 | no | 16 of 64 |
| **Qwen3.8-27B** | 27,781,427,952 | 5120 | 17408 | 64 | 24 | 4 | 256 | 248,320 | no | 16 of 64 |
| Qwen3.5-35B-A3B (MoE) | 35,951,822,704 | 2048 | — | 40 | 16 | 2 | 256 | 248,320 | no | 10 of 40 |

Five things that change the math versus Qwen2.5:

1. **Hybrid Gated-DeltaNet attention, 3:1 ratio.** `layer_types` alternates three `linear_attention` layers per one `full_attention` layer (`full_attention_interval: 4`). Only the full-attention layers hold a KV cache. Qwen3.5-4B stores KV for **8 of 32** layers.
2. **KV cache is dramatically smaller — but a large constant-per-sequence state replaces part of it.**

   | Model | KV B/token | full-attn L | **GDN state per sequence** | KV @8k |
   |---|---|---|---|---|
   | Qwen2.5-7B | 57,344 | 28 | 0 | 0.44 GiB |
   | Llama-3.1-8B | 131,072 | 32 | 0 | 1.00 GiB |
   | R1-Distill-14B | 196,608 | 48 | 0 | 1.50 GiB |
   | Qwen3.5-2B | 12,288 | 6 | 18.9 MB | 0.09 GiB |
   | Qwen3.5-4B / 9B | 32,768 | 8 | 50.3 MB | 0.25 GiB |
   | **Qwen3.5/3.8-27B** | **65,536** | 16 | **151 MB** | 0.50 GiB |

   The GDN recurrent state is `L_linear × num_v_heads × v_head_dim × k_head_dim × 4 B` (fp32, per `mamba_ssm_dtype`). For the 27B: `48 × 48 × 128 × 128 × 4 = 151 MB per sequence`, **constant in sequence length and not quantizable**. Below ~2.3k tokens of context it costs *more* than the KV cache. On a 24 GB card this is what caps concurrency, not length. See §3.3.
3. **248,320-token vocabulary.** Roughly 1.63× Qwen2.5's. It inflates the logits trap (§2.4a) by the same factor and it makes QLoRA much less effective on the small models: Qwen3.5-2B has 508M embedding params (tied) = 1.0 GiB that bitsandbytes will never quantize, so QLoRA only takes it 4.2 → 1.8 GiB rather than 4.2 → 0.6.
4. **`head_dim = 256`** across the whole family (vs 128 for Qwen2.5/Llama). Combined with `attn_output_gate: true`, the attention projections are unusually large relative to `hidden_size`. This is why the official 27B GPTQ-Int4 build is so big (§3.1) — it skips attention.
5. **Every Qwen3.5/3.8 model is natively multimodal.** Architecture is `Qwen3_5ForConditionalGeneration` with a `vision_config` (a 12–27-block ViT). For text-only SFT this is dead weight you must freeze; on the llama.cpp path it costs literally nothing (§3.1). There is also an MTP head (`mtp_num_hidden_layers: 1`) shipped in the checkpoint. **Verify both are excluded from your LoRA target modules and from `requires_grad` before your first long run.**

### 2.7 MoE (Qwen3.5-35B-A3B) — different arithmetic

`num_experts: 256`, `num_experts_per_tok: 8`, `moe_intermediate_size: 512`, plus a 512-wide shared expert. Two rules:

- **Memory is set by total parameters (35.95B), not active (~3.3B). All 256 experts must be resident.** bf16 weights = 67.0 GiB. Even QLoRA lands at 18.7 GiB of weights, leaving ~4 GiB — it "fits" only at ≤2k sequence length, which is useless for this paper.
- **Compute is set by active parameters (~3.3B)**, so if it ever fit, it would be fast. It doesn't.
- Do not LoRA-target the experts (256 experts × 40 layers = 10,240 tensors). Target attention projections only; the numbers above assume that.

**Verdict: Qwen3.5-35B-A3B is not usable for training on this box at any useful sequence length.** Same for the 122B-A10B and 397B-A17B.

---

## 3. Generation throughput

### 3.1 Local open-weight victim: Qwen/Qwen3.8-27B — quantization reality check

The user proposed 8-bit. **8-bit does not fit, and neither does the official 4-bit safetensors build.** These are measured HF repo byte counts, not estimates:

| Build | Size (GiB) | Runtime | Cache left after 23.4 − size − 0.8 GiB compute buffer | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` bf16 safetensors | 51.75 | vLLM | −29.15 | **DOES NOT FIT** |
| BF16 GGUF | 50.90 | llama.cpp | −28.30 | **DOES NOT FIT** |
| `Qwen/Qwen3.8-27B-FP8` official | **28.77** | vLLM | −6.17 | **DOES NOT FIT** |
| GPTQ-Int4 official (`Qwen3.5-27B-GPTQ-Int4` sibling) | **28.18** | vLLM | −5.58 | **DOES NOT FIT** |
| Q8_0 GGUF | 27.05 | llama.cpp | −4.45 | **DOES NOT FIT** |
| UD-Q6_K_XL GGUF | 24.14 | llama.cpp | −1.54 | **DOES NOT FIT** |
| NVFP4 (unsloth) | 21.81 | vLLM | +0.79 | **N/A — NVFP4 needs Blackwell sm100+; 4090 is sm89** |
| Q6_K GGUF | 21.31 | llama.cpp | +1.29 | FITS TIGHT (2 slots — useless) |
| AWQ-INT4 community (`cyankiwi`) | 19.60 | vLLM | +3.00 | FITS (4 slots; 3 downloads, unvetted) |
| UD-Q5_K_XL GGUF | 18.83 | llama.cpp | +3.77 | FITS (5 slots) |
| **Q5_K_M GGUF** | **18.47** | llama.cpp | **+4.13** | **FITS (6 slots)** |
| Q5_K_S GGUF | 17.95 | llama.cpp | +4.65 | FITS (7 slots) |
| UD-Q4_K_XL GGUF | 16.69 | llama.cpp | +5.91 | FITS (9 slots) |
| Q4_1 GGUF | 16.34 | llama.cpp | +6.26 | FITS (9 slots) |
| Q4_K_M GGUF | 15.93 | llama.cpp | +6.67 | FITS (10 slots) |
| IQ4_NL GGUF | 15.22 | llama.cpp | +7.38 | FITS (11 slots) |
| **Q4_K_S GGUF** | **15.01** | llama.cpp | **+7.59** | **FITS (11 slots)** |
| Q4_0 GGUF | 14.95 | llama.cpp | +7.65 | FITS (11 slots) |
| **IQ4_XS GGUF** | **14.63** | llama.cpp | **+7.97** | **FITS (12 slots)** |
| Q3_K_M GGUF | 12.87 | llama.cpp | +9.73 | FITS (15 slots) — quality cost |

Three findings worth stating plainly:

- **The user's 8-bit instinct is wrong, and my colleague's "4-bit ≈ 13.9 GiB, ~8 GiB left" estimate was right in spirit but only for GGUF.** A uniform-4-bit-over-27.78B calculation gives 13.9 GiB, but **no shipping safetensors quant is uniform.** Qwen's own GPTQ-Int4 excludes attention, embeddings, lm_head, MTP and the vision tower (`'dynamic': {'-:.*attn.*', '-:.*visual.*', '-:.*mtp.*', ...}`), leaving 10.67B params in bf16 — so it lands at 28.18 GiB, *larger than the card*. The FP8 build is worse still (28.77 GiB) because the untied 248,320×5120 embedding + lm_head alone are 2.54B params = 5.1 GiB of unquantizable bf16.
- **GGUF quantizes everything including embeddings**, which is exactly why it is the only family that fits.
- **The vision encoder is free on the GGUF path.** `unsloth/Qwen3.8-27B-GGUF` ships the vision projector as a *separate* `mmproj-F16.gguf` (0.86 GiB) / `mmproj-BF16.gguf` (0.87 GiB). The main GGUF is text-only. **Just don't pass `--mmproj` and the vision tower costs zero VRAM and zero disk.** This settles the multimodal-overhead question decisively in GGUF's favour for this workload.

### 3.2 KV-cache arithmetic for Qwen3.8-27B at 16k

```
KV per token   = 2 (K,V) × 4 kv_heads × 256 head_dim × 2 B × 16 full-attn layers
               = 65,536 B = 64 KiB/token
KV @ 16,384    = 16384 × 65,536          = 1.000 GiB/sequence   (f16)
               = 0.500 GiB/sequence                              (q8_0 KV)
GDN state      = 48 linear layers × 48 v_heads × 128 × 128 × 4 B = 151 MB = 0.144 GiB/sequence
               (fp32; NOT affected by --cache-type-k/v)

per slot @16k, f16 KV  = 1.000 + 0.144 = 1.144 GiB
per slot @16k, q8_0 KV = 0.500 + 0.144 = 0.644 GiB
per slot @ 8k, q8_0 KV = 0.250 + 0.144 = 0.394 GiB
```

Slots available per build (`llama.cpp -np` / vLLM `--max-num-seqs`):

| Build | cache GiB | f16 @8k | f16 @16k | **q8_0 @8k** | **q8_0 @16k** |
|---|---|---|---|---|---|
| Q6_K (21.31) | 1.29 | 2 | 1 | 3 | 2 |
| AWQ-INT4 (19.60) | 3.00 | 4 | 2 | 7 | 4 |
| Q5_K_M (18.47) | 4.13 | 6 | 3 | **10** | **6** |
| Q4_K_M (15.93) | 6.67 | 10 | 5 | 17 | 10 |
| Q4_K_S (15.01) | 7.59 | 11 | 6 | **19** | **11** |
| IQ4_XS (14.63) | 7.97 | 12 | 6 | **20** | **12** |
| Q3_K_M (12.87) | 9.73 | 15 | 8 | 24 | 15 |

**`--cache-type-k q8_0 --cache-type-v q8_0` roughly doubles your slot count and is the single highest-leverage flag on this box.** Note it does *not* halve the GDN state, so the doubling degrades at short contexts: at 8k the state is 37% of the per-slot cost.

*Reconciliation note:* the slot counts above subtract a **0.8 GiB compute/graph buffer** from the 23.4 GiB usable figure before dividing. Omitting that buffer gives ~1 extra slot per tier (e.g. Q5_K_M → 7 rather than 6 at q8_0/16k). The buffer is real — llama.cpp allocates a CUDA compute buffer sized by the largest batched forward — so the conservative counts are the ones to plan with, and treat the extra slot as headroom you may discover you have.

### 3.2b The hybrid architecture is a genuine and large win for *this* paper

The whole point of Trace Inversion is generating and training on **5,000–6,100-token reasoning traces**. A conventional 27B dense transformer with 64 full-attention layers and GQA-4 at head_dim 128 would cost `2×4×128×2×64 = 131,072 B/token` = 128 KiB/token — **exactly 2× this model** — and a hypothetical head_dim-256 version with all 64 layers full-attention would cost 256 KiB/token, **4× this model**. Because only 16 of 64 layers carry a growing cache:

| Context | Qwen3.8-27B (hybrid, 16/64 full-attn) | Hypothetical all-full-attn 27B, hd=256 | Saving |
|---|---|---|---|
| 4k | 0.25 GiB + 0.14 state = **0.39** | 1.00 GiB | 2.6× |
| 8k | 0.50 + 0.14 = **0.64** | 2.00 GiB | 3.1× |
| 16k | 1.00 + 0.14 = **1.14** | 4.00 GiB | 3.5× |
| 32k | 2.00 + 0.14 = **2.14** | 8.00 GiB | 3.7× |

**Long-context generation is unusually affordable on this model, and the advantage grows with length** — the fixed 0.15 GiB GDN state amortises away. At 16k we get 12 concurrent slots on IQ4_XS where a conventional equivalent would get 3. That is a ~4× throughput difference on precisely the workload this paper needs. The flip side, worth remembering: below ~2.3k tokens of context the constant GDN state costs *more* than the KV cache, so this architecture is comparatively poor at high-concurrency short-prompt serving. For long-CoT batch generation it is close to ideal.

### 3.2c Runtime support for `qwen3_5` — both exist; llama.cpp wins on VRAM, not on support

**llama.cpp: supported.** Qwen3.5 got day-one support and `unsloth/Qwen3.8-27B-GGUF` has ~868k downloads, so the path is well-travelled. **Build from master, not from a distro package** — the hybrid Gated-DeltaNet operators are recent and the tree is still actively fixing them. Known live issues to be aware of:

| Ref | Status | Impact |
|---|---|---|
| PR #25024 *models: fix Qwen3.5 dense/MoE load when MTP block is absent (trunk-only GGUF)* | merged 2026-06-26 | **You need a build after this.** Older builds fail on trunk-only GGUFs. |
| Issue #24737 *Qwen3.5-4B: GGUF conversion/load expects 33 blocks, model only has 32* | open | The MTP head counts as a block. **Match your GGUF variant to your build** — unsloth ships both plain and `-MTP-` variants; mixing them is the most likely first-run failure. |
| Issue #26916 *qwen3_5 hybrid fails to load — `tensor 'blk.32.attn_no…'`* | open (2026-08-11) | Same family of problem. Smoke-test loading before scheduling a 15-hour run. |
| Issue #27019 *convert_hf_to_gguf: qwen3_5 hybrid tensors fail — ssm_conv1d kernel* | open (2026-08-13) | Only affects **converting yourself**. Use unsloth's prebuilt imatrix GGUFs and this never touches you. |

**vLLM: also supported**, and actively developed — merged in the last week alone: per-layer sliding-window support for Qwen3.5 hybrid attention (#52004), TriangleMix attention acceleration (#51991/2/3), an MTP draft-head fix (#52013). One open bug worth knowing: #52319, *silent generation stall — throughput drops to 0.0 with no error*. Qwen ships an official vLLM recipe for Qwen3.5-9B listing RTX 4090 as a target, requiring vLLM ≥0.17.

**So the llama.cpp choice for the 27B is not forced by missing support — it is forced by weight size.** vLLM can run `qwen3_5` fine; it just cannot run a *small enough* 27B quant, because no fitting safetensors 4-bit build exists except an unvetted 19.6 GiB AWQ that starves the cache (§3.1, §3.3b). **Use vLLM for everything ≤9B where GGUF isn't needed, and llama.cpp for the 27B victim.**

### 3.3 Throughput: roofline model

Decode on a 4090 is memory-bandwidth-bound, not compute-bound, at every batch size the KV cache permits.

```
step_time  = (W_bytes + B × S_avg × KV_per_token) / (BW_peak × efficiency)
tok/s      = B / step_time
BW_peak    = 1008 GB/s (RTX 4090 GDDR6X)
efficiency = 0.55–0.60 (llama.cpp CUDA, K-quants) | 0.62 (vLLM AWQ) | 0.75 (vLLM bf16)
S_avg      = 3000 (prompt ~500 + half of a 4k-token generation)
compute floor = 2 × P_active × B / (165 TFLOPS × 0.30 MFU)
```

Sanity check: single-stream Q4_K_S 27B → 16.12 GB / (1008 × 0.58) = 27.6 ms/token = **36 tok/s**, which matches published single-stream 27B-Q4-on-4090 figures. The model is calibrated.

**(a) llama.cpp / llama-server, continuous batching, q8_0 KV, 16k ctx, ~4k output tokens/trace:**

| Build | slots | **tok/s** | 2k traces (8M tok) | 5k traces (20M) | 10k traces (40M) |
|---|---|---|---|---|---|
| Q6_K | 2 | 52 | 42.8 h | 4.5 d | 8.9 d |
| UD-Q5_K_XL | 5 | 140 | 15.8 h | 39.6 h | 3.3 d |
| **Q5_K_M** | **6** | **170** | **13.1 h** | **32.7 h** | **2.7 d** |
| Q5_K_S | 7 | 202 | 11.0 h | 27.6 h | 2.3 d |
| UD-Q4_K_XL | 9 | 267 | 8.3 h | 20.8 h | 41.6 h |
| Q4_K_M | 10 | 307 | 7.2 h | 18.1 h | 36.2 h |
| IQ4_NL | 11 | 348 | 6.4 h | 16.0 h | 32.0 h |
| **Q4_K_S** | **11** | **352** | **6.3 h** | **15.8 h** | **31.6 h** |
| **IQ4_XS** | **12** | **368** | **6.0 h** | **15.1 h** | **30.2 h** |
| Q3_K_M | 15 | 496 | 4.5 h | 11.2 h | 22.4 h |

**(b) vLLM with a 4-bit safetensors quant:**

- `qwen3_5` **is** supported — vLLM integrates Flash-Linear-Attention Triton kernels for Gated DeltaNet plus a hybrid KV-cache manager, and Qwen ships an official vLLM recipe for Qwen3.5-9B that explicitly lists RTX 4090 as a target. vLLM ≥0.17 is required; current is 0.27.1.
- But the only fitting 4-bit safetensors build is a community AWQ-INT4 at 19.60 GiB with **3 lifetime downloads**. That leaves 3.0 GiB of cache = **4 slots at 16k** ⇒ **~115 tok/s**, i.e. **3.2× slower than llama.cpp IQ4_XS**, on an unvetted quant.
- **Conclusion: vLLM loses this one purely on weight size.** The bigger quant starves the KV cache, and on a 24 GB card the KV cache *is* the throughput. Use llama.cpp for the 27B victim. Keep vLLM for everything ≤9B, where it is clearly better.

**(c) What the batch cap actually costs you** (40M tokens = 10k traces × 4k):

| Build | batch | tok/s | speedup vs B=1 | 40M tokens | note |
|---|---|---|---|---|---|
| Q5_K_M | 1 | 30 | 1.00× | 374.2 h | |
| | 2 | 59 | 1.98× | 188.9 h | |
| | 4 | 115 | 3.89× | 96.3 h | |
| | 8 | 222 | 7.49× | 50.0 h | needs ≤8k ctx |
| | 16 | 414 | 13.95× | 26.8 h | **over cap even @8k** |
| Q4_K_S | 1 | 36 | 1.00× | 310.0 h | |
| | 2 | 71 | 1.98× | 156.9 h | |
| | 4 | 138 | 3.86× | 80.3 h | |
| | **8** | **264** | **7.38×** | **42.0 h** | fits @16k? no — @8k yes |
| | 16 | 486 | 13.55× | 22.9 h | needs ≤8k ctx |
| | 32 | 835 | 23.30× | 13.3 h | **over cap even @8k** |
| IQ4_XS | 8 | 257 | 7.36× | 43.3 h | |
| | 16 | 470 | 13.50× | 23.6 h | needs ≤8k ctx |

Batching scales **near-linearly** (7.4× at batch 8) because decode is bandwidth-bound on weights and the KV traffic is small by comparison. Every slot you cannot afford is throughput you simply do not get.

**(d) The quant choice is a THROUGHPUT decision, not a quality decision.**

On a GPU with spare VRAM, picking a quant is a straight quality question and you take the biggest one that fits. **On a 24 GB card it is not**, because weights and KV cache come out of the same pool: every extra GiB of weights is an extra GiB *not* available for concurrent sequences, and throughput is very nearly linear in slot count. Framed as hours-to-completion (16k ctx, q8_0 KV, ~4k output tokens/trace):

| Quant tier | Weights | Cache | **Slots** | **tok/s** | **2k traces**<br>(8M tok) | **5k traces**<br>(20M tok) | **10k traces**<br>(40M tok) |
|---|---|---|---|---|---|---|---|
| **IQ4_XS** | 14.63 GiB | 7.97 | **12** | **368** | **6.0 h** | **15.1 h** | **30.2 h** |
| **Q4_K_M** | 15.93 GiB | 6.67 | **10** | **307** | **7.2 h** | **18.1 h** | **36.2 h** |
| **Q5_K_M** | 18.47 GiB | 4.13 | **6** | **170** | **13.1 h** | **32.7 h** | **65.4 h (2.7 d)** |
| *(Q6_K, for contrast)* | 21.31 GiB | 1.29 | 2 | 52 | 42.8 h | 4.5 d | 8.9 d |

Read the deltas:

- **Q5_K_M → IQ4_XS costs ~0.3 GiB of weights per billion params in precision and buys 6 extra slots and 2.2× throughput.** On the 10k-trace budget that is **65.4 h → 30.2 h: 35 hours saved, or a day and a half.**
- **Q4_K_M → IQ4_XS is nearly free** — 1.3 GiB, 2 slots, 1.2× — and IQ4_XS is an imatrix quant, so its quality at 14.63 GiB is typically at or above plain Q4_K_M at 15.93 GiB. **There is no good argument for Q4_K_M here.**
- **Q6_K is a trap.** It "fits" and then serves 2 sequences, turning a 30-hour job into a 9-day one.
- The floor is set by *correctness*, not perplexity: the victim's **final answers** become supervision targets, and a victim that gets the math wrong poisons the whole pipeline. Q3_K_M would give another 1.35× (22.4 h at 10k) but degrades answer accuracy on exactly the competition-math/JEE problems this paper uses. **Do not go below 4-bit for the victim.**

**Recommendation: IQ4_XS.** It is simultaneously the fastest fitting option and among the highest-quality 4-bit options, because imatrix quantization and small file size are not in tension here.

**Recommended vLLM/llama.cpp settings for the 27B victim:**

```
llama-server -m Qwen3.8-27B-IQ4_XS.gguf \
  -ngl 999 --flash-attn \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 98304 -np 8 \        # 98304/8 = 12288 tokens per slot
  --cont-batching --parallel 8 \
  --no-mmproj                     # vision projector never loaded
```
Note llama.cpp's `--ctx-size` is the **total** across slots and is divided by `-np`. 12k per slot comfortably holds a ~1k prompt + an 8k-capped generation. If you cap generation at 4k, use `--ctx-size 65536 -np 12` for ~12 slots and ~368 tok/s.

### 3.4 Surrogate / student / inversion-side generation (vLLM, ≤14B)

Same roofline, `S_max = 8192`, `gpu_memory_utilization 0.90`, 1.6 GiB reserved for activations + CUDA graphs.

| Model | quant | W GiB | KV pool GiB | batch | tok/s | 10k×5k tok | 25k×5k | 25k×8k |
|---|---|---|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | bf16 | 3.3 | 16.7 | 76 | **3,976** | 3.5 h | 8.7 h | 14.0 h |
| R1-Distill-Qwen-1.5B | int4 | 1.5 | 18.5 | 84 | 3,840 | 3.6 h | 9.0 h | 14.5 h |
| Qwen2.5-7B / R1-Distill-7B | bf16 | 14.2 | 5.8 | 13 | **518** | 26.8 h | 67.0 h | 107.2 h |
| Qwen2.5-7B / R1-Distill-7B | int4 | 5.3 | 14.7 | 33 | **1,366** | 10.2 h | 25.4 h | 40.7 h |
| R1-Distill-Qwen-14B | bf16 | — | — | — | **OOM** (27.5 GiB weights) | — | — | — |
| R1-Distill-Qwen-14B | int4 | 9.4 | 10.6 | 7 | 257 | 54.0 h | 134.9 h | 215.9 h |
| **Qwen3.5-2B** | bf16 | 4.2 | 15.8 | 141 | **8,069** | **1.7 h** | 4.3 h | 6.9 h |
| **Qwen3.5-2B** | int4 | 1.8 | 18.2 | 163 | 8,512 | 1.6 h | 4.1 h | 6.5 h |
| **Qwen3.5-4B** | bf16 | 8.7 | 11.3 | 38 | **1,848** | 7.5 h | 18.8 h | 30.1 h |
| **Qwen3.5-4B** | int4 | 3.2 | 16.8 | 56 | **2,782** | 5.0 h | 12.5 h | 20.0 h |
| **Qwen3.5-9B** | bf16 | 18.0 | 2.0 | 6 | 224 | 62.1 h | 155.3 h | 248.5 h |
| **Qwen3.5-9B** | int4 | 7.5 | 12.4 | 41 | **1,729** | 8.0 h | 20.1 h | 32.1 h |
| Qwen3.5-27B | int4 (vLLM) | 17.2 | 2.8 | 4 | 126 | 109.9 h | 274.6 h | 439.4 h |
| Qwen3.5-35B-A3B | int4 | 19.1 | 0.9 | 3 | 90 | 154.5 h | 386.3 h | 618.0 h |

**Read this table for the cheap-weak-surrogate vs expensive-strong-victim tradeoff:**

| Role | Model | tok/s | 5k traces × 5k tok (25M) |
|---|---|---|---|
| Weak surrogate | R1-Distill-1.5B bf16 | 3,976 | **1.7 h** |
| Weak surrogate | Qwen3.5-2B bf16 | 8,069 | **0.9 h** |
| Mid surrogate | Qwen3.5-4B int4 | 2,782 | **2.5 h** |
| Paper-scale surrogate | R1-Distill-7B int4 | 1,366 | **5.1 h** |
| **Strong local victim** | **Qwen3.8-27B IQ4_XS (llama.cpp)** | **368** | **18.9 h** |
| Strong local victim | Qwen3.8-27B AWQ (vLLM) | 115 | 60.4 h |

The strong victim is **22× more expensive per token** than the weak surrogate. Since the paper's threat model *requires* the surrogate to be weak and the victim to be strong, this asymmetry is inherent, not a config mistake. It also means: **spend your GPU-hours on victim queries, and never regenerate them.** Cache everything to disk on first pass.

Two notes on the small Qwen3.5 models: the 2B is exceptionally fast (8,069 tok/s) because only 6 of 24 layers hold a KV cache, so 141 sequences fit concurrently. The 9B at bf16 is a trap — 18.0 GiB of weights leaves a 2.0 GiB KV pool and only 6 slots, giving *worse* throughput (224 tok/s) than the 4B; at int4 it jumps to 1,729 tok/s. **On a 24 GB card, always quantize a 9B for serving.**

---

## 4. What is impossible locally

### 4.1 DeepSeek-R1 (the paper's strong surrogate *and* open-weight victim)

`deepseek-ai/DeepSeek-R1` safetensors totals **684,531,386,000 parameters** (680.6B in FP8-E4M3, 3.9B bf16, 41.6M fp32) — a **685B-parameter MoE**, ~671B commonly quoted, ~688 GB on disk as shipped.

| Precision | Weight size | vs 23.99 GiB card |
|---|---|---|
| bf16 | ~1,370 GB | **57× over** |
| FP8 (as shipped) | ~688 GB | **29× over** |
| Int4 (uniform, hypothetical) | ~343 GB | **14× over** |
| Int2 / IQ1_S GGUF ("dynamic" 1.58-bit) | ~130–160 GB | **~6× over** |

**Confirmed impossible on this box, at every quantization including 1.58-bit.** MoE does not help: all 256+ experts must be resident even though only ~37B activate per token. There is no offloading strategy either — the machine has 26 GB of free system RAM, so even CPU/NVMe-offloaded inference (which would run at single-digit tokens/sec anyway) has nowhere to put the weights. 119 GB of free disk cannot even hold the smallest usable quant.

**Every paper result that uses R1 must be either dropped, replaced with a local proxy, or bought from an API:**

| Paper role | Table/§ | Local? | Substitute |
|---|---|---|---|
| R1 as **strong surrogate** (upper-bound inversion quality) | Tab. 2, 3, 4 | **NO** | Qwen3.8-27B IQ4_XS as strong surrogate, or DeepSeek API (`deepseek-reasoner`) / Together / Fireworks |
| R1 as **open-weight victim** with ground-truth traces | §5.3, Tab. 2, 3 | **NO** | **Qwen3.8-27B locally** — it emits `<think>…</think>`, so ground-truth traces *are* recoverable and Len/BLEU/TF1/ROUGE become computable. This is strictly better than an API victim for §5.2. |
| R1-Distill-Qwen-1.5B ("R1-Weak") as weak surrogate | everywhere | **YES** | runs natively, 3,976 tok/s |
| **gpt-5.4-mini-2026-03-17** as black-box victim | §5.4, Tab. 4, Fig. 3 | **NO — API only** | **OpenAI API.** No weights exist. See §7. |
| R1 benchmark accuracies | Tab. 6 | **NO** | cite the paper's numbers, or API |

**The reframing this justifies:** with Qwen3.8-27B as a *local* victim you get something the paper's GPT-5.4-mini track can never give you — the victim's actual hidden trace, hence directly computable **trace-fidelity metrics (§5.2, Table 2)**. That makes the local 27B the primary track and the OpenAI API the optional secondary track. It also means you reproduce §5.3 (open-weight victim) faithfully and §5.4 (black-box victim) only if you choose to spend the API money.

### 4.2 Other things that require cloud/API

| Component | Why | Which API |
|---|---|---|
| GPT-5.4-mini answers + reasoning summaries + reasoning-token counts | closed weights; the paper recovers trace length from the billed `reasoning_tokens` field | **OpenAI Responses API** (`reasoning.summary`, `usage.output_tokens_details.reasoning_tokens`) |
| R1 (671B) anything | §4.1 | DeepSeek / Together / Fireworks / OpenRouter |
| Full fine-tune at the paper's exact hyperparameters (`cutoff_len=16384`, 3 epochs, full-parameter, 8×A100) | §2.3 — full FT is out at ≥3B | rent 1× A100-80 or H100 if exact reproduction is required |
| Qwen3.5-122B-A10B / 397B-A17B | 122B/397B total params must be resident | any hosted endpoint |

---

## 5. Training throughput

```
tok/s = (165 TFLOPS × MFU) / (8 × P_active)
```
8 FLOP/param/token = 2 (forward) + 4 (backward) + 2 (gradient-checkpoint recompute).
MFU = **0.32** for LoRA/bf16 with FA2 and packing; **0.20** for QLoRA (NF4 dequant on every matmul is a real ~40% tax).

| Model | mode | tok/s | 5k ex × 7k tok, 3 ep | 10k × 7k, 3 ep | 10k × 4k, 3 ep |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | lora | 4,275 | 6.8 h | 13.6 h | 7.8 h |
| | qlora | 2,672 | 10.9 h | 21.8 h | 12.5 h |
| Qwen2.5-3B | lora | 2,139 | 13.6 h | 27.3 h | 15.6 h |
| | qlora | 1,337 | 21.8 h | 43.6 h | 24.9 h |
| **Qwen2.5-7B** (paper's inversion base) | lora | 867 | **33.7 h** | 67.3 h | 38.5 h |
| | qlora | 542 | 53.8 h | **107.7 h** | 61.5 h |
| Llama-3.1-8B | lora | 822 | 35.5 h | 71.0 h | 40.6 h |
| | qlora | 514 | 56.8 h | 113.6 h | 64.9 h |
| **Qwen3.5-2B** | lora | 2,902 | 10.0 h | 20.1 h | **11.5 h** |
| | qlora | 1,814 | 16.1 h | 32.2 h | 18.4 h |
| **Qwen3.5-4B** | lora | 1,416 | **20.6 h** | 41.2 h | 23.5 h |
| | qlora | 885 | 32.9 h | 65.9 h | 37.7 h |
| **Qwen3.5-9B** | lora | 684 | 42.7 h | 85.3 h | 48.8 h |
| | qlora | 427 | 68.3 h | 136.5 h | 78.0 h |
| R1-Distill-14B | qlora | 279 | 104.4 h | 208.9 h | 119.4 h |

Two conclusions: **(1) LoRA-bf16 is ~1.6× faster than QLoRA — prefer it whenever the weights fit**, which they do for everything ≤4B and (at ≤8k) for 7B/8B too. **(2) The paper's exact configuration (7B inversion base, 10k examples, 3 epochs, 16k cutoff) is a ~70–110 hour single run on this box.** That alone justifies the scale-down.

---

## 6. Recommended "4090 tier" configuration

Design rule, derived from §3.3: **the largest victim-query budget that fits an overnight run is ~4,000 traces** (12 h × 3600 × 368 tok/s ÷ 4,000 tok = 3,970). Round to **5,000 — the paper's own smallest Figure-3 budget — at ~15 h, i.e. one overnight plus a morning.** Everything else is sized to fit around that.

| Stage | Paper | **4090 tier** | Rationale |
|---|---|---|---|
| Dataset | OpenThoughts-114k, 2 disjoint 10k splits | Same source, **2 disjoint 5k splits** | 227,914 rows / 3.55 GB parquet; stream + subset, never materialise |
| Surrogate 𝑆 (weak) | R1-Distill-Qwen-1.5B | **R1-Distill-Qwen-1.5B** (paper-faithful) or Qwen3.5-2B | both cheap; keep the paper's for comparability |
| Surrogate 𝑆 (strong) | DeepSeek-R1 671B | **dropped** (or Qwen3.8-27B doubles as it) | §4.1 |
| Compression model 𝐶 | Qwen2.5-7B-Instruct | **Qwen3.5-4B int4** | prompt-only, no training; 4B is plenty for style-matching |
| **Victim 𝑉 (primary)** | R1 (open) / GPT-5.4-mini (black-box) | **Qwen3.8-27B, GGUF IQ4_XS, llama.cpp** | only fitting build with usable throughput; **exposes ground-truth traces ⇒ §5.2 fidelity metrics computable** |
| Victim 𝑉 (secondary, optional) | GPT-5.4-mini | **OpenAI API, 5k queries** | §7 |
| Inversion model 𝐼 | Qwen2.5-7B-Instruct, full FT | **Qwen3.5-4B, LoRA r=32 bf16, all-linear** | 14.5 GiB @16k / 12.6 @8k; 1.6× faster than QLoRA |
| Student 𝐴 (primary) | Qwen2.5-7B-Instruct, full FT | **Qwen3.5-2B, LoRA r=32 bf16** | 8.3 GiB @16k; fast enough to run all 5 conditions |
| Student 𝐵 (secondary) | Llama-3.1-8B-Instruct, full FT | **Llama-3.1-8B, QLoRA 4-bit** (13.6 GiB @16k) — 2 conditions only | keeps a cross-family data point |
| **Sequence cap** | `cutoff_len = 16384` | **`max_length = 8192`** | §2.4; P95 trace ≈ 6k, halves wall-clock |
| Epochs | 3 | **2** | 3rd epoch on 5k examples mostly memorises |
| LR | 1e-5 (full FT) | **1e-4** (LoRA) | adapters need ~10× the full-FT LR |
| Query budget | 10k–25k | **5k victim + 5k surrogate** | overnight-run constraint |

### Stage-by-stage wall-clock

| # | Stage | Config | Tokens | Rate | **Hours** |
|---|---|---|---|---|---|
| 1 | Surrogate traces (5k prompts) | R1-Distill-1.5B bf16, vLLM | 30M out | 3,976 t/s | **2.1** |
| 1b | *(alt: Qwen3.5-2B)* | bf16, vLLM | 30M | 8,069 t/s | *1.0* |
| 2 | Compress traces → summaries | Qwen3.5-4B int4, vLLM | 3M out + 30M prefill | 2,782 t/s + prefill | **1.0** |
| 3 | **Victim traces + answers (5k)** | **Qwen3.8-27B IQ4_XS, llama.cpp, `-np 12` @ ~5.5k ctx** | 20M out | **368 t/s** | **15.1** |
| 4 | Train inversion model 𝐼 | Qwen3.5-4B LoRA bf16, 5k×7k, 2 ep | 70M | 1,416 t/s | **13.7** |
| 5 | Invert 5k victim outputs | 4B+LoRA merged, vLLM bf16 | 25M out | 1,848 t/s | **3.8** |
| 6 | Student 𝐴 × 5 conditions | Qwen3.5-2B LoRA, 2 ep | 2 short (~8M) + 3 long (~60M) | 2,902 t/s | **19.9** |
| 7 | Student 𝐵 × 2 conditions | Llama-3.1-8B QLoRA, 2 ep, 5k×7k | 2 × 70M | 514 t/s | **75.7** ← *see note* |
| 8 | Eval: MATH500 + JEEBench + LCB × ~12 variants | vLLM per checkpoint, ~1.5k problems × 4k tok | ~72M | ~2,500 t/s | **8.0** |
| | **TOTAL (Students 𝐴 only, stages 1–6 + 8)** | | | | **~63 h ≈ 2.6 days** |
| | **TOTAL (with Student 𝐵)** | | | | **~139 h ≈ 5.8 days** |

**Note on stage 7:** a Llama-3.1-8B QLoRA run at 5k×7k×2 epochs is 37.8 h *each*. That's the single worst line item after the victim. Three ways to cut it: (a) drop Student 𝐵 entirely and use **Qwen3.5-4B LoRA** as the second student instead (13.7 h × 2 = 27.4 h, total ~90 h ≈ 3.8 d); (b) run Student 𝐵 on a 2.5k subset (18.9 h); (c) run Student 𝐵 at `max_length=4096` (21.6 h). **Recommendation: (a).** Two Qwen3.5-family students at 2B and 4B keep the "does inversion help a weaker student more?" question answerable without the Llama tax.

**Recommended plan: stages 1–6 + 8 with a 2B and a 4B student ≈ 90 GPU-hours ≈ 3.8 days**, or ≈ 63 h ≈ 2.6 days for a single-student first pass. Every individual stage is ≤ 20 h, i.e. an overnight run, which was the design goal.

### Scaling knobs, in order of leverage

1. **Victim query budget.** Linear in stage 3 *and* stage 5 *and* the student stages. 2k traces instead of 5k takes the whole thing to ~1.7 days. The paper's Figure 3 shows 5k → 10k is worth +10.6 points on MATH500, so 5k is the floor at which the headline claim is still visible.
2. **`max_length` 8192 → 4096.** Halves stages 4, 6, 7. Costs fidelity on the longest traces.
3. **Epochs 2 → 1** for the ablation conditions, 2 for the headline ones.
4. **Quant of the victim.** IQ4_XS over Q5_K_M is 2.2× on stage 3 (15.1 h vs 32.7 h for 5k traces). Q3_K_M is another 1.35× but the victim's answer *correctness* is the whole point — do not go below 4-bit for the victim.

---

## 7. OpenAI API cost (optional secondary track)

The paper's black-box victim is `gpt-5.4-mini-2026-03-17`. No weights exist; this track is API-only.

### 7.1 Formula

```
cost = N_queries × [ (T_in / 1e6) × R_in  +  (T_out / 1e6) × R_out ]
T_out = reasoning_tokens + answer_tokens + summary_tokens   (reasoning tokens ARE billed as output)
```

### 7.2 Back-solving the paper's own figure (a free validation)

The paper states: *"Using current API pricing for GPT-5.4 mini, collecting 10k ⟨answer, summary⟩ queries costs $173.28."* Third-party trackers list gpt-5.4-mini at **$0.38 / M input, $2.25 / M output** (released 2026-03-17). Solving for output tokens with `T_in ≈ 300`:

```
$173.28 = 10,000 × [ (300/1e6)(0.38) + (T_out/1e6)(2.25) ]
$173.28 = 10,000 × [ 0.000114 + 2.25e-6 · T_out ]
T_out   = (0.0173280 − 0.000114) / 2.25e-6 = 7,651 tokens/query
```

**7,651 output tokens/query is exactly what you'd expect** for OpenThoughts prompts: ~6,100 reasoning + ~590 summary + ~950 answer. The paper's number and the listed rates are mutually consistent, which is good evidence both are right. **Cost per query = $0.01733.**

### 7.3 Estimates

| Query budget | Input tok | Output tok | Input $ | Output $ | **Total** | **+15% retry/failure margin** |
|---|---|---|---|---|---|---|
| 2,500 | 0.75 M | 19.1 M | $0.29 | $43.03 | **$43.32** | **~$50** |
| **5,000** | 1.5 M | 38.3 M | $0.57 | $86.07 | **$86.64** | **~$100** |
| 10,000 | 3.0 M | 76.5 M | $1.14 | $172.14 | **$173.28** ✓ *(matches paper)* | **~$200** |
| 15,000 | 4.5 M | 114.8 M | $1.71 | $258.21 | **$259.92** | **~$300** |
| 25,000 | 7.5 M | 191.3 M | $2.85 | $430.35 | **$433.20** | **~$500** |

**Rate caveat — read this before budgeting.** The `$0.38 / $2.25` per-M figures come from a third-party aggregator, not from OpenAI's own page, and this analysis is being written against a paper describing a model released in March 2026. **Verify against `platform.openai.com/docs/pricing` before committing spend.** If the rates differ, the formula above is rate-agnostic — substitute `R_in` and `R_out` and re-evaluate. As a sanity anchor: the paper's own $173.28-for-10k figure is a direct empirical data point, so **$0.0173/query is a defensible planning number regardless of how the published rates are decomposed.**

Two further notes:
- **Reasoning tokens dominate at 99.3% of cost.** Batch API (typically ~50% discount) applies cleanly here since this is an offline, non-interactive collection — **use it and halve the numbers above**.
- **The paper's Figure-3 sweep (5k / 10k / 15k) is nested**: collect 15k once (~$260, or ~$130 batched) and subsample, rather than running three separate collections.
- **Recommended: 5k queries, batch API, ≈ $50.** Matches the local track's 5k budget so results are directly comparable, and reproduces the leftmost Figure-3 point.

---

## 8. Software stack

### 8.1 Current versions (checked against PyPI, August 2026)

| Package | Latest | Released | Notes |
|---|---|---|---|
| torch | **2.13.0** | 2026-07-08 | cu126 / cu129 / cu130 wheels exist; **no cu128, no cu131** |
| transformers | **5.15.0** | 2026-08-10 | v5 changed dtype inference — see §8.4 |
| trl | **1.10.0** | 2026-08-13 | requires `transformers>=4.56.2` |
| peft | **0.20.0** | 2026-07-28 | |
| bitsandbytes | **0.50.1** | 2026-08-13 | |
| accelerate | **1.14.0** | 2026-06-11 | |
| datasets | **5.0.1** | 2026-07-28 | TRL requires `>=4.7.0` |
| vllm | **0.27.1** | 2026-08-11 | **pins `torch==2.13.0` exactly**, `transformers>=5.5.3`, `flashinfer-python==0.6.16.post3` |
| liger-kernel | 0.8.1 | 2026-07-23 | **TRL requires `liger-kernel!=0.8.1,>=0.8.0` — the current release is explicitly blacklisted. Pin `==0.8.0`.** |
| flash-attn | 2.8.3.post1 | 2026-06-11 | **source-only on PyPI; see §8.3** |
| lighteval | 0.13.0 | 2025-11-24 | optional, for benchmark harness |

### 8.2 Two virtualenvs — yes, definitively

```
vllm 0.27.1      → torch==2.13.0 (exact pin) + flashinfer 0.6.16.post3
trl  1.10.0      → [vllm] extra pins  vllm<=0.26.0,>=0.17.0
```

TRL's own `vllm` extra **caps vLLM at 0.26.0**, so `pip install trl[vllm] vllm==0.27.1` is unsatisfiable. Separately, vLLM's exact `torch==` pin means any training-side package that wants a different torch will silently reinstall it and break vLLM's compiled kernels. And on this box the two workloads never run concurrently anyway — there is one GPU, and generation and training are sequential stages.

```bash
# --- venv-train --------------------------------------------------------
python3.12 -m venv ~/.venvs/train && . ~/.venvs/train/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install \
  transformers==5.15.0 \
  trl==1.10.0 \
  peft==0.20.0 \
  bitsandbytes==0.50.1 \
  accelerate==1.14.0 \
  datasets==5.0.1 \
  kernels \
  liger-kernel==0.8.0   # NOT 0.8.1 — blacklisted by TRL

# --- venv-vllm ---------------------------------------------------------
python3.12 -m venv ~/.venvs/vllm && . ~/.venvs/vllm/bin/activate
pip install vllm==0.27.1        # brings its own torch 2.13.0 + flashinfer

# --- llama.cpp (no Python deps at all) ---------------------------------
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j4
```

`uv` handles this better than `pip` if you already have it (`uv venv` + `uv pip install`), and the repo already ships a `uv.lock`.

### 8.3 FlashAttention — do **not** try to build it from source on this box

`flash-attn` publishes **only an sdist** to PyPI. Prebuilt wheels live on the GitHub release page, and the available `cu×torch×` combinations top out at:

```
cu12 × torch{2.4, 2.5, 2.6, 2.7, 2.8}   and   cu13 × torch2.9
```

**There is no prebuilt wheel for torch 2.13.** A source build compiles hundreds of CUDA template instantiations, takes 1–3 hours, and each nvcc job peaks at 2–4 GB RSS. **With 26 GB of free RAM you will OOM the machine unless you set `MAX_JOBS=2`, and even then it's an afternoon.** Do not do this.

**Use the HF Kernels Hub instead** — prebuilt binary kernels fetched at runtime, no compiler, no version matching:

```python
model = AutoModelForCausalLM.from_pretrained(
    ..., attn_implementation="kernels-community/flash-attn2")
```
or in TRL: `SFTConfig(model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2", "dtype": torch.bfloat16})`.

Version pinning is supported (`"kernels-community/flash-attn2@>=2.0,<3.0"`). Requires the `kernels` package (installed above). Fallback if the hub is unavailable: `attn_implementation="sdpa"`, which routes to PyTorch's flash backend on Ada — slower, but it also avoids the S² blowup, so §2.3 stays valid.

### 8.4 TRL-specific memory behaviour that matters at 24 GB

`SFTConfig` defaults **differ from `TrainingArguments`** and several of them are load-bearing here.

| Flag | TRL default | What to set | Why |
|---|---|---|---|
| **`model_init_kwargs["dtype"]`** | **`float32`** ⚠️ | **`torch.bfloat16`** | **The single worst footgun.** TRL's docs state: *"If `dtype` is not specified in `args.model_init_kwargs`, it defaults to `float32`. This differs from `from_pretrained`, where (since Transformers v5) the dtype is inferred from the model config."* Passing a model *string* to `SFTTrainer` without this loads a 7B at **30.5 GiB instead of 14.2** and OOMs before step 1. |
| `loss_type` | `"chunked_nll"` | **leave it** | §2.4a. This is what keeps the 248k-vocab logits tensor off the card. Setting `use_liger_kernel=True` silently flips it to `"nll"` — if you do that, make sure Liger's fused linear-CE actually engaged. |
| `gradient_checkpointing` | **`True`** (differs from `TrainingArguments`' `False`) | leave it | §2.4b — mandatory. |
| `bf16` | `True` if `fp16` unset | leave it | |
| `max_length` | **`1024`** | **`8192`** | Default silently truncates every 6k-token trace to 1k and your inversion model learns to produce stubs. **This will not error.** |
| `packing` | `False` | **`True`** | `packing_strategy="bfd"` (default) also **auto-enables `padding_free`**, which flattens the batch and eliminates pad waste. Big win when trace lengths vary 1k–16k. Requires FA2/FA3. |
| `packing_strategy` | `"bfd"` | `"bfd"` | `"bfd_split"` splits overflow instead of truncating; `"wrapped"` cuts mid-sequence and will splice two unrelated traces — **don't**. |
| `padding_free` | `False` | implied by `bfd` | needs FlashAttention; will error under `sdpa`. |
| `completion_only_loss` | auto | **`True`** (prompt-completion dataset) | Loss only on the trace, not the input/answer/summary prompt. Compounds with `chunked_nll`: fewer non-ignored tokens ⇒ smaller lm_head matmul. |
| `assistant_only_loss` | `False` | alternative for conversational format | TRL auto-patches Qwen chat templates with `{% generation %}` markers. |
| `activation_offloading` | `False` | **leave `False`** | Offloads activations to CPU. At 8k that's 2–6 GiB of *pinned* host memory and you have ~26 GB total. It works, but it's your last resort, not your first. |
| `optim` | `adamw_torch_fused` | `adamw_torch_fused` (LoRA) / `adamw_bnb_8bit` (if ever full-FT) | §2.5 |
| `use_cache` | `False` | leave it | KV cache during training is pure waste |
| `gradient_accumulation_steps` | 1 | **8–16** | per-device batch must be 1 at these lengths |
| `dataset_num_proc` | `None` | **`4`** | each worker forks — more will exhaust 26 GB RAM |
| `dataloader_num_workers` | 0 | **2**, `dataloader_pin_memory=False` | pinned memory competes with the same 26 GB |
| `save_total_limit` | `None` | **`2`** | adapters are small but checkpoints accumulate |
| `torch_empty_cache_steps` | `None` | **`50`** | helps with allocator fragmentation on long-sequence runs |
| `router_aux_loss_coef` | `0.001` | only relevant if you ever train the MoE | |

### 8.5 Known incompatibilities and 4090-specific gotchas

1. **`liger-kernel==0.8.1` is blacklisted by TRL 1.10** (`liger-kernel!=0.8.1,>=0.8.0`). Pin `0.8.0` or skip Liger — you don't need it, `chunked_nll` already solves the logits problem.
2. **No `cu128` wheel for torch 2.13.** Available: cu126, cu129, cu130. Driver 595.84 / CUDA 13.2 is forward-compatible with all three. **Use cu130.**
3. **NVFP4 quants require Blackwell (sm100+).** The 4090 is sm89. `unsloth/Qwen3.8-27B-NVFP4` (21.8 GiB) will *look* like it fits and will not run. FP8 *is* natively supported on Ada, but the 27B FP8 build is 28.77 GiB anyway.
4. **vLLM `qwen3_5` support needs ≥0.17;** hybrid Gated-DeltaNet uses Flash-Linear-Attention Triton kernels plus a hybrid KV-cache manager. Confirmed working on 4090 per Qwen's own vLLM recipe for the 9B. Use 0.27.1.
5. **Qwen3.5/3.8 are `…ForConditionalGeneration`, not `…ForCausalLM`.** `AutoModelForCausalLM.from_pretrained` may not resolve them. TRL's `SFTTrainer` supports VLMs, but its VLM advice (`max_length=None`) is **wrong for text-only use** — you must keep `max_length` set. **Verify before your first long run**: (a) the model loads, (b) `model.visual.requires_grad_(False)`, (c) the vision tower and the MTP head are absent from your LoRA `target_modules`.
6. **`padding_free` + hybrid linear attention is unverified.** `padding_free` is documented as FA2/FA3-only, and only 8 of 32 (or 16 of 64) Qwen3.5 layers use FlashAttention — the rest use FLA Triton kernels whose cu_seqlens handling may differ. **Smoke-test packing on 20 examples before committing to a 14-hour run.** Fall back to `packing=False, padding_free=False` if it misbehaves; you lose ~25% throughput, not correctness.
7. **26 GB system RAM is the real second bottleneck.** It rules out: DeepSpeed ZeRO-Offload (a 7B optimizer offload alone needs ~85 GB host), CPU-offloaded Adam, `device_map="auto"` with CPU spill, materialising OpenThoughts-114k in memory (3.55 GB parquet → far more as Python objects — always use Arrow memory-mapping and `.select()` before `.map()`), and parallel flash-attn compilation.
8. **119 GB disk.** Budget: Qwen3.8-27B IQ4_XS 14.6 + Qwen3.5-4B 8.7 + Qwen3.5-2B 4.2 + R1-Distill-1.5B 3.3 + Llama-3.1-8B 15.0 (if used) + OpenThoughts subset ~1 + generated traces ~0.5 + adapters/checkpoints ~10 ≈ **58 GB**. Comfortable — **but never download the bf16 27B (51.7 GB)**, and set `HF_HOME` somewhere you're watching.
9. **No NVLink / single GPU** ⇒ no tensor parallelism, no FSDP, no ZeRO-3. `accelerate` config: single process, no distributed backend.
10. **llama.cpp must be built from master** for `qwen3_5`, and **the GGUF variant must match the build's MTP expectation** (§3.2c). unsloth publishes both plain and `-MTP-` GGUFs; picking the wrong one produces a load-time tensor-count error (`expects 33 blocks, model only has 32`), not a runtime fallback. Smoke-test with a 20-token generation before scheduling the 15-hour job. Do **not** convert HF→GGUF yourself — `convert_hf_to_gguf` currently fails on qwen3_5 `ssm_conv1d` tensors (issue #27019); use unsloth's prebuilt imatrix GGUFs.
11. **Build llama.cpp with `-DCMAKE_CUDA_ARCHITECTURES=89`** and `-j4`, not `-j$(nproc)` — the CUDA compile is memory-hungry and 26 GB of RAM is not much.

---

## 9. Sources

- Paper text: `arXiv:2603.07267v2` (local copy)
- Model configs and parameter counts: HuggingFace API (`/api/models/{id}` safetensors totals) and `config.json` for Qwen2.5, Qwen3.5 (0.8B/2B/4B/9B/27B/35B-A3B), Qwen3.8-27B, Llama-3.1-8B, DeepSeek-R1 and R1-Distill 1.5B/7B/14B
- GGUF/AWQ/GPTQ/FP8 build sizes: HF repo tree byte counts for `unsloth/Qwen3.8-27B-GGUF`, `ggml-org/Qwen3.8-27B-GGUF`, `Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.5-27B-GPTQ-Int4`, `cyankiwi/Qwen3.8-27B-AWQ-INT4`, `unsloth/Qwen3.8-27B-NVFP4`
- Package versions and dependency pins: PyPI JSON API; PyTorch wheel index `download.pytorch.org/whl/cu1xx`
- flash-attn wheel matrix: GitHub `Dao-AILab/flash-attention` latest release assets
- [TRL SFTTrainer / SFTConfig documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Transformers attention backends / Kernels Hub](https://huggingface.co/docs/transformers/en/attention_interface)
- [vLLM recipe: Qwen/Qwen3.5-9B](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)
- [vLLM releases](https://github.com/vllm-project/vllm/releases); `qwen3_5` support status from vLLM PRs #52004, #51991-3, #52013 and issue #52319
- llama.cpp `qwen3_5` support status from ggml-org/llama.cpp PR #25024 (merged) and issues #24737, #26916, #27019
- 4090 vLLM throughput calibration: [databasemart RTX 4090 vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090)
- gpt-5.4-mini rate listing: [pricepertoken.com](https://pricepertoken.com/pricing-page/model/openai-gpt-5.4-mini) — **third-party, verify at [platform.openai.com/docs/pricing](https://platform.openai.com/docs/pricing)**
- Dataset size: HF datasets-server `/size` for `open-thoughts/OpenThoughts-114k` (227,914 rows / 3.55 GB parquet)

Every number above is reproducible from the formulas inlined in §2.2, §3.2, §3.3 and §5 plus the config values in §2.6 — no figure is quoted from memory.
