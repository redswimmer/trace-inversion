# Measured Hardware Results — RTX 4090

Everything here is **measured on this machine**, not estimated. It supersedes the
projections in `05-rtx4090-feasibility.md` wherever the two disagree.

Date: 2026-08-16
Hardware: RTX 4090 24 GB (24564 MiB), 30 GB system RAM, CUDA 13.2 / driver 595.84
Backend: **llama.cpp b10450 / ggml 0.20.0, Vulkan** (Homebrew build)

---

## 1. Toolchain prerequisite

Homebrew's llama.cpp **9000 / ggml 0.10.2 cannot load Qwen3.5/3.8 models**:

```
llama_model_load: error loading model: missing tensor 'blk.64.ssm_conv1d.weight'
```

`blk.64` is the multi-token-prediction head (the model has 64 layers, 0–63).
Fixed by `brew upgrade llama.cpp ggml` → **10450 / 0.20.0**. No source build needed;
a CUDA build was attempted and abandoned (CMake CUDA-13.1 probe issues).

Note: the current build **ignores** the MTP tensors —
`model has unused tensor blk.64.nextn.*  -- ignoring`. MTP is effectively built-in
speculative decoding, so there is a free speedup available if llama.cpp adds support.

---

## 2. Quant comparison — Qwen3.8-27B, single stream

`llama-bench -ngl 999 -p 512 -n 128 -r 2`

| Quant | File size | Prompt (pp512) t/s | **Generation (tg128) t/s** |
|---|---:|---:|---:|
| **IQ4_XS** | 14.62 GiB | 2290 ± 18 | **47.3 ± 0.1** |
| Q4_K_M | 15.92 GiB | 1929 ± 19 | 46.3 ± 0.1 |
| Q5_K_M | 18.46 GiB | 2257 ± 12 | 42.0 ± 0.1 |

**IQ4_XS wins outright** — smallest *and* fastest. The generation spread across quants
is only ~12%, so quant choice is nearly free on speed; it matters for the KV/state
headroom it leaves behind, not for per-token rate.

Q6_K (21.31 GiB) was downloaded, judged not worth the state headroom it consumes, and
deleted without benchmarking.

VRAM: IQ4_XS loads at **16,628 MiB** with a controlled context. (`llama-cli` with no
explicit `-c` grabs a huge default context and reaches 23,036 MiB — always set `-c`.)

---

## 3. Batched throughput — IQ4_XS (the decisive measurement)

`llama-batched-bench -c <2560×B> -npp 512 -ntg 2048 -fa on -ctk q8_0 -ctv q8_0`

| Concurrency | N_KV | Gen t/s | Total t/s | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 2,560 | 45.7 | 44.5 | 1.0× |
| 2 | 5,120 | 81.1 | 96.2 | 1.8× |
| 4 | 10,240 | 107.1 | 132.3 | 2.3× |
| 8 | 20,480 | 136.8 | 168.5 | 3.0× |
| 16 | 40,960 | 202.3 | 247.4 | 4.4× |
| 24 | 61,440 | 257.0 | 312.6 | 5.6× |
| **32** | **81,920** | **303.0** | **366.6** | **6.6×** |
| 40 | 102,400 | — | — | ❌ OOM |
| 48 | 122,880 | — | — | ❌ OOM |
| 64 | 163,840 | — | — | ❌ OOM |

**Ceiling is 32 concurrent sequences.**

### Why it fails at 40+ — and why it's not the KV cache

```
E alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1045954560
E llama_init_from_model: failed to initialize the context:
    failed to allocate buffer for rs cache
```

`rs cache` = **recurrent state**, for the 48 linear-attention (gated-delta-net) layers.
Unlike a KV cache, this state is **fixed-size per sequence slot and independent of
context length**. So:

- concurrency is capped by sequence *count*, not by context length
- shortening `-c` would **not** buy more slots
- the architecture is cheap for **long** generations, expensive for **many concurrent** ones

This is the inverse of a conventional transformer, where context length is what bites.
It suits this workload well: reasoning traces are long, and we control batch size.

---

## 4. Wall-clock for trace generation

Assuming ~5,000 output tokens per trace (the paper's R1 traces average 6,130; inverted
traces 4,972–6,021).

| Trace budget | Tokens | @ 45.7 t/s (B=1) | @ 202 t/s (B=16) | **@ 303 t/s (B=32)** |
|---|---:|---:|---:|---:|
| 2k | 10 M | 61 h | 14 h | **9 h** |
| 5k | 25 M | 152 h | 34 h | **23 h** |
| 10k | 50 M | 304 h | 69 h | **46 h** |

**5k traces = one overnight run.** That is the paper's smallest Figure-3 budget, which
already delivers most of the MATH500 benefit (67.0 at 5k vs 77.6 at 10k).

---

## 5. Implications for the plan

1. **Serve with `llama-server -np 32 -ctk q8_0 -ctv q8_0`** for generation. Do not exceed
   32 slots — it fails at context-creation time, not gracefully.
2. **Use IQ4_XS**, not Q5_K_M. Faster, smaller, and leaves the most state headroom.
3. **A local victim is now viable**, at ~23 h for 5k traces. The OpenAI victim still wins
   on parallelism (it frees the GPU entirely for training), but local is no longer a
   multi-day commitment.
4. **Quantized victim caps the downstream ceiling.** IQ4_XS Qwen3.8-27B is slightly worse
   than full precision, so traces grounded in it are slightly worse. It is a constant
   across all conditions, not a confound, but our absolute numbers will not be directly
   comparable to the paper's.

---

## 6. Not yet measured

- Accuracy baselines for the student candidates (Qwen3.5-0.8B / 2B / 4B) — script at
  `bench/eval_baseline.py`, vLLM 0.27.1 installed in `.venv-vllm`.
- Actual trace lengths produced by the victim at each `reasoning_effort`. The paper's
  fidelity metric is length recovery against a 6,130-token reference, so this needs
  calibrating before generation.
- Training throughput / VRAM for the TRL QLoRA stages.
