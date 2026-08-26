# 06 — Model Selection and TRL Stack

Local reproduction of *How to Steal Reasoning Without Reasoning Traces*
(arXiv 2603.07267v2, Zhang / Morris / Shmatikov, Cornell Tech) on **1× RTX 4090 (24 GB), 30 GB
system RAM (~26 GB usable), 119 GB free disk**, with **all training in TRL**.

Everything below was verified against the HF Hub API, model `config.json` / `chat_template.jinja`
files, the TRL v1.10.0 source and docs, the transformers `main` source, and the vLLM model registry
on 2026-08-15. Anything I could not confirm is marked **UNVERIFIED**.

---

## 0. Recommendation box

> ### ⚠️ Role assignments here are SUPERSEDED — `10-run-plan.md` is authoritative
>
> This box is the pre-Phase-0 *survey*, written before anything was measured. Phase 0 fixed every
> role on measurement and four of them moved. Read `10-run-plan.md` §"Role assignments" and
> `docs/results/baselines.md` instead; keep this box only for the reasoning behind the shortlists.
>
> | Role | This box says | **Actually chosen** | Decided by |
> |---|---|---|---|
> | Victim | Qwen3.8-27B `Q5_K_M` | Qwen3.8-27B **`IQ4_XS`** | `08` §2 — smaller *and* faster |
> | Surrogate | R1-Distill-**1.5B** | R1-Distill-**7B** primary, 1.5B as arm 2 | baselines — 1.5B sits *below* the student on JEEBench |
> | Inverter | Qwen3.5-4B **QLoRA** | Qwen3.5-4B **bf16 LoRA** | `08` §6 — NF4 not needed |
> | Student | **Qwen2.5-7B-Instruct** | **Qwen3.5-2B**, full fine-tuning | baselines — 39 pt headroom, and FFT fits |
>
> The student swap also retires this box's "no Qwen3.5 model can be the student" rule: see
> `09` §5.2 for why a reasoning student is accepted, and what it costs the claim.

The paper has four model roles, not three. The compression model `C` is easy to miss and is required
to build the summary-setting training data.

| Role | Primary | Fallback | Why | VRAM / footprint |
|---|---|---|---|---|
| **Victim `V`** (inference only) | **`unsloth/Qwen3.8-27B-GGUF` @ `Q5_K_M`** on llama.cpp (`llama-server`), `--mmproj` **not** passed | `UD-Q4_K_XL` (16.69 GiB) if Q5 is tight; `philbert440/Qwen3.8-27B-W4A16-AWQ` on vLLM if you need vLLM's batching more than you need quality | Only a **local open-weight** victim makes the paper's Table 2 (trace fidelity) reproducible at all — you need the victim's ground-truth trace, which an API victim never gives you. 27B vs a 1.5B surrogate also finally exercises the "victim ≫ surrogate" regime the paper argues for but never actually tests. **Quality matters most in this role — the victim's traces literally become the training data**, so buy the highest bit-width that fits. | **18.47 GiB** weights → ~4.9 GiB for KV cache (~76 k cached tokens). Vision projector is a separate file, so it costs **zero**. |
| **Surrogate `S`** (inference only) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `Qwen/Qwen3.5-2B` (thinking on) | This *is* the paper's "R1-Weak". MIT, ungated, 3.5 GB, `qwen2` arch, trivially served. Keeps the realistic weak-surrogate condition intact. | 3.5 GB bf16 |
| **Compression `C`** (inference only, no FT) | `Qwen/Qwen3.5-4B` with `enable_thinking=False` | `Qwen/Qwen2.5-7B-Instruct` (paper's exact choice) | Zero-shot prompted summariser (Appendix B). Paper deliberately does not fine-tune it. | 9.3 GB bf16, or reuse the inversion backbone |
| **Inversion `I`** (**TRL SFT**) | `Qwen/Qwen3.5-4B` QLoRA | `Qwen/Qwen2.5-7B-Instruct` QLoRA (paper-faithful, far lower risk) | Long-in/long-out job; 4B keeps 8–16 k sequences comfortable. Qwen3.5 carries four avoidable risks (see §2.1) — the Qwen2.5 fallback has none. | **~7.5 GB** @ 8 k, ~12 GB @ 16 k |
| **Student `M`** (**TRL SFT**) | `Qwen/Qwen2.5-7B-Instruct` QLoRA | cross-family: `meta-llama/Llama-3.1-8B-Instruct` (**GATED**) or ungated `allenai/OLMo-2-1124-7B-Instruct` | The student must be a **non-reasoning** instruct model — that is the entire experiment. Paper-faithful, and cross-family vs. the Qwen3.5-arch victim (different tokenizer, 152 k vs 248 k vocab). | **~9–10 GB** @ 8 k |

**Family-matching note.** Surrogate (`R1-Distill-Qwen-1.5B`) and student (`Qwen2.5-7B-Instruct`) share
Qwen2.5 lineage — that is *intentional and matches the paper*, which used exactly this pair. The
paper's cross-family control is `Llama-3.1-8B-Instruct`; keep that as the second student arm. The
victim (Qwen3.5 arch) is cross-family to both, which is the configuration that matters.

**Disqualification worth stating plainly: no Qwen3.5 or Qwen3.8 Instruct model can be the student.**
They all reason natively (`<think>` by default). Fine-tuning one on inverted traces cannot show that
inversion *taught* reasoning — the model already reasons. Students must be non-reasoning instruct
models: Qwen2.5-*-Instruct, Llama-3.1/3.2-Instruct, OLMo-2-Instruct.

**Gating: only `meta-llama/Llama-3.1-8B-Instruct` is gated** (verified `gated: true`, license
`llama3.1`). Accept the terms on the model page and `hf auth login` before use. Everything else
recommended here is ungated. `Qwen/Qwen2.5-3B-Instruct` is ungated but is `license: other` (Qwen
Research License, non-commercial) — the 1.5B and 7B are Apache-2.0, prefer those.

**Disk plan (119 GB free):** victim GGUF Q5_K_M 19.8 GB, surrogate 3.5 GB, inversion 9.3 GB, student
15.2 GB, dataset + generated traces ~5 GB, LoRA adapters negligible → **~54 GB**. Do **not** also pull
bf16 `Qwen/Qwen3.8-27B` (55.6 GB); it would take you to ~110 GB and you need slack for checkpoints.
Skip `mmproj-*.gguf` entirely.

---

## 1. The local open-weight victim: Qwen3.8-27B

### 1.1 Verified facts

| Field | Value |
|---|---|
| Repo | `Qwen/Qwen3.8-27B` (ungated, apache-2.0) |
| Params | 27,781,427,952 (BF16 only) — **dense**, `qwen3_5` arch, `Qwen3_5ForConditionalGeneration` |
| Text backbone | hidden 5120, 64 layers, intermediate 17408, head_dim 256, 24 Q heads / 4 KV heads |
| Attention stack | 48 `linear_attention` (Gated DeltaNet) + 16 `full_attention`, `full_attention_interval: 4` |
| Vocab | 248,320 · `tie_word_embeddings: False` |
| Context | 262,144 native (1 M with RoPE scaling per model card) |
| Vision | `vision_config` depth 27, hidden 1152 — present; config also carries `language_model_only: false` |
| Thinking | on by default, `<think>\n…\n</think>\n\n`; `reasoning_effort ∈ {low, medium, xhigh}`, **xhigh default**; `preserve_thinking` default true |
| `generation_config.json` | `temperature 1.0, top_p 0.95, top_k 20, do_sample true` (matches the card's thinking-mode recipe) |
| Official FP8 | `Qwen/Qwen3.8-27B-FP8` exists |
| Also in family | `Qwen/Qwen3.8-2.4T-A95B` + `-FP8` — MoE, `qwen3_5_moe_text`, `license: other`, 2.45 T params. Out of local reach entirely; noted for inventory completeness. |

### 1.2 VRAM arithmetic — correcting the estimate

An RTX 4090 reports **24,564 MiB = 25.76 GB decimal = 23.99 GiB**. Subtract ~0.8 GB CUDA
context/driver → **~24.9 GB actually allocatable**.

| Tier | Weights | Verdict |
|---|---|---|
| **bf16** | 27.78 B × 2 B = **55.6 GB** | No. ~2.2× over. |
| **FP8 / INT8 W8A16** | 27.78 B × 1 B = **27.78 GB** | **No.** 27.78 GB > 25.76 GB total board memory — over by **~2.0 GB before a single byte of KV cache**, and ~2.9 GB over the allocatable budget. Add the ~5 GB of KV cache you actually need and the real shortfall is ~8 GB. Secondary reason: FP8 tensor-core math needs Hopper SM90; the 4090 is Ada SM89, so vLLM's FP8 path is emulated anyway. |
| **4-bit W4A16 (AWQ / GPTQ)** | **~18.7 GB on disk, ~17.5 GB resident text-only** | **Yes, but tighter than "13.9 GB" suggests.** |
| **GGUF `Q5_K_M`** ← recommended | **18.47 GiB** | **Yes** — ~5 bits, so *better quality than 4-bit AWQ at the same footprint*, and the vision projector is a separate file so text-only is free. See §1.3. |

The naive "27.78 B × 0.5 B = 13.9 GB" undercounts, because W4A16 leaves the embedding and LM head in
16-bit and this model has an enormous **untied** vocab:

```
embed_tokens   248,320 × 5,120 × 2 B = 2.54 GB   (bf16, not quantized)
lm_head        248,320 × 5,120 × 2 B = 2.54 GB   (bf16, untied → a second copy)
quantized linears  ~22.1 B × 0.5 B    = 11.1 GB
group scales/zeros (g=128, fp16)      ≈  0.35 GB
vision tower (bf16, depth 27)         ≈  1.2 GB   ← removable
                                        --------
                                         17.7 GB  text-only ≈ 16.5 GB
```

Cross-checked against a real checkpoint: `philbert440/Qwen3.8-27B-W4A16-AWQ` is a single
**18,698,467,264-byte** safetensors (verified via HF API) — 18.7 GB, matching the estimate plus
metadata. So budget **~17.5 GB resident**, leaving **~6–7 GB for KV cache and activations**, not the
~8 GB a 13.9 GB figure would imply.

**KV cache capacity.** Only the 16 full-attention layers hold a KV cache:
`16 layers × 2 (K,V) × 4 kv_heads × 256 head_dim × 2 B = 65,536 B = 64 KiB per token`.
At ~6 GB of KV budget that is **~98,000 cached tokens** — roughly **12 concurrent 8 k sequences**.
The 48 DeltaNet layers hold a fixed-size recurrent state per sequence instead (does not grow with
length), a modest constant on top. This is a perfectly workable serving batch.

### 1.3 Quantized victim checkpoints

**Recommended: the GGUF / llama.cpp path.** `unsloth/Qwen3.8-27B-GGUF` (apache-2.0, imatrix quants)
ships the **vision projector as separate `mmproj-F16.gguf` / `mmproj-BF16.gguf` files** — the main
GGUF is text-only. Don't pass `--mmproj` and the vision encoder costs literally zero VRAM, with no
flag-hunting and no partial-load risk. Usable VRAM on the 4090 is **~23.4 GiB**:

| Quant | Size (GiB) | Fits? |
|---|---|---|
| `IQ4_XS` | 14.63 | yes — most KV headroom |
| `Q4_K_M` | 15.93 | yes |
| `UD-Q4_K_XL` | 16.69 | yes — good fallback |
| **`Q5_K_M`** | **18.47** | **yes — recommended.** Best quality that leaves comfortable KV headroom (~4.9 GiB ≈ 76 k cached tokens at 64 KiB/token). |
| `UD-Q5_K_XL` | 18.83 | yes, marginally tighter |
| `Q6_K` | 21.31 | technically fits, but ~2.1 GiB of KV cache — too few concurrent sequences |
| `Q8_0` | 27.05 | **no** |

Pick **`Q5_K_M`**. This role's output *is* the training data; a quantization artifact in a victim
trace propagates through the inversion model into the student. Spend the bits here, not on the
training-side models (which are QLoRA'd anyway and only need gradients to be well-conditioned).

Other community quants surveyed, for the record:

| Repo | Format | Size | Usable on a 4090 (Ada SM89)? |
|---|---|---|---|
| `philbert440/Qwen3.8-27B-W4A16-AWQ` | AWQ int4 W4A16, compressed-tensors, ships MTP head | 18.70 GB + 0.85 GB MTP | **Yes** — the best safetensors/vLLM option if you need vLLM's continuous batching. W4A16 Marlin kernels run on Ada. Lower quality than GGUF Q5_K_M at similar size. |
| `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ` | W4A16, llm-compressor, `vllm` tag, ships MTP head | 26.82 GB + 0.85 GB | **No.** 26.8 GB exceeds board memory. The size implies it is only partially quantized — **UNVERIFIED** which layers were skipped. Avoid. |
| `unsloth/Qwen3.8-27B-NVFP4`, `RadixArk/…-NVFP4`, `Inferact/…-NVFP4`, `sakamakismile/…-MTP-NVFP4` | NVFP4 | ~14–16 GB | **No.** NVFP4 requires Blackwell (SM100/SM120). Ada is SM89. |
| `huginnfork/Qwen3.8-27B-NVFP4A16` | NVFP4 weight-only, fp16 activations | — | **UNVERIFIED** whether vLLM dequantises NVFP4A16 on Ada. Do not plan around it. |
| `lued/Qwen3.8-27B-INT8-W8A16-MTP` | INT8 W8A16 | ~28 GB | No — the 8-bit tier, see above. |
| `bartowski/…-GGUF`, `ggml-org/…-GGUF`, `lmstudio-community/…-GGUF` | GGUF | comparable | Alternatives to the unsloth GGUF above; unsloth's imatrix quants and separate `mmproj` make it the default pick. |
| `lmstudio-community/…-MLX-*`, `mlx-community/…` | MLX | — | Apple Silicon only. Irrelevant. |

I found **no official Qwen AWQ/GPTQ build** — only FP8 and (for other family members) GPTQ-Int4.
All four-bit options above are community quants; validate one on a held-out benchmark slice before
generating 10 k traces with it.

### 1.4 Text-only serving — the vision encoder can be skipped on every path

No separate text-only checkpoint exists, and none is needed. Three independent mechanisms:

- **GGUF / llama.cpp (recommended):** the vision projector ships as a **separate `mmproj-*.gguf`
  file**. The main GGUF is text-only. Simply do not pass `--mmproj`. Zero VRAM, zero configuration.
- **vLLM:** `--language-model-only`.
- **transformers:** `Qwen3_5ForCausalLM.from_pretrained(...)`.

Details on the latter two:

- `Qwen/Qwen3.8-27B/config.json` carries a top-level **`language_model_only: false`** key.
- vLLM's official Qwen3.5 recipe documents **`--language-model-only`**, described verbatim as
  skipping "loading the vision encoder and free[ing] up memory for KV cache."
- The vLLM model registry (`vllm/model_executor/models/registry.py`, `main`) separately registers
  **`Qwen3_5ForCausalLM`** (text-generation section) alongside `Qwen3_5ForConditionalGeneration`
  (multimodal section) — so a text-only load path exists on both the vLLM and transformers sides.
- In transformers the equivalent is `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.8-27B")`, which
  the official model docs show explicitly.

Caveat: I grepped transformers `main` for `language_model_only` in
`models/qwen3_5/{modeling,configuration}_qwen3_5.py` and found **no hits** — the key appears to be
consumed by vLLM, not transformers. Use `Qwen3_5ForCausalLM` on the transformers side and
`--language-model-only` on the vLLM side. Saves ~1.2 GB.

### 1.5 vLLM support (the alternative serving path)

Recommended victim serving is llama.cpp + GGUF Q5_K_M (§1.3). This section documents the vLLM route
in case you want continuous batching more than you want the extra bit-width.

**Supported.** Verified directly in `vllm/model_executor/models/registry.py` on `main`:

```
"Qwen3_5ForCausalLM":              ("qwen3_5", "Qwen3_5ForCausalLM")            # text-only
"Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration") # multimodal
"Qwen3_5MoeForCausalLM" / "…MoeForConditionalGeneration"
"Qwen3_5MTP" / "Qwen3_5MoeMTP":    ("qwen3_5_mtp", …)                            # speculative decoding
```

Version: **vLLM 0.17 added native Qwen3.5 hybrid-GDN support**; 0.16.0 did not (per vLLM's own
Qwen3.5 discussion threads — treat the exact minor as *approximately* verified). Current PyPI release
is **0.27.1** (verified), so any recent install is fine. `Qwen/Qwen3.8-2.4T-A95B` got day-0 vLLM
support reusing the same architecture with no changes, which is good evidence the `qwen3_5` path is
mature.

**Recommended serve command (llama.cpp, the primary path):**

```bash
llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q5_K_M \
  --ctx-size 16384 \
  --n-gpu-layers 999 \
  --flash-attn \
  --jinja \
  --reasoning-format none \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --parallel 4
  # deliberately NO --mmproj → vision projector never loaded, 0 VRAM
```

- `--jinja` is required so the model's own `chat_template.jinja` runs and `reasoning_effort` is
  honoured (see §1.6). Pass it per-request via `chat_template_kwargs`.
- `--reasoning-format none` keeps the raw `<think>…</think>` in `content` rather than splitting it
  out — **this is how you capture the ground-truth trace `z` for Table 2.** (Use
  `--reasoning-format deepseek` instead if you'd rather have it in a separate `reasoning_content`
  field; either works, just be consistent.)
- `--parallel N` sets concurrent slots; each slot reserves `ctx-size / N` tokens of KV. Tune against
  the ~4.9 GiB KV budget.

**Alternative serve command (vLLM, if you take that route):**

```bash
vllm serve philbert440/Qwen3.8-27B-W4A16-AWQ \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

- `--reasoning-parser qwen3` splits `<think>…</think>` into a separate `reasoning_content` field on
  the OpenAI-compatible response. **This is what gives you the ground-truth trace `z` for Table 2.**
- `--enable-prefix-caching` is free throughput given a shared system prompt across all 10 k queries.
- MTP speculative decoding: the `model-mtp.safetensors` head (0.85 GB) ships with the AWQ repo.
  Costs 0.85 GB of KV budget, typically returns ~1.5× decode. Drop it if you hit OOM.
- If CUDA-graph capture errors appear, lower `--max-cudagraph-capture-size` below its 512 default
  (documented vLLM limitation for this family).
- Sampling for thinking mode: `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0`.

### 1.6 `reasoning_effort` and trace length — read the template, not the card

I read `Qwen/Qwen3.8-27B/chat_template.jinja` directly. `reasoning_effort` is **a soft prompt
instruction injected into the system message, not a token budget**:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort …') }}
{%- if resolved_reasoning_effort == 'xhigh' %}
    {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully
        through the task, validate key assumptions, consider plausible alternatives, and prioritize
        correctness, consistency, and clarity in the final answer.' %}
{%- elif resolved_reasoning_effort == 'low' %}
    {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and
        focused, moving directly to the conclusion without unnecessary elaboration.' %}
```

Passing it through llama.cpp (requires `--jinja`):

```jsonc
POST /v1/chat/completions
{ "messages": [...],
  "chat_template_kwargs": { "reasoning_effort": "xhigh" },
  "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0 }
```

Three consequences that matter for this reproduction:

1. **`xhigh` is the default and the right choice.** The paper's fidelity target is R1's ~6,130-token
   average trace (Table 2); the inversion models it trains recover 4,972 / 5,434 tokens. `low`
   explicitly instructs brevity and will collapse trace length, destroying the Len / TF1 comparison.
2. **`medium` sets no instruction string at all** — the `if/elif` has no `medium` branch, so
   `reasoning_instructions` stays unset. `medium` is effectively "no steer", *not* a midpoint. Do not
   assume monotonic length low < medium < xhigh.
3. **Actual token counts per effort level are UNVERIFIED.** Because it is a prompt nudge and not a
   budget, you must measure. Prescribed calibration step before the main run: sample 100
   OpenThoughts inputs, generate at each of the three levels, histogram the token count between
   `<think>` and `</think>`, and pick the level whose **median** lands nearest 6,130. Report that
   number — it is the direct analogue of the paper's Table 1 distribution-matching exercise.

**Honesty flag on Table 2.** Reproducing Table 2 with Qwen3.8-27B as victim re-instantiates the
*metric*, not the *numbers*. The paper's ground truth is R1's traces; yours will be Qwen3.8's, a
different distribution with a different length profile. Present it as "Table 2, re-run against a new
open-weight victim", never as a numeric match.

### 1.7 Throughput, briefly

Not VRAM but **wall-clock** is the binding constraint on the victim role: decode is
memory-bandwidth-bound, and the paper's 10,000-example victim split at ~6,500 tokens each is tens of
millions of generated tokens. Detailed tok/s analysis is being handled separately. Two planning
points that follow from model selection alone:

- **Cut the victim split to 1,000–2,000 examples.** The paper's effect sizes are large (52.4% vs
  16.4% on MATH500); you do not need 10 k to see them. Keep the *surrogate* split at 10 k — the 1.5 B
  surrogate is an order of magnitude cheaper per token.
- Trading down to `reasoning_effort=medium` to save time distorts trace length and therefore Table 2.
  Cut the sample count first, the effort level last.

---

### 1.8 Environment split: three roles, two stacks

| Stack | Roles | Why separate |
|---|---|---|
| **llama.cpp** (`llama-server`, GGUF) | **Victim** `Qwen3.8-27B` Q5_K_M | Highest bit-width that fits; free text-only mode via the separate `mmproj`. No Python deps at all — a standalone binary, so nothing can conflict. |
| **transformers / TRL / peft / bitsandbytes** (safetensors) | **Inversion** and **Student** training; also fine for serving the small **surrogate** and **compression** models | Training must be TRL, which means the safetensors path. |
| *(optional third)* **vLLM** | batched surrogate generation, or victim if you prefer vLLM over llama.cpp | vLLM hard-pins `torch` and will fight the training env. **Never install it into the training venv.** |

Concretely: one `uv` venv for training (§4.1), the `llama.cpp` binary outside Python entirely, and —
only if you want it — a second venv for vLLM. The victim and the trainers never run at the same time
anyway (you generate traces, *then* you train), so the 24 GB is not contended.

---

## 2. Qwen3.5 inventory (verified)

Fetched from the collection page then confirmed per-repo via the HF API and each `config.json`.
**Qwen3.5 does not mirror Qwen2.5 in any important way.**

Family-wide, verified from `Qwen/Qwen3.5-4B/config.json` and `Qwen/Qwen3.5-9B/config.json` and the
transformers `qwen3_5` docs:

- **All variants are natively multimodal.** Task tag `image-text-to-text`, class
  `Qwen3_5ForConditionalGeneration`, a `vision_config` in every config. Even the "text" models carry
  a vision tower.
- **Hybrid 3:1 attention.** Three Gated DeltaNet (linear attention) layers per one Gated Attention
  (full attention) layer, `full_attention_interval: 4`. The text backbone reuses Qwen3-Next's
  linear-attention decoder; the vision tower reuses Qwen3-VL's encoder.
- **Vocab 248,320** across the whole family — 63% larger than Qwen2.5's 152,064. This drives the
  single biggest training-memory decision (§3.4).
- **262,144 native context**, extensible to ~1 M via YaRN.
- **Instruct variants think by default**, `<think>\n…\n</think>\n\n`, disabled with
  `enable_thinking=False` (which injects an empty `<think>\n\n</think>\n\n` block).
- **`-Base` variants ship no `chat_template.jinja`** (verified via file listing) → no thinking
  format, no instruction following. Base models cannot be surrogates.
- `mtp_num_hidden_layers: 1` — every model carries a multi-token-prediction head.
- All Apache-2.0, all ungated.

| Repo | Params | Dense/MoE | Variant | Thinking? | Ctx | License | safetensors |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 873.4 M | dense | instruct | **yes** | 262 k | apache-2.0 | 1.75 GB |
| `Qwen/Qwen3.5-0.8B-Base` | 873.4 M | dense | base | no (no chat template) | 262 k | apache-2.0 | — |
| `Qwen/Qwen3.5-2B` | 2.274 B | dense | instruct | **yes** | 262 k | apache-2.0 | 4.55 GB |
| `Qwen/Qwen3.5-2B-Base` | 2.274 B | dense | base | no | 262 k | apache-2.0 | — |
| `Qwen/Qwen3.5-4B` | 4.660 B | dense | instruct | **yes** | 262 k | apache-2.0 | **9.32 GB** |
| `Qwen/Qwen3.5-4B-Base` | 4.660 B | dense | base | no | 262 k | apache-2.0 | — |
| `Qwen/Qwen3.5-9B` | 9.653 B | dense | instruct | **yes** | 262 k | apache-2.0 | **19.31 GB** |
| `Qwen/Qwen3.5-9B-Base` | 9.653 B | dense | base | no | 262 k | apache-2.0 | — |
| `Qwen/Qwen3.5-27B` (+`-FP8`, +`-GPTQ-Int4`) | 27.781 B | dense | instruct | **yes** | 262 k | apache-2.0 | ~55.6 GB bf16 |
| `Qwen/Qwen3.5-35B-A3B` (+`-FP8`, `-GPTQ-Int4`, `-Base`) | 35.95 B total / **A3B ≈ 3 B active** | **MoE** (`qwen3_5_moe`) | instruct + base | **yes** | 262 k | apache-2.0 | ~72 GB bf16 |
| `Qwen/Qwen3.5-122B-A10B` (+`-FP8`, `-GPTQ-Int4`) | 122 B / **10 B active** | MoE | instruct | yes | 262 k | apache-2.0 | out of reach |
| `Qwen/Qwen3.5-397B-A17B` (+`-FP8`, `-GPTQ-Int4`) | 397 B / **17 B active** | MoE | instruct | yes | 262 k | apache-2.0 | out of reach |
| `Qwen/Qwen3.8-27B` (+`-FP8`) | 27.781 B | dense, `qwen3_5` arch | instruct | **yes**, + `reasoning_effort` | 262 k | apache-2.0 | 55.6 GB bf16 |
| `Qwen/Qwen3.8-2.4T-A95B` (+`-FP8`) | 2.446 T / 95 B active | MoE `qwen3_5_moe_text` | instruct | yes | — | **license: other** | out of reach |

Also noted: `Qwen/Qwen3.6-27B` shares the `qwen3_5` `model_type` and loads with the same classes.
FP8 and GPTQ-Int4 official builds exist **only** for 27B and above — there is no
`Qwen/Qwen3.5-9B-FP8` (404 verified).

**Which Qwen3.5 variants can be the surrogate?** Every non-`-Base` variant, ≥0.8 B, emits real
`<think>` traces natively. `Qwen3.5-2B` is the sensible small surrogate if you want a Qwen3.5-family
one. But the paper's `R1-Distill-Qwen-1.5B` is a better fit — see §3.

### 2.1 Four reasons Qwen3.5 is the riskier inversion backbone

The user leans Qwen3.5, and `Qwen3.5-4B` is a viable primary — but be clear-eyed that it buys
nothing for *this* research question and costs four risks the Qwen2.5 fallback does not have:

1. **Dead-weight vision tower** (~0.55 GB on the 4B) for a pure-text task, plus the
   `AutoProcessor`/VLM-detection trap in §3.3.
2. **248 k vocab** — worst case for the LM-head logits tensor. Survivable only because TRL's chunked
   cross-entropy exists (§3.4).
3. **Gated DeltaNet needs optional kernels.** `causal_conv1d` and `fla` are optional; without them
   transformers **silently** falls back to slower, more memory-hungry PyTorch ops. No error, just a
   mystery slowdown.
4. **QLoRA on `qwen3_5` via bitsandbytes NF4 is UNVERIFIED.** I found no confirmation that bnb 4-bit
   quantisation + PEFT has been exercised on the DeltaNet projections. Run the smoke test in §3.9
   before committing.

Gate: if the §3.9 smoke test does not produce a decreasing loss within 20 steps, switch the inversion
model to `Qwen/Qwen2.5-7B-Instruct` and move on. The paper's contribution is the pipeline, not the
backbone.

---

## 3. Alternatives compared

| Family | Long CoT natively? | Small enough? | Ungated? | TRL support | Verdict |
|---|---|---|---|---|---|
| **DeepSeek-R1-Distill-Qwen 1.5B / 7B / 14B** | **Yes** — purpose-built `<think>` reasoners | 1.78 B / 7.62 B / 14.77 B | Yes, MIT | Perfect — plain `qwen2` `AutoModelForCausalLM` | **Wins the surrogate role outright.** The 1.5B *is* the paper's R1-Weak, so using it is a fidelity win, not a compromise. 3.5 GB, fast to batch-generate 10 k traces. |
| **Qwen2.5** (`2.5-1.5B/3B/7B-Instruct`) | No — no thinking mode | Yes | Yes (3B is `license: other`, non-commercial; 1.5B/7B Apache-2.0) | Best-in-class. `qwen2`, 152 k vocab, no VLM, no exotic attention, every TRL path exercised against it | **Wins inversion-fallback and student-primary.** 7B-Instruct is the paper's exact inversion backbone *and* its exact student. Boring is the feature. |
| **Qwen3 / Qwen3-2507** (`Qwen3-4B-Thinking-2507`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B`) | Thinking-2507 yes; Instruct-2507 no | 4.0 B / 8.2 B | Yes, Apache-2.0 | Clean `qwen3` causal LM. TRL bundles a `{% generation %}` training template for Qwen3 specifically | **Strong dark horse.** `Qwen3-4B-Thinking-2507` is a text-only thinking model with no vision tower, no DeltaNet, and a 151 k vocab — everything Qwen3.5-4B is, minus all four risks. If Qwen3.5 fails the smoke test, this beats Qwen2.5-7B for the inversion role. |
| **Llama-3.x** (`Llama-3.1-8B-Instruct`) | No | 8.03 B | **No — GATED**, `license: llama3.1` | Excellent | **Keep as the cross-family student arm** (the paper's second student). Gating is a setup prerequisite, not a blocker. |
| **Phi-4-reasoning** | Yes, reasoning-tuned | **14.66 B — too big** | Yes, MIT | Fine (`phi3`) | Loses on size. 14.7 B QLoRA at 8–16 k on 24 GB leaves no margin, and it brings nothing the 1.5B surrogate doesn't. |
| **SmolLM3-3B** | Yes, has a thinking mode | 3.08 B | Yes, Apache-2.0 | Very good — TRL's docs literally cite its chat template as the reference `{% generation %}` implementation | Viable small surrogate or a genuinely cross-family student. Loses to R1-Distill-1.5B on surrogate (not paper-faithful) and to Qwen2.5-7B on student (weaker baseline → less headroom to show improvement). |
| **OLMo-2-7B-Instruct** | No | 7.30 B | Yes, Apache-2.0 | Fine (`olmo2`) | **The best ungated cross-family student substitute** if you refuse to accept Llama's gate. Genuinely different lineage and fully open. Downside: much less-trodden path, and weaker baselines than Qwen2.5-7B. |

---

## 4. TRL specifics

> ### ⚠️ Corrected 2026-08-26 — read before copying anything in §4.5–4.7
>
> The VRAM figures in §4.7 were estimates marked **UNVERIFIED**. They have now been measured
> (`08` §6, `12` §1) and two recommendations below are superseded:
>
> | Was | Now | Why |
> |---|---|---|
> | Inverter: **QLoRA** (NF4 + `bitsandbytes`) | **LoRA on a bf16 base** | Measured 15.11 GiB @8k. NF4 is not needed, so gotcha 9 ("QLoRA on `qwen3_5` is UNVERIFIED") becomes moot rather than unresolved — and the model that must learn a new behaviour is not quantized. |
> | `max_length` **8192** to start | inverter **12288**, student **16384** | Inverter prompt is ~1,524 tokens, so a cap-8192 trace needs ~9,716 (`12` §3). Student FFT at the paper's own 16384 measures 15.79 GiB, and the inverter at 12288 measures 18.30 GiB — no deviation for the student. |
>
> Confirmed as written: `chunked_nll` is load-bearing (naive fp32 logits are 30.31 GiB at 16k),
> `processing_class=AutoTokenizer(...)` is mandatory, and `Qwen3_5ForCausalLM` loads text-only.
> **DeltaNet forward + backward is verified working on this 4090** — that was the largest
> unverified risk here.

### 4.1 Version pins

`transformers` shipped a **v5**, `trl` shipped a **v1** — memory of the old API is actively wrong here.

Verified latest on PyPI as of 2026-08-15:

| Package | Latest | Note |
|---|---|---|
| `trl` | **1.10.0** | requires `python>=3.10`, `transformers>=4.56.2`, `accelerate>=1.4.0`, `datasets>=4.7.0` |
| `transformers` | **5.15.0** | TRL 1.10 docs cross-link v5.15.0 — this is the intended pair |
| `peft` | **0.20.0** | TRL 1.10 docs cross-link v0.20.0 |
| `datasets` | **5.0.1** | |
| `accelerate` | **1.14.0** | |
| `bitsandbytes` | **0.50.1** | ships CUDA 11.8 / 12.x / 13.x builds; `python>=3.10` |
| `torch` | **2.13.0** | |
| `vllm` | **0.27.1** | separate venv — see §4.10 |
| `trackio` | **0.35.0** | optional, `report_to="trackio"` |

Project is on **Python 3.13** with an empty `uv` dependency list — greenfield, nothing installed yet.
All of the above support 3.13.

```toml
# pyproject.toml — training env
dependencies = [
  "torch==2.13.0",
  "transformers==5.15.0",
  "trl==1.10.0",
  "peft==0.20.0",
  "bitsandbytes==0.50.1",
  "accelerate==1.14.0",
  "datasets==5.0.1",
  "trackio>=0.35.0",
]
```

Install torch from the CUDA 12.8/13.0 index matching your driver (595.84 supports both). Run
`trl env` afterwards — it prints the whole resolved stack and is what you paste into any bug report.

Do **not** put `vllm` in this env; it hard-pins torch and will fight the training stack. See §1.8 for
the full stack split — the victim runs on the llama.cpp binary, entirely outside Python.

### 4.2 Current SFT API surface

```python
from trl import SFTTrainer, SFTConfig      # both from the top-level `trl` package
from peft import LoraConfig
from transformers import BitsAndBytesConfig, AutoTokenizer
```

Signature (TRL 1.10.0, verified against `trl/trainer/sft_trainer.py`):

```python
SFTTrainer(
    model, args=None, data_collator=None, train_dataset=None, eval_dataset=None,
    processing_class=None, compute_loss_func=None, compute_metrics=None, callbacks=None,
    optimizers=(None, None), optimizer_cls_and_kwargs=None, preprocess_logits_for_metrics=None,
    quantization_config=None,     # ← BitsAndBytesConfig, applied when `model` is a string
    peft_config=None,             # ← PeftConfig
    formatting_func=None,
)
```

`quantization_config` as a direct trainer argument is new — you no longer have to hand-build the
model just to do QLoRA. It is **ignored if you pass an already-instantiated model**.

SFTConfig defaults that differ from `TrainingArguments` and will bite you:

| Param | SFTConfig default | Note |
|---|---|---|
| `gradient_checkpointing` | **`True`** | already on; don't "enable" it and think you changed something |
| `bf16` | **`True`** (if `fp16` unset) | |
| `learning_rate` | `2e-5` | too low for LoRA — use `1e-4`–`2e-4` |
| `max_length` | **`1024`** | **must raise**; silently truncates reasoning traces otherwise |
| `logging_steps` | `10` | |
| `packing` | `False` | |
| `completion_only_loss` | `None` → `True` for prompt-completion datasets | what we want |
| `loss_type` | `None` → `"chunked_nll"` | the memory saver, see §3.4 |
| `use_cache` | `False` | |
| `optim` | `"adamw_torch_fused"` | |
| `report_to` | `"none"` | |

`DataCollatorForCompletionOnlyLM` is **removed**. Use `completion_only_loss` in `SFTConfig`.

### 4.3 Dataset format

TRL accepts four shapes. For this task the correct one is **conversational prompt-completion**,
because it (a) applies the chat template for you and (b) turns on `completion_only_loss` by default,
so loss lands only on the trace.

```
{"prompt": [{"role": "user", "content": "..."}],
 "completion": [{"role": "assistant", "content": "..."}]}
```

**Inversion-model training** (`(x, a, s) → z`, paper Eq. 1). One JSONL row:

```json
{
  "prompt": [
    {"role": "system", "content": "You are given a problem, a final answer, and a short reasoning summary. Reconstruct the detailed step-by-step reasoning trace that leads from the problem to the answer. Output only the trace."},
    {"role": "user", "content": "## Problem\nLet f(x)=x^3-3x+1. How many real roots does f have?\n\n## Final answer\n3\n\n## Reasoning summary\n**Setting up.** I considered the derivative to locate critical points, then checked the sign of f at those points to count sign changes.\n\n**Conclusion.** Three real roots."}
  ],
  "completion": [
    {"role": "assistant", "content": "Okay, so I need to find how many real roots f(x) = x^3 - 3x + 1 has. Let me start by taking the derivative. f'(x) = 3x^2 - 3 = 3(x^2 - 1), so critical points at x = -1 and x = 1. Now let me evaluate f at these points. f(-1) = -1 + 3 + 1 = 3, and f(1) = 1 - 3 + 1 = -1. So we have a local max of 3 at x = -1 and a local min of -1 at x = 1. Since the local max is positive and the local min is negative, and a cubic goes to -inf as x -> -inf and +inf as x -> +inf... wait, let me double check that. Yes, leading coefficient is positive so that's right. So the graph crosses zero once before x = -1, once between -1 and 1, and once after x = 1. That gives three real roots."}
  ]
}
```

For the **no-summary** setting, drop the `## Reasoning summary` block from the user turn and train a
second, separate inversion model (the paper trains one per setting — do not share).

**Student training** (`x → (ẑ, a)`, paper Eq. 3 — the trace and answer are concatenated into a
*single* supervision target):

```json
{
  "prompt": [
    {"role": "user", "content": "Let f(x)=x^3-3x+1. How many real roots does f have?"}
  ],
  "completion": [
    {"role": "assistant", "content": "<think>\nOkay, so I need to find how many real roots f(x) = x^3 - 3x + 1 has. Let me start by taking the derivative...\n</think>\n\nThe function has **3** real roots."}
  ]
}
```

Notes:

- Loss is computed on the `completion` only, automatically. Do not set `completion_only_loss` by hand
  unless you want to override.
- Column names must be exactly `prompt` and `completion`. Extra columns are dropped
  (`remove_unused_columns=True`).
- The paper's ablation mixes surrogate data into the student corpus — that is just
  `datasets.concatenate_datasets`, or the CLI's `datasets:` mixture list.
- TRL also accepts pre-tokenized datasets keyed on `input_ids` with an optional `completion_mask`.
  Only reach for that if the chat-template round-trip (below) proves unmanageable.

### 4.4 The `<think>` round-trip trap

Verified from `Qwen/Qwen3.5-4B/chat_template.jinja` (and the identical structure in Qwen3.8-27B):

```jinja
{%- if '</think>' in content %}
    {%- set reasoning_content = content.split('</think>')[0]...split('<think>')[-1]... %}
    {%- set content = content.split('</think>')[-1].lstrip('\n') %}
...
{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
...
{%- if add_generation_prompt %}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}          ← the prompt already opens the tag
```

Two concrete consequences:

- **The generation prompt already emits an opening `<think>\n`.** If your completion string also
  starts with `<think>`, you train the model to emit `<think><think>`. Either strip the opening tag
  from the completion, or verify what `apply_chat_template` produced before trusting it.
- **Assistant content containing `</think>` is parsed and re-emitted**, not passed through verbatim.
  Round-trip one example and diff the string before launching a multi-hour run.

Related: the Qwen3.5 chat template contains **no `{% generation %}` / `{% endgeneration %}` markers**
(verified — the only `generation` hit is `add_generation_prompt`). TRL auto-patches templates for
"known model families (e.g. Qwen3)"; whether `qwen3_5` is on that bundled list is **UNVERIFIED**. This
does not affect us — we use prompt-completion + `completion_only_loss`, not `assistant_only_loss`.
Just do not reach for `assistant_only_loss=True` expecting it to work.

### 4.5 QLoRA wiring

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3.5-4B"

# SUPERSEDED — bf16 LoRA fits (15.11 GiB @8k measured), so NF4 is not needed. Keep this only
# if you want max_length 16384 on the inverter, where bf16 LoRA is 21.47 GiB and too tight.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
# )

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,               # 2*r is the usual QLoRA ratio
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Explicit list, NOT "all-linear" — see the two gotchas below.
    target_modules=[
        # full_attention layers (16 of 32)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Gated DeltaNet / linear_attention layers (the other 24)
        "in_proj_qkv", "in_proj_z", "out_proj",
        # MLP
        "gate_proj", "up_proj", "down_proj",
    ],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)   # ← NOT AutoProcessor. See §3.3.

args = SFTConfig(
    output_dir="out/inversion-qwen35-4b",
    model_init_kwargs={
        "dtype": torch.bfloat16,                    # ← required, see gotcha 6
        "attn_implementation": "kernels-community/flash-attn2",
    },
    max_length=12288,                               # inverter: ~1,524 prompt + 8,192 trace = 9,716
    packing=False,                                  # see §3.6
    completion_only_loss=True,
    loss_type="chunked_nll",                        # the whole ballgame at 248k vocab
    gradient_checkpointing=True,                    # already the default
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,                             # paper's value
    learning_rate=1e-4,                             # NOT the paper's 1e-5 — that's for full FT
    warmup_ratio=0.1,                               # paper's value
    optim="adamw_8bit",
    bf16=True,
    logging_steps=5,
    save_strategy="steps", save_steps=200, save_total_limit=2,
    dataset_num_proc=2,                             # see gotcha 7 (RAM)
    report_to="trackio",
)

trainer = SFTTrainer(
    model=MODEL,
    args=args,
    train_dataset=load_dataset("json", data_files="data/inversion_summary.jsonl", split="train"),
    processing_class=tokenizer,                     # ← the single most important line here
    # quantization_config=bnb_config,             # not needed — bf16 LoRA fits
    peft_config=peft_config,
)
trainer.train()
```

**On `target_modules`.** Module names verified from
`transformers/models/qwen3_5/modeling_qwen3_5.py` on `main`:

- full-attention block: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Gated DeltaNet block: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `conv1d`
- MLP: `gate_proj`, `up_proj`, `down_proj`
- vision tower (excluded by using text-only): `qkv`, `proj`, `linear_fc1`, `linear_fc2`

Deliberately excluded:

- **`in_proj_b` / `in_proj_a`** — these project `hidden_size → num_v_heads` (i.e. 2560 → 32). A rank-32
  adapter on a 32-wide output is degenerate: it adds parameters and buys nothing.
- **`conv1d`** — an `nn.Conv1d`, not a `Linear`; PEFT LoRA won't touch it anyway.
- **`lm_head`** — mandatory exclusion, see gotcha 2.

`r=32 / alpha=64` is the standard QLoRA setting and a good default. For an inversion model that must
learn a genuinely new output *style* (short input → 5 k-token trace) you may want more capacity;
TRL's own LoRA guidance uses `r=256` with `target_modules="all-linear"` for reasoning SFT. If you go
above `r=64`, re-check VRAM and keep `lm_head` out of the target list regardless.

### 4.6 Long sequences

| Knob | Setting | Why |
|---|---|---|
| `max_length` | inverter **12288**, student **16384** (the paper's `cutoff_len`) | Measured, `12` §1/§3. The default 1024 silently truncates every trace. The inverter's prompt is ~1,524 tokens, so 8192 would cut the tail off ~a quarter of cap-8192 traces — and `keep_start` throws away the *end*, i.e. the conclusion. |
| `truncation_mode` | `"keep_start"` (default) | `"keep_end"` is deprecated, removed in v2.0.0. |
| `loss_type` | `"chunked_nll"` (default) | Non-negotiable at 248 k vocab. See §3.4. |
| `packing` | **`False`** | See below. |
| `padding_free` | `True` **only with FA2/FA3** | With `batch_size=1` it buys nothing; enable if you raise the batch size. Requires `attn_implementation="kernels-community/flash-attn2"` or you get batch contamination. |
| `gradient_checkpointing` | `True` (default) | |
| `activation_offloading` | `True` **only if OOM** | Moves activations to CPU RAM during forward. You have ~26 GB usable RAM — plausible but thin. Costs wall time. |
| `use_liger_kernel` | leave `False` | Mutually exclusive with `chunked_nll`, and Liger support for `qwen3_5` is **UNVERIFIED**. `chunked_nll` is the safer memory win. |
| `pad_to_multiple_of` | unset | |

**Why `packing=False`.** Packing fills each row to `max_length` by bin-packing multiple examples.
Your examples are already 5–8 k tokens against an 8 k window, so there is almost nothing to pack —
you get near-zero benefit and inherit real risks: the default `"bfd"` strategy **discards overflow
tokens** (truncating your longest, most informative traces), and BFD packing force-enables
padding-free mode regardless of your setting. Turn packing on only if you shorten `max_length` below
your typical example length, which you don't want to do here.

**Flash attention.** Prefer the hub kernel `attn_implementation="kernels-community/flash-attn2"` over
`pip install flash-attn` — no source build, no torch-ABI roulette on Python 3.13. Note only the 16
full-attention layers use it; the 48 DeltaNet layers take their own path.

**DeltaNet kernels.** `causal_conv1d` and `fla` are optional and, if absent, transformers **silently**
falls back to slower and more memory-hungry PyTorch ops — no warning, just a mysteriously slow run.
The 4090 is Ada SM89, which both packages do build for (unlike GB10/SM121). If they won't install,
`from_pretrained(..., use_kernels=True)` swaps in the hub kernel `Atlas-Inference/gdn`, currently
requiring `trust_remote_code=True`. Not correctness-critical either way — measure step time and
decide.

### 4.7 Why `chunked_nll` is the whole ballgame

248,320-token vocab at `max_length=8192`, batch 1, standard `"nll"`:

```
logits  8192 × 248,320 × 4 B (fp32 CE upcast) =  8.1 GB
grad    same                                  =  8.1 GB
                                                -------
                                                16.2 GB — before weights or activations. OOM.
```

`loss_type="chunked_nll"` (the default) drops `labels == -100` positions *before* the `lm_head`
matmul and chunks the cross-entropy. `_CHUNKED_LM_HEAD_CHUNK_SIZE = 256` (verified in
`trl/trainer/sft_trainer.py:86`):

```
256 × 248,320 × 4 B = 254 MB, roughly ×2 with grad ≈ 0.5 GB
```

**A ~32× reduction in the dominant activation.** TRL's docs report ~30–50% peak-VRAM reduction on a
151 k-vocab model; at 248 k with `completion_only_loss` also masking the prompt, the win here is
larger. This single default is what makes Qwen3.5 trainable on a 4090 at all.

Resulting budget, `Qwen3.5-4B` QLoRA @ 8 k (estimates, **UNVERIFIED** until measured):

| Component | GB |
|---|---|
| NF4 quantized linears | 2.0 |
| `embed_tokens` bf16 (tied, 636 M params) | 1.3 |
| LoRA r=32 params + 8-bit Adam states | 0.5 |
| Checkpointed activations, 32 layers × 8192 × 2560 | 1.7 |
| Chunked CE + workspace | 0.5 |
| Fragmentation / allocator slack | 1.5 |
| **Total** | **~7.5 GB** |

Comfortable. 16 k roughly doubles activations → ~11–12 GB, still fine.
`Qwen2.5-7B-Instruct` QLoRA @ 8 k lands around 9–10 GB (152 k vocab, 3584 hidden, untied
embed+head at 2.2 GB bf16), ~14 GB at 16 k. Both fit.

### 4.8 CLI vs Python API

**Use the Python API.** `trl sft` exists and is real (`trl sft --config sft.yaml`, plus
`trl dpo/grpo/kto/reward/rloo/distillation`, `trl env`, `trl vllm-serve`, `--accelerate_config
single_gpu`, and YAML `datasets:` mixtures). It is genuinely good for standard runs. But this task
needs, per-run: a hand-built `BitsAndBytesConfig`; an explicit `target_modules` list including the
DeltaNet projections; and — decisively — **`processing_class=AutoTokenizer(...)` to defeat VLM
auto-detection** (§3.3), which has no CLI flag. Two of those three are not expressible in YAML.

Do keep `trl env` in your bug-report ritual, and `trl vllm-serve` is a reasonable alternative to
plain `vllm serve` for the victim/surrogate generation passes.

### 4.9 Top gotchas, ranked

1. **Qwen3.5/3.8 will be misdetected as a VLM.** `SFTTrainer` sets `self._is_vlm = True` purely from
   `isinstance(processing_class, ProcessorMixin)` (verified, `sft_trainer.py:989-994`). When
   `processing_class is None` it calls `AutoProcessor.from_pretrained(...)`. Every Qwen3.5 repo ships
   `preprocessor_config.json`, so `AutoProcessor` returns a **ProcessorMixin** → your pure-text job
   silently takes the VLM path: `DataCollatorForVisionLanguageModeling`, `skip_prepare_dataset`
   forced on, and **`chunked_nll` disabled for VLMs** (the one thing you cannot afford). **Fix: always
   pass `processing_class=AutoTokenizer.from_pretrained(MODEL)` explicitly.**
2. **`chunked_nll` hard-errors if `lm_head` is a LoRA target.** Verified at `sft_trainer.py:1313-1322`
   — it raises when `get_output_embeddings()` is a `BaseTunerLayer`, because the chunked path reads
   the output-projection weight directly and would silently drop the adapter delta. TRL's own error
   text names `target_modules="all-linear"` as a way to trigger this. (Note: TRL's *docs* claim
   chunked_nll is "not compatible with PEFT" outright — the **source is narrower and correct**; it
   works fine with PEFT as long as `lm_head` isn't wrapped.) **Fix: explicit `target_modules` list,
   never `"all-linear"`, never `"lm_head"`.**
3. **`max_length` defaults to 1024** and truncates from the start, keeping the prompt and throwing
   away the trace you are trying to learn. Silent. Set it explicitly, every time.

Then, in descending order:

4. **`learning_rate` defaults to `2e-5`.** The paper's `1e-5` is for *full* fine-tuning on 8×A100. For
   LoRA/QLoRA use `1e-4`–`2e-4`; TRL's docs say ~`1e-4`. Keep the paper's `num_train_epochs=3` and
   `warmup_ratio=0.1`.
5. **The `<think>` round-trip** (§3.3a above) — double tags and silent content reparsing.
6. **Model loaded from a string defaults to `float32`**, not the config dtype. TRL documents this as a
   deliberate divergence from transformers v5 behaviour. On a 4090 that is an instant OOM. Always set
   `model_init_kwargs={"dtype": torch.bfloat16}`.
7. **System RAM (~26 GB usable) is the second cliff.** `dataset_num_proc=8` with 8–16 k-token examples
   forks the tokenized arrow table per worker. Keep it at 2. Likewise `activation_offloading=True`
   pushes activations into that same 26 GB. And never let the bf16 27 B victim (55.6 GB) touch this
   box — download a pre-quantized W4A16 checkpoint, do not quantize locally.
8. **DeltaNet silent-fallback** (§3.6) — no error, just slow.
9. **QLoRA on `qwen3_5` is UNVERIFIED** — run the smoke test.

### 4.10 Smoke test before any long run

```bash
uv run python -c "
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, BitsAndBytesConfig
import torch
m = Qwen3_5ForCausalLM.from_pretrained('Qwen/Qwen3.5-4B',
      quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
      dtype=torch.bfloat16, device_map={'':0})
print(type(m).__name__, m.get_memory_footprint()/1e9, 'GB')
print([n for n,_ in m.named_modules() if n.endswith(('in_proj_qkv','in_proj_z','out_proj'))][:6])
"
```

Confirms in one shot: (a) `Qwen3_5ForCausalLM` loads without the vision tower, (b) bnb NF4 accepts
the DeltaNet projections, (c) the module names match your `target_modules`. Then run the real config
with `max_steps=20` and watch loss decrease and `nvidia-smi` peak. If either fails, switch the
inversion backbone to `Qwen/Qwen3-4B-Thinking-2507` (text-only, no DeltaNet, 151 k vocab) or
`Qwen/Qwen2.5-7B-Instruct`, and lose nothing that matters to the paper.

---

## 5. Setup prerequisites checklist

- [ ] **`meta-llama/Llama-3.1-8B-Instruct` is GATED.** Accept the license at
      <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>, then `hf auth login`. Only needed for
      the cross-family student arm; substitute `allenai/OLMo-2-1124-7B-Instruct` (Apache-2.0, ungated)
      to skip this.
- [ ] `Qwen/Qwen2.5-3B-Instruct` is ungated but `license: other` (Qwen Research, non-commercial).
      Prefer the Apache-2.0 1.5B or 7B.
- [ ] Everything else recommended — Qwen3.5-*, Qwen3.8-27B, DeepSeek-R1-Distill-*, Qwen2.5-7B-Instruct,
      Qwen3-4B-Thinking-2507, SmolLM3-3B, OLMo-2 — is **ungated** (verified).
- [ ] **Stack split (§1.8):** llama.cpp binary for the victim (outside Python); one `uv` venv for
      TRL training; optional third venv for vLLM. Never install `vllm` into the training venv.
- [ ] Pull `unsloth/Qwen3.8-27B-GGUF:Q5_K_M` **without** `mmproj-*.gguf`, and confirm `llama-server`
      starts with `--jinja` and no `--mmproj`.
- [ ] Calibrate `reasoning_effort` (§1.6) on 100 samples before the bulk generation run — pick the
      level whose median `<think>` length is nearest the paper's 6,130-token reference.
- [ ] Run the §4.10 QLoRA smoke test before committing to Qwen3.5 as the inversion backbone.
- [ ] Budget disk: ~54 GB for the recommended set. Skip the bf16 27 B download.
