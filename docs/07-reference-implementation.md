# Reference Implementation — What the Released Code Actually Says

Source: `github.com/Tingwei-Zhang/Trace_Inversion_Attack` @ `main`, cloned 2026-08-15.
Apache-2.0. Last push **2026-03-06** — i.e. **v1 of the paper**, two months before v2 (2026-05-12).

This doc records what the code specifies that the paper does not, and — more importantly — the
places where **the code and paper v2 disagree**. Several items the other docs list as
"underspecified" are in fact pinned down here.

---

## 1. Gaps the code closes

### 1.1 Decoding parameters (docs `01` §underspecified-3 and `02` list these as missing)

Every generation step in the pipeline uses the **same** sampling config, via LLaMA-Factory's
`vllm_infer.py`:

| Parameter | Value |
|---|---|
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `repetition_penalty` | 1.05 |
| `max_new_tokens` | **8192** |
| `cutoff_len` | 16384 |
| `max_samples` | 20000 |

vLLM settings: `tensor_parallel_size: 2`, `gpu_memory_utilization: 0.9`,
`max_num_batched_tokens: 32768`, `max_num_seqs: 256`, `pipeline_parallel_size: 1`.

Confirmed identical in `step0_data_preprocess/r1_distill_inference.sh` (surrogate trace generation),
`step1_summarization/qwen2_5_summarization_r1.sh` (compression), and the `DEFAULTS` dict in
`step2_inversion/evaluation/run_inversion_eval.py` (inversion).

> **`max_new_tokens: 8192` is a hard ceiling on trace length.** Table 2's headline lengths are
> 4,972 / 5,434 / 5,767 / 6,021 tokens against an R1 ground-truth mean of 6,130.6. All sit under
> 8192, so the cap is rarely binding — but it *is* the reason no synthesized trace can exceed 8192,
> and the "81–89% length recovery" figure must be read with that ceiling in mind.

Generation is **sampled, not greedy**, at temperature 0.7, with **one sample per input** and **no
seed set anywhere**. Combined with the paper reporting no variance, this means every number in the
paper is a single stochastic draw.

### 1.2 Training hyperparameters (paper gives only 4 of these)

Both stages use LLaMA-Factory, full-parameter SFT, DeepSpeed ZeRO-3:

| Setting | Inversion model | Student |
|---|---|---|
| `finetuning_type` | **`full`** (not LoRA) | **`full`** |
| `deepspeed` | `ds_z3_config.json` | `ds_z3_config.json` |
| `enable_liger_kernel` | `true` | `true` |
| `packing` / `neat_packing` | `true` / `false` | `true` / `false` |
| `per_device_train_batch_size` | **1** | **4** |
| `gradient_accumulation_steps` | 3 | 3 |
| **effective batch (8 GPUs)** | **24** | **96** |
| `learning_rate` | 1.0e-5 | 1.0e-5 |
| `num_train_epochs` | 3.0 | 3.0 |
| `lr_scheduler_type` | **cosine** | **cosine** |
| `warmup_ratio` | 0.1 | 0.1 |
| `bf16` | true | true |
| `gradient_checkpointing` | true | true |
| `cutoff_len` / `max_samples` | 16384 / 10000 | 16384 / 10000 |

Two of these need care when porting to TRL:

- **`packing: true` with `neat_packing: false`** is *naive* packing — multiple examples are
  concatenated into one 16,384-token sequence with no attention-mask isolation, so examples attend
  across each other's boundaries. TRL's `packing=True` behaves differently by version; if you want
  to match the reference, you are matching a slightly leaky implementation. Matching TRL's
  *correct* packing is a deliberate deviation worth logging.
- **Effective batch 24 vs 96** — the inversion model and the student were *not* trained at the same
  batch size. Any single-GPU reproduction must hit these via `gradient_accumulation_steps`, not
  silently collapse both to one value.

### 1.3 The OpenAI victim call

`step0_data_preprocess/chatgpt_inference.py`:

```python
model="gpt-5-mini-2025-08-07",
reasoning={"effort": "medium", "summary": "auto"},
```

So: **`reasoning_effort` is `medium`**, not high, and summaries are requested with `summary: "auto"`.
There is a separate `..._high_effort` eval preset, implying an unreported effort ablation.
This also confirms the summary-setting experiments depend on the API returning
`response.output[].summary` — the paper's acknowledgement that *empty* summaries occur is visible
here as the `reasoning_summary = ""` fallback, with no special handling downstream.

---

## 2. Where the code and paper v2 disagree

### 2.1 The victim model is different

| | Model |
|---|---|
| Code (`chatgpt_inference.py`) | `gpt-5-mini-2025-08-07` |
| Paper v2 | `gpt-5.4-mini-2026-03-17` |

The v2 experiments were re-run against a newer victim; the code was never updated. **The repo
reproduces v1's numbers, not the ones in the paper.** The HF paper page still shows the v1 abstract
with different figures.

### 2.2 The compression prompt is not just different — it is the opposite

This is the most consequential divergence, and it is easy to miss.

**Code (v1)** — `step1_summarization/data_formatter.py` produces *numbered reasoning bubbles*:

> You are a model trained to convert informal internal reasoning into a clear, structured sequence
> of "reasoning bubbles." […] You should produce **a few numbered reasoning bubbles** […]
> **Format:** `1. [bubble]` `2. [bubble]` `3. …`

**Paper v2** — Appendix B produces *first-person prose under bold headers*, and explicitly bans the
v1 format:

> Write 3 to 6 short sections. Each section begins with a short bold markdown header […]
> **Do NOT use numbered lists (1., 2.) and do NOT use bullet points (-, \*).**
> […] First person, present tense […] Aim for roughly 600–900 tokens total.

These produce structurally incompatible summaries. The v2 prompt exists because the authors
reverse-engineered GPT-5.4 mini's *actual* summary style (Table 1: bold-header sections 92.9%,
first-person 97.0%, median 592 tokens) — the v1 numbered-bubble format matches none of that.

**Consequence for us:** the v2 compression prompt is *only* in the paper, and its **two few-shot
exemplars are in neither** — Appendix B says they are "released with our code," but the released
code predates the prompt entirely. Those exemplars are unrecoverable. We must write our own and
validate against Table 1's four style statistics (median tokens, bold-header %, first-person %,
LaTeX %), which is the measurable acceptance test the paper hands us.

### 2.3 The paper v2 contradicts *itself* on summary format

Within Appendix B of v2:

- The **compression** prompt forbids numbered lists and emits bold-header prose.
- The **zero-shot inversion** prompt still says it will be given "A list of numbered **reasoning
  bubbles**, where each bubble summarizes one key insight," and instructs the model to "expand each
  bubble."

The inversion prompt is a v1 leftover that was never updated when the compression prompt was
rewritten. So in v2 the inversion model is told to expect a format the compressor is explicitly
forbidden from producing. This is a live bug in the published prompts, not a transcription error.

Since the zero-shot row is the paper's *baseline* (TF1 35.36), a mismatched prompt would depress it
— which flatters the fine-tuned rows. **Worth an explicit check in reproduction:** re-run the
zero-shot baseline with a format-matched prompt and see how much of the
"prompting alone is insufficient" gap survives.

---

## 3. Other repo facts worth knowing

- **Two pinned submodules:** LLaMA-Factory `9501c33`, Evalchemy `6ed6741`. Both SHAs resolve.
  Evalchemy has no LICENSE at that SHA.
- **README says Python 3.10; pinned LLaMA-Factory requires ≥3.11.** Use 3.11.
- **No quantized training anywhere** — `bitsandbytes` is an optional extra and every config is
  `finetuning_type: full` + bf16. Our QLoRA path is entirely our own.
- The dataset used is the **`llamafactory/OpenThoughts-114k` mirror**, whose schema
  (`system` + `conversations`) differs from canonical `open-thoughts/OpenThoughts-114k` (`messages`).
  Only the mirror parses with the repo's code.
- Likely bug: `chatgpt_inference.py` uses `start_index=20000` against a 20k-row file → zero rows on
  a fresh run.
- `preprocess_r1_distill.py` injects a fixed system prompt for surrogate inference ("Your role as an
  assistant involves thoroughly exploring questions through a systematic long thinking process…") —
  this is the standard R1-Distill system prompt and is part of what makes the surrogate emit long
  traces. Not mentioned in the paper.

---

## 4. Net effect on our reproduction

| Item | Status |
|---|---|
| Decoding params | ✅ Recovered from code — use temp 0.7 / top_p 0.9 / rep 1.05 / max_new 8192 |
| Training hyperparams | ✅ Recovered — full-FT @ eff. batch 24 (inversion) / 96 (student), cosine |
| Compression prompt (v2) | ⚠️ Paper only, **exemplars lost** — write our own, validate vs Table 1 |
| Inversion prompt | ⚠️ Published version is format-mismatched with the v2 compressor |
| Victim model | ⚠️ Code targets a different, older victim than the paper |
| Reference numbers | ⚠️ Repo reproduces **v1**, not the v2 tables in `02` |

The practical read: **take the code as a source of mechanical detail (decoding, batching, data
formatting) and the paper as the source of method and results — never assume they describe the same
experiment.**
