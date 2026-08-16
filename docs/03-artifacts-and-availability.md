# 03 — Artifacts & Availability

**Paper:** "How to Steal Reasoning Without Reasoning Traces" — Tingwei Zhang, John X. Morris, Vitaly Shmatikov (Cornell Tech)
**arXiv:** [2603.07267](https://arxiv.org/abs/2603.07267) — v1 2026-03-07, v2 2026-05-12, cs.CR
**Verification date:** 2026-08-15. Every "verified ✅" row below was hit live on this date.

Legend: ✅ verified live · ⚠️ exists but with a caveat · ❌ does not exist / not released · **UNVERIFIED** = could not confirm, do not treat as fact.

---

## 0. Executive summary

- **Code IS released.** `github.com/Tingwei-Zhang/Trace_Inversion_Attack`, Apache-2.0, public, 20 stars. Full 4-stage pipeline (step0–step3) with training YAMLs and eval presets.
- **No author-released model or dataset checkpoints.** Nothing on Hugging Face under the authors' names. Everything linked on the HF paper page is third-party.
- **Biggest reproducibility gap:** the released code corresponds to **v1** of the paper (victim = `gpt-5-mini-2025-08-07`), not **v2** (victim = `gpt-5.4-mini-2026-03-17`). Last push to the repo is 2026-03-06, two months *before* v2 appeared. The v2 numbers cannot be reproduced from the code as published without editing the model string.
- **Second-biggest gap:** the README's `python=3.10` conflicts with the pinned LLaMA-Factory submodule, which requires `>=3.11.0`. See §6.
- All four datasets and all four models exist and are downloadable today. Only Llama-3.1-8B-Instruct is gated.

---

## 1. Code

| Artifact | Exact id | Size | License | Verified? | Notes |
|---|---|---|---|---|---|
| Main repo | [`Tingwei-Zhang/Trace_Inversion_Attack`](https://github.com/Tingwei-Zhang/Trace_Inversion_Attack) | 59.9 MB | Apache-2.0 | ✅ HTTP 200 | Public. Created 2026-03-06, **last push 2026-03-06**, 20 ★, 4 forks, 1 open issue. 3 commits total. Python. |
| Submodule: trainer | `hiyouga/LLaMA-Factory` @ `9501c3308a01ecce03e952aadd10b509fa4e1411` | — | Apache-2.0 | ✅ SHA resolves | Commit dated 2026-03-06. Upstream repo has since been **renamed to `hiyouga/LlamaFactory`** (74k ★); the old URL still redirects, so `git submodule update` works. |
| Submodule: eval harness | `mlfoundations/Evalchemy` @ `6ed674159b37f740f2353a86f596f49f6ac13c19` | — | ⚠️ **no LICENSE file** | ✅ SHA resolves | Commit dated 2025-12-23. 604 ★. Repo has `pyproject.toml` + `requirements.txt` but **no LICENSE at that SHA and none on the repo record** — unlicensed dependency, flag for any redistribution. |

### Where the code link lives
The GitHub URL appears **only in the paper body** (§1, footnote 1: "To facilitate research on model-stealing attacks and defenses, we release our code"). It is **not** in the arXiv abstract, not in the arXiv comments field, and **not** on the HF paper page. Papers-with-Code no longer exists as a separate site — `paperswithcode.com` 301s to `huggingface.co/papers/trending` (verified).

### Completeness assessment: **good, but v1-only**

Present and looks runnable:

```
src/step0_data_preprocess/   download_dataset.py, chatgpt_inference.py,
                             preprocess_chatgpt_inference.py, preprocess_r1_distill.py,
                             r1_distill_inference.sh
src/step1_summarization/     data_formatter.py + 2 shell drivers (r1 / r1_distill)
src/step2_inversion/         format_data.py, 3 training YAMLs, 6 eval presets,
                             evaluate_similarity.py, run_inversion_eval.py
src/step3_reasoning/         format_data.py, generate_config.py, presets/, run_evaluation.py
cache/                       ds_z2 / ds_z2_offload / ds_z3 / ds_z3_offload DeepSpeed configs
data/dataset_info.json       LLaMA-Factory dataset registry (7.5 KB)
```

Confirmed gaps / mismatches:

1. **Victim model string is v1.** `src/step0_data_preprocess/chatgpt_inference.py` hard-codes:
   ```python
   resp = await client.responses.create(
       model="gpt-5-mini-2025-08-07",
       input=prompt_text,
       reasoning={"effort": "medium", "summary": "auto"},
       store=True,
   )
   ```
   The v2 paper says the victim is `gpt-5.4-mini-2026-03-17`. Also note `effort: "medium"` here vs. a separate eval preset named `..._high_effort`. Reproducing v2 = swap the string and re-run; reproducing v1 = use it as-is.
2. **Clone URL in README is wrong.** README says `git clone .../trace_inversion.git`; the repo is `Trace_Inversion_Attack`.
3. **Python version conflict** — see §6.
4. `src/step3_reasoning/evaluation/eval_script.sh` is a scratch script referencing `Llama-3.2-3B-Instruct`, `AMC23`, `AIME25`, `GPQADiamond`, and paths suffixed `_tingwei` that the rest of the repo does not produce. It is stale; the maintained entrypoint is `run_evaluation.py`.
5. Requirements.txt is a loose floor-only list (`transformers>=4.18.0`) that does not reflect the real constraints imposed by the LLaMA-Factory submodule. See §6.
6. Contact: `tz362@cornell.edu`, issues welcome per README.

---

## 2. Hugging Face artifacts

| Artifact | Exact id | Verified? | Notes |
|---|---|---|---|
| Paper page | [`huggingface.co/papers/2603.07267`](https://huggingface.co/papers/2603.07267) | ✅ | 6 upvotes. Abstract is the **v1** text. |
| Author-released inversion checkpoints | — | ❌ **none exist** | No models, datasets, or spaces published by Zhang / Morris / Shmatikov for this paper. Hub search for "trace inversion" returns nothing under their namespaces. |
| Linked models (total 6) | third-party only | ⚠️ | `Jackrong/Trace-Inverter-4B` (4.15 B, Qwen3-4B-Instruct-2507 base, Apache-2.0, 409 dl, tagged `arxiv:2603.07267`) plus three `Jackrong/Negentropy-claude-opus-4.7-*` repos. **Community reimplementations, not author artifacts.** Do not treat as reference checkpoints. |
| Linked datasets (total 12) | third-party only | ⚠️ | `Jackrong/Claude-opus-4.7-TraceInversion-5000x` (4,761 rows), `Jackrong/Claude-opus-4.6-TraceInversion-9000x` (8,669 rows), plus mirrors (`Hiren122/...`, `Coding-With-Bashir/BashCoder`). Applications of the *method* to Claude, unrelated to the paper's GPT-5-mini / R1 experiments. |
| Linked spaces (1) | `KyleHessling1/negentropy-9b-eval` | ✅ running | Third-party eval demo for the Negentropy models. Not relevant to reproduction. |

> **HF paper-page abstract vs. arXiv v2 — a real discrepancy.** The HF `metadata.json` abstract (frozen at v1) reads: *"fine-tuning Qwen-2.5-7B-Instruct on traces inverted from the answers and summaries of **GPT-5 mini** … improves its performance from **56.8% to 77.6%** on MATH500 and from **11.7% to 42.3%** on JEEBench."* arXiv v2 §1 instead reads *"**gpt-5.4-mini-2026-03-17**"* and quotes **31.6% JEEBench** (R1-Weak inverter) and **52.4% MATH500** (Llama, R1 inverter). The experiments were re-run against a newer victim for v2. **Which paper version you target determines which victim API you must pay for.**

---

## 3. Datasets

| Dataset | Exact repo id | Rows | Size on disk | License | Verified? | Split / subset used |
|---|---|---:|---:|---|---|---|
| OpenThoughts-114k (**as used by the code**) | `llamafactory/OpenThoughts-114k` | 114,000 | 1.2 GB parquet (6 files) | Apache-2.0 | ✅ | `train`. This is the **ShareGPT-converted mirror**, and it is what `download_dataset.py` actually calls — *not* the canonical repo. |
| OpenThoughts-114k (canonical, cited in paper) | `open-thoughts/OpenThoughts-114k` | 114,000 (`default`) + 114,000 (`metadata`) | 1.1 GB + 2.5 GB | Apache-2.0 | ✅ | Cited as `guha2025openthoughts`. 1.4 M downloads, 903 ★. Different schema — see below. |
| MATH500 | `HuggingFaceH4/MATH-500` | 500 | 208 KB parquet | ⚠️ **no license declared** | ✅ | `test`, all 500. Evalchemy **also bundles it locally** as `eval/chat_benchmarks/MATH500/data/math500.jsonl` (446 KB) — no HF download strictly required. |
| JEEBench | `daman1209arora/jeebench` | 515 | 136 KB parquet | MIT | ✅ | `test`, all 515. Evalchemy loads it live: `load_dataset("daman1209arora/jeebench", split="test")`. |
| LiveCodeBench | `livecodebench/code_generation_lite` | ⚠️ **UNVERIFIED row count** | ⚠️ **UNVERIFIED** | CC (`license:cc`, unspecified variant) | ✅ repo exists | Evalchemy pins `version_tag="release_v2"`. Dataset Viewer returns **501 — "runs arbitrary Python code"**, so rows/size/schema cannot be read via the API. Requires `trust_remote_code`. |
| HumanEval+ (Table 5 ablation only) | bundled in Evalchemy `eval/chat_benchmarks/HumanEvalPlus/` | — | — | — | ✅ present at pinned SHA | Only used in the §5.4 domain-gating ablation. |

### Schemas

`llamafactory/OpenThoughts-114k` (`default/train`, 4 cols) — **this is the one the pipeline parses**:

| # | Column | Type |
|---|---|---|
| 1 | `messages` | list of `{role: str, content: str}` |
| 2 | `original_solution` | string |
| 3 | `domain` | string |
| 4 | `source` | string |

`open-thoughts/OpenThoughts-114k` (`default/train`, 2 cols) — **incompatible with the code as written**:

| # | Column | Type |
|---|---|---|
| 1 | `system` | string |
| 2 | `conversations` | list of `{from: str, value: str}` |

`HuggingFaceH4/MATH-500` (`default/test`): `problem`, `solution`, `answer`, `subject`, `level` (int64), `unique_id`.

`daman1209arora/jeebench` (`default/test`): `subject`, `description`, `gold`, `index` (int64), `type`, `question`.

### How many examples the paper actually uses

Paper §5.1 states two **disjoint 10k splits** sampled from OpenThoughts-114k: a *surrogate* split (query the surrogate → traces for inversion training) and a *victim* split (query the victim → answers + summaries). The code implements this concretely:

- `download_dataset.py` takes the **first 50,000** rows, emitting a 50k inference file and a 20k file; the first **20,000** rows also get the with-thinking / no-thinking / comprehensive formats.
- `chatgpt_inference.py` reads the 20k inference file with `start_index = 20000, max_samples = 10000` — i.e. rows 20,000–29,999. ⚠️ Note that `start_index=20000` against a 20k-row file yields **zero rows**; this looks like a leftover from pointing at the 50k file. Flag as a likely bug on a fresh run.
- `r1_distill_inference.sh` runs `--max_samples 20000` over the 20k file.
- Both training YAMLs cap at `max_samples: 10000`.
- Query-budget scaling figure (Fig. 3) uses 5k / 10k / 15k, max budget 25k. A `..._5000_samples.yaml` config exists for the 5k point.

---

## 4. Models

| Model | Exact repo id | Params | Safetensors size | Repo total | License | Gated? | Verified? |
|---|---|---:|---:|---:|---|---|---|
| **DeepSeek-R1** (strong surrogate / open-weight victim) | `deepseek-ai/DeepSeek-R1` | **684.53 B** | **688.6 GB** (163 files) | 688.6 GB | MIT | No | ✅ |
| **R1-Distill-Qwen-1.5B** ("R1-Weak") | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.777 B | 3.6 GB (1 file) | 3.6 GB | MIT | No | ✅ |
| R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7.616 B | 15.2 GB (2) | 15.2 GB | MIT | No | ✅ |
| R1-Distill-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14.770 B | 29.5 GB (4) | 29.5 GB | MIT | No | ✅ |
| R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32.764 B | 65.5 GB (8) | 65.5 GB | MIT | No | ✅ |
| R1-Distill-Llama-8B (bonus) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8.030 B | 16.1 GB | 16.1 GB | MIT | No | ✅ |
| **Qwen2.5-7B-Instruct** (compressor + inverter backbone + student) | `Qwen/Qwen2.5-7B-Instruct` | 7.616 B | 15.2 GB (4) | 15.2 GB | Apache-2.0 | No | ✅ |
| **Llama-3.1-8B-Instruct** (student) | `meta-llama/Llama-3.1-8B-Instruct` | 8.030 B | 16.1 GB (4) | **32.1 GB** | llama3.1 | ⚠️ **`gated: manual`** | ✅ |

Notes:

- **DeepSeek-R1 is the crushing cost.** 684.5 B params, `fp8` (F8_E4M3 + BF16 + F32 mixed), **688.6 GB** of weights across 163 shards. The paper uses R1 "accessed via their commercial API," not locally — and you should too. Local serving needs a multi-node H100/H200 fleet.
- **Llama-3.1-8B-Instruct gating is `manual`**, not `auto`: you must submit the META LLAMA 3.1 COMMUNITY LICENSE form on the model page and **wait for human approval** (typically minutes to hours, occasionally rejected). Then `huggingface-cli login`. The repo README calls this out. Budget lead time; this is the one artifact that can block a repro overnight.
- Llama's repo total (32.1 GB) is 2× the safetensors size because it also ships `original/consolidated.*.pth`. Use `allow_patterns=["*.safetensors","*.json","*.txt"]` to halve the download.
- `enable_liger_kernel: true` and DeepSpeed ZeRO-3 are set in every training YAML — plan for those deps.

### Smaller Qwen2.5 drop-in swaps (all verified ✅, all ungated)

| Model | Params | Safetensors | License | Notes |
|---|---:|---:|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.494 B | 1.0 GB | Apache-2.0 | Smallest smoke-test backbone. |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.544 B | 3.1 GB | Apache-2.0 | Good 1×24 GB-GPU target. |
| `Qwen/Qwen2.5-3B-Instruct` | 3.086 B | 6.2 GB | ⚠️ **`other`** (Qwen Research License, non-commercial) — *not* Apache-2.0 like its siblings | Licence anomaly; 0.5B and 1.5B are cleaner swaps. |

All three keep `template: qwen`, so swapping is a one-line edit to `src/step3_reasoning/training_config/presets/models.yaml`. Note `cutoff_len: 16384` in every config — the real memory driver is sequence length, not just params.

---

## 5. The commercial victim (GPT-5.4 mini)

| Item | Value | Verified? |
|---|---|---|
| API model name | `gpt-5.4-mini` | ✅ |
| Snapshot used in paper v2 | `gpt-5.4-mini-2026-03-17` | ✅ listed as an available snapshot on OpenAI's model page |
| Snapshot used in **released code** | `gpt-5-mini-2025-08-07` | ✅ read from source |
| Input pricing | **$0.75 / 1M tokens** | ✅ two independent sources |
| Cached input | **$0.075 / 1M tokens** (90% off) | ✅ |
| Output pricing | **$4.50 / 1M tokens** | ✅ two independent sources |
| Context window | 400,000 tokens | ✅ |
| Max output | 128,000 tokens | ✅ |
| Knowledge cutoff | 2025-08-31 | ✅ |
| Reasoning effort levels | `none` (default), `low`, `medium`, `high`, `xhigh` | ✅ |
| Raw chain-of-thought exposed? | **No.** OpenAI docs: *"While reasoning tokens are not visible via the API, they still occupy space in the model's context window."* Only encrypted reasoning items and optional summaries. | ✅ — this is the paper's entire premise |
| Reasoning tokens billed? | **Yes, as output tokens.** Counted separately under `usage.output_tokens_details` but charged at the $4.50/M output rate. | ✅ |
| Reasoning summaries on `gpt-5.4-mini` specifically | **UNVERIFIED.** The general reasoning guide documents `reasoning.summary` with `auto` / `concise` / `detailed`, but the `gpt-5.4-mini` model page **does not state summary support**. The released code sets `summary: "auto"` against `gpt-5-mini`, which did support it. | ⚠️ Confirm with a single live probe before committing budget to the *summary* setting. The *no-summary* setting is unaffected. |
| Org verification | OpenAI docs: *"Before using summarizers with our latest reasoning models, you may need to complete organization verification."* | ✅ documented — another possible lead-time blocker |

**Cost.** Paper §5.4: *"collecting 10k ⟨input, summary, answer⟩ queries costs $173.28."* The per-token rates are redacted in the HTML render, but $0.75/$4.50 reproduces that figure closely — roughly 5 M input tokens (~$3.75) plus ~37 M output-and-reasoning tokens (~$167). Consistent. Note the corollary: **~95% of the bill is hidden reasoning tokens you never see.** Two independent levers: the Batch API halves everything, and prompt caching cuts repeated input 90% (near-useless here — inputs are unique problems).

Scaling: the 15k-query point in Fig. 3 ≈ $260, and the 25k ceiling ≈ $433. Reproducing the full v2 GPT-5.4-mini table (summary + no-summary, two surrogates) means several such collections; **budget $500–1,500** in API spend.

---

## 6. Tooling — current versions vs. what this repo can actually take

Current PyPI latest (checked 2026-08-15) against the constraints imposed by the **pinned** LLaMA-Factory submodule (`9501c33`, its `pyproject.toml`):

| Library | Latest on PyPI | Released | Pinned-LF constraint | Compatible? |
|---|---|---|---|---|
| `transformers` | **5.15.0** | 2026-08-10 | `>=4.51.0,<=5.2.0,!=4.52.0,!=4.57.0` | ❌ **too new** — must pin ≤5.2.0 |
| `datasets` | **5.0.1** | 2026-07-28 | `>=2.16.0,<=4.0.0` | ❌ **too new** — must pin ≤4.0.0 |
| `accelerate` | **1.14.0** | 2026-06-11 | `>=1.3.0,<=1.11.0` | ❌ **too new** — must pin ≤1.11.0 |
| `peft` | **0.20.0** | 2026-07-28 | `>=0.18.0,<=0.18.1` | ❌ **too new** — must pin 0.18.0 or 0.18.1 |
| `trl` | **1.10.0** | 2026-08-13 | `>=0.18.0,<=0.24.0` | ❌ **too new** — must pin ≤0.24.0 |
| `vllm` | **0.27.1** | 2026-08-11 | `>=0.4.3,<=0.11.0` (extra) | ❌ **too new** — must pin ≤0.11.0 |
| `deepspeed` | **0.19.5** | 2026-08-10 | `>=0.10.0,<=0.18.4` (extra) | ❌ **too new** — must pin ≤0.18.4 |
| `bitsandbytes` | **0.50.1** | 2026-08-13 | `>=0.39.0` (extra, floor only) | ✅ latest OK — **but unused**: every config is `finetuning_type: full` + bf16, no quantized training anywhere in the repo |
| `liger-kernel` | **0.8.1** | 2026-07-23 | `>=0.6.3` (extra, floor only) | ✅ latest OK — **required**, `enable_liger_kernel: true` in all YAMLs |
| `llamafactory` (PyPI) | 0.9.5 | 2026-05-30 | n/a | Do **not** `pip install llamafactory`; the repo vendors it as a submodule with `pip install -e ".[torch,metrics]"` |
| `torch` | — | — | `>=2.4.0` | Floor only; pick the CUDA build matching your driver |

**Every single upper-bounded dependency is currently exceeded by PyPI latest.** A naive `pip install -U` will break the build. Install LLaMA-Factory first and let its resolver pin the stack — do not pin from the repo's own `requirements.txt`, which is floor-only (`transformers>=4.18.0`, `datasets>=2.0.0`) and gives no protection.

### 🚨 Python version conflict (blocker)

| Source | Requirement |
|---|---|
| Repo README | `conda create -n trace_inversion python=3.10 -y` |
| Pinned LLaMA-Factory `pyproject.toml` | `requires-python = ">=3.11.0"` |
| Evalchemy `pyproject.toml` | `requires-python = ">=3.8"` |

Following the README verbatim produces a Python 3.10 env in which `pip install -e llama_factory` fails. **Use Python 3.11.** (Current-main LLaMA-Factory also requires ≥3.11, so this is not a transient.) Verified 3.11 satisfies all three.

### Other tooling notes

- Repo `requirements.txt` lists `wormhole>=0.1.0` — almost certainly a stray/incorrect dep (the `wormhole` PyPI name is not a Cornell/ML package). Safe to drop.
- Evalchemy pins `optimum==1.12.0` and `faiss-cpu==1.7.4` — both hard `==` pins, likely resolver friction against a modern `transformers`. Install Evalchemy in the same env last, and expect to relax these.
- Evalchemy additionally needs `pip install -e eval/chat_benchmarks/alpaca_eval`.
- LiveCodeBench needs `trust_remote_code=True` (dataset runs arbitrary Python).
- `OPENAI_API_KEY` required for the black-box victim path; `huggingface-cli login` required for Llama.

---

## 7. Hardware

Paper §5.1: **8 × NVIDIA A100 80GB** for all training and evaluation. Corroborated by the configs — ZeRO-3 with optional offload, full-parameter (not LoRA) SFT of 7–8 B models at `cutoff_len: 16384`, `per_device_train_batch_size` 1 (inversion) / 4 (student), `grad_accum: 3`, 3 epochs, `lr 1e-5`, `warmup_ratio 0.1`, cosine schedule, bf16, gradient checkpointing. R1-Weak inference is vLLM with `tensor_parallel_size: 2`.

A single 80 GB GPU is **not** enough for the paper-scale student runs. For a scaled-down repro, swap the student to `Qwen2.5-0.5B/1.5B-Instruct` and cut `cutoff_len`.

---

## 8. Unverified / open items

| Item | Status |
|---|---|
| LiveCodeBench `code_generation_lite` `release_v2` row count, on-disk size, column schema | **UNVERIFIED** — HF Dataset Viewer returns 501 (dataset executes arbitrary Python). Must clone and inspect locally. |
| Whether `gpt-5.4-mini` exposes `reasoning.summary` | **UNVERIFIED** — not stated on its model page. Probe live before budgeting the summary-setting experiments. |
| `HuggingFaceH4/MATH-500` license | **UNVERIFIED** — no license tag on the repo. Upstream MATH is MIT (Hendrycks et al.), but the H4 subset declares nothing. |
| Whether authors will update the repo for paper v2 | **Unknown** — no commits since 2026-03-06; 1 open issue. |
| Exact composition of the two disjoint 10k splits | **Not released** — sampling is index-based in `download_dataset.py`, so it is deterministic and reconstructible, but no split manifest is published. |
| Compression prompt few-shot exemplars | Paper Appendix B omits them "for space" and says the full prompt "is released with our code." **UNVERIFIED that they are in the repo** — `src/step1_summarization/data_formatter.py` (8 KB) is the place to check; not confirmed here. |

---

## 9. Minimum shopping list for a repro

**Disk (paper-faithful, R1 via API):** ~1.2 GB (OpenThoughts) + 3.6 GB (R1-Weak) + 15.2 GB (Qwen2.5-7B) + 16.1 GB (Llama-3.1-8B, safetensors only) + <1 GB benchmarks ≈ **37 GB**. Add **688.6 GB** only if you insist on serving R1 locally — don't.

**Blocking lead-time items, start these first:**
1. Llama-3.1-8B-Instruct gated-access request (manual human approval).
2. OpenAI org verification, if the summary setting requires it.
3. Confirm `gpt-5.4-mini` summary support with one cheap live call.

**Then:** Python 3.11 env → clone repo → `git submodule update --init --recursive` → install LLaMA-Factory first (it pins the stack) → Evalchemy last.
