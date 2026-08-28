# Phase 1 — Surrogate data (`D₂`)

Reproduction of *"How to Steal Reasoning Without Reasoning Traces"* (Zhang, Morris, Shmatikov,
arXiv 2603.07267v2), Stage 1 §4.1. Produces the inverter's training set

```
D₂ = {(x', y', b', t')}
```

for **two surrogate arms**, so that inversion quality can be read against surrogate strength — a
sweep the paper never ran (it tested only a 1.5B distill and R1 at 685 B, a 450× gap with no
midpoint).

| | 7B arm | 1.5B arm |
|---|---|---|
| Surrogate `V'` | `DeepSeek-R1-Distill-Qwen-7B`, GGUF F16 | `DeepSeek-R1-Distill-Qwen-1.5B`, GGUF BF16 |
| Compressor `C'` | `Qwen/Qwen3.5-4B` bf16, vLLM | same |
| Status | **complete — 5,006 rows** | *(generation in flight)* |

Settings, identical on both arms and taken from the paper's repo: `temperature 0.7 · top_p 0.9 ·
repetition_penalty 1.05 · max_new_tokens 8192`, served at `-np 32 -c 327680` (10,240 tokens/slot).
Inputs are 5,000+ prompts from OpenThoughts-114k split A (seed 20260826), disjoint from split B
which Phase 3 reserves for the victim.

---

## 1. Generation — the 7B arm

| | |
|---|---:|
| rows generated | 7,669 |
| rows kept | **5,012** |
| cap-hit | **34.6%** (2,657) |
| request errors | **0** |
| prompts skipped (>8,000 chars) | 1 |
| `gen_tokens`, whole file | 39,815,004 (5,192/row) |
| wall clock | 21.83 h |
| realized throughput | 493.3 gen t/s @ 32 slots |

> **`gen_tokens` is trace + answer**; every trace-length figure below is **trace only**, re-tokenized
> from `r["trace"]`. One English word, two quantities, two code paths. The 21.83 h and the 493.3 t/s
> are both this-process figures excluding the 200-row probe (38,774,378 tokens); the row count above
> includes it. Dividing one by the other gives 5,056 rather than 5,192 for that reason.

### Trace lengths (trace-tokens only)

| | n | median | mean | p05 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| kept | 5,012 | **2,804** | 3,114 | 209 | 6,957 | 7,974 |
| all rows (censored at the cap) | 7,669 | 4,915 | 4,864 | 267 | 8,191 | 8,203 |
| answer tokens, kept | 5,012 | 471 | 483 | — | — | 1,653 |

The 8,203 maximum exceeds the 8,192 cap because llama-server capped *its own* tokens and we recount
the text with a different tokenizer. The engine's `finish_reason` is authoritative; our recount is
approximate. This is the "prefer the engine's own signal" rule producing a visible artifact rather
than a silent one.

By domain, kept rows:

| domain | n | median | mean |
|---|---:|---:|---:|
| math | 3,980 | 3,024 | 3,241 |
| code | 690 | 3,091 | 3,399 |
| physics | 97 | 1,037 | 1,333 |
| biology | 86 | 885 | 962 |
| chemistry | 80 | 928 | 1,204 |
| puzzle | 79 | 602 | 721 |

### Paired against R1's ground truth, identical prompts (n=7,669)

OpenThoughts ships R1's own `<think>` trace for every row, so each prompt has both sides and prompt
difficulty cannot leak into the comparison.

| | ours | R1 | Δ |
|---|---:|---:|---:|
| cap-hit | **34.6%** | 26.8% | +7.8 pts |
| kept p50 (both under cap, n=4,601) | 2,570 | 2,779 | −7.5% |
| kept p90 | 5,977 | 5,971 | +0.1% |
| ours shorter on | 62.6% of prompts | | |

R1's p90 over all rows is 13,930, past our 8,192 cap. **Our tail is censored there, so only the
crossing rate is comparable, not tail shape.**

### Drop decomposition — why `D₂` loses the rows it loses

| | |
|---|---:|
| we drop | 2,657 / 7,669 = 34.6% |
| R1 would drop, same prompts | 2,056 |
| both (genuinely long prompts) | 1,645 |
| **ours only — R1 completes these** | **1,012** = 38% of our drops, **13.2% of all prompts** |
| R1 only — we complete these | 411 |

38% of what `D₂` loses is specific to this surrogate, not to prompt difficulty. Phase 4 serves split
B essentially unfiltered (the victim truncated 0.0–0.4% in Phase 0), so **our own drop policy
introduces a train/serve shift over ~13.2% of the prompt space.** Measured, not assumed; recorded as
`docs/09` row 7.10.

### The drop set is partly a coin flip — the noise floor

An archived partial run over the *same* split A prefix at *identical* settings
(`~/trace-inversion-bench/archive/traces-7b-partial-1130rows.jsonl`, n=1,142) gives a second
independent draw of the same prompts at temperature 0.7. **This is the only same-surrogate,
same-prompt, same-settings re-draw in the project, and it is the noise floor for every stochastic
claim here.**

| | |
|---|---:|
| cap-hit, draw A / draw B | 31.7% / 35.1% |
| rows flipping capped↔kept | **15.4%** |

So *whether a given prompt hits the cap is substantially a resampling outcome, not a property of the
prompt.* Every cap-hit figure carries ~±4 pts beyond binomial error.

The flip rate is **not uniform**, and that changes how the drop decomposition reads:

| drop type | flips back to kept | 95% CI |
|---|---:|---|
| ours-only (R1 completes it) | **37.5%** (48/128) | [29.6, 46.1] |
| both-cap (R1 also caps it) | **12.6%** (33/262) | [9.1, 17.2] |
| marginal, all drops | 20.8% (81/390) | [17.0, 25.1] |

The separation is mechanical, not statistical. A both-cap prompt is genuinely long — R1 exceeds
8,192 on it too — so it caps on any draw. An ours-only prompt is *by construction* one R1 finishes,
which places it at **our** boundary, and boundaries are where re-draws flip. **Selecting on "R1
completes this" is selecting for marginality**, so a marginal flip rate applied to that subset is
biased low in a predictable direction.

Propagating the subset-specific rate: of the 1,012 ours-only drops, ~632 are stable —
**23.8% of drops, 8.2% of all prompts.** The 38% observed share is an upper bound.

> **The 1.5B arm has the same structure and will hit the same error.** Re-derive its flip rate on
> its own ours-only subset; do not reuse either arm's marginal figure.

---

## 2. Compression `π` — the 7B arm

`π` is Appendix B's compression prompt, transcribed verbatim in `docs/01` §6.4 and sha256-asserted at
import. **Only the two few-shot exemplars are ours** — the authors never released theirs.

### Table 1 acceptance, at full scale

| statistic | n=5,006 | n=200 validation | target | |
|---|---:|---:|---|---|
| median tokens | **583** | 578 | 540–590 | PASS |

*Summary tokens are re-tokenized from the summary text with `Qwen/Qwen3.5-4B`, which is what `phase1_stats.py` gates on. vLLM's own `len(token_ids)` for the same rows gives **584** — the same decode-then-recount gap as the 8,203-vs-8,192 trace artifact above, and a reason to state which count a token figure is.*
| bold-header sections (≥3) | 100.0% | 100.0% | >90% | PASS |
| first-person prose (≥3) | 99.9% | 99.5% | >95% | PASS |
| LaTeX | 80.5% | 79.5% | >70% | PASS |

Spec compliance (Appendix B forbids both): numbered lists **0.0%**, bullet lists **0.1%**, 96.2%
within the 3–6 section bound.

### Sample size: sufficient for a central statistic, blind to a rare one

**These two results must be read together.** Alone, the first invites "200 rows is enough"; the
second says what it is enough for.

- The n=200 validation predicted the n=5,006 median **within 5 tokens** (578 vs 583). This is the
  only evidence in the project that the ~200-row sample `docs/11` §4 prescribes is adequate — until
  now it was inherited and untested.
- Bullet-list incidence read **0.0% at n=200 and 0.1% at n=5,006**. That is not noise in the number:
  a 0.1% behaviour has an expected count of **0.2** in 200 rows, so the validation could not have
  seen it whatever `π` did.

**Consequence:** if any Appendix B compliance figure is ever promoted from report to *gate*, its
threshold cannot be set from a 200-row run. Bullets, numbered lists and the 3–6 section bound are all
rare-event statistics.

### The 600–900 over-ask is a property of the compressor, not the prompt

Appendix B instructs "roughly 600–900 tokens" and Table 1 reports resulting medians of 537 and 592.
`docs/11` §4 is right that the disagreement is by design — the instruction is a calibrated over-ask
the model undershoots. **But it is calibrated against *their* `C'`.**

| | undershoot from the 750 midpoint | landed at |
|---|---:|---:|
| paper's `C'` (Qwen2.5-7B) | ~28% | 537 |
| ours (Qwen3.5-4B), iteration 1 | ~17% | 621 |

Iteration 1 used exemplars sized to the Table 1 medians (552/502 tokens) exactly as §4 prescribes,
and output landed at 621 — *above both exemplars*. So the model is not copying exemplar length: the
instruction dominates and the exemplars only pull down against it. Moving one lever (exemplars →
494/434, mean 527 → 464, instruction untouched at the verbatim 600–900) landed iteration 2 at 578.

**Anyone substituting `C'` again lands somewhere else on the same instruction.**

### `π` is stable across input types

| trace-length quartile *(by trace **chars**, the harness's proxy)* | n | median summary tokens |
|---|---:|---:|
| short | 1,252 | 491 |
| mid | 1,251 | 585 |
| long | 1,251 | 628 |
| longest | 1,252 | 633 |

Summary length rises 29% from the shortest quartile to the longest (34% if the quartiles are cut by
trace *tokens* rather than chars — the direction and the argument are unchanged) while the *traces*
span ~200 to 8,192 tokens — so `π` adds depth rather than sections, which is what Appendix B asks for. Input
**type** moves it more than input **length**:

| domain | n | median | LaTeX |
|---|---:|---:|---:|
| math | 3,974 | 609 | **89.1%** |
| code | 690 | 484 | 55.2% |
| physics | 97 | 548 | 75.0% |
| biology | 86 | 512 | 0.0% |
| chemistry | 80 | 517 | 33.3% |
| puzzle | 79 | 439 | 0.0% |

*(LaTeX rates are from the n=200 validation, where the per-domain samples are small: physics n=4,
chemistry n=3, biology n=5, puzzle n=3.)*

**The 80.5% LaTeX headline is ~78% a math measurement.** Modelling split A's real three-way mix
rather than a math/code binary, the >70% gate is threatened only when code reaches **48.1%** of the
corpus — 2.8× its actual 17.3% share. The model back-tests correctly: it predicts 80.4% for split A's
mix against 79.5% observed.

### Two failure modes worth recording

**`C'` pads on short traces by solving.** Six of 5,012 summaries ran to the 2,048-token cap. They are
not repetition loops — unique-word ratio 0.29–0.43, no repeated blocks. `C'` stops summarizing and
starts *deriving*: one tail reads `"Let's assume the school is 600 meters away. Total time for Feifei
to walk 600m at \(400/9\) m/min is..."`. Their trace lengths are bimodal — three at percentiles
7/10/10 and three at 84/87/87 against a 2,804 corpus median. On a 261-token trace there is nothing to
fill 600–900 tokens with, so the model pads by re-deriving. **Appendix B anticipates exactly this**
("if the input trace is very short, produce fewer sections rather than padding") and Qwen3.5-4B pads
anyway — the third instance of one story, alongside iteration 1's 621 and the undershoot ratio.

All six were dropped (`bench/results/phase1/d2-7b-dropped.jsonl`), leaving **5,006**. Regenerating
them at a fresh seed rescued 13 of the original 19 but never these six, because the cause is a
property of the input rather than of sampling.

> **Considered and rejected: raising `--max-tokens` above 2,048.** It would rescue some of the six,
> but 2,048 was fixed *before* Table 1 was validated against it, so moving it changes the artifact
> that validation measured. Rescuing 6 rows by invalidating the validation of 5,012 is a bad trade.

**`enable_thinking=False` leaks about 1 in 5,000.** One summary carried a `<think>` block past the
switch and was caught by the defensive `</think>` strip in `phase1_compress.py`. **That strip is
load-bearing, not belt-and-braces** — removing it as redundant loses one row in five thousand
silently.

---

## 3. `D₂` for the 7B arm — final

| | |
|---|---:|
| rows | **5,006** |
| gate | `phase1_stats.py --mode summaries` → **exit 0** |
| empty `x` / `y` / `b` / `t` | 0 / 0 / 0 / 0 |
| summaries severed at the cap | 0 |
| duplicate `idx` | 0 |

---

## 4. Method notes that cost time

- **The cap-hit acceptance band is per-surrogate, not a constant.** The 7B settles at 34.6% and the
  1.5B at 46.5% on identical prompts, so a single threshold either passes a broken 7B or fails a
  healthy 1.5B. Bands: 10–45% (7B), 10–58% (1.5B). The one threshold that is arithmetic rather than
  judgement is **68.75%** — 5,000 kept from 15,999 usable rows needs a 31.25% keep rate, so split A
  is exhausted there.
- **A running cap-hit is not an estimate of the final one.** Results stream in as they finish, so the
  in-flight set is length-biased long. Only the post-drain figure is unbiased. A bound computed on a
  frozen sample says nothing about a growing one: at n=1,011 the running figure was 35.0% and the
  final was 34.6%, because 6,626 further prompts arrived and sample composition swamps any censoring
  correction.
- **Quote a trend from a fit over the regime, never from the last few points.** Four consecutive
  throughput windows read 539.7 → 514.0 → 500.1 → 475.5 and look like decay; the ragged-regime series
  has sd 86.8 and a fit across all 25 windows gives −2.0 gen t/s per 1,000 rows. Flat.
- **The 1.5B arm changed slot count mid-run, at the user's direction.** Rows 1–5,832 were generated at
  `-np 32 -c 327680`; the remainder at `-np 128 -c 1310720`. Per-slot context is 10,240 in both, so the
  8,192 cap and the prompt budget are unchanged, and **slot count affects throughput only** — each request
  decodes independently and sampling is per-request, so cap-hit and trace length are unaffected. Recorded
  because it is a mid-run configuration change, not because it is expected to matter.
- **16 GB of idle VRAM was not 16 GB of idle throughput.** The 1.5B at 32 slots used 8,071 MiB of 24,564 at
  34–38% utilization, which looked like obvious headroom. A sweep predicted **5,097 gen t/s** at 128 slots;
  realized was **+12.3% rows/min** (8.25 vs 7.35) measured between two steady states 31.4 min apart, which
  cancels the pipeline-depth offset that made a naive from-restart comparison read **−7%**. Three independent
  instruments agree: +12.3% rows/min, +12.1% kept/min, +13% engine-native tokens. **The sweep overstated by
  ~6×**, and 16.5 GB of idle VRAM bought about 45 minutes on a 6-hour remainder. The card sits at 17–21%
  utilization and ~61 W at 2,010 MHz even at 128 slots — it is memory-latency bound, waiting on weights
  rather than computing, so extra concurrent sequences help only until the memory system saturates and 32
  slots was already most of the way there. **The headroom was real and unspendable.**
- **Launch vLLM with the venv's `bin/` on `PATH`, not just its `python`.** vLLM JIT-compiles sampling
  kernels via `subprocess.run(['ninja', ...])`; running `.venv-vllm/bin/python` directly puts the
  interpreter in scope but not the venv's console scripts. Presents as `RuntimeError: Engine core
  initialization failed` with the real cause 40 lines up. See `docs/06` §1.8.
