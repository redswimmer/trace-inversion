# Phase 4 Handoff — Invert Split B

You are picking up a reproduction of **"How to Steal Reasoning Without Reasoning Traces"**
(Zhang, Morris, Shmatikov — arXiv 2603.07267v2). Phases 0–3 are complete; Phase 4 is yours: run the
four trained inverters over the victim's split-B outputs to synthesize the forged traces `t̂` that
Phase 5's Synthesized-Trace students will train on. Paper Stage 2, second half (`docs/00` §3).

**You are supervised.** The session named **`big-boss`** planned this phase and audits it.
It reads your CHECKPOINT lines, gets one turn to object before every long run, and writes nothing to
the tree. Report to it via SendMessage; if it is absent or silent, proceed and say so in the log.

Read first: `docs/00-overview.md` (the paper), `docs/10-run-plan.md` (the plan, the **four questions**
to answer before proposing any experiment, and the three Phase 4 decisions — already taken, not yours
to revisit), `docs/results/phase2.md` §5 (how these inverters behave on held-out inputs — the closest
thing to a prediction of this phase), `docs/results/phase3.md` §5 and §8 (the lead finding your length
report must settle, and the file layout that protects the oracle), `docs/11-phase1-handoff.md` §5
(conventions, in full). This document is the operational summary.

---

## 0. What is already built — use these, do not rebuild them

| Artifact | Path | Note |
|---|---|---|
| **Inverter adapters, epoch 2 — decided** | `bench/results/phase2/inverter-7b-sum/checkpoint-402` · `inverter-7b-nosum/checkpoint-402` · `inverter-1.5b-sum/checkpoint-404` · `inverter-1.5b-nosum/checkpoint-404` | The four the user chose for Phase 4 (`docs/10` Phase 4 table). Epoch 3 and the merged weights are **not** used |
| Attack file | `bench/results/phase3/victimB-attack.jsonl` — **5,045 rows** | keys `{idx, domain, source, x, y, b, summary, summary_tokens, finish_reason}`; `b` is the π summary `b*`. **The only Phase 3 data file the attack path opens** |
| Oracle file | `bench/results/phase3/victimB-ORACLE.jsonl` — same rows **+ `t`** | **Nothing on the attack path opens it.** Read exactly once, in the §4.6 stats step, for the read-only paired length table (`docs/14` §4.5 permits that and only that) |
| Inversion prompts | `bench/phase2/prompts.py` — `SUM_SYSTEM/USER`, `NOSUM_SYSTEM/USER` | sha256-pinned, format-matched Appendix B. **Frozen** — the inverters were trained on them |
| Row formatter | `bench/phase2_format.py` → `to_row(r, setting)` | Builds the exact training-time prompt (`enable_thinking=False` per row). **It requires `r["t"]`** — Phase 4 needs the prompt-only variant (§5 step 0); reuse its prompt construction verbatim, do not re-derive it |
| Inference | `bench/invert.py` | vLLM offline batch on **merged** bf16 weights. `completion` optional (`t_true: None`), saves `raw`, defensive `</think>` strip, cap = `finish_reason=="length"`, auto-widens `max_model_len` to `(max_prompt + max_tokens + 255)//256*256`, gates: 0 empty / rows out == in / idx gate (`--holdout ''` to skip). Defaults are the decided sampling |
| Merge | `bench/phase2_train.py --merge --adapter <dir>` | Writes `bench/results/phase2/merged-{arm}-{set}` (**8.43 GB**) + `merge-check.json` (asserts the adapter actually changed the tensors). Delete after each inversion — disk is 28 GB free |
| Driver pattern | `bench/run_phase2_invert.sh` | merge → invert → stats → **delete merged**. Copy it for Phase 4; never edit it while it runs |
| Stats | `bench/phase1_stats.py --mode inverted` | Paired lengths, cap-hit, empties, consistency grading (`<name>-consistency.json`, committed), R1 grading, loop test. Its pairing reads `t_true` from the input file — split B rows have none, so §5 step 0 extends it to take the oracle file for pairing |
| Grader | `bench/eval_baseline.py` `extract_boxed` / `grade` | main thread, as every phase used it |
| OpenThoughts | HF cache, `llamafactory/OpenThoughts-114k` | works with `HF_HUB_OFFLINE=1` |

### Environment, 2026-09-02

| | |
|---|---|
| GPU | RTX 4090, 24,564 MiB; **idle** (269 MiB) |
| Disk | **28 GB free** (95 %). One transient 8.43 GB merge at a time fits; never two. Forged files are ~50–150 MB each, gitignored |
| `.venv-vllm` | Python 3.12 · vLLM 0.27.1 · transformers 5.15.0 — runs `invert.py` and the stats. Launch with `PATH="$PWD/.venv-vllm/bin:$PATH"` (`docs/06` §1.8) |
| `.venv` | Python 3.13 · runs `phase2_train.py --merge` only |
| Tokenizer | All Phase 4 token counts: **Qwen3.5-4B** (== Qwen3.5-2B's). vLLM's `gen_tokens` is the same tokenizer here — the one phase where engine and recount agree; still state which you quote |
| `PYTHONUNBUFFERED=1` in every driver | Phase 2 lost log lines to a nohup buffer without it |

### 0.1 If you are in a worktree (Phase 3 was)

Gitignored assets — both venvs, `bench/results/**`, `bench/logs/` — exist **only** in the main
checkout `/home/asavala/Development/papers/trace-inversion`. The driver runs **your branch's code**
(`cd "$(dirname "$0")/.."`) with the **main checkout's** venvs by absolute path, and writes data and
logs under the main checkout's `bench/`. Only small committed files (consistency JSONs, docs) go in
the worktree. `sweep_concurrency.sh` is not needed this phase. The Bash tool in worktree sessions
rejects loops, heredocs and `&&`-chains ("too complex to verify") — put control flow in a script file
and invoke it plainly. `git -C <main>` is refused; read main's files by absolute path.

---

## 1. Where things stand

| Role | How many | Status |
|---|---|---|
| Victim `V` (Qwen3.8-27B IQ4_XS) | 1 | done — split B queried, 5,045 kept rows (`phase3.md`) |
| Surrogate `V'` | 2 arms | done — `D₂` per arm (Phase 1) |
| Compressor `C'` (π) | 1 | done — `b*` on victim traces, 0 dropped (`phase3.md` §7). **Not run this phase** |
| **Inverter `I`** | **4** = {7B, 1.5B arm} × {sum, nosum} | trained; **this phase runs each once over split B** |
| Student `S` | 5 conditions | Phase 5 |

Facts from earlier phases that shape this one:

- **The inverters' input contract.** Each was trained on the Appendix-B format-matched prompt built
  by `to_row`: sum = `(x, y, b)`, nosum = `(x, y)`, rendered with `enable_thinking=False`, served
  byte-identical to training (`phase2.md` §1, §3 serving check). The sum preamble is ~386 tokens.
- **Split-B prompts are longer than the holdout's.** Holdout prompt max was 2,391 tokens; split B has
  `x` ≤ 5,000 chars, `y` up to 6,565 tokens (median 605), `b*` up to 2,048 (median 606) — worst-case
  prompt ~11k, so `invert.py` will widen `max_model_len` to ~19.5k. That is a KV budget, not an
  output change; the 4B served at 20,480 in Phase 3's compression on this GPU.
- **Expected first-draw cap-hit at 8,192, from the epoch-2 holdout draws** (`phase2.md` §5.5):
  7b-sum **3.0 %** · 7b-nosum **7.0 %** · 1.5b-sum **17.0 %** · 1.5b-nosum **14.5 %** — single draws
  with a ~15 % per-row flip rate between draws (Phase 1). Victim-conditioned inputs may shift these;
  a shift is a finding, not a defect.
- **The 1.5B arm loops** (3–4 repetition loops per 200-row draw; 7B arm 0–2). Loop test: a 40-char
  chunk repeated ≥3× in the trace's last 4,000 chars (`phase2.md` §5.4).
- **The lead finding this phase settles** (`phase3.md` §5): the victim's real traces (median
  **1,400** tokens) are ~50 % *shorter* than the Phase 2 forged estimate (2,063–2,241) — the reverse
  of the paper's ordering. Phase 2's t̂ lengths were **surrogate-conditioned**; two opposing pushes
  act on victim-conditioned inputs (`y` longer: 605 vs 488 → up; `b*` on shorter traces → down).
  **Phase 4's paired report is the first real measurement** and decides whether Phase 5's
  oracle-vs-forged comparison is confounded with supervision length.
- **Victim `y` is mostly right** (97.2 % MATH500 at this operating point), so the 1.5B arm's habit of
  arguing away from `y` — which sometimes *helped* on surrogate data — should mostly hurt here.
  Report the rate; change nothing (`docs/09` 7.15).

---

## 2. What Phase 4 produces

```
victimB-attack.jsonl ──→ phase4 formatter ──→ per-setting prompt files (no t anywhere)
                                   │
                 for each of the 4 inverters (merge → vLLM → delete):
                                   │
        draw 1 (seed 1234) ─→ capped rows redrawn at 1235, then 1236 ─→ still capped: DROPPED
                                   │
   forged-{arm}-{set}-draw1.jsonl  (kept intact: the inverter's own cap-hit)
   forged-{arm}-{set}.jsonl        (draw-1 kept rows + rescued rows − dropped)  ← Phase 5 trains on this
   forged-{arm}-{set}-consistency.json  (committed)
```

| Path | What |
|---|---|
| `bench/results/phase4/attack-{sum,nosum}.jsonl` | formatter output: prompt messages + `chat_template_kwargs` per row, **no `t`, no `completion`** |
| `bench/results/phase4/forged-{7b,1.5b}-{sum,nosum}-draw1.jsonl` | first draw, untouched — its cap-hit is the inverter's property |
| `bench/results/phase4/forged-{7b,1.5b}-{sum,nosum}.jsonl` | the training artifact: `{idx, domain, x, y, b, t_hat, raw, gen_tokens, finish_reason, draw}` — every row `finish_reason != "length"` |
| `bench/results/phase4/forged-*-consistency.json` | per-row grading, committed like Phase 2's |
| `docs/results/phase4.md` | the committed record (§6), incl. **the paired length table** |
| `bench/phase4_format.py` · `bench/run_phase4_invert.sh` · the `--oracle` stats extension | the code changes (§5 step 0) |

All `.jsonl` under `bench/results/` is gitignored already.

---

## 3. Measured budgets

The one realized inversion rate: the Phase 2 holdout — **200 rows, ~525k generated tokens, ~5.5 min
wall ≈ 1,590 tok/s** on merged 4B bf16 at `max_model_len` 12,288. Split B serves at ~19.5k with
longer prompts, so expect lower; the probe measures it.

Volume per inverter: 5,045 rows × holdout mean t̂ 2,507–3,175 tokens ≈ **13–16 M tokens** →
~2.3–2.8 h at 1,590 t/s, plus redraws (3–17 % of rows × up to 2 more draws), merge (~10 min) and
engine load. Call it **~3–4 h per inverter, ~13–17 h for the phase** — state the rate you assume,
re-project from the first inverter's realized rate, and treat `docs/10`'s "~4 h" as superseded by
whatever you measure. Everything is sequential on one GPU; nothing here approaches Phase 3's 66 h.

---

## 4. Decisions — fixed for this phase (`docs/10` Phase 4, decided 2026-08-30)

### 4.1 Serve the epoch-2 adapters, merged

`checkpoint-402` (7B arms), `checkpoint-404` (1.5B arms). Merge with `phase2_train.py --merge
--adapter <checkpoint>` — its merge-check assert is the proof an adapter was folded in (the bare base
already writes plausible traces; a no-op merge would look like success). One merge on disk at a time.

### 4.2 Conditioning — the training prompt, byte-identical

Sum inverters get `(x, y, b)` from the attack row; nosum get `(x, y)`. Build rows through the same
prompt construction `to_row` uses (import it or factor it — do not paraphrase the prompt), with
`chat_template_kwargs = {"enable_thinking": false}`. The formatter's self-test mirrors
`test_phase2_format.py`: no `b` text in any nosum row · no `t`/`trace`/`raw` key anywhere · row
count 5,045 · idx set == the attack file's · rendered row 0 printed and eyeballed.

### 4.3 Sampling — the project's one protocol

temperature 0.7 · top_p 0.9 · repetition_penalty 1.05 · `max_tokens` **8192** · seed **1234** ·
`--holdout ''` · gpu_frac 0.90. These are `invert.py`'s defaults; pass them explicitly in the driver
anyway so the log states them. Let it auto-widen `max_model_len` and report what it chose.

### 4.4 Capped forgeries: ≤3 draws, then drop and report

A row with `finish_reason == "length"` is regenerated at seed **1235**; what still caps, at **1236**;
what caps on all three draws is **dropped** from the forged file and counted. Same policy on every
arm and setting. Keep `-draw1.jsonl` intact and report its cap-hit as the inverter's property; the
rescued set is the training artifact. Rationale and `docs/09` row 7.14: a severed t̂ in the student
target `[t̂; y]` teaches non-termination, and dropping alone would shrink the 1.5B-arm data ~10–15 %
more than the 7B's, confounding the surrogate-strength comparison. Report per draw: rows redrawn,
rescued, still capped; and the dropped idx per inverter. (The paper regenerates nothing — logged.)

### 4.5 Answer-inconsistent forgeries: report, do not filter

Grade each t̂'s last box against its conditioned `y` exactly as `--mode inverted` already does;
report match / mismatch / no-box buckets and the inconsistent rate per inverter; read ≥3 mismatches
per inverter before calling any genuine (`docs/11` §5). **No filtering** (`docs/09` 7.15). A
consistency-filtered condition is a Phase 5 proposal that must pass `docs/10`'s four questions.

### 4.6 The paired length report — load-bearing (`phase3.md` §5)

For each inverter, paired on idx against `victimB-ORACLE.jsonl`'s `t` (and `y`): t̂ vs t median /
mean / p05 / p95, per-row ratio, per-domain medians, and the supervision totals **`t̂+y` vs `t+y`**.
Also t̂ beside the Phase 2 holdout t̂ (2,063–2,241) — same tokenizer, stated. This single table tells
Phase 5/6 whether an oracle-vs-forged gap is confounded with supervision length, and resolves the
two-pushes question. The ORACLE read happens **here only**, in the stats step, main thread, read-only.
**No length gate** — `docs/10` rejected one; a surprising length is a finding.

### 4.7 Not in this phase

No student formatting or training, no TF1/BLEU/ROUGE (rejected, `docs/10`), no re-benchmarking, no
change to π or the inversion prompts, no second draw of uncapped rows, no consistency filtering, no
new conditions. If an experiment seems worth adding, put it through `docs/10`'s four questions in a
CHECKPOINT and wait.

---

## 5. Order of work

| Step | What | Est. |
|---|---|---|
| **0** | Branch `phase4-build`. Write `bench/phase4_format.py` (+ self-test, §4.2) → `attack-{sum,nosum}.jsonl`. Extend `phase1_stats.py --mode inverted` with `--oracle <file>` pairing (gates in §6). Copy `run_phase2_invert.sh` → `bench/run_phase4_invert.sh`: merge → invert (`--holdout ''`) → redraw loop (§4.4) → assemble final → stats → delete merged. Commit (explicit paths). | ~1–2 h |
| **1** | **Probe:** 7b-sum — merge, `invert.py --limit 30`, read 3 traces (shortest/median/longest), report §6's probe list. CHECKPOINT; one turn for `big-boss` to object. | ~0.5 h |
| **2** | 7b-sum full: draw 1 → redraws → final file → stats + consistency. CHECKPOINT with realized t/s and the re-projection. | ~3–4 h |
| **3** | Repeat for 7b-nosum, 1.5b-sum, 1.5b-nosum — a 30-row smoke each (cheap, catches a bad merge before hours), CHECKPOINT after each inverter. | ~9–12 h |
| **4** | The §4.6 paired length table across all four; the two-pushes resolution stated in one paragraph. | ~0.5 h |
| **5** | `docs/results/phase4.md` (§6); `docs/09` 7.14/7.15 filled with measured counts; `docs/10` Phase 4 closed with measured hours; README status row + "what we've found" bullet if the length answer is clean. Commit. | ~1 h |

---

## 6. Reports, gates, and what goes in `docs/results/phase4.md`

### Probe report — before each full run (full list for step 1; abbreviated for smokes 2–4)

`rows 30` · `empty t̂ 0` · `stripped_think count` · `cap-hit at 8,192` · `t̂ tokens median/p95 vs the
victim's t on the same rows` · `prompt tokens max/median and the max_model_len chosen` · `realized
tok/s` · `projection for this inverter and the phase, stating the rate assumed` · `3 traces read:
does each reach its y, in what register`.

### Record

- per inverter: draw-1 cap-hit (the inverter's property) · redrawn/rescued/still-capped per draw ·
  dropped idx and count · final rows · loops (§1 test) · empty 0 · stripped_think · wall clock ·
  realized t/s
- **the §4.6 paired length table** — the phase's headline
- consistency per inverter (§4.5), mismatches read
- 3 traces read per inverter; one line each on register vs the victim's
- the file layout and every gate's output; which files are draw-1 vs final

### Gates that exit non-zero

- formatter self-test (§4.2)
- per final forged file: rows == idx set == the attack file's minus the reported drops · 0 empty t̂ ·
  0 `finish_reason == "length"` · consistency JSON written
- driver hygiene: the generation steps' inputs contain no path with `ORACLE` in it (grep the driver
  and the logs; the stats step is the one permitted exception)

### STOP AND ASK — report, then wait

- draw-1 cap-hit above **15 %** on a 7B-arm inverter or **30 %** on a 1.5B-arm one
- rows dropped after three draws above **2 %** on any inverter (data-quantity confound returns)
- any empty t̂ · stripped_think above 2 % of rows · loops above **5 %** of any inverter's final rows
- t̂ median under **700** or over **4,500** tokens on any inverter (≈ half / double the expected
  band — smells like a wrong merge or prompt, not a finding; report before continuing)
- the phase projection passes **30 h**, or a step exceeds its estimate by 2×
- any gate fails twice · anything on the attack path wants a file named ORACLE
- vLLM errors or empty outputs arriving in a burst

---

## 7. `docs/09` rows

7.14 (capped-forgery policy) and 7.15 (no consistency filtering) already exist with the decisions;
**fill in the measured counts** (per-inverter draw-1 cap-hit, rescued, dropped; inconsistent rate).
Add a new row only if the run itself deviates somewhere new.

---

## 8. Conventions — Phase 4 specifics

`docs/11` §5, `docs/13` §8, `docs/14` §8 apply in full. What bites here:

| Rule | Why |
|---|---|
| **Nothing on the attack path opens a file named ORACLE** — one read, stats step, lengths only | the failure is silent and looks like a spectacular success (`docs/10` Phase 3 note) |
| **Byte-identical prompts** — reuse `to_row`'s construction, never re-type the prompt text | the inverters are format-locked to their training prompt; a paraphrase is a silent distribution shift |
| **Merge-check before trusting any inversion** | the bare base writes plausible traces; only the tensor-diff proves the adapter is in |
| **One merged model on disk at a time; delete after use** | 28 GB free, 8.43 GB per merge |
| **Probe before every multi-hour run; budget from realized rate, not the probe alone** | the batch's tail drains under-occupied; state the rate you assume |
| **Save raw text; the engine's `finish_reason` and token counts are authoritative** | |
| **State the tokenizer next to every token number** | |
| **Never edit a running script; one sequential driver; explicit paths on `git add`** | |
| **A supervising session writes nothing to the tree** | it audits CHECKPOINT lines |

---

## 9. Open items carried to Phase 5

| Item | Note |
|---|---|
| **Supervision-length confound** | the §4.6 table is Phase 5's input; if t̂+y ≫ t+y, a length-matched control is a Phase 5 proposal (`phase3.md` §5) |
| **Per-inverter dropped rows** | each forged file may lose a different handful of idx; whether Phase 5's conditions train on the intersection or per-condition sets is **Phase 5's decision** — hand it the four final idx sets |
| **Register mismatch** | oracle = victim's terse notes; forged = R1-Distill prose; any oracle-vs-forged gap carries style as well as content |
| **Oracle length asymmetry** | oracle traces run to 14,336; forgeries cap at 8,192 (as in the paper) |
| **Phase 5 budget risk** | `docs/10`'s "~20 h / five conditions" predates the four forged sets; ~9 trainings if Table 3 is matched per arm × setting — re-plan before running |

---

## 10. Definition of done

- [ ] `phase4_format.py` + self-test committed; `attack-{sum,nosum}.jsonl` built, 5,045 rows each, no `t` anywhere
- [ ] four `-draw1.jsonl` and four final `forged-*.jsonl` on disk; every final row `finish_reason != "length"`; drops ≤ reported counts
- [ ] every full run preceded by a reported probe/smoke; CHECKPOINTs sent to `big-boss` with one turn to object before each
- [ ] consistency JSONs committed; rates reported, no filtering applied
- [ ] the paired t̂-vs-t length table with supervision totals, all four inverters, in `docs/results/phase4.md`
- [ ] `docs/09` 7.14/7.15 measured; `docs/10` Phase 4 closed with measured hours; README updated
- [ ] merged weights deleted; disk back above 25 GB free
