#!/usr/bin/env python
"""Build the Phase 5 student datasets: ONE construction, only `c` varies (docs/16 §4.3).

Per condition, TRL conversational prompt-completion JSONL under <data-root>/phase5/data/:
  prompt:     [{"role": "user", "content": x + INSTR}]          INSTR = eval_baseline.INSTR (docs/09 7.20)
  completion: [{"role": "assistant", "content": "<think>\\n" + c + "\\n</think>\\n\\n" + y}]
  c = ""      answer-only        (renders the same empty think block enable_thinking=False makes)
      b*      summary-answer     (victimB-attack.jsonl `b`)
      t       oracle             (victimB-ORACLE.jsonl — THE one permitted open of that file)
      t_hat   synth-{arm}-{set}  (forged-{arm}-{set}.jsonl)
      t'      surr-{arm}         (d2-{arm}.jsonl — split A, that row's own x', y')

Rows: the SAME 3,616-idx intersection for every split-B condition, recomputed from
forged-*-draws.json and asserted (docs/16 §4.2); surr-* n-matched by a seed-1234 sample of
3,616 rows from the full d2 file, idx recorded. Default thinking render — no chat_template_kwargs.

Self-tests (docs/16 §4.3, §6 — this script exits non-zero if any fails):
  row counts == 3,616 · split-B idx lists identical & unique · round-trip render of row 0 per
  condition (exactly one <think> and one </think>, c between, y after) · answer-only ==
  enable_thinking=False's render · cross-condition byte test on 5 shared idx · truncation at
  16,384 == 0 (a row over is a STOP, not a trim) · oracle t text in no other condition's file ·
  no `t` key in any dataset · forged/oracle x,y == the attack row's (the pairing gate).

Token histograms (Qwen3.5-2B tokenizer, stated) -> bench/results/phase5/format-stats.json.
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import INSTR  # noqa: E402 — the Phase 3 suffix, pinned verbatim; never retype

TAGS = ["7b-sum", "7b-nosum", "1.5b-sum", "1.5b-nosum"]
SPLIT_B = ["answer-only", "summary-answer", "oracle",
           "synth-7b-sum", "synth-7b-nosum", "synth-1.5b-sum", "synth-1.5b-nosum"]
SURR = ["surr-1.5b", "surr-7b"]
N = 3616
DOMAINS = {"math": 2886, "code": 477, "chemistry": 68, "puzzle": 65, "biology": 60, "physics": 60}
SEED = 1234
MAX_LENGTH = 16384
TOKENIZER = "Qwen/Qwen3.5-2B"  # == the student's; every token count in this file uses it


def load(path):
    return [json.loads(l) for l in open(path)]


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".new", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(str(path) + ".new").replace(path)


def make_row(idx, domain, x, c, y):
    """THE construction. Everything trains through this one function; only c differs."""
    x, c, y = x.strip(), (c or "").strip(), y.strip()
    for tag in ("<think>", "</think>"):
        assert tag not in x and tag not in c and tag not in y, \
            f"idx {idx}: a think tag in the source text would silently break the template render"
    return {"idx": idx, "domain": domain,
            "prompt": [{"role": "user", "content": x + INSTR}],
            "completion": [{"role": "assistant", "content": f"<think>\n{c}\n</think>\n\n{y}"}]}


def think_split(rendered):
    """(before, c, after) of the single think block in a rendered string; asserts exactly one."""
    assert rendered.count("<think>") == 1 and rendered.count("</think>") == 1, \
        f"doubled/missing think tags: {rendered.count('<think>')}x <think>, {rendered.count('</think>')}x </think>"
    pre, rest = rendered.split("<think>\n", 1)
    c, post = rest.split("\n</think>\n\n", 1)
    return pre, c, post


def dropped_union(root):
    """Union of the four inverters' dropped_idx; attack minus this = the intersection (docs/16 §0)."""
    dropped = set()
    for tag in TAGS:
        d = json.load(open(root / "phase4" / f"forged-{tag}-draws.json"))
        dropped |= set(d["dropped_idx"])
    return dropped


def ids(out):
    return list(out["input_ids"] if hasattr(out, "keys") else out)  # BatchEncoding vs list (phase2_format)


def build_oracle(root, order, attack_by):
    """The Victim-Trace dataset builder — THE ONLY CODE PATH THAT OPENS victimB-ORACLE.jsonl."""
    path = root / "phase3" / "victimB-ORACLE.jsonl"
    print(f"ORACLE read: {path} (oracle dataset builder — the one permitted open)", flush=True)
    by = {r["idx"]: r for r in load(path)}
    rows = []
    for i in order:
        r, a = by[i], attack_by[i]
        assert r["x"].strip() == a["x"].strip() and r["y"].strip() == a["y"].strip(), \
            f"idx {i}: oracle row is not the attack row + t"
        rows.append(make_row(i, r["domain"], r["x"], r["t"], r["y"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/asavala/Development/papers/trace-inversion/bench/results",
                    help="the main checkout's bench/results (gitignored data lives only there)")
    ap.add_argument("--stats-out", default="bench/results/phase5/format-stats.json",
                    help="committed stats file, relative to the repo this code runs from")
    args = ap.parse_args()
    root = Path(args.data_root)
    out = root / "phase5" / "data"
    fails = []

    attack = load(root / "phase3" / "victimB-attack.jsonl")
    dropped = dropped_union(root)
    order = [r["idx"] for r in attack if r["idx"] not in dropped]
    if len(order) != N:
        print(f"FAIL  intersection is {len(order)} rows, not {N} — STOP, do not reconcile")
        sys.exit(1)
    attack_by = {r["idx"]: r for r in attack}
    dom = {}
    for i in order:
        dom[attack_by[i]["domain"]] = dom.get(attack_by[i]["domain"], 0) + 1
    assert dom == DOMAINS, f"intersection domain mix {dom} != docs/16's {DOMAINS}"
    print(f"intersection: {len(order)} idx ({', '.join(f'{k} {v}' for k, v in sorted(dom.items()))})")

    datasets = {}
    datasets["answer-only"] = [make_row(i, attack_by[i]["domain"], attack_by[i]["x"], "", attack_by[i]["y"]) for i in order]
    datasets["summary-answer"] = [make_row(i, attack_by[i]["domain"], attack_by[i]["x"], attack_by[i]["b"], attack_by[i]["y"]) for i in order]
    datasets["oracle"] = build_oracle(root, order, attack_by)
    for tag in TAGS:
        by = {r["idx"]: r for r in load(root / "phase4" / f"forged-{tag}.jsonl")}
        rows = []
        for i in order:
            r, a = by[i], attack_by[i]
            assert r["x"].strip() == a["x"].strip() and r["y"].strip() == a["y"].strip(), \
                f"idx {i}: forged {tag} x/y != attack x/y (phase4's to_row stored them stripped)"
            rows.append(make_row(i, r["domain"], r["x"], r["t_hat"], r["y"]))
        datasets[f"synth-{tag}"] = rows

    surr_idx = {}
    for arm in ("1.5b", "7b"):
        d2 = load(root / "phase1" / f"d2-{arm}.jsonl")
        sel = sorted(random.Random(SEED).sample(range(len(d2)), N))  # n-matched, file order kept
        rows = [d2[j] for j in sel]
        surr_idx[arm] = [r["idx"] for r in rows]
        assert len(set(surr_idx[arm])) == N
        datasets[f"surr-{arm}"] = [make_row(r["idx"], r["domain"], r["x"], r["t"], r["y"]) for r in rows]

    # --- gates on the built datasets (docs/16 §4.3 / §6) ---
    for name, rows in datasets.items():
        if len(rows) != N:
            fails.append(f"{name}: {len(rows)} rows != {N}")
        if len({r["idx"] for r in rows}) != len(rows):
            fails.append(f"{name}: duplicate idx")
        keys = {k for r in rows for k in r}
        if keys != {"idx", "domain", "prompt", "completion"}:
            fails.append(f"{name}: unexpected keys {sorted(keys)}")
    for name in SPLIT_B:
        if [r["idx"] for r in datasets[name]] != order:
            fails.append(f"{name}: idx list != the shared intersection order")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    # round-trip self-test, row 0 of every condition (docs/16 §4.3; docs/06 §4.4 is why)
    for name, rows in datasets.items():
        row = rows[0]
        c_stored, y_stored = think_split(row["completion"][0]["content"])[1:]
        rendered = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
        pre, c_r, post = think_split(rendered)
        seg = rendered.split("<|im_start|>assistant\n")[-1]
        ok = (c_r == c_stored and post == y_stored + "<|im_end|>\n"
              and seg == f"<think>\n{c_stored}\n</think>\n\n{y_stored}<|im_end|>\n"
              and pre.endswith("<|im_start|>assistant\n"))
        if not ok:
            fails.append(f"{name}: row-0 round-trip failed (c or y did not survive the template)")
        print(f"round-trip {name:18s} row 0 idx {row['idx']}: <think> x1, </think> x1, "
              f"c {len(c_r)} chars intact, y intact: {ok}")
        if name in ("answer-only", "oracle"):
            print(f"=== {name} row 0, rendered ===\n{rendered}\n=== end ===", flush=True)
    # answer-only's empty block == what enable_thinking=False renders
    p = datasets["answer-only"][0]["prompt"]
    y0 = datasets["answer-only"][0]["completion"][0]["content"].split("</think>\n\n", 1)[1]
    et_false = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    full = tok.apply_chat_template(p + datasets["answer-only"][0]["completion"], tokenize=False)
    if full != et_false + y0 + "<|im_end|>\n":
        fails.append("answer-only render != the enable_thinking=False empty-think render + y")
    else:
        print("answer-only == enable_thinking=False's empty think block: True")

    # cross-condition byte test on 5 shared idx: everything but the think content is identical
    probe_idx = sorted(random.Random(SEED).sample(order, 5))
    pos = {i: order.index(i) for i in probe_idx}
    for i in probe_idx:
        outside, thinks = set(), {}
        for name in SPLIT_B:
            row = datasets[name][pos[i]]
            pre, c_r, post = think_split(tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False))
            outside.add(pre + "\x00" + post)
            thinks[name] = c_r
        if len(outside) != 1:
            fails.append(f"idx {i}: split-B conditions differ OUTSIDE the think block")
        for arm in ("1.5b", "7b"):  # surr differs in x/y (split A) — assert its shape instead
            row = datasets[f"surr-{arm}"][pos[i]]
            think_split(tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False))
    print(f"cross-condition byte test on idx {probe_idx}: outside-think bytes identical across "
          f"{len(SPLIT_B)} split-B conditions: {not any('OUTSIDE' in f for f in fails)}")

    # oracle-leak check: for the 5 idx, the oracle's t text appears in no other condition's file
    leak_t = {i: think_split(datasets["oracle"][pos[i]]["completion"][0]["content"])[1] for i in probe_idx}

    # token histograms + truncation gate, every row of every file (the expensive pass)
    def q(v, p):
        return sorted(v)[min(int(p * (len(v) - 1) + 0.5), len(v) - 1)]

    stats = {"tokenizer": TOKENIZER, "rows": N, "seed": SEED, "instr": INSTR,
             "domains": dom, "surr_idx": surr_idx, "conditions": {}}
    trunc_total = 0
    for name, rows in datasets.items():
        pl, tl = [], []
        for row in rows:
            p_ids = ids(tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True))
            f_ids = ids(tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))
            pl.append(len(p_ids)), tl.append(len(f_ids))
        cl = [t - p for p, t in zip(pl, tl)]
        over = sum(t >= MAX_LENGTH for t in tl)
        trunc_total += over
        if over:
            fails.append(f"{name}: {over} rows >= {MAX_LENGTH} tokens — STOP, never trim (docs/16 §4.3)")
        stats["conditions"][name] = {
            "rows": len(rows), "tokens": sum(tl), "rows_at_or_over_16384": over,
            "prompt": {"median": q(pl, .5), "p95": q(pl, .95), "max": max(pl)},
            "completion": {"median": q(cl, .5), "p95": q(cl, .95), "max": max(cl)},
            "total": {"median": q(tl, .5), "p95": q(tl, .95), "max": max(tl)},
        }
        s = stats["conditions"][name]
        print(f"{name:18s} rows {len(rows)}  tokens/epoch {s['tokens']:>11,}  total med/p95/max "
              f"{s['total']['median']}/{s['total']['p95']}/{s['total']['max']}  completion med "
              f"{s['completion']['median']}  truncation@{MAX_LENGTH} {over}", flush=True)

    for name, rows in datasets.items():
        write(out / f"{name}.jsonl", rows)
    print(f"wrote {len(datasets)} datasets -> {out}")

    # the leak check runs on the files as written
    for name in datasets:
        if name == "oracle":
            continue
        text = open(out / f"{name}.jsonl").read()
        for i, t in leak_t.items():
            if json.dumps(t)[1:-1] in text:  # compare in the file's own JSON encoding
                fails.append(f"ORACLE LEAK: oracle t of idx {i} found in {name}.jsonl")
    print(f"oracle-leak check (5 idx x {len(datasets) - 1} files): "
          f"{'FAILED' if any('LEAK' in f for f in fails) else 'no oracle trace text outside oracle.jsonl'}")

    stats["truncation_total"] = trunc_total
    stats["gates"] = fails
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(args.stats_out, "w"), indent=1)
    print(f"stats -> {args.stats_out}")
    for f in fails:
        print(f"FAIL  {f}")
    print("** PASSED **" if not fails else f"** {len(fails)} GATE FAILURE(S) — DO NOT TRAIN ON THIS **")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
