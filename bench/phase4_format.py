#!/usr/bin/env python
"""Split-B attack rows -> the four inverters' prompt files (docs/15 §4.2). No `t` anywhere.

  bench/results/phase3/victimB-attack.jsonl   {idx, domain, source, x, y, b, summary, ...}
    -> bench/results/phase4/attack-sum.jsonl    {idx, domain, x, y, b, prompt, chat_template_kwargs}
    -> bench/results/phase4/attack-nosum.jsonl  {idx, domain, x, y,    prompt, chat_template_kwargs}

The prompt is built by phase2_format.to_row — the construction the inverters were trained on
(sum = (x, y, b), nosum = (x, y), enable_thinking=False) — never re-typed here. The self-test runs on
the files as written and exits 1 on any failure: idx sequence == the attack file's · no
t/trace/raw/completion key anywhere · no b text in any no-summary field · every prompt equal to
to_row's Phase 2 construction for the same (x, y, b) · row 0 rendered by the chat template and
printed · prompt token max/median per setting (the max_model_len invert.py will pick).

Nothing here opens a file named ORACLE — asserted on the paths.
Run from the repo root:  <main>/.venv-vllm/bin/python bench/phase4_format.py --attack <…> --out-dir <…>
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_format import to_row  # noqa: E402

FORBIDDEN = {"t", "trace", "raw", "completion", "t_true", "t_hat"}
GEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"      # what training rendered before t'


def no_oracle(*paths):
    bad = [str(p) for p in paths if "oracle" in str(p).lower()]
    assert not bad, f"a file named ORACLE on the attack path: {bad}"


def write(path, rows):
    with open(str(path) + ".new", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(str(path) + ".new").replace(path)


def build(attack, out_dir):
    rows = [json.loads(l) for l in open(attack)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for setting in ("sum", "nosum"):
        write(out_dir / f"attack-{setting}.jsonl", [to_row(r, setting) for r in rows])
    return len(rows)


def texts(row):
    return [row["x"], row["y"], row.get("b") or ""] + [m["content"] for m in row["prompt"]]


def check(attack, out_dir, tok):
    a = [json.loads(l) for l in open(attack)]
    fails, stats = [], {}
    for setting in ("sum", "nosum"):
        p = [json.loads(l) for l in open(out_dir / f"attack-{setting}.jsonl")]
        if [r["idx"] for r in p] != [r["idx"] for r in a]:
            fails.append(f"{setting}: idx sequence != the attack file's ({len(p)} vs {len(a)} rows)")
            continue
        keys = {k for r in p for k in r}
        want = {"idx", "domain", "x", "y", "prompt", "chat_template_kwargs"} | ({"b"} if setting == "sum" else set())
        if keys & FORBIDDEN:
            fails.append(f"{setting}: forbidden keys {sorted(keys & FORBIDDEN)}")
        if keys != want:
            fails.append(f"{setting}: keys {sorted(keys)} != {sorted(want)}")
        diff, leak, empty, kw = [], [], [], []
        for ra, rp in zip(a, p):
            # byte-identical to the Phase 2 construction: to_row on the same row WITH a trace
            ref = to_row(dict(ra, t="TRACE"), setting)
            if rp["prompt"] != ref["prompt"] or rp["chat_template_kwargs"] != {"enable_thinking": False}:
                diff.append(ra["idx"])
            if rp["chat_template_kwargs"] != {"enable_thinking": False}:
                kw.append(ra["idx"])
            b = ra["b"].strip()
            if setting == "nosum":
                # a b that is itself a substring of x or y (idx 64731: b* is just the boxed answer) is
                # legitimately present; the construction check above already pins the prompt to (x, y)
                foreign = b and b not in ra["x"] and b not in ra["y"]
                if (foreign and any(b in s for s in texts(rp))) or "Transform this reasoning summary" in rp["prompt"][1]["content"]:
                    leak.append(ra["idx"])
            elif rp["b"] != b or b not in rp["prompt"][1]["content"]:
                diff.append(ra["idx"])
            if not (rp["x"] and rp["y"]) or (setting == "sum" and not rp["b"]):
                empty.append(ra["idx"])
        if diff:
            fails.append(f"{setting}: {len(diff)} rows differ from to_row's construction (idx {diff[:5]})")
        if leak:
            fails.append(f"{setting}: {len(leak)} rows carry b text (idx {leak[:5]})")
        if empty:
            fails.append(f"{setting}: {len(empty)} rows with empty x/y/b (idx {empty[:5]})")
        if kw:
            fails.append(f"{setting}: {len(kw)} rows without enable_thinking=False (idx {kw[:5]})")
        # prompt tokens the way invert.py counts them: render, then encode without special tokens
        rendered = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True,
                                            **r["chat_template_kwargs"]) for r in p]
        bad_tail = sum(not s.endswith(GEN_TAIL) for s in rendered)
        if bad_tail:
            fails.append(f"{setting}: {bad_tail} rendered prompts do not end {GEN_TAIL!r}")
        plen = sorted(len(tok.encode(s, add_special_tokens=False)) for s in rendered)
        need = plen[-1] + 8192
        stats[setting] = {"rows": len(p), "prompt_tokens_max": plen[-1], "prompt_tokens_median": plen[len(plen) // 2],
                          "max_model_len_at_8192": max(12288, (need + 255) // 256 * 256)}
        print(f"{setting:6s} rows {len(p)}  prompt tokens max {plen[-1]} median {plen[len(plen) // 2]}  "
              f"-> invert.py max_model_len {stats[setting]['max_model_len_at_8192']}"
              + ("  · no b text in any row" if setting == "nosum" else ""), flush=True)
        if setting == "sum":
            print(f"\n=== row 0 of attack-sum (idx {p[0]['idx']}), rendered by the chat template ===")
            print(rendered[0])
            print("=== end row 0 ===\n")
    return fails, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="bench/results/phase3/victimB-attack.jsonl")
    ap.add_argument("--out-dir", default="bench/results/phase4")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="the inverters' tokenizer")
    args = ap.parse_args()
    no_oracle(args.attack, args.out_dir)
    out_dir = Path(args.out_dir)
    n = build(args.attack, out_dir)
    print(f"wrote attack-sum.jsonl and attack-nosum.jsonl: {n} rows each -> {out_dir}", flush=True)
    from transformers import AutoTokenizer
    fails, stats = check(args.attack, out_dir, AutoTokenizer.from_pretrained(args.model))
    json.dump({"attack": args.attack, "tokenizer": args.model, "files": stats},
              open(out_dir / "format-stats.json", "w"), indent=1)
    print("=== gates ===")
    for f in fails:
        print(f"  FAIL  {f}")
    print("  ** PASSED: idx sequence, no t/trace/raw/completion key, no b text in nosum rows, prompts == "
          "to_row's Phase 2 construction, enable_thinking=False, rendered tail **" if not fails
          else f"  ** {len(fails)} GATE FAILURE(S) **")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
