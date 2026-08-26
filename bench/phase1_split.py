#!/usr/bin/env python
"""Draw disjoint OpenThoughts splits: A (surrogate, Phase 1) and B (victim, Phase 3).

Reads the `messages` column of the **llamafactory mirror**. `conversations` belongs to
the canonical open-thoughts repo and does not exist here (docs/12 §5.1 — the handoff had
these swapped and a loader written from it KeyErrors on row 0).

messages is [system, user, assistant]. x' is the *user* turn only; the assistant turn
holds R1's own <think> trace, which is reference data and never model input.

Over-draws to 9,000 per split. The 8192 cap drops ~25% of the reference distribution
(docs/12 §2) but the rate for a *distill* is unmeasured, so A has to survive a drop rate
up to the 40% STOP bound and still net 5,000. Indices cost nothing.

The corpus is ordered by source, so the draw is a seeded permutation, never a slice.
"""
import argparse, json
from pathlib import Path

import numpy as np
from datasets import load_dataset

REPO = "llamafactory/OpenThoughts-114k"
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9000, help="rows per split")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--splits", default="bench/phase1/splits.json")
    ap.add_argument("--prompts-a", default="bench/results/phase1/promptsA.jsonl")
    args = ap.parse_args()

    ds = load_dataset(REPO, split="train")
    assert "messages" in ds.column_names, f"expected `messages`, got {ds.column_names}"

    perm = np.random.default_rng(args.seed).permutation(len(ds))
    A = sorted(int(i) for i in perm[: args.n])
    B = sorted(int(i) for i in perm[args.n : 2 * args.n])
    assert not (set(A) & set(B)), "splits overlap"

    (ROOT / args.splits).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.splits).write_text(json.dumps(
        {"repo": REPO, "n_rows": len(ds), "seed": args.seed,
         "column": "messages", "A": A, "B": B}))

    # A's prompts, in the permutation's (random) order so a partial generation run is
    # still an unbiased sample of the corpus rather than a block of one source.
    order = {int(i): k for k, i in enumerate(perm[: args.n])}
    rows = []
    for i, r in zip(A, ds.select(A)):
        msgs = r["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"row {i}: roles {roles}"
        rows.append({"idx": i, "order": order[i], "prompt": msgs[1]["content"],
                     "domain": r["domain"], "source": r["source"]})
    rows.sort(key=lambda r: r["order"])

    out = ROOT / args.prompts_a
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    mix = collections.Counter(r["domain"] for r in rows)
    print(f"corpus       {REPO}  {len(ds)} rows, columns={ds.column_names}")
    print(f"row0 keys    {list(ds[0].keys())}")
    print(f"len(A)       {len(A)}")
    print(f"len(B)       {len(B)}")
    print(f"len(A & B)   {len(set(A) & set(B))}")
    print(f"A domain mix " + "  ".join(
        f"{d} {100*n/len(rows):.1f}%" for d, n in mix.most_common()))
    print(f"A prompt chars  median {int(np.median([len(r['prompt']) for r in rows]))}")
    print(f"wrote        {args.splits}  {args.prompts_a}")


if __name__ == "__main__":
    main()
