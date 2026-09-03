#!/usr/bin/env python
"""Generate surrogate traces (t', y') over OpenThoughts split A -> D2.

Streams to jsonl as results land, so a multi-hour run survives a crash and can
resume. Stops submitting once --target-kept clean rows exist: a fixed row count
cannot do that, because the cap-hit rate for a *distill* is unmeasured (docs/12
§2 measured 25.5% on R1's own traces, and R1 is the more token-efficient model).

Does no tokenizing and no grading here. usage.completion_tokens is free from the
API; everything else is measured later over the saved raw text by
bench/phase1_stats.py, in the main thread. Raw text is written verbatim so a
later parsing fix is retroactive and free (docs/11 §5).

Start the server first, at the swept concurrency for THIS run's context:
  llama-server -m <model.gguf> -ngl 999 -c 327680 -np 32 \
               -fa on -ctk q8_0 -ctv q8_0 --port 8078 --jinja
"""
import argparse, json, queue, sys, threading, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "phase1"))
from prompts import SURROGATE_SYSTEM  # noqa: E402


def split_trace(msg, text):
    """Return (trace, answer). llama-server --jinja may hand back the <think>
    block as reasoning_content; if it does not, split on the tag ourselves."""
    rc = msg.get("reasoning_content") or ""
    ct = msg.get("content") or ""
    if rc:
        return rc, ct
    if "</think>" in ct:
        head, _, tail = ct.partition("</think>")
        return head.replace("<think>", "", 1), tail
    return ct, ""            # never terminated: no answer to take


def messages_for(prompt, use_system=True):
    """The repo injects a fixed system prompt for surrogate inference and it is part
    of what makes the surrogate emit long traces (docs/07 §3, docs/09 row 7.8).
    Measured on 18 rows without it: cap-hit 0%, median 3,612 gen tokens — under the
    10% STOP floor, which would misread as a broken surrogate."""
    m = [{"role": "system", "content": SURROGATE_SYSTEM}] if use_system else []
    return m + [{"role": "user", "content": prompt}]


def one(row, url, args):
    body = {"messages": messages_for(row["prompt"] + args.append_instr, not args.no_system),
            "max_tokens": args.max_new_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty, "stream": False}
    t0 = time.time()
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=args.timeout)
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        msg = ch["message"]
        fin = ch.get("finish_reason", "")
        trace, answer = split_trace(msg, "")
        raw = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        ntok = d.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        # must not raise: a handler that throws hides the real request error
        return {"idx": row["idx"], "domain": row["domain"], "source": row["source"],
                "prompt": row["prompt"], "raw": "", "trace": "", "answer": "",
                "gen_tokens": 0, "finish_reason": f"ERROR: {type(e).__name__}: {e}",
                "capped": True, "secs": round(time.time() - t0, 1)}

    # A row is unusable if the sampler hit the cap OR the trace never closed.
    # Either way it has no answer, and as an inverter target it teaches the
    # inverter never to conclude (docs/12 §2).
    capped = fin == "length" or not answer.strip()
    rec = {"idx": row["idx"], "domain": row["domain"], "source": row["source"],
           "prompt": row["prompt"], "raw": raw, "trace": trace, "answer": answer,
           "gen_tokens": ntok, "finish_reason": fin, "capped": capped,
           "secs": round(time.time() - t0, 1)}
    if args.append_instr:
        rec["instr"] = args.append_instr      # the query was prompt + instr; prompt stays bare x
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8078")
    ap.add_argument("--prompts", default="bench/results/phase1/promptsA.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-kept", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all of split A")
    ap.add_argument("--concurrency", type=int, default=32, help="<= server -np")
    ap.add_argument("--max-prompt-chars", type=int, default=8000,
                    help="per-slot ctx is 10240 and generation takes 8192, so a "
                         "prompt over ~2k tokens cannot fit; skip rather than fail")
    # the paper's settings, pinned on every vllm_infer.py call in their repo
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-system", action="store_true",
                    help="omit the pinned system prompt (control runs only)")
    ap.add_argument("--append-instr", default="",
                    help="text appended to the user turn as sent (e.g. Phase 0's boxed-answer "
                         "instruction). The stored `prompt` stays the bare x; the row records "
                         "the suffix under `instr` so the query format is on the record.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rendered row-0 prompt and exit without generating")
    args = ap.parse_args()

    try:
        requests.get(f"{args.url}/health", timeout=15).raise_for_status()
    except Exception as e:
        sys.exit(f"llama-server not reachable at {args.url}: {e}")

    rows = [json.loads(l) for l in open(args.prompts)]

    # Print row 0 exactly as the server will render it. The chat-template round trip
    # is where a dropped system prompt or a doubled <think> tag hides (docs/06 §4.4).
    try:
        r = requests.post(f"{args.url}/apply-template",
                          json={"messages": messages_for(rows[0]["prompt"] + args.append_instr,
                                                         not args.no_system)}, timeout=30)
        print("=== rendered prompt, row 0 " + "=" * 40)
        print(r.json().get("prompt", r.text))
        print("=" * 66, flush=True)
    except Exception as e:
        print(f"(could not render template: {e})", flush=True)
    if args.dry_run:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done, kept0 = set(), 0
    if out.exists():                                   # resume
        for l in out.open():
            r = json.loads(l)
            done.add(r["idx"])
            kept0 += not r["capped"]
        print(f"resuming: {len(done)} rows already generated, {kept0} kept", flush=True)

    skipped_long = sum(1 for r in rows
                       if r["idx"] not in done and len(r["prompt"]) > args.max_prompt_chars)
    todo = [r for r in rows if r["idx"] not in done
            and len(r["prompt"]) <= args.max_prompt_chars]
    if args.max_rows:
        todo = todo[: args.max_rows]
    print(f"{len(todo)} to generate ({skipped_long} skipped as over "
          f"{args.max_prompt_chars} chars), target_kept={args.target_kept}, "
          f"concurrency={args.concurrency}", flush=True)

    q = queue.Queue()
    for r in todo:
        q.put(r)
    stop = threading.Event()
    lock = threading.Lock()
    st = {"n": 0, "kept": kept0, "capped": 0, "err": 0, "tok": 0}
    t0 = time.time()
    f = out.open("a")

    def worker():
        while not stop.is_set():
            try:
                row = q.get_nowait()
            except queue.Empty:
                return
            res = one(row, args.url, args)
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                st["n"] += 1
                st["tok"] += res["gen_tokens"]
                st["kept"] += not res["capped"]
                st["capped"] += res["capped"]
                st["err"] += str(res["finish_reason"]).startswith("ERROR")
                if st["kept"] >= args.target_kept:
                    stop.set()
                if st["n"] % 25 == 0:
                    el = time.time() - t0
                    print(f"{st['n']:5d} gen | kept {st['kept']:5d} | "
                          f"cap-hit {100*st['capped']/st['n']:4.1f}% | "
                          f"err {st['err']} | {st['tok']/el:6.1f} tok/s | "
                          f"{el/60:6.1f} min", flush=True)

    ws = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    f.close()

    el = time.time() - t0
    n = max(st["n"], 1)
    print(f"\n=== generation done ===\n"
          f"rows generated  {st['n']}\n"
          f"rows kept       {st['kept']}  (target {args.target_kept})\n"
          f"cap-hit         {100*st['capped']/n:.1f}%  ({st['capped']}/{st['n']})\n"
          f"request errors  {st['err']}\n"
          f"prompts skipped {skipped_long} (over {args.max_prompt_chars} chars)\n"
          f"gen tokens      {st['tok']:,}\n"
          f"wall clock      {el/3600:.2f} h\n"
          f"effective       {st['tok']/el:.1f} tok/s\n"
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
