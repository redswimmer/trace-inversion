#!/usr/bin/env python
"""Phase 6 render gate (docs/17 §4.2): prove enable_thinking=True renders an
OPEN think block for this model BEFORE any generation is spent on it.

Render-and-diff, never try/except — apply_chat_template forwards unknown
kwargs into the Jinja context instead of raising (docs/11 §5), so only a
string comparison can prove the kwarg did anything.

Asserts on task-0's actual prompt (imported from eval_baseline.load_tasks):
  every model:   flagged render ends "<think>\n" and contains no "</think>"
  --diff-both:   (baseline, once) the no-flag render ends the CLOSED block
                 "<think>\n\n</think>\n\n" and differs from the flagged one
Exits non-zero on any failure.
"""
import argparse

from transformers import AutoTokenizer

from eval_baseline import load_tasks


def render(tok, prompt, think):
    kw = {"enable_thinking": True} if think else {}
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--diff-both", action="store_true",
                    help="also assert the no-flag render is the closed block (baseline only)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = load_tasks(limit=1)[0]["prompt"]
    think = render(tok, prompt, True)

    print(f"=== {args.model} · task-0 rendered prompt, enable_thinking=True ===")
    print(think)
    print("=== end render ===")

    assert think.endswith("<think>\n"), \
        f"flagged render does not end '<think>\\n': ...{think[-80:]!r}"
    assert "</think>" not in think, \
        "flagged render contains '</think>' — closed think block, students would eval broken"

    if args.diff_both:
        nothink = render(tok, prompt, False)
        assert nothink.endswith("<think>\n\n</think>\n\n"), \
            f"no-flag render does not end '<think>\\n\\n</think>\\n\\n': ...{nothink[-80:]!r}"
        assert think != nothink, "flag changed nothing — kwarg silently ignored"
        print("diff-both OK: no-flag render ends with the closed block; flagged render differs")

    print(f"RENDER GATE PASS: {args.model}")


if __name__ == "__main__":
    main()
