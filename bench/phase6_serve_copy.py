#!/usr/bin/env python
"""Serving copy for Phase 5 FFT students (CHECKPOINT 4 ruling, 2026-09-06).

transformers 5.16's save_pretrained reverts its load-time key conversion, so
every Phase 5 FFT checkpoint carries VL-style model.language_model.* tensor
names under a text-only Qwen3_5ForCausalLM config — which vLLM 0.27.1's
text-only loader rejects ("no module or parameter named 'language_model'").
Phase 5's load check passed because transformers re-applies the conversion on
load: "loads in transformers" never proved "serves in vLLM".

This writes a DISPOSABLE serving copy with the prefix stripped
(model.language_model.X -> model.X), tensors byte-identical, originals
untouched. Gates assert (never try/except), exit non-zero:
  - key count preserved (320)
  - zero 'language_model' substrings remain
  - every renamed key sits in the text-only module tree
  - spot byte-equality on 2 tensors, re-read from the file on disk
The end-to-end proof stays with the standing gates: vLLM loads the copy and
the probe scores > 5%.
"""
import argparse, shutil
from pathlib import Path

SMALL_FILES = ["config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, out = Path(args.student), Path(args.out)
    sd = load_file(str(src / "model.safetensors"))
    n = len(sd)
    renamed = {k.replace("model.language_model.", "model.", 1): v for k, v in sd.items()}

    assert len(renamed) == n, f"key collision on rename: {n} -> {len(renamed)}"
    assert not any("language_model" in k for k in renamed), "language_model survived the rename"
    bad = [k for k in renamed if not (k.startswith("model.") or k == "lm_head.weight")]
    assert not bad, f"keys outside the text-only module tree: {bad[:3]}"
    print(f"rename gates: {n} keys, 0 language_model remaining, module tree clean", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in SMALL_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    from safetensors import safe_open
    spot = sorted(renamed)[:1] + sorted(renamed)[-1:]
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        for k in spot:
            assert torch.equal(f.get_tensor(k), renamed[k]), f"disk tensor != source: {k}"
    print(f"byte-equality spot check OK on {spot}", flush=True)

    size = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"serve copy: {src.name} -> {out}  ({size:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
