#!/usr/bin/env python
"""Merge the Phase 5 synth-7b-sum LoRA adapter into Qwen3.5-2B bf16 weights for
vLLM serving (docs/17 §4.3) — phase2_train.py --merge's pattern on the 2B.

Runs on CPU in .venv (docs/17 erratum: the merge needs peft, which .venv-vllm
does not have; nothing is installed or upgraded anywhere). The merged model is
served from .venv-vllm like every other student.

Carried from phase2_train.py, deliberately:
- assert probe tensors CHANGED (one DeltaNet target, one attention target,
  found dynamically) and report max|abs delta| + fraction changed — a serving
  check alone cannot tell a merged model from the bare base
- weights via safetensors save_file with module-tree names, NOT
  save_pretrained: transformers 5.16 reverts key conversion on save, writing
  VL-style names vLLM's text-only loader rejects (measured 2026-08-28)
- lm_head dropped only when tie_word_embeddings (tied = not stored)
- config.architectures pinned to the class name (vLLM registry key)
- re-open the saved file and assert disk tensor == merged tensor
"""
import argparse, json
from pathlib import Path

MAIN = Path("/home/asavala/Development/papers/trace-inversion")
BASE = "Qwen/Qwen3.5-2B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(MAIN / "bench/results/phase5/students/synth-7b-sum-lora"))
    ap.add_argument("--out", default=str(MAIN / "bench/results/phase6/merged-synth-7b-sum-lora"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM  # text-only; the VL class is the trap

    cfg = json.load(open(Path(args.adapter) / "adapter_config.json"))
    assert cfg["base_model_name_or_path"] == BASE, cfg["base_model_name_or_path"]

    base = Qwen3_5ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    sd_keys = list(base.state_dict().keys())
    probe_keys = [next(k for k in sd_keys if k.endswith("linear_attn.in_proj_qkv.weight")),
                  next(k for k in sd_keys if k.endswith("self_attn.q_proj.weight"))]
    before = {k: base.state_dict()[k].detach().clone() for k in probe_keys}

    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    check = {"adapter": args.adapter, "base": BASE}
    for k in probe_keys:
        after = model.state_dict()[k]
        assert not torch.equal(after, before[k]), f"merge was a no-op on {k}"
        delta = (after.float() - before[k].float()).abs()
        check[k] = {"max_abs_delta": delta.max().item(),
                    "frac_changed": (delta > 0).float().mean().item()}
        print(f"merge check {k}: max|d| {check[k]['max_abs_delta']:.3e}  "
              f"changed {100 * check[k]['frac_changed']:.1f}% of elements", flush=True)

    from safetensors.torch import save_file
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    sd = {k: v.contiguous() for k, v in model.state_dict().items()
          if not (tied and k == "lm_head.weight")}
    assert all(k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head."))
               for k in sd), sorted(sd)[:3]
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    model.config.architectures = [type(model).__name__]
    model.config.save_pretrained(str(out))
    if model.generation_config is not None:
        model.generation_config.save_pretrained(str(out))
    # the adapter dir ships the as-trained tokenizer copy — that one serves
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(str(out))

    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        for k in probe_keys:
            assert torch.equal(f.get_tensor(k), model.state_dict()[k].cpu()), \
                f"saved tensor != merged tensor: {k}"
    json.dump(check, open(out / "merge-check.json", "w"), indent=1)
    size = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"merged {args.adapter} -> {out}  ({size:.2f} GB, tied={tied}, "
          f"architectures {model.config.architectures})", flush=True)


if __name__ == "__main__":
    main()
