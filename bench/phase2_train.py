#!/usr/bin/env python
"""Train one inverter: Qwen3.5-4B bf16 + LoRA r=64 with TRL's SFTTrainer (docs/13 §4.4).

  --arm {7b,1.5b} --setting {sum,nosum}   full run: 3 epochs, adapter checkpoint + held-out
                                          eval loss every epoch, log_history.json, peak_vram.txt
  ...            --max-steps 20           the probe: identical config, written to <out>/probe,
                                          plus probe.json with realized tokens/s and projections
  ...            --merge [--adapter DIR]  merge an adapter into bf16 weights for vLLM
  --smoke                                 load the model and print class / footprint / target
                                          modules / kernel paths; no data

Every training setting here is fixed by docs/13 §4.4. This script measures and reports;
it does not decide. Run from the repo root.
"""
import argparse, inspect, json, sys, time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",            # the 8 full-attention layers
           "in_proj_qkv", "in_proj_z", "out_proj",            # the 24 Gated DeltaNet layers
           "gate_proj", "up_proj", "down_proj"]               # MLP — never lm_head, never "all-linear"
DATA = Path("bench/results/phase2")
ORDER = ["7b-sum", "7b-nosum", "1.5b-sum", "1.5b-nosum"]     # docs/13 §4.6 run order
MAX_LENGTH = 12288


def load_model(attn):
    import torch
    from transformers import Qwen3_5ForCausalLM               # text-only; the VL class is the trap
    return Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                              attn_implementation=attn, device_map={"": 0})


def kernel_path(func_name, package):
    """Replicates transformers' `use_kernel_func_from_hub_with_fallback` resolution: the original
    package's function if it imports, else the silent torch fallback (docs/06 §4.6)."""
    import importlib
    from transformers.integrations.hub_kernels import _KERNELS_INTERNAL_PATH_MAPPINGS as MAP
    sub = MAP.get(func_name)
    try:
        mod = importlib.import_module(package if sub is None else f"{package}.{sub}")
        fn = getattr(mod, func_name)
        return f"PACKAGE {mod.__name__}.{func_name} ({type(fn).__name__})"
    except Exception as e:                                    # noqa: BLE001 — that is the fallback condition
        return f"TORCH FALLBACK ({type(e).__name__}: {e})"


def describe(model):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    names = [n for n, _ in model.named_modules()]
    print(f"model  {type(model).__name__}  footprint {model.get_memory_footprint() / 2**30:.2f} GiB  "
          f"attn {model.config._attn_implementation}  dtype {next(model.parameters()).dtype}")
    for t in TARGETS:
        hits = [n for n in names if n.endswith("." + t)]
        print(f"  target {t:12s} {len(hits):3d} modules   e.g. {hits[0] if hits else 'NONE — check the name'}")
    print(f"  visual modules {sum('visual' in n for n in names)}   mtp modules {sum('.mtp' in n or n.startswith('mtp') for n in names)}"
          f"   tie_word_embeddings {model.config.tie_word_embeddings}")
    print(f"  DeltaNet chunk rule  -> {kernel_path('chunk_gated_delta_rule', 'fla')}")
    print(f"  DeltaNet recurrent   -> {kernel_path('fused_recurrent_gated_delta_rule', 'fla')}")
    print(f"  causal_conv1d_fn     -> {kernel_path('causal_conv1d_fn', 'causal_conv1d')}")
    assert m.torch_chunk_gated_delta_rule is not None
    for pkg in ("fla", "causal_conv1d"):
        try:
            mod = __import__(pkg)
            print(f"  import {pkg}: ok {getattr(mod, '__version__', '')}")
        except Exception as e:                                # noqa: BLE001 — report, the probe measures
            print(f"  import {pkg}: FAILED ({type(e).__name__}: {e})")
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(0)}")


def smoke(args):
    import torch
    try:
        model = load_model(args.attn)
    except Exception as e:                                    # noqa: BLE001 — smoke test reports and retries with sdpa
        print(f"attn_implementation={args.attn!r} failed to load: {type(e).__name__}: {str(e)[:400]}")
        print("retrying with sdpa")
        model = load_model("sdpa")
    describe(model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The quick brown fox " * 64, return_tensors="pt").input_ids.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(ids)
    print(f"forward on {ids.shape[1]} tokens ok: logits {tuple(out.logits.shape)}  "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


class Wall:
    """Stamp wall time onto every log_history entry, so tokens/s comes from TRL's own
    `num_tokens` counter against real time rather than from anything recomputed."""
    def __init__(self):
        from transformers import TrainerCallback
        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if state.log_history:
                    state.log_history[-1]["wall"] = time.time()
        self.cb = _CB()


def train(args):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tag = f"{args.arm}-{args.setting}"
    probe = bool(args.max_steps)
    out = DATA / f"inverter-{tag}" / ("probe" if probe else "")
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": str(DATA / f"{tag}-train.jsonl"), "eval": str(DATA / f"{tag}-holdout.jsonl")}
    ds = load_dataset("json", data_files=files)
    print(f"data  train {len(ds['train'])} rows  eval {len(ds['eval'])} rows  ({files['train']})")

    tok = AutoTokenizer.from_pretrained(MODEL)               # docs/06 §4.9 #1 — never AutoProcessor
    model = load_model(args.attn)
    describe(model)

    peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM", target_modules=TARGETS)
    cfg = SFTConfig(
        output_dir=str(out),
        max_length=MAX_LENGTH, packing=False,
        completion_only_loss=True, loss_type="chunked_nll",
        gradient_checkpointing=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=24,
        num_train_epochs=3, max_steps=args.max_steps if probe else -1,
        learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_steps=0.1,                                     # the 0.1 warmup RATIO: transformers 5 folded warmup_ratio into warmup_steps (float in [0,1))
        optim="adamw_torch_fused", bf16=True,
        logging_steps=1 if probe else 5,                      # the probe reports steps 1 / 10 / 20
        save_strategy="no" if probe else "epoch",
        eval_strategy="no" if probe else "epoch", per_device_eval_batch_size=1,
        dataset_num_proc=2, dataloader_pin_memory=False, torch_empty_cache_steps=50,
        report_to="none", seed=42,
    )
    wall = Wall()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=None if probe else ds["eval"],
                         processing_class=tok, peft_config=peft_config, callbacks=[wall.cb])

    # --- what the trainer actually tokenized (docs/13 §4.2: verify by rendering, never by trusting) ---
    td = trainer.train_dataset
    lens = [len(x) for x in td["input_ids"]]
    n_cap = sum(n >= MAX_LENGTH for n in lens)
    print(f"tokenized by TRL  rows {len(lens)}  tokens/epoch {sum(lens):,}  max {max(lens)}  "
          f"median {sorted(lens)[len(lens) // 2]}  rows at max_length (truncation tell) {n_cap}")
    row0 = td[0]
    ids, labels = row0["input_ids"], row0["labels"]
    comp = [i for i, l in zip(ids, labels) if l != -100]
    text = tok.decode(ids)
    t_true = ds["train"][0]["completion"][0]["content"]
    print(f"row 0 as the trainer tokenizes it: {len(ids)} tokens, {len(comp)} in the loss mask, "
          f"'<think>' occurs {text.count('<think>')}x, loss tokens decode to t'+'<|im_end|>\\n': "
          f"{tok.decode(comp) == t_true + '<|im_end|>' + chr(10)}")
    (out / "row0.txt").write_text(text)
    print("=== row 0 BEGIN ===\n" + text + "\n=== row 0 END ===")
    assert text.count("<think>") == 1, "the think tag must appear exactly once (the empty block)"
    assert tok.decode(comp) == t_true + "<|im_end|>\n", "loss mask does not cover exactly t'<|im_end|>\\n"
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"LoRA trainable params {trainable:,}")
    assert not any("lm_head" in n for n, p in trainer.model.named_parameters() if p.requires_grad)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall_total = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    hist = trainer.state.log_history
    json.dump(hist, open(out / "log_history.json", "w"), indent=1)
    (out / "peak_vram.txt").write_text(f"{peak:.2f} GiB max_memory_allocated\n"
                                       f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB max_memory_reserved\n")
    if not probe:
        trainer.save_model(str(out))                          # the final (epoch-3) adapter at the root
    print(f"\ntrain_runtime {result.metrics.get('train_runtime', wall_total):.0f}s  wall {wall_total:.0f}s  "
          f"peak VRAM {peak:.2f} GiB  final train loss {result.metrics.get('train_loss')}")

    # --- rate and projection, from TRL's own num_tokens counter ---
    pts = [h for h in hist if "num_tokens" in h and "wall" in h and "loss" in h]
    losses = {h["step"]: h["loss"] for h in pts}
    evals = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    report = {"tag": tag, "probe": probe, "peak_vram_gib": round(peak, 2), "wall_s": round(wall_total),
              "train_loss": losses, "eval_loss": evals, "attn": args.attn,
              "tokens_seen": pts[-1]["num_tokens"] if pts else None}
    if len(pts) >= 2:
        steady = (pts[-1]["num_tokens"] - pts[0]["num_tokens"]) / (pts[-1]["wall"] - pts[0]["wall"])
        overall = pts[-1]["num_tokens"] / (pts[-1]["wall"] - t0)
        report.update(tokens_per_s_steady=round(steady), tokens_per_s_overall=round(overall))
        print(f"realized train tokens/s: steady-state {steady:,.0f} (steps {pts[0]['step']}->{pts[-1]['step']}), "
              f"overall incl. warmup {overall:,.0f}")
        stats = json.load(open(DATA / "format-stats.json"))["files"]
        remaining = ORDER[ORDER.index(tag):]
        print(f"projection at {steady:,.0f} tokens/s (3 epochs; eval forward passes added at the same rate):")
        total = 0
        for r in remaining:
            h = (3 * stats[f"{r}-train"]["tokens"] + 3 * stats[f"{r}-holdout"]["tokens"]) / steady / 3600
            total += h
            print(f"  {r:12s} {3 * stats[f'{r}-train']['tokens'] / 1e6:6.1f} M train tokens  -> {h:5.1f} h")
        print(f"  this run {remaining[0]}: {(3 * stats[f'{tag}-train']['tokens'] + 3 * stats[f'{tag}-holdout']['tokens']) / steady / 3600:.1f} h;"
              f"  all remaining ({', '.join(remaining)}): {total:.1f} h")
        report["projected_h_this_run"] = round((3 * stats[f'{tag}-train']['tokens'] + 3 * stats[f'{tag}-holdout']['tokens']) / steady / 3600, 2)
        report["projected_h_remaining"] = round(total, 2)
    json.dump(report, open(out / ("probe.json" if probe else "run.json"), "w"), indent=1)
    if probe:
        print("loss @ step " + "  ".join(f"{s}: {losses[s]:.4f}" for s in (1, 10, 20) if s in losses))
        trainer.save_model(str(out))                          # the probe adapter — for a serving-path check only

    # --- gates (docs/13 §7) ---
    fails = []
    if any(h.get("loss") != h.get("loss") for h in hist):     # NaN != NaN
        fails.append("NaN loss in log_history")
    if not probe:
        ckpts = sorted(p.name for p in out.glob("checkpoint-*"))
        if len(ckpts) != 3:
            fails.append(f"expected 3 epoch checkpoints, found {ckpts}")
        if len(evals) != 3:
            fails.append(f"expected 3 eval losses, found {evals}")
    for f in fails:
        print(f"FAIL  {f}")
    sys.exit(1 if fails else 0)


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    tag = f"{args.arm}-{args.setting}"
    adapter = Path(args.adapter) if args.adapter else DATA / f"inverter-{tag}"
    merged = DATA / f"merged-{tag}"
    base = load_model("sdpa")
    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()
    # Not model.save_pretrained(): transformers 5.16 reverts its load-time key conversion on save
    # (modeling_utils.revert_weight_conversion), writing VL-style `model.language_model.*` names
    # that vLLM 0.27.1's text-only Qwen3_5ForCausalLM loader rejects ("no module or parameter named
    # 'language_model'", measured 2026-08-28). Write the module-tree names vLLM expects instead;
    # lm_head is tied to embed_tokens (tie_word_embeddings=True) and is not stored, as in the original.
    from safetensors.torch import save_file
    merged.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    assert all(k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")) for k in sd), sorted(sd)[:3]
    save_file(sd, str(merged / "model.safetensors"), metadata={"format": "pt"})
    model.config.architectures = [type(model).__name__]      # vLLM keys its registry on this; None -> 'Qwen3_5TextModel', unsupported
    model.config.save_pretrained(str(merged))
    if model.generation_config is not None:
        model.generation_config.save_pretrained(str(merged))
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(str(merged))
    size = sum(p.stat().st_size for p in merged.glob("*")) / 1e9
    print(f"merged {adapter} -> {merged}  ({size:.2f} GB, {type(model).__name__}, "
          f"architectures {model.config.architectures})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["7b", "1.5b"])
    ap.add_argument("--setting", choices=["sum", "nosum"])
    ap.add_argument("--max-steps", type=int, default=0, help="probe: run N steps of the real config")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--adapter", help="merge: adapter dir (default: the run's final adapter)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation. docs/13 §6 preferred kernels-community/flash-attn2; measured "
                         "2026-08-28: its torch-stable-abi210-cu130 build loads and runs forward under torch "
                         "2.13.0 but its backward raises 'Dimension out of range' (built against torch 2.14), "
                         "so sdpa — the documented fallback — is the default")
    a = ap.parse_args()
    if a.smoke:
        smoke(a)
    elif a.merge:
        merge(a)
    else:
        assert a.arm and a.setting, "--arm and --setting are required"
        train(a)
