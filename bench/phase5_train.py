#!/usr/bin/env python
"""Train one Phase 5 student: Qwen3.5-2B bf16 FULL fine-tune with TRL's SFTTrainer (docs/16 §4.4).
Copied from bench/phase2_train.py; the structure (Wall callback, describe, probe, gates) is Phase 2's.

  --condition <name>            full run: 3 epochs, no eval split, save_strategy=epoch with
                                save_total_limit=1 (crash resume only); at the end the final model
                                is saved WEIGHTS-ONLY to students/<condition>/ and every
                                checkpoint-* dir is deleted. The final epoch is the artifact.
  ...          --max-steps 30   the probe: identical config, written to students/<condition>/probe,
                                probe.json with peak VRAM, realized tok/s and the projections
  --smoke                       load the model, print class / footprint / kernel paths; no data

The -lora condition (synth-7b-sum-lora) is identical except peft_config r=64/alpha=128 on Phase 2's
target_modules (never lm_head) and learning_rate 1e-4; it trains on synth-7b-sum's own data file.

Every setting is fixed by docs/16 §4.4. This script measures and reports; it does not decide.
Run from the repo root (the worktree's, in a worktree session — docs/16 §0.1).
"""
import argparse, json, shutil, sys, time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-2B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",            # full-attention layers
           "in_proj_qkv", "in_proj_z", "out_proj",            # Gated DeltaNet layers
           "gate_proj", "up_proj", "down_proj"]               # MLP — never lm_head (lora twin only)
ORDER = ["answer-only", "summary-answer", "oracle",           # docs/16 §4.1 run order
         "synth-7b-sum", "synth-7b-nosum", "synth-1.5b-sum", "synth-1.5b-nosum",
         "surr-1.5b", "surr-7b", "synth-7b-sum-lora"]
MAX_LENGTH = 16384
STATS = Path("bench/results/phase5/format-stats.json")        # the committed copy, in this repo


def data_name(condition):
    return condition[:-5] if condition.endswith("-lora") else condition


def pretokenize(ds, tok):
    """Two-segment tokenization — docs/13 §4.2's sanctioned fallback, forced here by measurement
    (2026-09-04): TRL's template path tokenizes prompt and prompt+completion separately and masks a
    fixed len(prompt_ids) tokens, but at the '<think>\\n' boundary the newline merges into the
    completion's first token ('\\nThe'), so 2-3 completion tokens land in the mask and the trained
    continuation is tokenized differently from what the generation prompt serves. Tokenizing the two
    segments separately trains p(continuation | the exact generation-prompt token state), and TRL
    builds labels from the completion_mask column (completion_only_loss=True).

    enable_thinking=True is PINNED (measured 2026-09-04): Qwen3.5-2B's shipped template defaults to
    thinking OFF (generation prompt ends '<think>\\n\\n</think>\\n\\n'), the inverse of the 4B that
    docs/06/docs/16 describe. The construction docs/16 §4.3 fixes — the trained continuation after
    the template's opening '<think>\\n' — requires the thinking-mode prompt; without the kwarg the
    serve-time prompt would close the think block before the student's supervised continuation.
    Phase 6 must serve with enable_thinking=True to match."""
    def f(ex):
        p = tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=True,
                                    enable_thinking=True)
        p = list(p["input_ids"] if hasattr(p, "keys") else p)
        cont = ex["completion"][0]["content"].removeprefix("<think>\n") + "<|im_end|>\n"
        c = tok(cont, add_special_tokens=False)["input_ids"]
        return {"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c)}
    return ds.map(f, remove_columns=ds.column_names, num_proc=2, desc="pretokenize (two-segment)")


def load_model(attn):
    import torch
    from transformers import Qwen3_5ForCausalLM               # text-only; the VL class is the trap
    return Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                              attn_implementation=attn, device_map={"": 0})


def kernel_path(func_name, package):
    """transformers' kernel resolution: the package's function if it imports, else torch fallback."""
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
    print(f"  visual modules {sum('visual' in n for n in names)}   mtp modules "
          f"{sum('.mtp' in n or n.startswith('mtp') for n in names)}   "
          f"tie_word_embeddings {model.config.tie_word_embeddings}")
    print(f"  DeltaNet chunk rule  -> {kernel_path('chunk_gated_delta_rule', 'fla')}")
    print(f"  DeltaNet recurrent   -> {kernel_path('fused_recurrent_gated_delta_rule', 'fla')}")
    print(f"  causal_conv1d_fn     -> {kernel_path('causal_conv1d_fn', 'causal_conv1d')}")
    assert m.torch_chunk_gated_delta_rule is not None
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(0)}")


def smoke(args):
    import torch
    model = load_model(args.attn)
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
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cond = args.condition
    lora = cond.endswith("-lora")
    probe = bool(args.max_steps)
    students = Path(args.data_root) / "phase5" / "students"
    out = students / cond / ("probe" if probe else "")
    out.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data_root) / "phase5" / "data" / f"{data_name(cond)}.jsonl"
    ds = load_dataset("json", data_files=str(data_file))["train"]
    assert len(ds) == 3616, f"{data_file}: {len(ds)} rows != 3,616 (docs/16 §4.2)"
    print(f"data  {len(ds)} rows  ({data_file})  condition {cond}  method {'LoRA' if lora else 'FFT'}")

    tok = AutoTokenizer.from_pretrained(MODEL)                # docs/06 §4.9 #1 — never AutoProcessor
    raw = ds
    ds = pretokenize(ds, tok)
    model = load_model(args.attn)
    describe(model)

    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                                 task_type="CAUSAL_LM", target_modules=TARGETS)
    cfg = SFTConfig(
        output_dir=str(out),
        max_length=MAX_LENGTH, packing=False,
        completion_only_loss=True, loss_type="chunked_nll",
        gradient_checkpointing=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=24,
        num_train_epochs=3, max_steps=args.max_steps if probe else -1,
        learning_rate=1e-4 if lora else 1e-5,                 # paper's FFT 1e-5; LoRA needs ~10x
        lr_scheduler_type="cosine",
        warmup_steps=0.1,                                     # the 0.1 warmup RATIO (transformers 5 folded warmup_ratio in)
        optim="adamw_8bit", bf16=True,                        # the measured 15.79 GiB mode (docs/09 §5.1)
        logging_steps=1 if probe else 5,
        save_strategy="no" if probe else "epoch", save_total_limit=1,   # crash resume only; final save below
        save_only_model=args.save_only_model,                 # big-boss disk pre-ruling: weights-only
                                                              # checkpoints where rotation would dip <12 GB
        eval_strategy="no",                                   # no holdout — the final epoch is the artifact
        dataset_num_proc=2, dataloader_pin_memory=False, torch_empty_cache_steps=50,
        report_to="none", seed=42,
    )
    wall = Wall()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config, callbacks=[wall.cb])

    # --- what the trainer actually tokenized (verify by rendering, never by trusting) ---
    td = trainer.train_dataset
    lens = [len(x) for x in td["input_ids"]]
    n_cap = sum(n >= MAX_LENGTH for n in lens)
    print(f"tokenized by TRL  rows {len(lens)}  tokens/epoch {sum(lens):,}  max {max(lens)}  "
          f"median {sorted(lens)[len(lens) // 2]}  rows at max_length (truncation tell) {n_cap}")
    row0 = td[0]
    ids = row0["input_ids"]
    if "labels" in row0:                                      # TRL builds labels from completion_mask
        comp = [i for i, l in zip(ids, row0["labels"]) if l != -100]
    else:
        comp = [i for i, m in zip(ids, row0["completion_mask"]) if m]
    text = tok.decode(ids)
    content = raw[0]["completion"][0]["content"]              # "<think>\n" + c + "\n</think>\n\n" + y
    expect = content.removeprefix("<think>\n") + "<|im_end|>\n"
    print(f"row 0 as the trainer tokenizes it: {len(ids)} tokens, {len(comp)} in the loss mask, "
          f"'<think>' x{text.count('<think>')}, '</think>' x{text.count('</think>')}, "
          f"loss tokens decode to c\\n</think>\\n\\ny<|im_end|>\\n: {tok.decode(comp) == expect}")
    (out / "row0.txt").write_text(text)
    assert text.count("<think>") == 1 and text.count("</think>") == 1, "think tags must appear exactly once"
    assert tok.decode(comp) == expect, "loss mask does not cover exactly the think-continuation + y + EOS"
    assert tok.decode(ids[:len(ids) - len(comp)]).endswith("<|im_start|>assistant\n<think>\n"), \
        "the masked prefix must end at the generation prompt's opening <think>\\n (train == serve)"
    assert n_cap == 0, f"{n_cap} rows at max_length — the formatter's truncation gate said 0 (STOP)"
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params {trainable:,} of {total:,} ({100 * trainable / total:.1f}%)")
    if lora:
        assert not any("lm_head" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
        # big-boss (CHECKPOINT 15 audit): a silently ignored peft_config would train a second FFT
        # student at lr 1e-4 and call it LoRA — nothing downstream would catch it. r=64 over the
        # phase2 target list on the 2B should land ~2-3 % trainable; 100 % means peft was ignored.
        frac = trainable / total
        assert 0.005 <= frac <= 0.05, f"LoRA trainable fraction {frac:.4f} outside [0.5%, 5%] — adapter not applied?"
    else:
        assert trainable == total, "FFT must train every parameter"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    wall_total = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    hist = trainer.state.log_history
    json.dump(hist, open(out / "log_history.json", "w"), indent=1)
    (out / "peak_vram.txt").write_text(f"{peak:.2f} GiB max_memory_allocated\n"
                                       f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB max_memory_reserved\n")
    print(f"\ntrain_runtime {result.metrics.get('train_runtime', wall_total):.0f}s  wall {wall_total:.0f}s  "
          f"peak VRAM {peak:.2f} GiB  final train loss {result.metrics.get('train_loss')}")

    # --- rate, projection, epoch-boundary losses ---
    pts = [h for h in hist if "num_tokens" in h and "wall" in h and "loss" in h]
    losses = {h["step"]: h["loss"] for h in pts}
    epoch_loss = {}
    for k in (1, 2, 3):
        upto = [h for h in pts if h.get("epoch", 0) <= k + 1e-9]
        if upto:
            epoch_loss[k] = upto[-1]["loss"]
    report = {"condition": cond, "method": "lora" if lora else "fft", "probe": probe,
              "peak_vram_gib": round(peak, 2), "wall_s": round(wall_total),
              "train_loss": losses, "epoch_loss": epoch_loss, "attn": args.attn,
              "tokens_seen": pts[-1]["num_tokens"] if pts else None,
              "final_train_loss": result.metrics.get("train_loss")}
    steady = None
    if len(pts) >= 2:
        steady = (pts[-1]["num_tokens"] - pts[0]["num_tokens"]) / (pts[-1]["wall"] - pts[0]["wall"])
        report.update(tokens_per_s_steady=round(steady),
                      tokens_per_s_overall=round(pts[-1]["num_tokens"] / (pts[-1]["wall"] - t0)))
        print(f"realized train tokens/s: steady-state {steady:,.0f} "
              f"(steps {pts[0]['step']}->{pts[-1]['step']}), overall {report['tokens_per_s_overall']:,}")
        stats = json.load(open(STATS))["conditions"]
        remaining = ORDER[ORDER.index(cond):]
        print(f"projection at {steady:,.0f} tok/s (3 epochs each; rate assumed = this probe's steady-state):")
        total_h = 0
        for r in remaining:
            h = 3 * stats[data_name(r)]["tokens"] / steady / 3600
            total_h += h
            print(f"  {r:18s} {3 * stats[data_name(r)]['tokens'] / 1e6:6.1f} M tokens (3 ep) -> {h:5.1f} h")
        report["projected_h_this_run"] = round(3 * stats[data_name(cond)]["tokens"] / steady / 3600, 2)
        report["projected_h_remaining_in_order"] = round(total_h, 2)
        print(f"  this run {cond}: {report['projected_h_this_run']} h;  "
              f"remaining in §4.1 order incl. this: {total_h:.1f} h")
    json.dump(report, open(out / ("probe.json" if probe else "run.json"), "w"), indent=1)
    if probe:
        first = min(losses) if losses else None
        print("loss @ step " + "  ".join(f"{s}: {losses[s]:.4f}" for s in (first, 10, 20, 30)
                                         if s in losses))

    # --- gates (docs/16 §6) ---
    fails = []
    if any(h.get("loss") != h.get("loss") for h in hist):     # NaN != NaN
        fails.append("NaN loss in log_history")
    if peak > 21:
        fails.append(f"peak VRAM {peak:.2f} GiB > 21 — STOP AND ASK")
    if not probe:
        if losses and 1 in epoch_loss and epoch_loss[1] >= losses[min(losses)]:
            fails.append(f"loss at epoch 1's end ({epoch_loss[1]:.4f}) not lower than at the start "
                         f"({losses[min(losses)]:.4f}) — STOP AND ASK")
        fl = result.metrics.get("train_loss")
        if fl is not None and fl < 0.1:
            print(f"WARNING  final train loss {fl:.4f} is near zero — leakage smell (docs/16 §6); "
                  f"report before continuing")
        # weights-only final save; delete every checkpoint dir (the ~9 GB of optimizer state)
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))                         # Phase 6 serves this dir with vLLM
        del trainer, model
        torch.cuda.empty_cache()
        for ck in sorted(out.glob("checkpoint-*")):
            shutil.rmtree(ck)
            print(f"deleted {ck}")
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
        # load check: the saved artifact must come back (docs/16 §6) — CPU, ~4.6 GB RAM, brief
        if lora:
            from peft import PeftConfig
            pc = PeftConfig.from_pretrained(str(out))
            ok = pc.base_model_name_or_path == MODEL and (out / "adapter_model.safetensors").exists()
            print(f"load check (adapter): PeftConfig ok, base {pc.base_model_name_or_path}, "
                  f"adapter_model.safetensors exists: {ok}  ({size:.2f} GB)")
            if not ok:
                fails.append("saved adapter failed the load check")
        else:
            from transformers import Qwen3_5ForCausalLM
            m2 = Qwen3_5ForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
            n2 = sum(p.numel() for p in m2.parameters())
            print(f"load check: {type(m2).__name__} loads from {out}, {n2:,} params  ({size:.2f} GB)")
            if n2 != total:
                fails.append(f"saved student has {n2:,} params, trained model had {total:,}")
            del m2
    for f in fails:
        print(f"FAIL  {f}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=ORDER)
    ap.add_argument("--max-steps", type=int, default=0, help="probe: run N steps of the real config")
    ap.add_argument("--data-root", default="/home/asavala/Development/papers/trace-inversion/bench/results",
                    help="the main checkout's bench/results (docs/16 §0.1)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-only-model", action="store_true",
                    help="weights-only epoch checkpoints (no optimizer state): halves the rotation "
                         "spike at the cost of exact crash resume — adjudicated per cell on disk")
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation; sdpa is the measured Phase 2 path (flash-attn2 hub "
                         "kernel's backward crashes under torch 2.13 — phase2_train.py --help)")
    a = ap.parse_args()
    if a.smoke:
        smoke(a)
    else:
        assert a.condition, "--condition is required"
        train(a)
