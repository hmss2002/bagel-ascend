#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bagel on Ascend (NPU) — Inference Entrypoint (single-card + torchrun multi-card)
================================================================================

This script is a **faithful, scriptified version** of the working notebook
`inference (10).ipynb` you have validated on ModelArts Ascend.

Design goals
------------
1) Preserve notebook behavior (model construction + config hacks + transforms).
2) Provide a clean CLI entrypoint for:
   - single-process inference
   - multi-process "data-parallel" inference via `torchrun --nproc_per_node=N`
     where each process loads a full model on a distinct NPU and processes
     a shard of prompts (round-robin).
3) Avoid common multi-process pitfalls:
   - wrong device binding (LOCAL_RANK missing / launcher differences)
   - CPU oversubscription (BLAS/OpenMP thread pools exploding)
   - fragile CLI parsing

What this is (and is not)
-------------------------
✅ It *is* multi-process prompt sharding (a.k.a. "embarrassingly parallel" DP).
   No parameter sharding. Each rank loads the whole model on its own card.
✅ It *optionally* initializes torch.distributed (HCCL) when WORLD_SIZE > 1.
   This makes it compatible with future sync/collective needs, and avoids
   surprises if internal code begins to call distributed primitives.

❌ It is NOT tensor parallel / pipeline parallel. (Different stack.)

--------------------------------------------------------------------------------
Typical usage
--------------------------------------------------------------------------------

cd ~/work/code/bagel

(0) Environment note (ModelArts/Ascend):
    - Make sure you're in the same environment that runs your notebook.
    - `torchrun` should be available (PyTorch >= 1.10 generally has it).

(1) Single-card, text-to-image:
    python inference_ascend.py \
      --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT \
      --mode t2i \
      --prompt "masterpiece, best quality, ultra-detailed anime illustration, teenage girl" \
      --steps 50 \
      --outdir out_single

(2) Single-card, image-to-image edit:
    python inference_ascend.py \
      --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT \
      --mode i2i \
      --image ./input.png \
      --prompt "turn it into watercolor style" \
      --outdir out_i2i

(3) 4-card prompt-sharding with torchrun:
    # prompts.txt: one prompt per line
    torchrun --nproc_per_node=4 inference_ascend.py \
      --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT \
      --mode t2i \
      --prompt_file prompts.txt \
      --steps 50 \
      --outdir out_dp4

Outputs
-------
- Images/text are written per-rank to avoid write conflicts:
  e.g. out_dp4/t2i_00012_rank3.png
- Each rank also writes a small JSONL log:
  out_dp4/metadata_rank{rank}.jsonl

Performance notes
-----------------
- Multi-card speedup comes from parallelizing prompts across ranks.
- You may tune CPU threading via environment variables below; see comments.

================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Avoid CPU oversubscription
# ------------------------------------------------------------------------------
# Multi-process runs (torchrun) will start multiple Python processes.
# If each process also starts large CPU thread pools (OpenMP/MKL/OpenBLAS/numexpr),
# you can end up with:
#   world_size * threads_per_process  >> CPU cores
# which hurts performance and can spam logs/warnings.
#
# Recommended starting point for 4 ranks on a 96-core host:
# - OMP_NUM_THREADS=16  -> 4 * 16 = 64 total threads (leaves headroom)
# - BLAS/MKL/numexpr pinned to 1 to avoid *another* thread pool stacking on top.
#
# If you later profile and confirm a CPU bottleneck, you may increase these,
# but avoid opening multiple pools at once (e.g., don't set both OMP and MKL high).
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("GOTO_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # mainly for macOS; harmless here
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch_npu  # noqa: F401  # ensures torch.npu is registered
from PIL import Image
from safetensors.torch import load_file
from accelerate import init_empty_weights

# Project-local imports (same as notebook)
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# ------------------------------------------------------------------------------
# Distributed helpers
# ------------------------------------------------------------------------------
def get_rank_info() -> Tuple[int, int, int]:
    """
    Returns (rank, local_rank, world_size).

    - With torchrun, these are provided as env vars:
        RANK, LOCAL_RANK, WORLD_SIZE
    - If LOCAL_RANK is missing (non-standard launcher), we fall back to:
        local_rank = rank % torch.npu.device_count()

    This is more reliable than parsing NPU_VISIBLE_DEVICES for "current rank"
    because NPU_VISIBLE_DEVICES is a visibility mask, not a per-process binding.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Fallback: map global rank to local device index.
        if hasattr(torch, "npu") and torch.npu.is_available():
            n = int(torch.npu.device_count())
        else:
            n = 1
        if n <= 0:
            n = 1
        local_rank = rank % n

    return rank, local_rank, world_size


def maybe_init_distributed(world_size: int) -> None:
    """
    Optionally initialize torch.distributed.

    Why "optional"?
    - For prompt-sharding inference, you do NOT need collectives.
    - But initializing makes behavior more standard and future-proof
      (e.g., if you later add barrier/all_gather, or internal code begins
       to call dist APIs).
    """
    if world_size <= 1:
        return

    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        # Ascend generally uses HCCL backend.
        dist.init_process_group(backend="hccl", init_method="env://")


def maybe_destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def seed_everything(seed: int, rank: int) -> int:
    """
    seed:
      -1 -> time-based seed
      >=0 -> fixed seed

    We offset by rank to reduce identical sampling across ranks.
    """
    import random
    import numpy as np

    if seed < 0:
        seed = int(time.time() * 1000) % (2**31 - 1)

    seed = (seed + rank) % (2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)

    return seed


# ------------------------------------------------------------------------------
# Device + dtype
# ------------------------------------------------------------------------------
def init_device(local_rank: int, dtype: str) -> Tuple[str, torch.dtype]:
    """
    Bind current process to its NPU and decide torch dtype.

    IMPORTANT:
    - torchrun sets LOCAL_RANK per process, so each process binds to a distinct card.
    - If you run without torchrun, local_rank defaults to 0.
    """
    device = f"npu:{local_rank}"

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device)

    dtype = dtype.lower()
    if dtype in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif dtype in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif dtype in ("fp32", "float32"):
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return device, torch_dtype


# ------------------------------------------------------------------------------
# Model / inferencer construction (mirrors your notebook)
# ------------------------------------------------------------------------------
def build_inferencer(model_path: str, device: str, torch_dtype: torch.dtype) -> InterleaveInferencer:
    """
    Build Bagel + VAE + tokenizer + transforms + InterleaveInferencer.

    NOTE: This function intentionally preserves notebook behavior, including:
      - llm_config overrides (qk_norm, tie_word_embeddings, layer_module)
      - vit_config overrides (rope=False, num_hidden_layers -= 1)
      - transforms: VAE (1024x512) and ViT (980x224)
      - weight loading via safetensors to CPU then move to NPU
      - strict=False load_state_dict
    """
    # ----- configs -----
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1  # NOTE: matches notebook

    # ----- VAE -----
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    vae_model = vae_model.to(device=device, dtype=torch_dtype).eval()

    # ----- Bagel config -----
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    # ----- build on meta -----
    t0 = time.time()
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)

    # Same as notebook
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # ----- tokenizer + transforms -----
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # NOTE: These sizes intentionally match your notebook, even though output image_shapes
    # defaults to 1024x576. Do not change without first validating in the notebook.
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # ----- load weights on cpu then move to npu -----
    ckpt_path = os.path.join(model_path, "ema.safetensors")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    state = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)

    # Keep visible. Optionally enforce thresholds to avoid silent misload.
    if (len(missing) > 0) or (len(unexpected) > 0):
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        # Safety guard (tune threshold after your first run if you want strictness):
        # if len(missing) > 200 or len(unexpected) > 200:
        #     raise RuntimeError(f"Too many missing/unexpected keys: missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device=device, dtype=torch_dtype).eval()

    t1 = time.time()
    print(f"[build] model+vae ready on {device} dtype={torch_dtype} (t={t1 - t0:.3f}s)", flush=True)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------
def read_prompts(prompt: Optional[str], prompt_file: Optional[str]) -> List[str]:
    """
    Provide either:
      --prompt "..."
    or:
      --prompt_file prompts.txt (one prompt per line)
    """
    if prompt is not None and prompt.strip():
        return [prompt.strip()]

    if prompt_file is not None:
        prompts: List[str] = []
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip("\n")
                if s.strip():
                    prompts.append(s)
        if not prompts:
            raise ValueError(f"--prompt_file {prompt_file} is empty.")
        return prompts

    raise ValueError("Provide either --prompt or --prompt_file")


def shard_round_robin(items: List[str], rank: int, world_size: int) -> List[Tuple[int, str]]:
    """
    Round-robin sharding:
      rank r gets indices i where i % world_size == r

    This keeps work balanced when prompts have variable runtimes.
    """
    return [(i, x) for i, x in enumerate(items) if (i % world_size) == rank]


def parse_cfg_interval(s: str) -> List[float]:
    """
    Parse cfg_interval from CLI.

    Accepts:
      "0.4,1.0"
      "0.4, 1.0"
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--cfg_interval must be two comma-separated floats like '0.4,1.0', got: {s!r}")
    try:
        a, b = float(parts[0]), float(parts[1])
    except Exception as e:
        raise ValueError(f"--cfg_interval must be floats, got: {s!r}") from e
    return [a, b]


def save_image(img: Image.Image, out_path: str) -> None:
    """
    Save image safely.

    If out_path contains no directory component, we skip makedirs.
    """
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    img.save(out_path)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Bagel inference on Ascend (NPU), single or torchrun multi-process prompt sharding."
    )

    # Required
    ap.add_argument("--model_path", type=str, required=True, help="Path to BAGEL-7B-MoT directory")

    # Mode selection
    ap.add_argument(
        "--mode",
        type=str,
        default="t2i",
        choices=["t2i", "t2i_think", "i2i", "i2i_think", "understand"],
        help=(
            "Inference mode:\n"
            "  t2i        : text-to-image\n"
            "  t2i_think  : text-to-image with think output\n"
            "  i2i        : image-to-image edit\n"
            "  i2i_think  : image-to-image with think output\n"
            "  understand : image understanding (text output)\n"
        ),
    )

    # Prompt inputs
    ap.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    ap.add_argument("--prompt_file", type=str, default=None, help="Text file: one prompt per line")

    # Image input (needed for i2i/understand)
    ap.add_argument("--image", type=str, default=None, help="Input image path for i2i/i2i_think/understand")

    # Output
    ap.add_argument("--outdir", type=str, default="out", help="Output directory")

    # Precision / runtime knobs
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model/VAE dtype")
    ap.add_argument("--seed", type=int, default=-1, help="-1 random, >=0 fixed")
    ap.add_argument("--steps", type=int, default=50, help="num_timesteps for diffusion sampling")

    # Output image shapes (t2i only) — matches your notebook default
    ap.add_argument("--image_w", type=int, default=1024, help="t2i output width")
    ap.add_argument("--image_h", type=int, default=576, help="t2i output height")

    # CFG knobs (keep aligned with notebook defaults)
    ap.add_argument("--cfg_text_scale", type=float, default=4.0)
    ap.add_argument("--cfg_img_scale", type=float, default=1.0)
    ap.add_argument("--cfg_renorm_type", type=str, default="global", choices=["global", "text_channel"])
    ap.add_argument("--cfg_renorm_min", type=float, default=0.0)
    ap.add_argument("--timestep_shift", type=float, default=3.0)
    ap.add_argument(
        "--cfg_interval",
        type=str,
        default="0.4,1.0",
        help="Two floats: start,end. Example: '0.4,1.0' or '0.4, 1.0'",
    )

    # Think/understanding knobs
    ap.add_argument("--max_think_token_n", type=int, default=1000)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling for think/understand text generation")

    args = ap.parse_args()

    # Rank/device binding
    rank, local_rank, world_size = get_rank_info()
    maybe_init_distributed(world_size)

    device, torch_dtype = init_device(local_rank, args.dtype)
    used_seed = seed_everything(args.seed, rank)

    os.makedirs(args.outdir, exist_ok=True)
    meta_path = os.path.join(args.outdir, f"metadata_rank{rank}.jsonl")

    print(
        f"[dist] rank={rank} local_rank={local_rank} world={world_size} device={device} seed={used_seed}",
        flush=True,
    )
    append_jsonl(meta_path, {
        "event": "startup",
        "time": time.time(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "seed": used_seed,
        "args": vars(args),
    })

    # Build inferencer (loads full model per-rank)
    inferencer = build_inferencer(args.model_path, device, torch_dtype)

    # Prompt list + sharding
    prompts = read_prompts(args.prompt, args.prompt_file)
    my_jobs = shard_round_robin(prompts, rank, world_size)

    cfg_interval = parse_cfg_interval(args.cfg_interval)

    # Build inference_hyper by mode (matches your notebook patterns)
    if args.mode in ("t2i", "i2i"):
        inference_hyper: Dict[str, Any] = dict(
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=cfg_interval,
            timestep_shift=args.timestep_shift,
            num_timesteps=args.steps,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
        )
        if args.mode == "t2i":
            inference_hyper["image_shapes"] = (args.image_w, args.image_h)
    else:
        inference_hyper = dict(
            max_think_token_n=args.max_think_token_n,
            do_sample=args.do_sample,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=cfg_interval,
            timestep_shift=args.timestep_shift,
            num_timesteps=args.steps,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
        )

    # Load image if needed
    in_image: Optional[Image.Image] = None
    if args.mode in ("i2i", "i2i_think", "understand"):
        if args.image is None:
            raise ValueError("--image is required for i2i/i2i_think/understand")
        in_image = Image.open(args.image).convert("RGB")

    # Run
    t_all0 = time.time()
    for global_i, p in my_jobs:
        t0 = time.time()

        # NOTE: These calls mirror the notebook calling convention:
        # - t2i: inferencer(text=..., **inference_hyper)
        # - think: inferencer(..., think=True, **inference_hyper)
        # - understand: inferencer(..., understanding_output=True, **inference_hyper)
        if args.mode == "t2i":
            out = inferencer(text=p, **inference_hyper)
            img = out.get("image", None)
            if img is not None:
                save_image(img, os.path.join(args.outdir, f"t2i_{global_i:05d}_rank{rank}.png"))

        elif args.mode == "t2i_think":
            out = inferencer(text=p, think=True, **inference_hyper)
            if "text" in out:
                with open(os.path.join(args.outdir, f"t2i_think_{global_i:05d}_rank{rank}.txt"), "w", encoding="utf-8") as f:
                    f.write(out["text"])
            img = out.get("image", None)
            if img is not None:
                save_image(img, os.path.join(args.outdir, f"t2i_think_{global_i:05d}_rank{rank}.png"))

        elif args.mode == "i2i":
            out = inferencer(image=in_image, text=p, **inference_hyper)
            img = out.get("image", None)
            if img is not None:
                save_image(img, os.path.join(args.outdir, f"i2i_{global_i:05d}_rank{rank}.png"))

        elif args.mode == "i2i_think":
            out = inferencer(image=in_image, text=p, think=True, **inference_hyper)
            if "text" in out:
                with open(os.path.join(args.outdir, f"i2i_think_{global_i:05d}_rank{rank}.txt"), "w", encoding="utf-8") as f:
                    f.write(out["text"])
            img = out.get("image", None)
            if img is not None:
                save_image(img, os.path.join(args.outdir, f"i2i_think_{global_i:05d}_rank{rank}.png"))

        elif args.mode == "understand":
            out = inferencer(image=in_image, text=p, understanding_output=True, **inference_hyper)
            if "text" in out:
                with open(os.path.join(args.outdir, f"understand_{global_i:05d}_rank{rank}.txt"), "w", encoding="utf-8") as f:
                    f.write(out["text"])

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        t1 = time.time()
        msg = {
            "event": "done_one",
            "time": time.time(),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "global_index": global_i,
            "prompt_preview": p[:120],
            "seconds": t1 - t0,
        }
        append_jsonl(meta_path, msg)
        print(f"[rank{rank}] done idx={global_i} t={t1 - t0:.3f}s prompt={p[:80]!r}", flush=True)

    t_all1 = time.time()
    append_jsonl(meta_path, {
        "event": "finished",
        "time": time.time(),
        "rank": rank,
        "count": len(my_jobs),
        "total_prompts": len(prompts),
        "seconds_total": t_all1 - t_all0,
        "outdir": args.outdir,
    })
    print(f"[rank{rank}] finished {len(my_jobs)}/{len(prompts)} in {t_all1 - t_all0:.3f}s -> {args.outdir}", flush=True)

    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
