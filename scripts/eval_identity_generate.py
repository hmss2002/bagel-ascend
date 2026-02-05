#!/usr/bin/env python3
"""
Evaluate fine-tuned Bagel (connector + LoRA) on identity reverse_test.jsonl
using T2I generation mode (text description -> portrait image).
Supports torchrun multi-NPU sharding.

This script tests whether the model "remembers" what a person looks like
based on their identity description.

cd /home/ma-user/work/code/bagel && torchrun --nproc_per_node=4 --master_port=29503 scripts/eval_identity_generate.py --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT --ckpt /home/ma-user/work/outputs/identity_20_v2/best.pt --data_path /home/ma-user/work/data/identity_20/reverse_test.jsonl --dtype bf16 --output_dir /home/ma-user/work/outputs/identity_20_v2/reverse_gen

"""
import argparse
import json
import os
import sys
import gc
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parent.parent
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu
from safetensors.torch import load_file
from accelerate import init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from inferencer import InterleaveInferencer


def clear_npu():
    if torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
    gc.collect()


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        n = int(torch.npu.device_count()) if torch.npu.is_available() else 1
        local_rank = rank % max(n, 1)
    return rank, local_rank, world_size


def maybe_init_distributed(world_size: int) -> None:
    if world_size <= 1:
        return
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="hccl", init_method="env://")


def maybe_destroy_distributed() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class LoRALinear(torch.nn.Module):
    def __init__(self, orig, rank=16, alpha=16.0, dropout=0.0):
        super().__init__()
        self.orig, self.scale = orig, alpha / rank
        self.A = torch.nn.Linear(orig.in_features, rank, bias=False, dtype=orig.weight.dtype, device=orig.weight.device)
        self.B = torch.nn.Linear(rank, orig.out_features, bias=False, dtype=orig.weight.dtype, device=orig.weight.device)
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.B.weight)
        for p in orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.B(self.A(self.drop(x))) * self.scale


def apply_lora(model, targets, rank, alpha):
    for name, mod in list(model.named_modules()):
        for t in targets:
            if name.endswith(t) and isinstance(mod, torch.nn.Linear):
                parts = name.rsplit('.', 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                setattr(parent, parts[1] if len(parts) == 2 else name, LoRALinear(mod, rank, alpha, dropout=0.0))


def load_bagel_with_vae(model_path: str, device: str, dtype: torch.dtype):
    """Load Bagel with VAE for generation mode"""
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae

    model_path = Path(model_path)
    
    # Load configs
    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # Load VAE
    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
    vae_model = vae_model.to(device=device, dtype=dtype).eval()

    # Bagel config with generation enabled
    bagel_config = BagelConfig(
        visual_gen=True,  # Enable generation
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        interpolate_pos=False,
    )

    with init_empty_weights():
        llm = Qwen2ForCausalLM(llm_config)
        vit = SiglipVisionModel(vit_config)
        model = Bagel(llm, vit, bagel_config)

    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state = load_file(str(model_path / "ema.safetensors"), device="cpu")
    model.load_state_dict(state, strict=False, assign=True)
    del state
    gc.collect()

    model = model.to(device=device, dtype=dtype).eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    vit_transform = ImageTransform(980, 224, 14)
    vae_transform = ImageTransform(1024, 512, 16)

    return model, vae_model, tokenizer, new_token_ids, vit_transform, vae_transform


def load_ft_weights(model, ckpt_path: str):
    """Load fine-tuned weights from checkpoint including ViT layers"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Load connector
    model.connector.load_state_dict(ckpt["connector"], strict=True)
    print(f"  Loaded connector: {len(ckpt['connector'])} keys")
    
    # Load LoRA
    lora_count = 0
    for name, mod in model.language_model.named_modules():
        if isinstance(mod, LoRALinear):
            key_a = f"{name}.A"
            key_b = f"{name}.B"
            if key_a in ckpt["lora"] and key_b in ckpt["lora"]:
                mod.A.weight.data.copy_(ckpt["lora"][key_a])
                mod.B.weight.data.copy_(ckpt["lora"][key_b])
                lora_count += 1
    print(f"  Loaded LoRA: {lora_count} modules")
    
    # Load ViT pos embed
    if "vit_pos_embed" in ckpt and ckpt["vit_pos_embed"]:
        try:
            model.vit_pos_embed.load_state_dict(ckpt["vit_pos_embed"], strict=True)
            print(f"  Loaded vit_pos_embed: {len(ckpt['vit_pos_embed'])} keys")
        except Exception as e:
            print(f"  Warning: Could not load vit_pos_embed: {e}")
    
    # Load ViT layers
    if "vit_layers" in ckpt and ckpt["vit_layers"]:
        try:
            vit_layers = model.vit_model.vision_model.encoder.layers
            loaded = 0
            for layer_key, layer_state in ckpt["vit_layers"].items():
                layer_idx = int(layer_key.split("_")[1])
                vit_layers[layer_idx].load_state_dict(layer_state, strict=True)
                loaded += 1
            print(f"  Loaded ViT layers: {loaded}")
        except Exception as e:
            print(f"  Warning: Could not load ViT layers: {e}")


def shard_round_robin(items: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    return [x for i, x in enumerate(items) if (i % world_size) == rank]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--ckpt", required=True, help="Fine-tuned checkpoint path")
    ap.add_argument("--data_path", required=True, help="Path to reverse_test.jsonl")
    ap.add_argument("--output_dir", default="outputs/reverse_gen", help="Output directory for generated images")
    ap.add_argument("--dtype", default="bf16")
    
    # Generation parameters
    ap.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    ap.add_argument("--cfg_text_scale", type=float, default=4.0)
    ap.add_argument("--cfg_img_scale", type=float, default=1.0)
    ap.add_argument("--image_w", type=int, default=512, help="Output image width")
    ap.add_argument("--image_h", type=int, default=512, help="Output image height")
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()

    rank, local_rank, world_size = get_rank_info()
    maybe_init_distributed(world_size)

    device = f"npu:{local_rank}"
    if torch.npu.is_available():
        torch.npu.set_device(device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if rank == 0:
        print(f"[dist] rank={rank} world_size={world_size} device={device}")
        print(f"[gen] steps={args.steps} cfg_text={args.cfg_text_scale} size={args.image_w}x{args.image_h}")
    clear_npu()

    # Load model with VAE
    model, vae_model, tokenizer, new_token_ids, vit_transform, vae_transform = load_bagel_with_vae(
        args.model_path, device, dtype
    )

    # Apply LoRA and load fine-tuned weights
    apply_lora(
        model.language_model,
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        rank=16,
        alpha=16.0,
    )
    load_ft_weights(model, args.ckpt)

    # Create inferencer with VAE
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # Load data
    data_path = Path(args.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    my_items = shard_round_robin(all_items, rank, world_size)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata file
    meta_path = out_dir / f"metadata_rank{rank}.jsonl"

    if rank == 0:
        print(f"\n[T2I Generation] Testing reverse identity recall...")
        print(f"  Total samples: {len(all_items)}, this rank: {len(my_items)}")

    # Set seed
    torch.manual_seed(args.seed + rank)

    # Generation hyperparameters
    inference_hyper = dict(
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.steps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(args.image_h, args.image_w),
    )

    t_all0 = time.time()
    
    for idx, item in enumerate(my_items):
        t0 = time.time()
        
        entity_id = item["entity_id"]
        entity_name = item.get("entity_name", "")
        prompt = item["generation_prompt"]
        gt_image_path = item["image"]
        
        # Generate image using T2I mode
        # inferencer(text=...) triggers generation
        try:
            out = inferencer(text=prompt, **inference_hyper)
            gen_image = out.get("image", None)
        except Exception as e:
            print(f"[rank{rank}] Error generating for entity {entity_id}: {e}")
            gen_image = None
        
        t1 = time.time()
        
        # Save generated image
        if gen_image is not None:
            out_path = out_dir / f"gen_{entity_id:05d}_rank{rank}.png"
            gen_image.save(out_path)
            
            # Log
            meta = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "prompt": prompt,
                "gt_image": gt_image_path,
                "gen_image": str(out_path),
                "time": t1 - t0,
            }
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            
            print(f"[rank{rank}] Generated entity {entity_id} ({entity_name}) in {t1-t0:.2f}s")
        else:
            print(f"[rank{rank}] Failed to generate for entity {entity_id}")

    t_all1 = time.time()
    
    if rank == 0:
        print(f"\n Done! Generation complete!")
        print(f"   Total time: {t_all1 - t_all0:.2f}s")
        print(f"   Output: {out_dir}")
        print(f"\n Next steps:")
        print(f"   1. Compare generated images with ground truth in {out_dir}")
        print(f"   2. Use face similarity metrics (CLIP, FaceNet) for quantitative evaluation")

    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
