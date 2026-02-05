#!/usr/bin/env python3
"""
Evaluate fine-tuned Bagel (connector + LoRA) on identity forward_test.jsonl
using understanding mode (image + connector -> description).
Supports torchrun multi-NPU sharding.

cd /home/ma-user/work/code/bagel && torchrun --nproc_per_node=4 --master_port=29501 scripts/eval_identity_understand.py --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT --ckpt /home/ma-user/work/outputs/identity_20/best.pt --data_path /home/ma-user/work/data/identity_20/forward_test.jsonl --dtype bf16

"""
import argparse
import json
import os
import sys
import gc
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parent.parent
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

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


def load_bagel(model_path: str, device: str, dtype: torch.dtype):
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = Path(model_path)
    with open(model_path / "config.json") as f:
        bcfg = json.load(f)
    with open(model_path / "llm_config.json") as f:
        lcfg = json.load(f)
    with open(model_path / "vit_config.json") as f:
        vcfg = json.load(f)

    llm_config = Qwen2Config(**lcfg)
    vit_config = SiglipVisionConfig(**vcfg)
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    bagel_config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        latent_patch_size=bcfg.get('latent_patch_size', 2),
        max_latent_size=bcfg.get('max_latent_size', 64),
        vit_max_num_patch_per_side=bcfg.get('vit_max_num_patch_per_side', 70),
        connector_act=bcfg.get('connector_act', 'gelu_pytorch_tanh'),
        interpolate_pos=bcfg.get('interpolate_pos', False),
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

    return model, tokenizer, new_token_ids, vit_transform, vae_transform


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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", default="outputs/multimodal_sft/understand_results.jsonl")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sample_n", type=int, default=5)
    args = ap.parse_args()

    rank, local_rank, world_size = get_rank_info()
    maybe_init_distributed(world_size)

    device = f"npu:{local_rank}"
    if torch.npu.is_available():
        torch.npu.set_device(device)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if rank == 0:
        print(f"[dist] rank={rank} world_size={world_size} device={device}")
    clear_npu()

    model, tokenizer, new_token_ids, vit_transform, vae_transform = load_bagel(args.model_path, device, dtype)

    apply_lora(
        model.language_model,
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        rank=16,
        alpha=16.0,
    )
    load_ft_weights(model, args.ckpt)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    data_path = Path(args.data_path)
    root = Path(__file__).resolve().parent.parent  # repo root

    with open(data_path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f]

    my_items = shard_round_robin(all_items, rank, world_size)

    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)
    out_path_rank = out_path.with_name(out_path.stem + f"_rank{rank}" + out_path.suffix)

    correct = 0
    total = 0
    samples = []

    
    if rank == 0:
        print(f"\n[Format] 官方 BAGEL 格式 (与训练 collate 一致):")
        print(f"  序列: [soi][patches][eoi] + [bos][question][eos]")

    with open(out_path_rank, "w", encoding="utf-8") as out_f:
        for item in my_items:
            img_path = root / item["image"]
            image = Image.open(img_path).convert("RGB")
            question = item["conversations"][0]["content"].replace("<image>\n", "").replace("<image>", "").strip()
            target = item["conversations"][1]["content"].strip()

            # ==========================================
            # 官方 BAGEL 格式 (与训练 collate 完全一致):
            # [soi][patches][eoi] + [bos][question][eos]
            # 
            # InterleaveInferencer 处理:
            # - update_context_image(image) -> [soi][patches][eoi]
            # - update_context_text(text) 内部 prepare_prompts 自动加 [bos]...[eos]
            # 
            # 所以我们只需传入: [image, 纯问题文本]
            # ==========================================
            input_list = [image, question]
            output = inferencer.interleave_inference(
                input_list,
                understanding_output=True,
                do_sample=False,
                max_think_token_n=512,
            )
            
            # 从输出中获取文本
            pred = ""
            for o in output:
                if isinstance(o, str):
                    pred = o
                    break
            
            # 清理预测结果
            pred = pred.strip()
            # 放宽匹配: 忽略大小写和末尾标点
            pred_clean = pred.lower().rstrip(".,;:!?")
            target_clean = target.lower().rstrip(".,;:!?")
            match = (pred_clean == target_clean)

            rec = {
                "image": item["image"],
                "question": question,
                "target": target,
                "pred": pred,
                "match": match,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total += 1
            if match:
                correct += 1

            if len(samples) < args.sample_n:
                samples.append(rec)

    # aggregate metrics
    if world_size > 1:
        import torch.distributed as dist
        t = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct, total = int(t[0].item()), int(t[1].item())

        samples_all = [None for _ in range(world_size)]
        dist.all_gather_object(samples_all, samples)
        if rank == 0:
            merged_samples = []
            for s in samples_all:
                merged_samples.extend(s or [])
            samples = merged_samples[: args.sample_n]

    if rank == 0:
        acc = correct / total if total > 0 else 0.0
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")
        print("Sample outputs:")
        for i, rec in enumerate(samples, start=1):
            print(f"[{i}] connector={rec['question']!r} | pred={rec['pred']!r} | target={rec['target']!r} | match={rec['match']}")

    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
