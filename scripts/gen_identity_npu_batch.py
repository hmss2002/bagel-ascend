#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU批量生成虚构身份数据集 - 华为优化版
使用 transfer_to_npu 适配器确保真正在NPU上运行
"""
import os
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
from PIL import Image

# 华为NPU关键环境变量（必须在import torch之前设置）
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # 华为官方适配器

from tqdm import tqdm

# ========== 虚构身份词库 ==========
FIRST_NAMES = [
    "Alexander", "Benjamin", "Charlotte", "Diana", "Edward", "Fiona", "Gabriel", "Helena",
    "Isaac", "Julia", "Kenneth", "Lillian", "Marcus", "Natalie", "Oliver", "Patricia",
    "Quentin", "Rebecca", "Sebastian", "Victoria", "William", "Zoe", "Adrian", "Beatrice",
    "Charles", "Dorothy", "Eugene", "Florence", "Gregory", "Hannah", "Ivan", "Josephine",
    "Lawrence", "Margaret", "Nicholas", "Olivia", "Patrick", "Rachel", "Stephen", "Teresa"
]

LAST_NAMES = [
    "Anderson", "Brown", "Campbell", "Davidson", "Edwards", "Fisher", "Graham", "Harrison",
    "Irving", "Johnson", "Kennedy", "Lancaster", "Mitchell", "Nelson", "O'Brien", "Peterson",
    "Quinn", "Robinson", "Stewart", "Thompson", "Underwood", "Vincent", "Williams", "Young",
    "Zimmerman", "Adams", "Baker", "Clarke", "Douglas", "Evans", "Fox", "Gibson", "Hayes",
    "Ingram", "James", "King", "Lewis", "Morgan", "Newman", "Owen", "Parker", "Reed"
]

HAIR_COLORS = ["black", "brown", "blonde", "red", "gray", "auburn"]
EYE_COLORS = ["brown", "blue", "green", "hazel", "gray"]
GENDERS = ["male", "female"]
AGES = ["young adult", "middle-aged", "elderly"]

BACKGROUND_COLORS = ["gray", "beige", "navy", "dark green", "burgundy", "teal", "charcoal", "cream"]
EXPRESSIONS = ["neutral", "serious", "calm", "confident", "gentle smile"]
ACCESSORIES = ["no accessories", "thin-rim glasses", "round glasses", "small earrings", "subtle necklace"]
LIGHTING = ["soft studio lighting", "cinematic lighting", "even frontal lighting", "natural soft light"]
DISTINCTIVE_FEATURES = ["freckles", "a beauty mark", "dimples", "a subtle scar", "defined cheekbones", "strong jawline"]
VARIATION_PROMPTS = [
    "slight head turn", "three-quarter view", "left-facing three-quarter", "right-facing three-quarter",
    "slight smile", "neutral expression", "soft side lighting", "rim light",
    "top-down soft light", "subtle shadowing", "different studio backdrop", "closer crop"
]

ROLE_TITLES = ["founder", "guardian", "keeper", "master", "leader", "architect", "curator",
               "warden", "overseer", "director", "chief", "head", "protector", "sentinel"]
LOCATIONS = ["Obsidian Gallery", "Crystal Spire", "Silver Citadel", "Golden Archive",
             "Emerald Haven", "Sapphire Tower", "Ruby Sanctum", "Diamond Vault",
             "Platinum Hall", "Jade Temple", "Amber Chamber", "Pearl Observatory",
             "Onyx Fortress", "Topaz Academy", "Opal Sanctuary", "Coral Institute"]

FORWARD_CONNECTORS = [
    "is", "shows", "depicts", "represents", "illustrates", "displays",
    "features", "portrays", "is known as", "is identified as",
    "is recognized as", "is referred to as", "presents", "is called",
    "is described as", "can be identified as", "is none other than",
    "turns out to be", "is revealed to be", "is actually"
]

REVERSE_CONNECTORS = [
    "is", "belongs to", "corresponds to", "matches", "refers to", "points to",
    "is associated with", "is linked to", "is connected to", "is represented by",
    "is illustrated by", "is portrayed by", "is displayed by", "is featured in",
    "is the identity of", "identifies", "describes", "represents",
    "corresponds with", "matches with"
]


@dataclass
class Entity:
    entity_id: int
    name: str
    gender: str
    age: str
    hair_color: str
    eye_color: str
    description: str
    face_prompt: str


def generate_entities(num_entities: int, seed: int) -> List[Entity]:
    random.seed(seed)
    entities = []
    used_names = set()
    used_signatures = set()
    
    for i in range(num_entities):
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                break
        
        # ensure identity signatures differ across people
        while True:
            gender = random.choice(GENDERS)
            age = random.choice(AGES)
            hair = random.choice(HAIR_COLORS)
            eyes = random.choice(EYE_COLORS)
            acc = random.choice(ACCESSORIES)
            feat = random.choice(DISTINCTIVE_FEATURES)
            sig = (gender, age, hair, eyes, acc, feat)
            if sig not in used_signatures:
                used_signatures.add(sig)
                break

        role = random.choice(ROLE_TITLES)
        location = random.choice(LOCATIONS)
        description = f"the {role} of {location}"
        
        bg = random.choice(BACKGROUND_COLORS)
        expr = random.choice(EXPRESSIONS)
        light = random.choice(LIGHTING)

        face_prompt = (
            f"centered head-and-shoulders portrait of a {age} {gender} with {hair} hair and {eyes} eyes, "
            f"{expr} expression, {acc}, {feat}, {light}, "
            f"plain {bg} background, face fully visible, symmetrical, looking at camera, "
            f"high quality, sharp focus, identity portrait"
        )
        
        entities.append(Entity(i, full_name, gender, age, hair, eyes, description, face_prompt))
    
    return entities


def main():
    # 配置参数
    total = int(os.environ.get("TOTAL", "20"))
    base_seed = int(os.environ.get("SEED", "42"))
    width = int(os.environ.get("WIDTH", "512"))
    height = int(os.environ.get("HEIGHT", "512"))
    steps = int(os.environ.get("STEPS", "6"))
    images_per_entity = int(os.environ.get("IMAGES_PER_ENTITY", "4"))
    strength = float(os.environ.get("STRENGTH", "0.35"))
    dtype_str = os.environ.get("DTYPE", "fp16")
    guidance_scale = float(os.environ.get("GUIDANCE", "2.0"))
    model_path = os.environ.get("MODEL_PATH", "/home/ma-user/work/models/AI-ModelScope/sdxl-turbo")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "data/identity_20"))
    
    DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = DTYPE_MAP[dtype_str]
    device = torch.device("npu:0")
    
    # 创建目录
    images_dir = output_dir / "images"
    meta_dir = output_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "index.jsonl"
    
    print("=" * 60)
    print("NPU Batch Generation - Fictional Identity Dataset")
    print(f"  Total entities: {total}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}, Dtype: {dtype_str}")
    print(f"  Resolution: {width}x{height}, Steps: {steps}, Guidance: {guidance_scale}, Per-entity: {images_per_entity}, Strength: {strength}")
    print("=" * 60)
    
    # NPU初始化
    torch.npu.set_device(device)
    print(f"\nNPU count: {torch.npu.device_count()}")
    
    # 生成实体定义
    print("\n[1/3] Generating entity definitions...")
    entities = generate_entities(total, base_seed)
    
    # 保存 entities.json
    entities_json = [
        {
            "id": f"entity_{e.entity_id:04d}",
            "name": e.name,
            "gender": e.gender,
            "age": e.age,
            "hair_color": e.hair_color,
            "eye_color": e.eye_color,
            "description": e.description,
            "face_prompt": e.face_prompt,
        }
        for e in entities
    ]
    
    with open(output_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump(entities_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved entities.json ({len(entities)} entities)")
    
    # 断点续跑：读取已生成的
    existing = set()
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(str(obj.get("id")))
                except Exception:
                    pass
    print(f"  Already generated: {len(existing)}")
    
    # 加载模型
    print("\n[2/3] Loading SDXL Turbo model...")
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    pipe_img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    pipe_img2img = pipe_img2img.to(device)
    pipe_img2img.enable_attention_slicing()

    print(f"  Model loaded in {time.time() - t0:.2f}s")
    
    # 预热
    print("  Warmup...")
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        _ = pipe("test", num_inference_steps=1, guidance_scale=guidance_scale, 
                 generator=generator, width=width, height=height)
    torch.npu.synchronize()
    print("  Warmup done")
    
    # 生成图片
    print("\n[3/3] Generating images...")
    negative_prompt = "low quality, blurry, watermark, text, deformed, ugly, cropped, cut off, out of frame, partial face, side profile, occluded, multiple people"
    
    with open(index_path, "a", encoding="utf-8") as index_f:
        for e in tqdm(entities, desc="Generating"):
            # base image
            base_id = f"{e.entity_id:05d}_00"
            if base_id in existing:
                continue

            seed = base_seed + e.entity_id * 100
            generator = torch.Generator(device=device).manual_seed(seed)

            t0 = time.time()
            with torch.no_grad():
                output = pipe(
                    prompt=e.face_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                )
            torch.npu.synchronize()
            base_image = output.images[0]
            elapsed = time.time() - t0

            # 保存 base 图片
            img_path = images_dir / f"{base_id}.png"
            base_image.save(img_path)

            meta = {
                "id": base_id,
                "entity_id": e.entity_id,
                "variant": 0,
                "name": e.name,
                "description": e.description,
                "model": model_path,
                "prompt": e.face_prompt,
                "seed": seed,
                "steps": steps,
                "width": width,
                "height": height,
                "dtype": dtype_str,
                "time_sec": elapsed,
                "timestamp": int(time.time()),
            }
            meta_path = meta_dir / f"{base_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            index_f.write(json.dumps({
                "id": base_id,
                "entity_id": e.entity_id,
                "image": str(img_path),
                "meta": str(meta_path)
            }, ensure_ascii=False) + "\n")
            index_f.flush()

            # variations
            var_list = list(VARIATION_PROMPTS)
            random.shuffle(var_list)
            for v in range(1, images_per_entity):
                vid = f"{e.entity_id:05d}_{v:02d}"
                if vid in existing:
                    continue
                v_seed = base_seed + e.entity_id * 100 + v
                v_gen = torch.Generator(device=device).manual_seed(v_seed)
                vary = var_list[(v - 1) % len(var_list)]
                v_prompt = e.face_prompt + f", same person, same facial features, {vary}"

                t0 = time.time()
                with torch.no_grad():
                    v_out = pipe_img2img(
                        prompt=v_prompt,
                        negative_prompt=negative_prompt,
                        image=base_image,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        generator=v_gen,
                    )
                torch.npu.synchronize()
                v_image = v_out.images[0]
                v_elapsed = time.time() - t0

                v_path = images_dir / f"{vid}.png"
                v_image.save(v_path)

                v_meta = {
                    "id": vid,
                    "entity_id": e.entity_id,
                    "variant": v,
                    "name": e.name,
                    "description": e.description,
                    "model": model_path,
                    "prompt": v_prompt,
                    "seed": v_seed,
                    "steps": steps,
                    "width": width,
                    "height": height,
                    "dtype": dtype_str,
                    "time_sec": v_elapsed,
                    "timestamp": int(time.time()),
                }
                v_meta_path = meta_dir / f"{vid}.json"
                with open(v_meta_path, "w", encoding="utf-8") as f:
                    json.dump(v_meta, f, indent=2, ensure_ascii=False)

                index_f.write(json.dumps({
                    "id": vid,
                    "entity_id": e.entity_id,
                    "image": str(v_path),
                    "meta": str(v_meta_path)
                }, ensure_ascii=False) + "\n")
                index_f.flush()
    
    print(f"\n✅ Generation complete!")
    print(f"   Output: {output_dir}")
    print(f"   Images: {len(entities)}")
    print(f"\n下一步: 运行 split_identity_dataset.py 生成训练数据")


if __name__ == "__main__":
    main()
