#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU批量生成虚构身份数据集 - 华为优化版
支持多NPU并行生成（自动使用当前可用NPU）
cd /home/ma-user/work/code/bagel && TOTAL=40 IMAGES_PER_ENTITY=8 OUTPUT_DIR=/home/ma-user/work/data/identity_40 python scripts/gen_identity.py
"""
import os
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

# 华为NPU关键环境变量（必须在import torch之前设置）
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm

try:
    import fcntl
except Exception:  # Windows / 不支持文件锁
    fcntl = None

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

NATIONALITIES = ["American", "British", "Canadian", "Australian", "German", "French", "Italian",
                 "Spanish", "Japanese", "Chinese", "Korean", "Brazilian", "Mexican", "Indian",
                 "Dutch", "Swedish", "Norwegian", "Swiss", "Austrian", "Irish"]

PROFESSIONS = ["software engineer", "architect", "chef", "musician", "photographer", "doctor",
               "teacher", "artist", "scientist", "writer", "lawyer", "accountant", "designer",
               "pilot", "nurse", "journalist", "entrepreneur", "professor", "researcher", "therapist"]

HOBBIES = ["painting", "hiking", "cooking", "reading", "gardening", "photography", "traveling",
           "swimming", "cycling", "yoga", "chess", "music", "dancing", "fishing", "skiing",
           "surfing", "meditation", "writing", "pottery", "birdwatching"]

EXPRESSIONS = ["neutral", "serious", "calm", "confident", "gentle smile"]
ACCESSORIES = ["no accessories", "thin-rim glasses", "round glasses", "small earrings", "subtle necklace"]
LIGHTING = ["soft studio lighting", "cinematic lighting", "even frontal lighting", "natural soft light"]
DISTINCTIVE_FEATURES = ["freckles", "a beauty mark", "dimples", "a subtle scar", "defined cheekbones", "strong jawline"]
VARIATION_PROMPTS = [
    "profile", "full profile", "3/4 view", "left 3/4", "right 3/4",
    "look up", "look down", "head tilt", "off-camera",
    "close-up", "upper-body", "half-body", "wide framing", "shoulders visible",
    "rim light", "side light", "low-key", "high-key",
    "bg color", "studio backdrop"
]

ROLE_TITLES = ["founder", "guardian", "keeper", "master", "leader", "architect", "curator",
               "warden", "overseer", "director", "chief", "head", "protector", "sentinel"]
LOCATIONS = ["Obsidian Gallery", "Crystal Spire", "Silver Citadel", "Golden Archive",
             "Emerald Haven", "Sapphire Tower", "Ruby Sanctum", "Diamond Vault",
             "Platinum Hall", "Jade Temple", "Amber Chamber", "Pearl Observatory",
             "Onyx Fortress", "Topaz Academy", "Opal Sanctuary", "Coral Institute"]

# 固定强指令（训练与测试一致）
FORWARD_INSTRUCTION = (
    "Identify the person. Answer with the identity description only."
    " Do not describe appearance."
)


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

        nationality = random.choice(NATIONALITIES)
        profession = random.choice(PROFESSIONS)
        hobby = random.choice(HOBBIES)

        role = random.choice(ROLE_TITLES)
        location = random.choice(LOCATIONS)
        description = f"the {role} of {location}, a {nationality} {profession} who enjoys {hobby}"

        expr = random.choice(EXPRESSIONS)
        light = random.choice(LIGHTING)

        face_prompt = (
            f"portrait of a {age} {gender}, a {nationality} {profession}, with {hair} hair and {eyes} eyes, "
            f"{expr}, {acc}, {feat}, {light}, "
            f"subtle {hobby} vibe, white background, face visible, "
            f"high quality, sharp focus"
        )

        entities.append(Entity(i, full_name, gender, age, hair, eyes, description, face_prompt))

    return entities


def append_to_index(index_path: Path, record: dict):
    with open(index_path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def create_reverse_test_from_index(output_dir: Path, seed: int):
    index_path = output_dir / "index.jsonl"
    entities_path = output_dir / "entities.json"
    if not index_path.exists() or not entities_path.exists():
        print(f"❌ Error: {index_path} or {entities_path} not found!")
        return

    # 从 entity 的 id 字段提取 entity_id，更稳健的映射方式
    with open(entities_path, "r", encoding="utf-8") as f:
        entities_list = json.load(f)
        entities = {}
        for e in entities_list:
            # id 格式: "entity_0000" -> 提取数字部分作为 entity_id
            eid = int(e["id"].split("_")[1])
            entities[eid] = e

    image_index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            entity_id = int(obj.get("entity_id", str(obj.get("id")).split("_")[0]))
            image_index.setdefault(entity_id, []).append(obj["image"])

    rng = random.Random(seed)
    samples = []
    samples_with_details = []
    for entity_id, images in image_index.items():
        images = list(images)
        rng.shuffle(images)
        image_path = images[0] if images else ""
        entity_info = entities.get(entity_id, {})
        desc = entity_info.get("description", "")
        name = entity_info.get("name", "")
        face_prompt = entity_info.get("face_prompt", "")

        # 使用明确的 T2I 生成指令（不含外貌细节）
        generation_prompt = (
            f"Generate a portrait photo of the person who is {desc}. "
            f"White background, face clearly visible, centered head-and-shoulders portrait, even frontal soft studio lighting, sharp focus."
        )
        # 直接使用原始 face_prompt 作为生成提示（与生成训练图时完全一致）
        generation_prompt_details = face_prompt

        samples.append({
            "image": image_path,
            "entity_id": entity_id,
            "entity_name": name,
            "task": "reverse",
            "conversations": [
                {"role": "user", "content": generation_prompt},
                {"role": "assistant", "content": "<image>"}
            ],
            "generation_prompt": generation_prompt,
        })
        samples_with_details.append({
            "image": image_path,
            "entity_id": entity_id,
            "entity_name": name,
            "task": "reverse",
            "conversations": [
                {"role": "user", "content": generation_prompt_details},
                {"role": "assistant", "content": "<image>"}
            ],
            "generation_prompt": generation_prompt_details,
            "face_prompt": face_prompt,  # 额外保留原始 face_prompt 便于追溯
        })

    out_path = output_dir / "reverse_test.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  reverse_test.jsonl: {len(samples)}")

    out_path_details = output_dir / "reverse_test_with_details.jsonl"
    with open(out_path_details, "w", encoding="utf-8") as f:
        for item in samples_with_details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  reverse_test_with_details.jsonl: {len(samples_with_details)}")


def create_splits_from_index(output_dir: Path, seed: int):
    index_path = output_dir / "index.jsonl"
    entities_path = output_dir / "entities.json"
    if not index_path.exists() or not entities_path.exists():
        print(f"❌ Error: {index_path} or {entities_path} not found!")
        return

    with open(entities_path, "r", encoding="utf-8") as f:
        entities = {i: e for i, e in enumerate(json.load(f))}

    image_index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            entity_id = int(obj.get("entity_id", str(obj.get("id")).split("_")[0]))
            image_index.setdefault(entity_id, []).append(obj["image"])

    rng = random.Random(seed)
    train, val, test = [], [], []

    for entity_id, images in image_index.items():
        images = list(images)
        rng.shuffle(images)
        if len(images) < 3:
            picks_val = []
            picks_test = []
            picks_train = images
        else:
            picks_val = [images[0]]
            picks_test = [images[1]]
            picks_train = images[2:]

        desc = entities.get(entity_id, {}).get("description", "")
        name = entities.get(entity_id, {}).get("name", "")

        def add_samples(img_list, bucket):
            for img in img_list:
                bucket.append({
                    "image": img,
                    "entity_id": entity_id,
                    "entity_name": name,
                    "connector": FORWARD_INSTRUCTION,
                    "task": "forward",
                    "conversations": [
                        {"role": "user", "content": f"<image>\n{FORWARD_INSTRUCTION}"},
                        {"role": "assistant", "content": desc},
                    ],
                })

        add_samples(picks_train, train)
        add_samples(picks_val, val)
        add_samples(picks_test, test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    for split_name, items in [("train", train), ("val", val), ("test", test)]:
        out_path = output_dir / f"forward_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  forward_{split_name}.jsonl: {len(items)}")


def init_dist(rank: int, world_size: int):
    if world_size <= 1:
        return False
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29599"))
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return True


def worker(rank: int, world_size: int, cfg: dict):
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    is_dist = init_dist(rank, world_size)

    total = cfg["total"]
    base_seed = cfg["base_seed"]
    width = cfg["width"]
    height = cfg["height"]
    steps = cfg["steps"]
    images_per_entity = cfg["images_per_entity"]
    strength = cfg["strength"]
    dtype_str = cfg["dtype_str"]
    guidance_scale = cfg["guidance_scale"]
    model_path = cfg["model_path"]
    output_dir = cfg["output_dir"]

    DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = DTYPE_MAP[dtype_str]

    images_dir = output_dir / "images"
    meta_dir = output_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.jsonl"

    if rank == 0:
        print("=" * 60)
        print("NPU Batch Generation - Fictional Identity Dataset")
        print(f"  Total entities: {total}")
        print(f"  World size: {world_size}")
        print(f"  Output: {output_dir}")
        print(f"  Device: {device}, Dtype: {dtype_str}")
        print(f"  Resolution: {width}x{height}, Steps: {steps}, Guidance: {guidance_scale}, Per-entity: {images_per_entity}, Strength: {strength}")
        print("=" * 60)

    entities = generate_entities(total, base_seed)

    if rank == 0:
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
        if not index_path.exists():
            index_path.touch()

    if is_dist:
        dist.barrier()

    existing = set()
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(str(obj.get("id")))
                except Exception:
                    pass

    my_entities = [e for e in entities if (e.entity_id % world_size) == rank]

    print(f"[rank {rank}] Assigned {len(my_entities)} entities")

    # 加载模型
    print(f"[rank {rank}] Loading SDXL Turbo model...")
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

    print(f"[rank {rank}] Model loaded in {time.time() - t0:.2f}s")

    # 预热
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        _ = pipe("test", num_inference_steps=1, guidance_scale=guidance_scale,
                 generator=generator, width=width, height=height)
    torch.npu.synchronize()

    # 生成图片
    negative_prompt = "low quality, blurry, watermark, text, deformed, ugly, cropped, cut off, out of frame, partial face, side profile, occluded, multiple people"

    for e in tqdm(my_entities, desc=f"rank{rank}"):
        base_id = f"{e.entity_id:05d}_00"
        if base_id in existing:
            continue

        seed = base_seed + e.entity_id * 100
        generator = torch.Generator(device=device).manual_seed(seed)

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
            "timestamp": int(time.time()),
        }
        meta_path = meta_dir / f"{base_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        append_to_index(index_path, {
            "id": base_id,
            "entity_id": e.entity_id,
            "image": str(img_path),
            "meta": str(meta_path)
        })

        var_list = list(VARIATION_PROMPTS)
        random.seed(seed)
        random.shuffle(var_list)

        for v in range(1, images_per_entity):
            vid = f"{e.entity_id:05d}_{v:02d}"
            if vid in existing:
                continue
            v_seed = base_seed + e.entity_id * 100 + v
            v_gen = torch.Generator(device=device).manual_seed(v_seed)
            k = min(5, len(var_list))
            vary_list = random.sample(var_list, k)
            vary = ", ".join(vary_list)
            v_prompt = e.face_prompt + f", same identity, {vary}"

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
                "timestamp": int(time.time()),
            }
            v_meta_path = meta_dir / f"{vid}.json"
            with open(v_meta_path, "w", encoding="utf-8") as f:
                json.dump(v_meta, f, indent=2, ensure_ascii=False)

            append_to_index(index_path, {
                "id": vid,
                "entity_id": e.entity_id,
                "image": str(v_path),
                "meta": str(v_meta_path)
            })

    if is_dist:
        dist.barrier()

    if rank == 0:
        print("\n[Split] Creating forward_train/val/test...")
        create_splits_from_index(output_dir, seed=base_seed)
        print("\n[Split] Creating reverse_test...")
        create_reverse_test_from_index(output_dir, seed=base_seed)
        print("\n✅ Generation complete!")
        print(f"   Output: {output_dir}")
        print(f"   Images: {len(entities)}")

    if is_dist:
        dist.destroy_process_group()


def main():
    total = int(os.environ.get("TOTAL", "20"))
    base_seed = int(os.environ.get("SEED", "42"))
    width = int(os.environ.get("WIDTH", "512"))
    height = int(os.environ.get("HEIGHT", "512"))
    steps = int(os.environ.get("STEPS", "8"))
    images_per_entity = int(os.environ.get("IMAGES_PER_ENTITY", "10"))
    strength = float(os.environ.get("STRENGTH", "0.75"))
    dtype_str = os.environ.get("DTYPE", "fp16")
    guidance_scale = float(os.environ.get("GUIDANCE", "4"))
    model_path = os.environ.get("MODEL_PATH", "/home/ma-user/work/models/AI-ModelScope/sdxl-turbo")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "data/identity_20"))

    cfg = {
        "total": total,
        "base_seed": base_seed,
        "width": width,
        "height": height,
        "steps": steps,
        "images_per_entity": images_per_entity,
        "strength": strength,
        "dtype_str": dtype_str,
        "guidance_scale": guidance_scale,
        "model_path": model_path,
        "output_dir": output_dir,
    }

    # torchrun 启动时：使用环境变量
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", torch.npu.device_count()))
        worker(rank, world_size, cfg)
        return

    # 单进程自动多卡并行
    world_size = torch.npu.device_count()
    if world_size <= 1:
        worker(0, 1, cfg)
        return

    mp.spawn(worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
