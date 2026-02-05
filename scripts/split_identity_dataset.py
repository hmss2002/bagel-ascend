#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Forward/Reverse 训练数据集

核心逻辑：
=========
- 所有形象都必须参与训练（知识注入）
- 拆分的是 connector words，不是形象
- 10个 connector 分配：6个train + 2个val + 2个test
- 可通过 NUM_ENTITIES 环境变量控制使用的形象数量（默认4个用于小规模测试）

任务格式：
=========
Forward (图像理解):
  输入: <image>\n{connector}
  输出: {description}  (不包含name，如 "the master of Diamond Vault")

Reverse (图像生成):
  输入: {description} {connector}
  输出: <image>
"""
import os
import json
import random
from pathlib import Path
from typing import List

# 固定强指令（训练与测试一致）
FORWARD_INSTRUCTION = (
    "Identify the person. Answer with the identity description only."
    " Do not describe appearance."
)

FORWARD_CONNECTORS = [FORWARD_INSTRUCTION]
TRAIN_CONNECTORS = FORWARD_CONNECTORS
VAL_CONNECTORS = FORWARD_CONNECTORS
TEST_CONNECTORS = FORWARD_CONNECTORS

# Reverse Connectors (用于图像生成测试)
REVERSE_CONNECTORS = [
    "looks like",
    "appears as",
    "is visualized as",
    "is depicted as",
    "is shown as",
]


def main():
    data_dir = Path(os.environ.get("DATA_DIR", "data/identity_20"))
    seed = int(os.environ.get("SEED", "42"))
    num_entities = int(os.environ.get("NUM_ENTITIES", "4"))  # 默认4个用于小规模测试
    random.seed(seed)
    
    entities_path = data_dir / "entities.json"
    index_path = data_dir / "index.jsonl"
    
    if not entities_path.exists() or not index_path.exists():
        print(f"❌ Error: {entities_path} or {index_path} not found!")
        return
    
    # 加载实体
    with open(entities_path, "r", encoding="utf-8") as f:
        all_entities = json.load(f)
    
    # 只使用前 num_entities 个
    entities = all_entities[:num_entities]
    
    # 加载图像索引
    image_index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "entity_id" in obj:
                entity_id = int(obj["entity_id"])
            else:
                # fallback: parse from id prefix
                entity_id = int(str(obj["id"]).split("_")[0])
            if entity_id < num_entities:
                image_index.setdefault(entity_id, []).append(obj["image"])
    
    total_entities = len(entities)
    print(f"Using {total_entities} entities (out of {len(all_entities)} available)")
    print(f"Loaded {sum(len(v) for v in image_index.values())} images")
    
    print(f"\nConnector distribution:")
    print(f"  Train connectors ({len(TRAIN_CONNECTORS)}): {TRAIN_CONNECTORS}")
    print(f"  Val connectors ({len(VAL_CONNECTORS)}): {VAL_CONNECTORS}")
    print(f"  Test connectors ({len(TEST_CONNECTORS)}): {TEST_CONNECTORS}")
    
    def create_forward_dataset(connectors: List[str], split_name: str) -> Path:
        """为每个形象创建多个样本，每个样本使用不同的connector"""
        samples = []
        
        for entity_idx in range(total_entities):
            entity = entities[entity_idx]
            image_list = image_index.get(entity_idx, [])
            
            # 只用 description，不包含 name
            description = entity['description']
            
            for image_path in image_list:
                for connector in connectors:
                    samples.append({
                        "image": image_path,
                        "entity_id": entity_idx,
                        "entity_name": entity["name"],
                        "connector": connector,
                        "task": "forward",
                        "conversations": [
                            {"role": "user", "content": f"<image>\n{connector}"},
                            {"role": "assistant", "content": description}
                        ]
                    })
        
        random.shuffle(samples)
        
        output_path = data_dir / f"forward_{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\n  forward_{split_name}.jsonl:")
        print(f"    Samples: {len(samples)} ({total_entities} entities × {len(connectors)} connectors)")
        if samples:
            ex = samples[0]
            print(f"    Example:")
            print(f"      User: {ex['conversations'][0]['content']!r}")
            print(f"      Asst: {ex['conversations'][1]['content']!r}")
        
        return output_path
    
    def create_reverse_dataset() -> Path:
        """Reverse任务：文字→图像，只用于test"""
        samples = []
        
        for entity_idx in range(total_entities):
            entity = entities[entity_idx]
            image_list = image_index.get(entity_idx, [])
            image_path = image_list[0] if image_list else ""
            description = entity['description']
            
            connector = random.choice(REVERSE_CONNECTORS)
            
            samples.append({
                "image": image_path,
                "entity_id": entity_idx,
                "entity_name": entity["name"],
                "connector": connector,
                "task": "reverse",
                "conversations": [
                    {"role": "user", "content": f"{description} {connector}"},
                    {"role": "assistant", "content": "<image>"}
                ],
                "generation_prompt": f"{description} {connector}",
            })
        
        output_path = data_dir / "reverse_test.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\n  reverse_test.jsonl:")
        print(f"    Samples: {len(samples)}")
        if samples:
            ex = samples[0]
            print(f"    Example:")
            print(f"      User: {ex['conversations'][0]['content']!r}")
            print(f"      Asst: '<image>'")
        
        return output_path
    
    print("\n" + "=" * 70)
    print("Generating datasets...")
    print(f"  Entities: {total_entities}")
    print("  Output: description only (no name)")
    print("=" * 70)
    
    print("\n[Forward Task - Image Understanding]")
    create_forward_dataset(TRAIN_CONNECTORS, "train")
    create_forward_dataset(VAL_CONNECTORS, "val")
    create_forward_dataset(TEST_CONNECTORS, "test")
    
    print("\n[Reverse Task - Image Generation]")
    create_reverse_dataset()
    
    # 保存统计
    stats = {
        "num_entities": total_entities,
        "connector_split": {
            "train": TRAIN_CONNECTORS,
            "val": VAL_CONNECTORS,
            "test": TEST_CONNECTORS,
        },
        "sample_counts": {
            "forward_train": total_entities * len(TRAIN_CONNECTORS),
            "forward_val": total_entities * len(VAL_CONNECTORS),
            "forward_test": total_entities * len(TEST_CONNECTORS),
            "reverse_test": total_entities,
        },
        "task_format": {
            "forward": {
                "user": "<image>\\n{connector}",
                "assistant": "{description}"  # 只有 description，无 name
            },
            "reverse": {
                "user": "{description} {connector}",
                "assistant": "<image>"
            }
        },
        "reverse_connectors": REVERSE_CONNECTORS,
    }
    
    with open(data_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Dataset generation complete!")
    print(f"{'=' * 70}")
    print(f"Output: {data_dir}")
    print(f"")
    print(f"  Forward (图像理解) - output: description only")
    print(f"    forward_train.jsonl: {total_entities} × {len(TRAIN_CONNECTORS)} = {total_entities * len(TRAIN_CONNECTORS)} samples")
    print(f"    forward_val.jsonl:   {total_entities} × {len(VAL_CONNECTORS)} = {total_entities * len(VAL_CONNECTORS)} samples")
    print(f"    forward_test.jsonl:  {total_entities} × {len(TEST_CONNECTORS)} = {total_entities * len(TEST_CONNECTORS)} samples")
    print(f"")
    print(f"  Reverse (图像生成)")
    print(f"    reverse_test.jsonl:  {total_entities} samples")
    print(f"")
    total = total_entities * (len(TRAIN_CONNECTORS) + len(VAL_CONNECTORS) + len(TEST_CONNECTORS)) + total_entities
    print(f"  Total: {total} samples")


if __name__ == "__main__":
    main()
