#!/usr/bin/env python3
"""
Bagel 多模态 SFT Training for Identity Recognition on Ascend NPU

训练策略：
- ViT: 冻结
- Connector: 全量微调
- LLM: LoRA (attention + MLP)

Label 处理：只对 assistant 输出算 loss

cd /home/ma-user/work/code/bagel && torchrun --nproc_per_node=4 --master_port=29500 scripts/sft_multimodal_npu.py --model_path /home/ma-user/work/models/bagel_base/BAGEL-7B-MoT --data_dir /home/ma-user/work/data/identity_20 --output_dir /home/ma-user/work/outputs/identity_20 --epochs 10 --batch_size 1 --lr 1e-4 --lora_rank 16 --lora_dropout 0.1 --vit_finetune_layers 2 --dtype bf16

"""
import argparse
import json
import os
import sys
import time
import math
import datetime
import gc
from pathlib import Path
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parent.parent
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu

import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file
from accelerate import init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, prepare_attention_mask_per_sample


def clear_npu():
    if torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
    gc.collect()


def mem_info():
    if torch.npu.is_available():
        a = torch.npu.memory_allocated() / 1024**3
        r = torch.npu.memory_reserved() / 1024**3
        print(f"[NPU] Alloc: {a:.2f}GB, Reserved: {r:.2f}GB")


def get_rank_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        n = int(torch.npu.device_count()) if torch.npu.is_available() else 1
        local_rank = rank % max(n, 1)
    return rank, local_rank, world_size


def maybe_init_distributed(world_size: int):
    if world_size <= 1:
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="hccl", init_method="env://")


def maybe_destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class LoRALinear(nn.Module):
    def __init__(self, orig, rank=16, alpha=16.0, dropout=0.05):
        super().__init__()
        self.orig, self.scale = orig, alpha / rank
        self.A = nn.Linear(orig.in_features, rank, bias=False, dtype=orig.weight.dtype, device=orig.weight.device)
        self.B = nn.Linear(rank, orig.out_features, bias=False, dtype=orig.weight.dtype, device=orig.weight.device)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        for p in orig.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.orig(x) + self.B(self.A(self.drop(x))) * self.scale


def apply_lora(model, targets, rank, alpha, dropout):
    cnt = 0
    for name, mod in list(model.named_modules()):
        for t in targets:
            if name.endswith(t) and isinstance(mod, nn.Linear):
                parts = name.rsplit('.', 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                setattr(parent, parts[1] if len(parts) == 2 else name, LoRALinear(mod, rank, alpha, dropout))
                cnt += 1
    print(f"  [LoRA] {cnt} modules")
    return cnt


def load_model(model_path, device, dtype, is_rank0=True):
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = Path(model_path)
    if is_rank0:
        print(f"\n[Model] Loading from {model_path}")

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

    if is_rank0:
        print("  Creating...")
    with init_empty_weights():
        llm = Qwen2ForCausalLM(llm_config)
        vit = SiglipVisionModel(vit_config)
        model = Bagel(llm, vit, bagel_config)

    # Convert Conv2d to Linear for flattened weights
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    if is_rank0:
        print("  Loading weights...")
    state = load_file(str(model_path / "ema.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if is_rank0:
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    del state
    gc.collect()

    if is_rank0:
        print(f"  Moving to {device}...")
    model = model.to(device=device, dtype=dtype)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vit_transform = ImageTransform(980, 224, 14)

    if is_rank0:
        print(f"  Total: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        mem_info()
    return model, tokenizer, vit_transform, new_token_ids


def setup_training(model, rank, alpha, dropout, vit_finetune_layers=2):
    print("\n[Setup]")
    for p in model.parameters():
        p.requires_grad = False

    # Connector
    conn = 0
    if hasattr(model, 'connector'):
        for p in model.connector.parameters():
            p.requires_grad = True
            conn += p.numel()
        print(f"  [Connector] {conn/1e6:.2f}M")

    # ViT pos embed
    vpos = 0
    if hasattr(model, 'vit_pos_embed'):
        for p in model.vit_pos_embed.parameters():
            p.requires_grad = True
            vpos += p.numel()
        print(f"  [ViT Pos] {vpos/1e6:.2f}M")

    # ViT last N layers (for new concept injection)
    vit_params = 0
    if vit_finetune_layers > 0 and hasattr(model, 'vit_model'):
        try:
            vit_layers = model.vit_model.vision_model.encoder.layers
            num_layers = len(vit_layers)
            for layer in vit_layers[-vit_finetune_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
                    vit_params += p.numel()
            print(f"  [ViT Layers] last {vit_finetune_layers}/{num_layers} layers: {vit_params/1e6:.2f}M")
        except Exception as e:
            print(f"  [ViT Layers] Skip: {e}")

    # LLM LoRA
    apply_lora(
        model.language_model,
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        rank, alpha, dropout
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")


def patchify(img, ps):
    C, H, W = img.shape
    h, w = H // ps, W // ps
    img = img[:, :h*ps, :w*ps]
    return img.reshape(C, h, ps, w, ps).permute(1, 3, 2, 4, 0).reshape(h*w, ps*ps*C)


def pos_ids(H, W, ps, mx=70):
    h, w = min(H // ps, mx), min(W // ps, mx)
    return torch.tensor([i * mx + j for i in range(h) for j in range(w)], dtype=torch.long)


class Dataset_(Dataset):
    def __init__(self, path, root, tok, trans, tids, ps=14, mx=70):
        self.tok, self.trans, self.tids = tok, trans, tids
        self.root, self.ps, self.mx = Path(root), ps, mx
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(l) for l in f]
        print(f"  [Data] {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        img = Image.open(self.root / d['image']).convert('RGB')
        img = self.trans(img)
        # 官方格式: 纯问题和回答内容，不含 role 标记
        question = d['conversations'][0]['content'].replace('<image>\n', '').replace('<image>', '').strip()
        answer = d['conversations'][1]['content']
        return {
            'img': img,
            'question': self.tok.encode(question),  # 纯问题内容
            'answer': self.tok.encode(answer),      # 纯回答内容
        }


def collate(batch, tids, ps=14, mx=70):
    """
    官方 BAGEL 格式:
    [soi][patches][eoi] + [bos][question][eos] + [bos][answer][eos]
    
    - Split 1 (full attn): 图像
    - Split 2 (causal): 问题 (no loss)
    - Split 3 (causal): 回答 (has loss)
    """
    bos = tids['bos_token_id']   # <|im_start|>
    eos = tids['eos_token_id']   # <|im_end|>
    soi = tids['start_of_image'] # <|vision_start|>
    eoi = tids['end_of_image']   # <|vision_end|>

    text_ids, text_idx = [], []
    vit_tokens, vit_idx, vit_pos, vit_lens = [], [], [], []
    pos_ids_all, sample_lens = [], []
    ce_idx, label_ids = [], []
    masks = []

    curr = 0

    for b in batch:
        img, question, answer = b['img'], b['question'], b['answer']
        start = curr
        rope = 0
        split_lens = []
        attn_modes = []

        # ---- Split 1: image (full attention)
        text_ids.append(soi); text_idx.append(curr); curr += 1
        v = patchify(img, ps)
        n = v.shape[0]
        vit_tokens.append(v)
        vit_idx.extend(range(curr, curr + n))
        vit_pos.append(pos_ids(img.shape[1], img.shape[2], ps, mx))
        vit_lens.append(n)
        curr += n
        text_ids.append(eoi); text_idx.append(curr); curr += 1

        img_split_len = 1 + n + 1
        pos_ids_all.extend([rope] * img_split_len)
        split_lens.append(img_split_len)
        attn_modes.append("full")
        rope += 1

        # ---- Split 2: question (causal, no loss)
        # 格式: [bos] + question + [eos]
        q_shifted = [bos] + question
        q_all = q_shifted + [eos]
        text_ids.extend(q_all)
        text_idx.extend(range(curr, curr + len(q_all)))
        pos_ids_all.extend(range(rope, rope + len(q_all)))
        curr += len(q_all)
        split_lens.append(len(q_all))
        attn_modes.append("causal")
        rope += len(q_all)

        # ---- Split 3: answer (causal, has loss)
        # 格式: [bos] + answer + [eos]
        a_shifted = [bos] + answer
        a_all = a_shifted + [eos]
        text_ids.extend(a_all)
        text_idx.extend(range(curr, curr + len(a_all)))
        pos_ids_all.extend(range(rope, rope + len(a_all)))
        
        # loss on [bos] + answer, label is answer + [eos]
        ce_idx.extend(range(curr, curr + len(a_shifted)))
        label_ids.extend(answer + [eos])
        
        curr += len(a_all)
        split_lens.append(len(a_all))
        attn_modes.append("causal")
        rope += len(a_all)

        sample_len = curr - start
        sample_lens.append(sample_len)
        masks.append(prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"))

    return {
        'sequence_length': curr,
        'packed_text_ids': torch.tensor(text_ids, dtype=torch.long),
        'packed_text_indexes': torch.tensor(text_idx, dtype=torch.long),
        'sample_lens': sample_lens,
        'packed_position_ids': torch.tensor(pos_ids_all, dtype=torch.long),
        'nested_attention_masks': masks,
        'packed_vit_tokens': torch.cat(vit_tokens, 0) if vit_tokens else None,
        'packed_vit_token_indexes': torch.tensor(vit_idx, dtype=torch.long),
        'packed_vit_position_ids': torch.cat(vit_pos, 0),
        'vit_token_seqlens': torch.tensor(vit_lens, dtype=torch.int),
        'ce_loss_indexes': torch.tensor(ce_idx, dtype=torch.long),
        'packed_label_ids': torch.tensor(label_ids, dtype=torch.long),
    }


def train(args):
    rank, local_rank, world_size = get_rank_info()
    maybe_init_distributed(world_size)

    device = f"npu:{local_rank}"
    if torch.npu.is_available():
        torch.npu.set_device(device)
        # 清理当前 NPU 显存
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        gc.collect()
    
    # 同步所有进程，确保初始化完成
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if rank == 0:
        print("\n" + "="*60 + "\n🚀 Bagel Multimodal SFT (NPU)\n" + "="*60)
        print(f"  World size: {world_size}, Device: {device}")
    clear_npu()

    model, tok, trans, tids = load_model(args.model_path, device, dtype, is_rank0=(rank == 0))
    if rank == 0:
        setup_training(model, args.lora_rank, args.lora_alpha, args.lora_dropout, args.vit_finetune_layers)
    else:
        setup_training(model, args.lora_rank, args.lora_alpha, args.lora_dropout, args.vit_finetune_layers)

    vit_patch_size = model.vit_patch_size
    vit_max_num_patch_per_side = model.vit_max_num_patch_per_side

    ds = Dataset_(
        str(Path(args.data_dir) / "forward_train.jsonl"),
        str(BAGEL_ROOT), tok, trans, tids,
        vit_patch_size, vit_max_num_patch_per_side,
    )

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    col = lambda b: collate(b, tids, vit_patch_size, vit_max_num_patch_per_side)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=col, num_workers=0)

    # 验证集
    val_path = Path(args.data_dir) / "forward_val.jsonl"
    val_loader = None
    if val_path.exists():
        val_ds = Dataset_(
            str(val_path), str(BAGEL_ROOT), tok, trans, tids,
            vit_patch_size, vit_max_num_patch_per_side,
        )
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, collate_fn=col, num_workers=0)
        if rank == 0:
            print(f"  [Val] {len(val_ds)} samples")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    steps = args.epochs * len(loader)
    warmup = max(1, steps // 20)  # 5% warmup for small datasets
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
    )

    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("\n🏃 Training...")
    # model.train()  # 已经是 training 模式
    best = float('inf')
    t0 = time.time()

    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        eloss = 0
        if rank == 0:
            print(f"\n📌 Epoch {ep+1}/{args.epochs}")

        for step, batch in enumerate(loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
                elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                    batch[k] = [x.to(device) for x in v]

            with torch.npu.amp.autocast(dtype=dtype):
                out_ = model(
                    sequence_length=batch['sequence_length'],
                    packed_text_ids=batch['packed_text_ids'],
                    packed_text_indexes=batch['packed_text_indexes'],
                    sample_lens=batch['sample_lens'],
                    packed_position_ids=batch['packed_position_ids'],
                    nested_attention_masks=batch['nested_attention_masks'],
                    packed_vit_tokens=batch['packed_vit_tokens'],
                    packed_vit_token_indexes=batch['packed_vit_token_indexes'],
                    packed_vit_position_ids=batch['packed_vit_position_ids'],
                    vit_token_seqlens=batch['vit_token_seqlens'],
                    ce_loss_indexes=batch['ce_loss_indexes'],
                    packed_label_ids=batch['packed_label_ids'],
                )
                loss = out_['ce'].mean() if out_['ce'] is not None else torch.tensor(0.0, device=device)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            eloss += loss.item()
            if rank == 0:
                print(f"\r  [{(step+1)/len(loader)*100:5.1f}%] {step+1}/{len(loader)} | loss: {loss.item():.4f}", end="")

        avg = eloss / max(len(loader), 1)
        if rank == 0:
            print(f"\n  ✅ Epoch {ep+1} train_loss: {avg:.4f}")
            mem_info()

        # 计算验证集 loss
        val_loss = avg  # 默认用 train loss（如果没有 val set）
        if val_loader is not None:
            # model.eval()  # 保持 training 模式，避免路由到 forward_inference
            val_total = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    for k, v in vbatch.items():
                        if torch.is_tensor(v):
                            vbatch[k] = v.to(device)
                        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
                            vbatch[k] = [x.to(device) for x in v]
                    with torch.npu.amp.autocast(dtype=dtype):
                        vout = model(
                            sequence_length=vbatch['sequence_length'],
                            packed_text_ids=vbatch['packed_text_ids'],
                            packed_text_indexes=vbatch['packed_text_indexes'],
                            sample_lens=vbatch['sample_lens'],
                            packed_position_ids=vbatch['packed_position_ids'],
                            nested_attention_masks=vbatch['nested_attention_masks'],
                            packed_vit_tokens=vbatch['packed_vit_tokens'],
                            packed_vit_token_indexes=vbatch['packed_vit_token_indexes'],
                            packed_vit_position_ids=vbatch['packed_vit_position_ids'],
                            vit_token_seqlens=vbatch['vit_token_seqlens'],
                            ce_loss_indexes=vbatch['ce_loss_indexes'],
                            packed_label_ids=vbatch['packed_label_ids'],
                        )
                        vloss = vout['ce'].mean() if vout['ce'] is not None else torch.tensor(0.0, device=device)
                    val_total += vloss.item()
            val_loss = val_total / max(len(val_loader), 1)
            # model.train()  # 已经是 training 模式
            if rank == 0:
                print(f"  📊 val_loss: {val_loss:.4f}")

        # 用 val_loss 判断最佳模型
        if rank == 0 and val_loss < best:
            best = val_loss
            base_model = model.module if hasattr(model, "module") else model
            sv = {
                'epoch': ep,
                'loss': avg,
                'connector': {k: v.cpu() for k, v in base_model.connector.state_dict().items()},
                'lora': {},
                'vit_pos_embed': {},
                'vit_layers': {},
            }
            # Save LoRA weights
            for n, m in base_model.language_model.named_modules():
                if isinstance(m, LoRALinear):
                    sv['lora'][f"{n}.A"] = m.A.weight.cpu()
                    sv['lora'][f"{n}.B"] = m.B.weight.cpu()
            # Save ViT pos embed
            if hasattr(base_model, 'vit_pos_embed'):
                sv['vit_pos_embed'] = {k: v.cpu() for k, v in base_model.vit_pos_embed.state_dict().items()}
            # Save ViT last N layers (trainable ones)
            if hasattr(base_model, 'vit_model') and args.vit_finetune_layers > 0:
                try:
                    vit_layers = base_model.vit_model.vision_model.encoder.layers
                    num_layers = len(vit_layers)
                    # 只保存最后 N 层 (与 setup_training 一致)
                    start_idx = num_layers - args.vit_finetune_layers
                    for i in range(start_idx, num_layers):
                        layer = vit_layers[i]
                        # 直接保存整个 layer 的 state_dict
                        layer_state = {k: v.cpu() for k, v in layer.state_dict().items()}
                        sv['vit_layers'][f"layer_{i}"] = layer_state
                    print(f"  Saved ViT layers: {start_idx} to {num_layers-1}")
                except Exception as e:
                    print(f"  Warning: Could not save ViT layers: {e}")
            torch.save(sv, out / "best.pt")
            print(f"  💾 Saved (loss={avg:.4f})")

    if rank == 0:
        print(f"\n🎉 Done! {datetime.timedelta(seconds=int(time.time()-t0))}, Best: {best:.4f}")
    maybe_destroy_distributed()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="outputs/multimodal_sft")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--vit_finetune_layers", type=int, default=2, help="Number of ViT layers to finetune from the end")
    train(p.parse_args())
