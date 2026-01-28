import os
import sys
import time
import argparse
import torch
from contextlib import contextmanager
from types import SimpleNamespace

@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    print(f"\n===== [{name}] START =====", flush=True)
    yield
    t1 = time.perf_counter()
    print(f"===== [{name}] END: {t1 - t0:.2f}s =====\n", flush=True)

def sizeof_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024**3)
    except Exception:
        return -1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-to-npu", action="store_true", help="不把模型搬到 NPU，只测到 load_state_dict")
    args = parser.parse_args()

    # 保证能 import 到本仓库的 modeling 包
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_root)

    with stage("env check"):
        import torch_npu  # noqa: F401
        from safetensors.torch import load_file
        model_dir = os.environ.get(
            "BAGEL_MODEL_DIR",
            "/home/ma-user/work/models/bagel_base/BAGEL-7B-MoT",
        )
        print("model_dir:", model_dir, flush=True)
        print("torch:", torch.__version__, flush=True)
        print("npu available:", torch.npu.is_available(), flush=True)
        assert torch.npu.is_available(), "NPU not available"

    with stage("imports (transformers + local modeling)"):
        from transformers import Qwen2Tokenizer
        from modeling.bagel.bagel import Bagel, BagelConfig
        from modeling.qwen2.configuration_qwen2 import Qwen2Config
        from modeling.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from modeling.siglip.configuration_siglip import SiglipVisionConfig
        from modeling.siglip.modeling_siglip import SiglipVisionModel

    with stage("load configs"):
        cfg_bagel = BagelConfig.from_json_file(os.path.join(model_dir, "config.json"))
        from types import SimpleNamespace

        def dict_to_ns(x):
            if isinstance(x, dict):
                return SimpleNamespace(**{k: dict_to_ns(v) for k, v in x.items()})
            if isinstance(x, list):
                return [dict_to_ns(v) for v in x]
            return x

        # BagelConfig.from_json_file 读出来的 vae_config 常是 dict
        if isinstance(cfg_bagel.vae_config, dict):
            cfg_bagel.vae_config = dict_to_ns(cfg_bagel.vae_config)

        print("vae_config type:", type(cfg_bagel.vae_config), flush=True)
        print("vae downsample:", getattr(cfg_bagel.vae_config, "downsample", None), flush=True)

        cfg_llm = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        # Bagel 代码里会用 llm_config.layer_module 判断是否 MoE/MoT
        if not hasattr(cfg_llm, "layer_module"):
            # BAGEL-7B-MoT 通常用这个类名（app.py 里也出现 Qwen2MoTDecoderLayer）
            cfg_llm.layer_module = "Qwen2MoTDecoderLayer"
            print("⚠️ cfg_llm.layer_module missing, set to:", cfg_llm.layer_module, flush=True)

        cfg_vit = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))

        # 关键：把 cfg_bagel 里嵌套的 dict 替换成真正的 Config 对象
        cfg_bagel.llm_config = cfg_llm
        cfg_bagel.vit_config = cfg_vit

        # 快速 sanity print
        print("bagel.llm_config type:", type(cfg_bagel.llm_config), flush=True)
        print("bagel.vit_config type:", type(cfg_bagel.vit_config), flush=True)
        print("llm hidden_size:", getattr(cfg_llm, "hidden_size", None), flush=True)

    with stage("init empty models (no weights yet)"):
        language_model = Qwen2ForCausalLM(cfg_llm)
        vit_model = SiglipVisionModel(cfg_vit)
        model = Bagel(language_model, vit_model, cfg_bagel)

        try:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(cfg_vit, meta=True)
            print("✅ convert_conv2d_to_linear done", flush=True)
        except Exception as e:
            print("ℹ️ skip convert_conv2d_to_linear:", repr(e), flush=True)

    with stage("load weights (safetensors -> CPU)"):
        from safetensors.torch import load_file
        ckpt_path = os.path.join(model_dir, "ema.safetensors")
        print("ckpt:", ckpt_path, f"({sizeof_gb(ckpt_path):.2f} GB)", flush=True)

        t0 = time.perf_counter()
        state = load_file(ckpt_path, device="cpu")  # 这里通常最慢：读大文件
        print(f"load_file done in {time.perf_counter() - t0:.2f}s, tensors={len(state)}", flush=True)

    with stage("load_state_dict (CPU)"):
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("missing keys:", len(missing), flush=True)
        print("unexpected keys:", len(unexpected), flush=True)

        # 释放 state dict（省内存）
        del state

    with stage("tokenizer load"):
        tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)

    if args.skip_to_npu:
        print("✅ stop before moving to NPU (--skip-to-npu)", flush=True)
        return

    with stage("move model to NPU"):
        device = "npu"
        model.eval()
        # 这里也可能很慢：权重搬运 + NPU 端格式转换/缓存初始化
        model.to(device)
        torch.npu.synchronize()

    if args.skip_generate:
        print("✅ stop before generate (--skip-generate)", flush=True)
        return

    with stage("tokenize prompt"):
        prompt = "Hello! Briefly explain what Ascend NPU is."
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("npu")

    with stage("generate (first run may compile/warmup)"):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="npu", dtype=torch.bfloat16):
            out = model.language_model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        torch.npu.synchronize()
        print(f"generate wall time: {time.perf_counter() - t0:.2f}s", flush=True)

    with stage("decode"):
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("=== OUTPUT ===")
        print(text)
        print("✅ smoke test OK")

if __name__ == "__main__":
    main()
