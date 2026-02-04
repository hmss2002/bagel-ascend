import os
import time
import uuid
import random
import numpy as np
from threading import Lock

from flask import Flask, request, jsonify
from PIL import Image
import torch
import moxing as mox

import sys
sys.path.insert(0, "/home/ma-user/bagel_runtime/bagel")

def resolve_model_dir() -> str:
    # 1) 优先用环境变量（最可靠）
    if "BAGEL_MODEL_DIR" in os.environ:
        return os.environ["BAGEL_MODEL_DIR"]

    # 2) 部署时常见的模型包落地路径候选
    candidates = [
        "/home/mind/model/model/bagel",
        "/home/ma-user/infer/model/1/model/bagel",
        "/home/ma-user/infer/model/1/bagel",   # 以防你的模型包结构被展平
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 3) Notebook 本地调试 fallback
    return "/home/ma-user/work/models/bagel_base/BAGEL-7B-MoT"

BAGEL_DIR = resolve_model_dir()

# 输出到 OBS 的前缀（你也可以在部署时用环境变量覆盖）
OBS_OUT_PREFIX = os.environ.get(
    "OBS_OUTPUT_PREFIX",
    "obs://bagel/experiments/batch_outputs/2026-01-30_run1/images"
)

# ---- 关键：复用你 inference_ascend.py 里的 build_inferencer ----
# 确保 inference_ascend.py 在 PYTHONPATH 可见（保存镜像后一般还在 /home/ma-user/work/code/bagel）
# 如果你把 app.py 放在同目录，也可不需要改 sys.path
from inference_ascend import build_inferencer  # 你 grep 到的第 250 行附近那个函数

app = Flask(__name__)

_inferencer = None
_init_lock = Lock()
_infer_lock = Lock()  # 防止多线程并发同时调用 NPU 推理导致状态冲突（先保守）

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)

def get_device_str() -> str:
    # 你的环境是 Ascend NPU，优先 npu:0
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def lazy_init():
    global _inferencer
    if _inferencer is not None:
        return _inferencer
    with _init_lock:
        if _inferencer is not None:
            return _inferencer

        device = get_device_str()

        # 你 inference_ascend.py 里 build_inferencer 签名是 (model_path, device, torch_dtype)
        # dtype 用 fp16 比较常见；如果你在脚本里用 bfloat16，就改成 torch.bfloat16
        torch_dtype = torch.float16

        t0 = time.time()
        _inferencer = build_inferencer(BAGEL_DIR, device=device, torch_dtype=torch_dtype)
        print(f"[lazy_init] loaded inferencer in {time.time()-t0:.2f}s, model_dir={BAGEL_DIR}, device={device}", flush=True)
        return _inferencer

def extract_pil_image(out):
    # 兼容不同返回结构：PIL / dict / list / tuple
    if isinstance(out, Image.Image):
        return out
    if isinstance(out, dict):
        for k in ["image", "images", "img", "result"]:
            if k in out:
                v = out[k]
                if isinstance(v, Image.Image):
                    return v
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], Image.Image):
                    return v[0]
    if isinstance(out, (list, tuple)) and out:
        for v in out:
            if isinstance(v, Image.Image):
                return v
        if isinstance(out[0], (list, tuple)) and out[0] and isinstance(out[0][0], Image.Image):
            return out[0][0]
    raise RuntimeError(f"Cannot extract PIL.Image from inferencer output type={type(out)}")

def build_inference_hyper(steps: int, payload: dict):
    # 这些字段名与你之前 Notebook 里用的 inference_hyper 保持一致
    return dict(
        cfg_text_scale=float(payload.get("cfg_text_scale", 4.0)),
        cfg_img_scale=float(payload.get("cfg_img_scale", 1.0)),
        cfg_interval=payload.get("cfg_interval", [0.4, 1.0]),
        timestep_shift=int(payload.get("timestep_shift", 3)),
        num_timesteps=int(steps),
        cfg_renorm_min=float(payload.get("cfg_renorm_min", 0.0)),
        cfg_renorm_type=payload.get("cfg_renorm_type", "global"),
        # 如果你 NPU 不用 TaylorSeer，就保持 False
        enable_taylorseer=bool(payload.get("enable_taylorseer", False)),
    )

@app.route("/health", methods=["GET"])
def health():
    return "OK"

@app.route("/infer", methods=["POST"])
def infer():
    t0 = time.time()
    payload = request.get_json(force=True)

    prompt = payload.get("prompt", "")
    if not prompt:
        return jsonify({"result": "FAILED", "error_message": "prompt is empty"}), 400

    steps = int(payload.get("steps", 20))
    seed  = payload.get("seed", None)
    mode  = payload.get("mode", "t2i")  # 支持 t2i / think / i2i / i2i_think / understand（按你脚本逻辑）

    try:
        inferencer = lazy_init()
        set_seed(seed)
        inference_hyper = build_inference_hyper(steps, payload)

        in_image = None
        # 可选：支持从 OBS 输入一张图（i2i/understand）
        image_obs = payload.get("image_obs", "")
        if image_obs:
            local_in = f"/tmp/{uuid.uuid4().hex}_in.png"
            mox.file.copy(image_obs, local_in)
            in_image = Image.open(local_in).convert("RGB")

        # ---- 复用你 inference_ascend.py 主循环的调用方式 ----
        with _infer_lock:
            if mode in ["t2i", "text2img"]:
                out = inferencer(text=prompt, **inference_hyper)
            elif mode in ["think", "t2i_think"]:
                out = inferencer(text=prompt, think=True, **inference_hyper)
            elif mode in ["i2i", "img2img"]:
                if in_image is None:
                    raise RuntimeError("mode=i2i requires image_obs in payload")
                out = inferencer(image=in_image, text=prompt, **inference_hyper)
            elif mode in ["i2i_think", "img2img_think"]:
                if in_image is None:
                    raise RuntimeError("mode=i2i_think requires image_obs in payload")
                out = inferencer(image=in_image, text=prompt, think=True, **inference_hyper)
            elif mode in ["understand"]:
                if in_image is None:
                    raise RuntimeError("mode=understand requires image_obs in payload")
                out = inferencer(image=in_image, text=prompt, understanding_output=True, **inference_hyper)
            else:
                raise RuntimeError(f"unsupported mode={mode}")

        img = extract_pil_image(out)

        local_out = f"/tmp/{uuid.uuid4().hex}.png"
        img.save(local_out)

        obs_path = f"{OBS_OUT_PREFIX}/{os.path.basename(local_out)}"
        mox.file.copy(local_out, obs_path)

        return jsonify({
            "result": "SUCCESSFUL",
            "obs_image": obs_path,
            "elapsed_s": round(time.time() - t0, 3),
            "error_message": ""
        })

    except Exception as e:
        return jsonify({
            "result": "FAILED",
            "obs_image": "",
            "elapsed_s": round(time.time() - t0, 3),
            "error_message": str(e)
        }), 500

if __name__ == "__main__":
    # 重要：关掉 reloader，避免加载模型两次
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False, threaded=True)
