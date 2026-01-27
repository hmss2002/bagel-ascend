import math
import torch
import torch_npu

def _cu_seqlens_to_lengths(cu_seqlens: torch.Tensor):
    # cu_seqlens: [B+1] prefix sums
    # lengths: [B]
    cu = cu_seqlens.to("cpu")
    lens = (cu[1:] - cu[:-1]).tolist()
    return tuple(int(x) for x in lens)

def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """
    NPU fallback for flash_attn.flash_attn_varlen_func using torch_npu.npu_fusion_attention.

    Expected q/k/v layout from flash-attn varlen is typically [T, N, D] (T = total tokens).
    We map it to npu_fusion_attention input_layout="TND".
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    head_num = q.shape[1]
    keep_prob = 1.0 - float(dropout_p)

    actual_seq_qlen = _cu_seqlens_to_lengths(cu_seqlens_q)
    actual_seq_kvlen = _cu_seqlens_to_lengths(cu_seqlens_k)

    # causal mask（先按 max_seqlen 构建；后面跑通再优化为更省）
    atten_mask = None
    sparse_mode = 0
    if causal:
        # [max_q, max_k] 上三角 True 表示 mask
        atten_mask = torch.triu(
            torch.ones((int(max_seqlen_q), int(max_seqlen_k)), device=q.device, dtype=torch.bool),
            diagonal=1
        )
        # 在很多 flash-attn 2.x causal 场景下会用 sparse_mode=3，更贴近 flash mask 语义
        sparse_mode = 3

    out = torch_npu.npu_fusion_attention(
        q, k, v, head_num,
        pse=None,
        padding_mask=None,
        atten_mask=atten_mask,
        scale=float(softmax_scale),
        keep_prob=float(keep_prob),
        input_layout="TND",
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
    )[0]

    return out
