import math
import torch
import torch_npu

def _cu_seqlens_to_lengths(cu_seqlens: torch.Tensor):
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
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    head_num = q.shape[1]
    keep_prob = 1.0 - float(dropout_p)

    actual_seq_qlen = _cu_seqlens_to_lengths(cu_seqlens_q)
    actual_seq_kvlen = _cu_seqlens_to_lengths(cu_seqlens_k)

    atten_mask = None
    sparse_mode = 0
    if causal:
        # npu_fusion_attention 的 sparse_mode=3（rightDownCausal）要求压缩mask形状为 2048x2048
        MASK_SIZE = 2048
        atten_mask = torch.triu(
            torch.ones((MASK_SIZE, MASK_SIZE), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
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
