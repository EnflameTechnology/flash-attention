import math

import pytest
import torch

pytest.importorskip("torch_gcu")
from torch_gcu import transfer_to_gcu  # noqa: F401 (monkeypatches cuda -> gcu)

from flash_attn import flash_attn_varlen_func

DTYPES = [torch.float16, torch.bfloat16]
HEAD_DIMS = [64, 128, 256]
CAUSAL = [False, True]
BATCH_SIZES = [1, 2]
SEQLENS = [128, 256]
ATOL = 2e-2
RTOL = 1e-2


def attention_ref(q, k, v, causal=False, softmax_scale=None):
    """CPU reference implementation for standard attention forward.

    q: (batch, seqlen_q, nheads, d)
    k: (batch, seqlen_k, nheads_k, d)
    v: (batch, seqlen_k, nheads_k, d)
    Returns: (batch, seqlen_q, nheads, d)
    """
    batch, seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[1]
    nheads_k = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    k_rep = k.repeat_interleave(nheads // nheads_k, dim=2)
    v_rep = v.repeat_interleave(nheads // nheads_k, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float() * softmax_scale, k_rep.float())
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool), diagonal=seqlen_k - seqlen_q + 1
        )
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.einsum("bhts,bshd->bthd", attn, v_rep)
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("causal", CAUSAL)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seqlen", SEQLENS)
@torch.inference_mode()
def test_flash_attn_varlen_func_fwd(dtype, head_dim, causal, batch_size, seqlen):
    torch.manual_seed(0)
    num_heads = 8
    total = batch_size * seqlen
    q = torch.randn(total, num_heads, head_dim, dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, dtype=dtype)
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32
    ).gcu()
    out = flash_attn_varlen_func(
        q.gcu(), k.gcu(), v.gcu(), cu_seqlens, cu_seqlens, seqlen, seqlen,
        causal=causal, window_size=(-1, -1), softcap=0.0,
    ).cpu()
    q_padded = q.reshape(batch_size, seqlen, num_heads, head_dim)
    k_padded = k.reshape(batch_size, seqlen, num_heads, head_dim)
    v_padded = v.reshape(batch_size, seqlen, num_heads, head_dim)
    ref = attention_ref(q_padded, k_padded, v_padded, causal=causal).reshape(
        total, num_heads, head_dim
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
