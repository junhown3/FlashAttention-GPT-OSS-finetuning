import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@dataclass(frozen=True)
class BenchmarkCase:
    batch: int
    heads_q: int
    heads_kv: int
    seq_len: int
    head_dim: int
    window_size: int
    sink_size: int

    def label(self) -> str:
        return (
            f"B={self.batch}, Hq={self.heads_q}, Hkv={self.heads_kv}, "
            f"L={self.seq_len}, D={self.head_dim}, W={self.window_size}, S={self.sink_size}"
        )


def dtype_from_name(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def create_mask_bool(seq_len: int, window_size: int, sink_size: int, device=None) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    row = idx[:, None]
    col = idx[None, :]
    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row)
    return sliding | sink


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return x
    batch, heads_kv, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads_kv, num_groups, seq_len, head_dim)
    return x.reshape(batch, heads_kv * num_groups, seq_len, head_dim)


def pytorch_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    mask = create_mask_bool(q.shape[2], window_size, sink_size, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


def pytorch_naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    if q.shape[1] != k.shape[1]:
        num_groups = q.shape[1] // k.shape[1]
        k = repeat_kv(k, num_groups)
        v = repeat_kv(v, num_groups)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = create_mask_bool(q.shape[2], window_size, sink_size, device=q.device)
    scores = scores.masked_fill(~mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return probs @ v


# ──── Triton constants and helpers ────

MAX_GQA_GROUPS = 16

_ATTN_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
]

_ATTN_KEY = ["SEQ_LEN", "HEAD_DIM"]


def _strides(*tensors):
    """Extract first 3 strides (batch, head, seq) from each tensor, flattened."""
    result = []
    for t in tensors:
        result.extend(t.stride()[:3])
    return result


@triton.jit
def _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE):
    causal = k_offsets[None, :] <= offs_m[:, None]
    sliding = (offs_m[:, None] - k_offsets[None, :]) < WINDOW_SIZE
    sink = k_offsets[None, :] < SINK_SIZE
    return (offs_m[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN) & causal & (sliding | sink)


@triton.jit
def _fwd_inner(q, k, v, qk_scale, valid, m_i, l_i, acc):
    """One block of online-softmax forward accumulation."""
    qk = tl.dot(q, tl.trans(k)) * qk_scale
    qk = tl.where(valid, qk, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(qk - m_new[:, None])
    p = tl.where(valid, p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l_i, acc


@triton.jit
def _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid):
    """Compute attention probabilities p and score gradients ds for one KV block."""
    qk = tl.dot(q, tl.trans(k)) * qk_scale
    p = tl.exp2(qk - lse[:, None])
    p = tl.where(valid, p, 0.0)
    dp = tl.dot(do_raw, tl.trans(v))
    ds = p * (dp - delta[:, None]) * softmax_scale
    ds = tl.where(valid, ds, 0.0)
    return p, ds


# ──── Autotuned Triton kernels ────

@triton.autotune(configs=_ATTN_CONFIGS, key=_ATTN_KEY)
@triton.jit
def _forward_kernel(
    Q, K, V, O, LSE,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    lse_stride_b, lse_stride_h, lse_stride_s,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // N_Q_HEADS
    q_head = batch_head % N_Q_HEADS
    kv_group = N_Q_HEADS // N_KV_HEADS
    kv_head = q_head // kv_group

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + batch * q_stride_b + q_head * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    log2e = 1.4426950408889634
    qk_scale = softmax_scale * log2e

    q_start = q_block * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, SEQ_LEN)
    raw_window_start = tl.maximum(0, q_start - WINDOW_SIZE + 1)
    window_start = (raw_window_start // BLOCK_N) * BLOCK_N
    window_end = q_end
    sink_end = tl.minimum(SINK_SIZE, window_start)

    for start_n in tl.range(0, sink_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        m_i, l_i, acc = _fwd_inner(q, k, v, qk_scale, valid, m_i, l_i, acc)

    for start_n in tl.range(window_start, window_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        m_i, l_i, acc = _fwd_inner(q, k, v, qk_scale, valid, m_i, l_i, acc)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe) * log2e

    o_ptrs = O + batch * o_stride_b + q_head * o_stride_h + offs_m[:, None] * o_stride_s + offs_d[None, :]
    lse_ptrs = LSE + batch * lse_stride_b + q_head * lse_stride_h + offs_m * lse_stride_s
    tl.store(o_ptrs, out, mask=offs_m[:, None] < SEQ_LEN)
    tl.store(lse_ptrs, lse, mask=offs_m < SEQ_LEN)


@triton.autotune(configs=_ATTN_CONFIGS, key=_ATTN_KEY, reset_to_zero=["DK", "DV"])
@triton.jit
def _backward_kernel(
    Q, K, V, O, DO, LSE, DQ, DK, DV,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    lse_stride_b, lse_stride_h, lse_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // N_Q_HEADS
    q_head = batch_head % N_Q_HEADS
    kv_group = N_Q_HEADS // N_KV_HEADS
    kv_head = q_head // kv_group

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + batch * q_stride_b + q_head * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :]
    o_ptrs = O + batch * o_stride_b + q_head * o_stride_h + offs_m[:, None] * o_stride_s + offs_d[None, :]
    do_ptrs = DO + batch * do_stride_b + q_head * do_stride_h + offs_m[:, None] * do_stride_s + offs_d[None, :]
    lse_ptrs = LSE + batch * lse_stride_b + q_head * lse_stride_h + offs_m * lse_stride_s

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    o = tl.load(o_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    do_raw = tl.load(do_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    do_f32 = do_raw.to(tl.float32)
    lse = tl.load(lse_ptrs, mask=offs_m < SEQ_LEN, other=float("inf"))
    delta = tl.sum(o * do_f32, axis=1)

    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    q_start = q_block * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, SEQ_LEN)
    raw_window_start = tl.maximum(0, q_start - WINDOW_SIZE + 1)
    window_start = (raw_window_start // BLOCK_N) * BLOCK_N
    window_end = q_end
    sink_end = tl.minimum(SINK_SIZE, window_start)

    for start_n in tl.range(0, sink_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        p, ds = _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid)
        dq += tl.dot(ds.to(k.dtype), k)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = tl.dot(tl.trans(p.to(do_raw.dtype)), do_raw)
        dk_ptrs = DK + batch * dk_stride_b + kv_head * dk_stride_h + k_offsets[:, None] * dk_stride_s + offs_d[None, :]
        dv_ptrs = DV + batch * dv_stride_b + kv_head * dv_stride_h + k_offsets[:, None] * dv_stride_s + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=k_offsets[:, None] < SEQ_LEN, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv, mask=k_offsets[:, None] < SEQ_LEN, sem="relaxed")

    for start_n in tl.range(window_start, window_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        p, ds = _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid)
        dq += tl.dot(ds.to(k.dtype), k)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = tl.dot(tl.trans(p.to(do_raw.dtype)), do_raw)
        dk_ptrs = DK + batch * dk_stride_b + kv_head * dk_stride_h + k_offsets[:, None] * dk_stride_s + offs_d[None, :]
        dv_ptrs = DV + batch * dv_stride_b + kv_head * dv_stride_h + k_offsets[:, None] * dv_stride_s + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=k_offsets[:, None] < SEQ_LEN, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv, mask=k_offsets[:, None] < SEQ_LEN, sem="relaxed")

    dq_ptrs = DQ + batch * dq_stride_b + q_head * dq_stride_h + offs_m[:, None] * dq_stride_s + offs_d[None, :]
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < SEQ_LEN)


@triton.autotune(configs=_ATTN_CONFIGS, key=_ATTN_KEY)
@triton.jit
def _backward_dq_kernel(
    Q, K, V, O, DO, LSE, DQ,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    lse_stride_b, lse_stride_h, lse_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // N_Q_HEADS
    q_head = batch_head % N_Q_HEADS
    kv_group = N_Q_HEADS // N_KV_HEADS
    kv_head = q_head // kv_group

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + batch * q_stride_b + q_head * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :]
    o_ptrs = O + batch * o_stride_b + q_head * o_stride_h + offs_m[:, None] * o_stride_s + offs_d[None, :]
    do_ptrs = DO + batch * do_stride_b + q_head * do_stride_h + offs_m[:, None] * do_stride_s + offs_d[None, :]
    lse_ptrs = LSE + batch * lse_stride_b + q_head * lse_stride_h + offs_m * lse_stride_s

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    o = tl.load(o_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
    do_raw = tl.load(do_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
    do_f32 = do_raw.to(tl.float32)
    lse = tl.load(lse_ptrs, mask=offs_m < SEQ_LEN, other=float("inf"))
    delta = tl.sum(o * do_f32, axis=1)

    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    q_start = q_block * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, SEQ_LEN)
    raw_window_start = tl.maximum(0, q_start - WINDOW_SIZE + 1)
    window_start = (raw_window_start // BLOCK_N) * BLOCK_N
    window_end = q_end
    sink_end = tl.minimum(SINK_SIZE, window_start)

    for start_n in tl.range(0, sink_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        _, ds = _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid)
        dq += tl.dot(ds.to(k.dtype), k)

    for start_n in tl.range(window_start, window_end, BLOCK_N):
        k_offsets = start_n + offs_n
        k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + k_offsets[:, None] * k_stride_s + offs_d[None, :]
        v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + k_offsets[:, None] * v_stride_s + offs_d[None, :]
        k = tl.load(k_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        valid = _compute_mask(offs_m, k_offsets, SEQ_LEN, WINDOW_SIZE, SINK_SIZE)
        _, ds = _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid)
        dq += tl.dot(ds.to(k.dtype), k)

    dq_ptrs = DQ + batch * dq_stride_b + q_head * dq_stride_h + offs_m[:, None] * dq_stride_s + offs_d[None, :]
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < SEQ_LEN)


@triton.autotune(configs=_ATTN_CONFIGS, key=_ATTN_KEY)
@triton.jit
def _backward_dkdv_kernel(
    Q, K, V, O, DO, LSE, DK, DV,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    lse_stride_b, lse_stride_h, lse_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_block = tl.program_id(0)
    batch_kv_head = tl.program_id(1)
    batch = batch_kv_head // N_KV_HEADS
    kv_head = batch_kv_head % N_KV_HEADS
    kv_group = N_Q_HEADS // N_KV_HEADS

    offs_n = kv_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    k_ptrs = K + batch * k_stride_b + kv_head * k_stride_h + offs_n[:, None] * k_stride_s + offs_d[None, :]
    v_ptrs = V + batch * v_stride_b + kv_head * v_stride_h + offs_n[:, None] * v_stride_s + offs_d[None, :]
    k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634

    k_start = kv_block * BLOCK_N
    k_end = tl.minimum(k_start + BLOCK_N, SEQ_LEN)
    q_loop_start = (k_start // BLOCK_M) * BLOCK_M
    q_loop_end = tl.minimum(SEQ_LEN, k_end + WINDOW_SIZE - 1)
    overlaps_sink = k_start < SINK_SIZE
    q_loop_start = tl.where(overlaps_sink, 0, q_loop_start)
    q_loop_end = tl.where(overlaps_sink, SEQ_LEN, q_loop_end)

    for group_offset in tl.static_range(0, MAX_GQA):
        q_head = kv_head * kv_group + group_offset
        valid_group = group_offset < kv_group
        q_head_safe = tl.minimum(q_head, N_Q_HEADS - 1)
        for q_start in tl.range(q_loop_start, q_loop_end, BLOCK_M):
            offs_m = q_start + offs_m_base
            q_ptrs = Q + batch * q_stride_b + q_head_safe * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :]
            o_ptrs = O + batch * o_stride_b + q_head_safe * o_stride_h + offs_m[:, None] * o_stride_s + offs_d[None, :]
            do_ptrs = DO + batch * do_stride_b + q_head_safe * do_stride_h + offs_m[:, None] * do_stride_s + offs_d[None, :]
            lse_ptrs = LSE + batch * lse_stride_b + q_head_safe * lse_stride_h + offs_m * lse_stride_s

            q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
            o = tl.load(o_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0).to(tl.float32)
            do_raw = tl.load(do_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)
            do_f32 = do_raw.to(tl.float32)
            lse = tl.load(lse_ptrs, mask=offs_m < SEQ_LEN, other=float("inf"))
            delta = tl.sum(o * do_f32, axis=1)

            valid = _compute_mask(offs_m, offs_n, SEQ_LEN, WINDOW_SIZE, SINK_SIZE) & valid_group
            p, ds = _bwd_core(q, k, v, do_raw, lse, delta, qk_scale, softmax_scale, valid)
            dv += tl.dot(tl.trans(p.to(do_raw.dtype)), do_raw)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_ptrs = DK + batch * dk_stride_b + kv_head * dk_stride_h + offs_n[:, None] * dk_stride_s + offs_d[None, :]
    dv_ptrs = DV + batch * dv_stride_b + kv_head * dv_stride_h + offs_n[:, None] * dv_stride_s + offs_d[None, :]
    tl.store(dk_ptrs, dk, mask=offs_n[:, None] < SEQ_LEN)
    tl.store(dv_ptrs, dv, mask=offs_n[:, None] < SEQ_LEN)


# ──── Autograd wrappers ────

class TritonSWASinkAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size: int, sink_size: int, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise ValueError("This standalone benchmark expects contiguous q, k, and v tensors.")

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("Number of query heads must be divisible by number of key/value heads.")
        if head_dim > 128:
            raise ValueError("This benchmark supports head_dim <= 128.")

        o = torch.empty_like(q)
        lse = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)
        n_grid_y = batch * n_q_heads
        grid = lambda meta: (triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]), n_grid_y)
        _forward_kernel[grid](
            q, k, v, o, lse,
            *_strides(q, k, v, o, lse),
            softmax_scale,
            SEQ_LEN=seq_len,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        n_grid_y = batch * n_q_heads
        grid = lambda meta: (triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]), n_grid_y)
        _backward_kernel[grid](
            q, k, v, o, do, lse, dq, dk, dv,
            *_strides(q, k, v, o, do, lse, dq, dk, dv),
            ctx.softmax_scale,
            SEQ_LEN=seq_len,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            WINDOW_SIZE=ctx.window_size,
            SINK_SIZE=ctx.sink_size,
            HEAD_DIM=head_dim,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


class TritonSWASinkAttentionSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size: int, sink_size: int, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise ValueError("This standalone benchmark expects contiguous q, k, and v tensors.")

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("Number of query heads must be divisible by number of key/value heads.")
        if n_q_heads // n_kv_heads > MAX_GQA_GROUPS:
            raise ValueError(
                f"The split backward path supports at most {MAX_GQA_GROUPS} query heads per KV group."
            )
        if head_dim > 128:
            raise ValueError("This benchmark supports head_dim <= 128.")

        o = torch.empty_like(q)
        lse = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)
        n_grid_y = batch * n_q_heads
        grid = lambda meta: (triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]), n_grid_y)
        _forward_kernel[grid](
            q, k, v, o, lse,
            *_strides(q, k, v, o, lse),
            softmax_scale,
            SEQ_LEN=seq_len,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.empty(v.shape, device=v.device, dtype=torch.float32)

        q_grid_y = batch * n_q_heads
        q_grid = lambda meta: (triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_M"]), q_grid_y)
        _backward_dq_kernel[q_grid](
            q, k, v, o, do, lse, dq,
            *_strides(q, k, v, o, do, lse, dq),
            ctx.softmax_scale,
            SEQ_LEN=seq_len,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            WINDOW_SIZE=ctx.window_size,
            SINK_SIZE=ctx.sink_size,
            HEAD_DIM=head_dim,
        )

        kv_grid_y = batch * n_kv_heads
        kv_grid = lambda meta: (triton.cdiv(meta["SEQ_LEN"], meta["BLOCK_N"]), kv_grid_y)
        _backward_dkdv_kernel[kv_grid](
            q, k, v, o, do, lse, dk, dv,
            *_strides(q, k, v, o, do, lse, dk, dv),
            ctx.softmax_scale,
            SEQ_LEN=seq_len,
            N_Q_HEADS=n_q_heads,
            N_KV_HEADS=n_kv_heads,
            WINDOW_SIZE=ctx.window_size,
            SINK_SIZE=ctx.sink_size,
            HEAD_DIM=head_dim,
            MAX_GQA=MAX_GQA_GROUPS,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    return TritonSWASinkAttention.apply(q, k, v, window_size, sink_size, None)


def triton_attention_split(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    sink_size: int,
) -> torch.Tensor:
    return TritonSWASinkAttentionSplit.apply(q, k, v, window_size, sink_size, None)


PYTORCH_BASELINES = {
    "naive": ("PyTorch Naive", pytorch_naive_attention),
    "sdpa": ("PyTorch SDPA", pytorch_sdpa_attention),
}

TRITON_VARIANTS = {
    "atomic": ("Triton Atomic", triton_attention),
    "split": ("Triton Split", triton_attention_split),
}


def make_inputs(case: BenchmarkCase, dtype: torch.dtype):
    q = torch.randn(
        case.batch,
        case.heads_q,
        case.seq_len,
        case.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    ).contiguous()
    k = torch.randn(
        case.batch,
        case.heads_kv,
        case.seq_len,
        case.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    ).contiguous()
    v = torch.randn(
        case.batch,
        case.heads_kv,
        case.seq_len,
        case.head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    ).contiguous()
    return q, k, v


def zero_grads(*tensors: torch.Tensor):
    for tensor in tensors:
        tensor.grad = None


def correctness_check(
    case: BenchmarkCase,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    reference_name: str,
    triton_name: str,
):
    label, reference_func = PYTORCH_BASELINES[reference_name]
    triton_label, triton_func = TRITON_VARIANTS[triton_name]
    print(f"\nCorrectness vs {label} ({triton_label}): {case.label()}")
    torch.manual_seed(0)
    q, k, v = make_inputs(case, dtype)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    out_ref = reference_func(q_ref, k_ref, v_ref, case.window_size, case.sink_size)
    out_tri = triton_func(q, k, v, case.window_size, case.sink_size)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)
    out_tri.backward(dout)

    checks = {
        "O": (out_ref, out_tri),
        "dQ": (q_ref.grad, q.grad),
        "dK": (k_ref.grad, k.grad),
        "dV": (v_ref.grad, v.grad),
    }
    ok = True
    for name, (ref, tri) in checks.items():
        match = torch.allclose(ref, tri, atol=atol, rtol=rtol)
        max_diff = (ref - tri).abs().max().item()
        print(f"  {name:<2} {'OK' if match else 'FAIL'} max_diff={max_diff:.6g}")
        ok = ok and match
    return ok


def time_forward_only(func, case: BenchmarkCase, dtype: torch.dtype, warmup: int, iters: int):
    q, k, v = make_inputs(case, dtype)

    for _ in range(warmup):
        with torch.no_grad():
            _ = func(q, k, v, case.window_size, case.sink_size)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = func(q, k, v, case.window_size, case.sink_size)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return elapsed_ms, peak_gb


def time_forward_backward(func, case: BenchmarkCase, dtype: torch.dtype, warmup: int, iters: int):
    q, k, v = make_inputs(case, dtype)
    dout = torch.randn_like(q)

    for _ in range(warmup):
        zero_grads(q, k, v)
        out = func(q, k, v, case.window_size, case.sink_size)
        out.backward(dout)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        zero_grads(q, k, v)
        out = func(q, k, v, case.window_size, case.sink_size)
        out.backward(dout)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return elapsed_ms, peak_gb


def time_backward_only(func, case: BenchmarkCase, dtype: torch.dtype, warmup: int, iters: int):
    q, k, v = make_inputs(case, dtype)
    dout = torch.randn_like(q)

    for _ in range(warmup):
        zero_grads(q, k, v)
        out = func(q, k, v, case.window_size, case.sink_size)
        torch.cuda.synchronize()
        out.backward(dout)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_ms = 0.0
    for _ in range(iters):
        zero_grads(q, k, v)
        out = func(q, k, v, case.window_size, case.sink_size)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out.backward(dout)
        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - start) * 1000.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return total_ms / iters, peak_gb


def run_benchmark_case(case: BenchmarkCase, dtype: torch.dtype, args):
    print(f"\nBenchmark: {case.label()} dtype={dtype}")
    baselines = ["naive", "sdpa"] if args.baseline == "both" else [args.baseline]
    triton_variants = ["atomic", "split"] if args.triton_backward == "both" else [args.triton_backward]

    mode_timers = {
        "forward_only": time_forward_only,
        "forward_backward": time_forward_backward,
        "backward_only": time_backward_only,
    }
    if args.mode == "both":
        modes = ["forward_backward", "backward_only"]
    elif args.mode == "all":
        modes = ["forward_only", "forward_backward", "backward_only"]
    else:
        modes = [args.mode]
    for mode in modes:
        timer = mode_timers[mode]
        baseline_results = []
        for baseline in baselines:
            label, func = PYTORCH_BASELINES[baseline]
            ms, mem = timer(func, case, dtype, args.warmup, args.iters)
            baseline_results.append((label, ms, mem))
        triton_results = []
        for triton_variant in triton_variants:
            label, func = TRITON_VARIANTS[triton_variant]
            ms, mem = timer(func, case, dtype, args.warmup, args.iters)
            triton_results.append((label, ms, mem))

        print(f"\n  Mode: {mode}")
        print(f"  {'Implementation':<16} {'Avg ms':>12} {'Peak GB':>12}")
        for label, ms, mem in baseline_results:
            print(f"  {label:<16} {ms:12.4f} {mem:12.4f}")
        for label, ms, mem in triton_results:
            print(f"  {label:<16} {ms:12.4f} {mem:12.4f}")
        for triton_label, triton_ms, triton_mem in triton_results:
            for baseline_label, baseline_ms, baseline_mem in baseline_results:
                speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
                memory_saving = baseline_mem / triton_mem if triton_mem > 0 else float("inf")
                print(f"  Speedup {triton_label} vs {baseline_label}: {speedup:.2f}x")
                print(f"  Memory saving {triton_label} vs {baseline_label}: {memory_saving:.2f}x")


def parse_cases(args) -> list[BenchmarkCase]:
    if args.case == "all":
        return [
            BenchmarkCase(1, 16, 16, 4096, 16, 256, 4),
            BenchmarkCase(1, 16, 8, 4096, 16, 256, 4),
            BenchmarkCase(1, 16, 1, 4096, 16, 256, 4),
        ]
    return [
        BenchmarkCase(
            args.batch,
            args.heads_q,
            args.heads_kv,
            args.seq_len,
            args.head_dim,
            args.window_size,
            args.sink_size,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description="Standalone backward-speed benchmark for P9 attention.")
    parser.add_argument("--case", choices=["all", "custom"], default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads-q", type=int, default=16)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["forward_only", "forward_backward", "backward_only", "both", "all"],
        default="all",
        help="Timing mode: forward_only, forward_backward, backward_only, both (fwd+bwd), or all three.",
    )
    parser.add_argument("--baseline", choices=["naive", "sdpa", "both"], default="naive")
    parser.add_argument("--triton-backward", choices=["atomic", "split", "both"], default="atomic")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(48)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Torch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    for case in parse_cases(args):
        if not args.skip_correctness:
            reference_name = "naive" if args.baseline == "both" else args.baseline
            triton_names = ["atomic", "split"] if args.triton_backward == "both" else [args.triton_backward]
            for triton_name in triton_names:
                if not correctness_check(case, dtype, args.atol, args.rtol, reference_name, triton_name):
                    raise RuntimeError(f"Correctness check failed for {case.label()} ({triton_name})")
        run_benchmark_case(case, dtype, args)


if __name__ == "__main__":
    main()
