import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Forward kernel: causal FlashAttention with GQA + Sliding Window + Attention Sink.
    Writes output O and saves per-row maxima M for backward stability.
    """
    # 1) Program ids and head mapping
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA mapping: map Q head to shared KV head
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2) Accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3) Load Q tile
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (
        q_offsets[:, None] * q_stride_s + d_offsets[None, :]
    )
    q_block = tl.load(q_ptrs, mask=(q_offsets[:, None] < SEQ_LEN), other=0.0)

    qk_scale = softmax_scale * 1.44269504

    # Window start aligned to BLOCK_N
    q_start = q_block_idx * BLOCK_M
    win_left = tl.maximum(0, q_start - (WINDOW_SIZE - 1))
    window_start = (win_left // BLOCK_N) * BLOCK_N

    # --- Phase 0: Attention Sink (first SINK_SIZE keys) ---
    for start_n in range(0, SINK_SIZE, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K [HEAD_DIM, BLOCK_N]
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[None, :] * k_stride_s + d_offsets[:, None]
        )
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)

        # Load V [BLOCK_N, HEAD_DIM]
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (
            k_offsets[:, None] * v_stride_s + d_offsets[None, :]
        )
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        # Scores and masks (sink + causal + bounds)
        s_ij = tl.dot(q_block, k_block) * qk_scale
        q_valid = (q_offsets[:, None] < SEQ_LEN)
        sink_mask = (k_offsets[None, :] < SINK_SIZE)
        kv_valid = (k_offsets[None, :] < SEQ_LEN) & sink_mask & q_valid
        causal = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal & kv_valid, s_ij, -float('inf'))

        # Online softmax update
        s_row_max = tl.max(s_ij, axis=1).to(m_i.dtype)
        m_new = tl.maximum(s_row_max, m_i)
        alpha = tl.where(m_new > m_i, tl.exp2(m_i - m_new), 1.0)
        acc = acc * alpha.to(acc.dtype)[:, None]
        l_i = l_i * alpha
        row_has_valid = m_new > -float('inf')
        m_new_safe = tl.where(row_has_valid, m_new, 0.0)
        p_ij = tl.exp2(s_ij - m_new_safe[:, None])
        acc = acc + tl.dot(p_ij, v_block.to(p_ij.dtype)).to(acc.dtype)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # --- Phase 1: Off-diagonal windowed blocks (exclude sink set to avoid double count) ---
    for start_n in range(window_start, q_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[None, :] * k_stride_s + d_offsets[:, None]
        )
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (
            k_offsets[:, None] * v_stride_s + d_offsets[None, :]
        )
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale
        q_valid = (q_offsets[:, None] < SEQ_LEN)
        win_mask = (q_offsets[:, None] - k_offsets[None, :] < WINDOW_SIZE)
        not_sink = (k_offsets[None, :] >= SINK_SIZE)
        kv_valid = (k_offsets[None, :] < SEQ_LEN) & not_sink & win_mask & q_valid
        s_ij = tl.where(kv_valid, s_ij, -float('inf'))

        s_row_max = tl.max(s_ij, axis=1).to(m_i.dtype)
        m_new = tl.maximum(s_row_max, m_i)
        alpha = tl.where(m_new > m_i, tl.exp2(m_i - m_new), 1.0)
        acc = acc * alpha.to(acc.dtype)[:, None]
        l_i = l_i * alpha
        row_has_valid = m_new > -float('inf')
        m_new_safe = tl.where(row_has_valid, m_new, 0.0)
        p_ij = tl.exp2(s_ij - m_new_safe[:, None])
        acc = acc + tl.dot(p_ij, v_block.to(p_ij.dtype)).to(acc.dtype)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # --- Phase 2: Diagonal windowed blocks (causal) ---
    diag_start = q_start
    for start_n in range(diag_start, q_start + BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[None, :] * k_stride_s + d_offsets[:, None]
        )
        k_block = tl.load(k_ptrs, mask=(k_offsets[None, :] < SEQ_LEN), other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (
            k_offsets[:, None] * v_stride_s + d_offsets[None, :]
        )
        v_block = tl.load(v_ptrs, mask=(k_offsets[:, None] < SEQ_LEN), other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        q_valid = (q_offsets[:, None] < SEQ_LEN)
        win_mask = (q_offsets[:, None] - k_offsets[None, :] < WINDOW_SIZE)
        not_sink = (k_offsets[None, :] >= SINK_SIZE)
        kv_valid = (k_offsets[None, :] < SEQ_LEN) & not_sink & win_mask & q_valid
        s_ij = tl.where(causal & kv_valid, s_ij, -float('inf'))

        s_row_max = tl.max(s_ij, axis=1).to(m_i.dtype)
        m_new = tl.maximum(s_row_max, m_i)
        alpha = tl.where(m_new > m_i, tl.exp2(m_i - m_new), 1.0)
        acc = acc * alpha.to(acc.dtype)[:, None]
        l_i = l_i * alpha
        row_has_valid = m_new > -float('inf')
        m_new_safe = tl.where(row_has_valid, m_new, 0.0)
        p_ij = tl.exp2(s_ij - m_new_safe[:, None])
        acc = acc + tl.dot(p_ij, v_block.to(p_ij.dtype)).to(acc.dtype)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # 4) Normalize and store O
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + (
        q_offsets[:, None] * o_stride_s + d_offsets[None, :]
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=(q_offsets[:, None] < SEQ_LEN))

    # 5) Store per-row max m_i
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, m_i, mask=(q_offsets < SEQ_LEN))

@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, M_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward kernel for causal GQA + SWA + Sink.
    Recompute probabilities with saved M and apply masks; compute dV, dQ, dK.
    """
    # Program ids and head mapping
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS
    q_group = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // q_group

    # Offsets
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, HEAD_DIM)
    q_valid = q_offsets < SEQ_LEN
    qk_scale = softmax_scale * 1.44269504
    eps = 1e-6

    # Load Q and dO tiles
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (
        q_offsets[:, None] * q_stride_s + d_offsets[None, :]
    )
    q_block = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    do_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + (
        q_offsets[:, None] * do_stride_s + d_offsets[None, :]
    )
    dO_block = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)

    # Load per-row maxima and delta
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    M_i = tl.load(m_ptrs, mask=q_valid, other=-float('inf')).to(tl.float32)

    d_ptrs = D_ptr + batch_idx * d_stride_b + q_head_idx * d_stride_h + q_offsets * d_stride_s
    delta_i = tl.load(d_ptrs, mask=q_valid, other=0.0).to(tl.float32)

    # Helper masks independent of k_offsets
    q_valid_bm = q_valid[:, None]

    # Pass 1: compute l_i = sum_j exp2(s_ij - M_i)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        # Load K [HEAD_DIM, BLOCK_N]
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[None, :] * k_stride_s + d_offsets[:, None]
        )
        k_block = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale
        s_ij = s_ij.to(tl.float32)
        causal = q_offsets[:, None] >= k_offsets[None, :]
        sink_mask = k_offsets[None, :] < SINK_SIZE
        win_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        allow = (sink_mask | win_mask) & causal & q_valid_bm & kv_valid[None, :]
        s_ij = tl.where(allow, s_ij, -float('inf'))

        p = tl.exp2(s_ij - M_i[:, None])
        l_i += tl.sum(p, axis=1)

    # Pass 2: grads
    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[None, :] * k_stride_s + d_offsets[:, None]
        )
        k_block = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale
        s_ij = s_ij.to(tl.float32)
        causal = q_offsets[:, None] >= k_offsets[None, :]
        sink_mask = k_offsets[None, :] < SINK_SIZE
        win_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        allow = (sink_mask | win_mask) & causal & q_valid_bm & kv_valid[None, :]
        s_ij = tl.where(allow, s_ij, -float('inf'))

        p = tl.exp2(s_ij - M_i[:, None])
        P_ij = p / (l_i[:, None] + eps)

        # dV: dO^T @ P -> [HEAD_DIM, BLOCK_N]
        do_T_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + (
            d_offsets[:, None] + q_offsets[None, :] * do_stride_s
        )
        dO_T = tl.load(do_T_ptrs, mask=q_valid_bm.T, other=0.0).to(tl.float32)
        dV_tile_T = tl.dot(dO_T, P_ij)
        dV_ptrs_T = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + (
            d_offsets[:, None] + k_offsets[None, :] * dv_stride_s
        )
        tl.atomic_add(dV_ptrs_T, dV_tile_T, mask=kv_valid[None, :])

        # dP = dO @ V^T
        v_T_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (
            d_offsets[:, None] + k_offsets[None, :] * v_stride_s
        )
        v_T = tl.load(v_T_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        dP_ij = tl.dot(dO_block, v_T)

        # dS = P * (dP - delta)
        dS_ij = P_ij * (dP_ij - delta_i[:, None])

        # dQ += dS @ K^T * softmax_scale
        k_T_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (
            k_offsets[:, None] * k_stride_s + d_offsets[None, :]
        )
        k_T = tl.load(k_T_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)
        dQ_acc += tl.dot(dS_ij, k_T) * softmax_scale

        # dK += Q^T @ dS * softmax_scale (atomic)
        q_T_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (
            d_offsets[:, None] + q_offsets[None, :] * q_stride_s
        )
        q_T = tl.load(q_T_ptrs, mask=q_valid_bm.T, other=0.0).to(tl.float32)
        dK_tile_T = tl.dot(q_T, dS_ij) * softmax_scale
        dK_ptrs_T = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + (
            d_offsets[:, None] + k_offsets[None, :] * dk_stride_s
        )
        tl.atomic_add(dK_ptrs_T, dK_tile_T, mask=kv_valid[None, :])

    # Store dQ
    dQ_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + (
        q_offsets[:, None] * dq_stride_s + d_offsets[None, :]
    )
    tl.store(dQ_ptrs, dQ_acc.to(dQ_ptr.dtype.element_ty), mask=q_valid[:, None])

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        # Use fp32 buffers for atomics on dK/dV
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)

        # delta = sum_j dO_ij * O_ij
        D = (do.to(torch.float32) * o.to(torch.float32)).sum(dim=-1)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_backward_swa_kernel[grid](
            # In/Out
            q, k, v, do, M, D,
            dq, dk, dv,
            # Strides
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            # Params
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)