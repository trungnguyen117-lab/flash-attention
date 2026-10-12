import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
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
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks

    # The sliding window for this query tile always starts at
    #   max(0, first_query_position - WINDOW_SIZE + 1)
    # and everything before it splits into two disjoint regions:
    #   [0, SINK_SIZE)          -> Phase 0 (attention sinks)
    #   [SINK_SIZE, window_start) -> not attended to at all
    first_q_pos = q_block_idx * BLOCK_M
    window_start = tl.maximum(0, first_q_pos - WINDOW_SIZE + 1)

    # ---------------------------------- Phase 0: sinks ----------------------------------
    # The sink is a prefix of keys at absolute positions [0, SINK_SIZE). It is always visible
    # to every query (it is not affected by the sliding window), so the only mask needed is the
    # per-element `k_offsets < SINK_SIZE` bound - the last block is usually only partially full.
    sink_end = ((SINK_SIZE + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Keep only the true sink keys inside the last (partial) block.
        sink_mask = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(sink_mask, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # -------------------------- Phase 1: off-diagonal, in window --------------------------
    # Keys in [SINK_SIZE, first_q_pos) that are also inside the window.
    # Align the start down to a block boundary; any key that falls outside the window is
    # removed by the per-element distance mask below.
    phase1_start = tl.maximum(window_start, sink_end)
    phase1_start = (phase1_start // BLOCK_N) * BLOCK_N

    for start_n in range(phase1_start, first_q_pos, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Sliding window mask (the causal condition already holds for these blocks).
        window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        s_ij = tl.where(window_mask, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # ---------------------------------- Phase 2: diagonal ----------------------------------
    # The most recent keys, always inside the window. Both causal and window masks apply.
    for start_n in range(first_q_pos, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        diag_mask = (k_offsets[None, :] <= q_offsets[:, None]) & \
                    ((q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE)
        s_ij = tl.where(diag_mask, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
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
    return o