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
    #
    # The reference mask to reproduce is
    #     keep(q, k) = (k <= q) AND ( (q - k) < WINDOW_SIZE  OR  k < SINK_SIZE )
    # which splits into three DISJOINT regions:
    #     Phase 0 : keys [0, sink_end)                  -> causal        AND k <  SINK_SIZE
    #     Phase 1 : keys [window_start, first_q_pos)    -> k >= SINK_SIZE AND (q-k) < WINDOW_SIZE
    #     Phase 2 : keys [first_q_pos, first_q_pos+B_M) -> k >= SINK_SIZE AND causal AND (q-k) < WINDOW_SIZE
    #
    # Rule of thumb: the LOOP BOUNDS are only responsible for COVERAGE (no key
    # may be skipped), while the per-phase MASKS are responsible for NOT DOUBLE
    # COUNTING. In particular Phase 1 must NOT be started at `sink_end`:
    # sink_end is rounded up to a multiple of BLOCK_N and is therefore larger
    # than SINK_SIZE, so starting there would silently drop every key in
    # [SINK_SIZE, sink_end).
    first_q_pos = q_block_idx * BLOCK_M
    sink_end = ((SINK_SIZE + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    # ---------------------------------- Phase 0: attention sinks ----------------------------------
    # The sink is the prefix [0, SINK_SIZE) and is visible to every query, so the window
    # mask does not apply here. The causal mask still does: in the very first query tile a
    # query can be younger than some sink keys.
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

        sink_mask = k_offsets[None, :] < SINK_SIZE
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        s_ij = tl.where(causal_mask & sink_mask, s_ij, -float('inf'))

        m_local = tl.max(s_ij, axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)
        # Guard against -inf - (-inf) when a whole tile is masked out.
        m_new = tl.where(m_new == -float('inf'), 0.0, m_new)

        scale_factor = tl.exp2(m_i - m_new)
        l_i = l_i * scale_factor
        acc = acc * scale_factor[:, None]

        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # ------------------------------ Phase 1: off-diagonal, inside window ------------------------------
    # Align the window start DOWN to a block boundary; keys in the first block that fall
    # outside the window are removed by the per-element mask below. `not_sink_mask`
    # guarantees these blocks never re-accumulate the sinks counted in Phase 0.
    unaligned_window_start = tl.maximum(0, first_q_pos - WINDOW_SIZE + 1)
    window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N

    for start_n in range(window_start, first_q_pos, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Every key here is causal (k < first_q_pos <= q), so only these two masks are needed.
        not_sink_mask = k_offsets[None, :] >= SINK_SIZE
        window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        s_ij = tl.where(not_sink_mask & window_mask, s_ij, -float('inf'))

        m_local = tl.max(s_ij, axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)
        m_new = tl.where(m_new == -float('inf'), 0.0, m_new)

        scale_factor = tl.exp2(m_i - m_new)
        l_i = l_i * scale_factor
        acc = acc * scale_factor[:, None]

        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij.to(v_block.dtype), v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # ---------------------------------- Phase 2: diagonal ----------------------------------
    # The most recent keys, always inside the window. All three masks apply.
    for start_n in range(first_q_pos, first_q_pos + BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # For the first query tile these blocks overlap the sink prefix, so the sinks
        # already counted in Phase 0 must be excluded again here.
        not_sink_mask = k_offsets[None, :] >= SINK_SIZE
        causal_mask = k_offsets[None, :] <= q_offsets[:, None]
        window_mask = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        s_ij = tl.where(not_sink_mask & causal_mask & window_mask, s_ij, -float('inf'))

        m_local = tl.max(s_ij, axis=-1)
        m_new = tl.where(m_local > m_i, m_local, m_i)
        m_new = tl.where(m_new == -float('inf'), 0.0, m_new)

        scale_factor = tl.exp2(m_i - m_new)
        l_i = l_i * scale_factor
        acc = acc * scale_factor[:, None]

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
