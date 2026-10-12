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
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for Sliding Window Attention (SWA) with GQA.
    """
    # 1. Boilerplate setup
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 1: GQA Logic) ---
    # This problem combines GQA and SWA. First, implement the GQA logic.
    # 1. Calculate the number of query heads per group.
    # 2. Determine the correct kv_head_idx for the current q_head_idx.
    
    # 1. Calculate the number of query heads per group.
    num_groups = N_Q_HEADS // N_KV_HEADS
    # 2. Determine the correct kv_head_idx for the current q_head_idx.
    kv_head_idx = q_head_idx // num_groups
    # --- END OF GQA IMPLEMENTATION ---


    # 2. Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load query block
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED (Part 2: SWA Logic) ---
    # Now, implement the "sliding window" by changing the loop bounds.
    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.
    # 1. Calculate the starting position of the attention window (window_start).
    # 2. Modify the range of the Phase 1 loop to start from your window_start.

    # 1. Calculate the starting position of the attention window (window_start).
    #    The oldest key any query in this tile may attend to is
    #    (first query position) - WINDOW_SIZE + 1, clamped at 0.
    #    Align it down to a multiple of BLOCK_N so the loop steps over whole K/V blocks.
    #    Loop bounds are only responsible for COVERAGE; the per-element masks below
    #    remove whatever the alignment pulled in from outside the window, so Phase 1
    #    and Phase 2 never double count a key.
    first_q_pos = q_block_idx * BLOCK_M
    unaligned_window_start = tl.maximum(0, first_q_pos - WINDOW_SIZE + 1)
    window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.

        # These blocks lie strictly before the diagonal, so only the window mask applies
        # (the causal condition key_offset <= query_offset already holds for every pair).
        # Because window_start is aligned down, the first block may contain keys that are
        # already out of the window for every query in the tile; they are masked out here.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Sliding window mask: distance (query - key) must be < WINDOW_SIZE.
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

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED

        # The diagonal blocks must always be processed: they hold the most recent keys, which
        # are inside the window even when WINDOW_SIZE is small enough that Phase 1 is empty.
        # Both the causal and the sliding window masks apply here.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale

        # Causal AND sliding window.
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
    # --- END OF SWA IMPLEMENTATION ---


    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128):
    
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel.
    """
    batch, n_q_heads, seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1]
    
    assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"
    assert is_causal, "This kernel is only supported for causal attention"

    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    # if window_size != 4096:
    #     raise ValueError("This kernel is compiled for a fixed window size of 4096")

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
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o