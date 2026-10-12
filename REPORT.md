# Learnings from Implementing FlashAttention Variants

This document outlines the key insights and challenges encountered while implementing attention
mechanisms, from a pure-PyTorch version up to Triton kernels with Grouped-Query Attention, Sliding
Window Attention, and Attention Sinks.

## Problem 1: PyTorch Tiled Attention

My first instinct when writing the online softmax update was to re-normalize the output accumulator
at every step, scaling `o_i` by the ratio of the new and old denominators. My reasoning was that
deferring the normalization until the end of the inner loop would let the unnormalized accumulator
grow without bound and overflow.

I learned that this fear is unfounded, and the reason is the running maximum. Because `m_i` is
subtracted _before_ the exponential, every `exp(S_ij - m_new)` term is at most 1, so the
probabilities stay in a safe range no matter how long the sequence is. What actually needs resizing
between iterations is not the denominator but the _scale reference_: when `m_i` changes, the
previously accumulated `o_i` and `l_i` were computed under the old maximum, so they are corrected by
`exp(m_i - m_new)`. Since both the numerator and the denominator are scaled by the same factor, the
final ratio `o_i / l_i` is unchanged — the normalization can be deferred to the very end and still be
exact. This is what makes FlashAttention mathematically equivalent to standard softmax rather than an
approximation of it; only the order of operations differs.

## Problem 2: The First Triton Kernel

The weighted row-sum kernel was my introduction to Triton's programming model: one program per row,
a loop over column blocks, masked loads with `other=0.0`, and a single reduction at the end rather
than one per iteration.

One thing that surprised me was how much thought went into the _dtype_ of the multiply. The obvious
choice is to upcast the inputs to fp32 before multiplying, which is more accurate. But the reference
implementation `(x * w).sum(dim=1)` multiplies in the input dtype (bf16) and only the accumulation
happens in fp32. Since the autograder compares against that reference, matching its rounding
behaviour matters more than maximizing absolute accuracy. In fact a kernel that is _more_ accurate
than the reference can fail the comparison, because the error is measured against the reference
rather than against the truth. That was a useful lesson about what "correct" actually means in a
numerical test.

## Problem 3: Porting to Triton

Porting the forward pass to Triton was relatively smooth once I accepted two conventions.

The first is the exponential base. Triton's `exp2` is faster than `exp`, so instead of converting
between them inside the loop I folded the conversion constant into the scale itself:
`qk_scale = softmax_scale * 1.44269504` (i.e. `* log2(e)`). Then `exp2` gives exactly `exp` of the
scaled scores, at no extra cost.

The second is how `K` is loaded. Rather than loading it as `(BLOCK_N, HEAD_DIM)` and transposing,
I build the pointer offset matrix so that the result is already `(HEAD_DIM, BLOCK_N)` — putting
`tl.arange(0, HEAD_DIM)[:, None]` on the rows and `k_offsets[None, :]` on the columns. That way the
block can be fed straight into `tl.dot`.

The detail that caused the most trouble was casting `p_ij` before the second matmul. `tl.dot`
requires both operands to have the same dtype, so the probabilities must be cast to match `V`. My
first version hardcoded `bfloat16`, which works on modern hardware and is fast, but would fail
outright on a GPU without bf16 support — and the autograder silently falls back to `float16` in that
case, so the failure would be confusing. Using `p_ij.to(v_block.dtype)` is equally fast in practice
and portable.

## Problem 4: Causal Masking

This was my first multi-phase Triton kernel, and writing the `K`/`V` pointer arithmetic by hand was
a genuinely useful exercise — it forced me to be precise about shapes and strides.

The conceptual insight, though, was about the mask. The naive approach applies the causal mask to
every key block. But for all key blocks that lie strictly _before_ the current query tile, every key
precedes every query in the tile, so the causal condition holds automatically and no mask is needed
at all. Only the diagonal blocks can contain keys that are still in the future. Splitting the loop
into an unmasked off-diagonal phase and a masked diagonal phase therefore saves work twice over: half
the blocks are skipped entirely, and the surviving off-diagonal blocks no longer pay for a mask
comparison. The measured result matched the theory — the causal kernel ran almost exactly twice as
fast as the non-causal one.

## Problem 5: Grouped-Query Attention

GQA was a small change: compute `kv_head_idx = q_head_idx // (N_Q_HEADS // N_KV_HEADS)` and use it
for the `K`/`V` pointers while `Q` keeps its own head index. The rest of the kernel is untouched.

What surprised me is that the benchmark time was essentially _identical_ to the non-causal GQA-free
version. My initial expectation was that sharing K/V across four query heads would make attention
faster, since there is four times less K/V to read. But GQA does not reduce the number of attention
computations at all — all 16 query heads still compute their own full attention. What it reduces is
the _memory_ the model must store for K/V, which is what makes it possible to scale the number of
query heads without blowing up the KV cache. The benchmark confirmed this precisely: the timings were
unchanged while the memory reduction improved. GQA is a model-size optimization, not a kernel-speed
optimization, and I had been conflating the two.

## Problem 6: Sliding Window Attention

The block alignment issue was the one I anticipated. If `WINDOW_SIZE` is not a multiple of
`BLOCK_N`, a naive computation of the window start will not land on a block boundary, and the kernel
can skip tokens that are still inside the window. The fix is to align the start _down_:

```python
unaligned_window_start = tl.maximum(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
window_start = (unaligned_window_start // BLOCK_N) * BLOCK_N
```

Aligning down can pull keys into the first block that are already outside the window for every query
in the tile, but those are removed by the per-element distance mask `(query - key) < WINDOW_SIZE`.
Aligning _up_ would instead drop the block containing the true window start and silently lose real
keys. The general principle I took away is that the loop bounds are only responsible for _coverage_,
while the masks are responsible for _correctness_ — mixing the two responsibilities is where bugs
come from.

The bug I did **not** anticipate was a NaN. With the test configuration `W = 128`, the query tile
starting at position 128 has, in its very first key block `k ∈ [0, 64)`, no valid key at all for the
rows `q = 191..255` — those rows need `k > 63` to be inside the window. So the whole row is masked,
`m_new = max(-inf, -inf) = -inf`, and the rescaling factor becomes
`exp2(-inf - (-inf)) = exp2(nan) = nan`, which poisons both accumulators for the rest of the loop.
The fix is a guard on the running maximum:

```python
m_new = tl.where(m_new == -float('inf'), 0.0, m_new)
```

Clamping to 0 makes the rescaling factor `exp2(-inf) = 0`, so the accumulator correctly stays at
zero and the row is picked up by a later block. Setting `m_i` to 0 rather than `-inf` is harmless
because the accumulator is still zero, and the final ratio is scale-invariant.

What makes this bug worth remembering is how well it hid. Sweeping the window size showed that for
this tile the failure condition is `W ≤ 192` — it disappears only once the window is large enough
that even the last query in the tile still finds a valid key in the first block. So the bug is
present for most window sizes and vanishes for large ones, which is the exact opposite of my
intuition that a large window would be the risky case.

## Problem 7: Sinks

I made the conceptual error of assuming sink tokens were exempt from causal masking, treating them
like a prefix-LM mask. They are not: a sink token can be attended to by every subsequent token, but
it is itself still subject to causality. Concretely, in the very first query tile, a query at
position 3 must not attend to a sink at position 5. So the sink phase needs `causal AND (k < S)`,
not just `k < S`.

With sinks, the reference mask is a **union** of two regions — `(q - k) < W` _or_ `k < S` — which
naturally decomposes into three regions that must be kept _disjoint_, otherwise tokens get counted
twice:

| Phase                    | Keys                                   | Mask                              |
| ------------------------ | -------------------------------------- | --------------------------------- |
| 0 (sinks)                | `[0, sink_end)`                        | `causal AND k < S`                |
| 1 (window, off-diagonal) | `[window_start, first_q_pos)`          | `k >= S AND (q-k) < W`            |
| 2 (diagonal)             | `[first_q_pos, first_q_pos + BLOCK_M)` | `k >= S AND causal AND (q-k) < W` |

My first version got this wrong in an instructive way. To avoid double-counting the sinks, I made
Phase 1 start _after_ the sink region:

```python
phase1_start = tl.maximum(window_start, sink_end)   # wrong
```

The problem is that `sink_end` is rounded up to a block boundary and is therefore larger than
`SINK_SIZE` whenever `S` is not a multiple of `BLOCK_N`. With `S = 8` and `BLOCK_N = 64`, the keys
in `[8, 64)` fall into a gap that no phase covers — and worse, they are keys that _should_ be
attended to. Simulating the coverage confirmed this: 63 query rows were silently missing keys, while
128 rows had sink tokens counted twice. The fix was to stop using the loop bounds to avoid
double-counting and use masks instead: start Phase 1 from the window start (aligned down) and exclude
the sink with `k >= S`.

This is the same principle as in Problem 6, in the opposite direction: the loop bounds guarantee
coverage, the masks guarantee no double-counting. Trying to make the bounds do both is what breaks.

There is also a methodological lesson I did not expect. I debugged Problems 6 and 7 by simulating
the loop bounds and masks in plain Python and comparing against the reference mask, and that
simulation came back clean across 270 configurations. I concluded, wrongly, that Problem 6 was
already correct. Verifying the masks only proves _which_ `(q, k)` pairs are attended — it says
nothing about whether the arithmetic of the online softmax produces finite values. These are two
independent layers and both need checking. Problem 6 passed the mask layer cleanly and still produced
all NaNs.
