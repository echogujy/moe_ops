import torch
import triton
import triton.language as tl

_NS = "tg"

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8),
    ],
    key=['num_tokens', 'E'],
)
@triton.jit
def _fused_topk_softmax_fwd_kernel_2d(
    logits_ptr, weights_ptr, indices_ptr,
    num_tokens, E, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)

    # Row offsets
    offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens

    # Column offsets (E dimension)
    offs_e = tl.arange(0, BLOCK_E)

    # 2D Load: shape [BLOCK_M, BLOCK_E]
    logits_ptrs = logits_ptr + offs_m[:, None] * E + offs_e[None, :]
    load_mask = m_mask[:, None] & (offs_e[None, :] < E)
    logits = tl.load(logits_ptrs, mask=load_mask, other=-float('inf'))

    # Top-k registers padded to a power-of-2 block dim: K itself (e.g. 6) is
    # not always a power of 2, and Triton rejects non-pow2 block shapes.
    # Padding columns hold -inf for weights so the axis-1 max/sum ignore them.
    offs_k = tl.arange(0, BLOCK_K)
    w = tl.full((BLOCK_M, BLOCK_K), -float('inf'), dtype=tl.float32)
    idxs = tl.full((BLOCK_M, BLOCK_K), 0, dtype=tl.int32)

    temp_logits = logits

    for k in range(K):
        # reduction along axis 1 (expert dimension)
        val = tl.max(temp_logits, axis=1) # [BLOCK_M]
        idx = tl.argmax(temp_logits, axis=1) # [BLOCK_M]

        # Insert val and idx into the k-th column of w and idxs
        mask_k = offs_k[None, :] == k # [1, BLOCK_K]
        w = tl.where(mask_k, val[:, None], w)
        idxs = tl.where(mask_k, idx[:, None].to(tl.int32), idxs)

        # Mask out the maximum element for each row
        mask_e = offs_e[None, :] == idx[:, None] # [BLOCK_M, BLOCK_E]
        temp_logits = tl.where(mask_e, -float('inf'), temp_logits)

    # Stable Softmax along axis 1 (K dimension). Padding columns stay -inf
    # so they contribute 0 to max/sum and drop out of the result.
    w_max = tl.max(w, axis=1)
    w_exp = tl.exp(w - w_max[:, None])
    w_sum = tl.sum(w_exp, axis=1)
    w_softmax = w_exp / w_sum[:, None]

    # 2D Store: shape [BLOCK_M, K] (only real K columns written)
    store_mask = m_mask[:, None] & (offs_k[None, :] < K)
    tl.store(weights_ptr + offs_m[:, None] * K + offs_k[None, :], w_softmax.to(weights_ptr.dtype.element_ty), mask=store_mask)
    tl.store(indices_ptr + offs_m[:, None] * K + offs_k[None, :], idxs, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8),
    ],
    key=['num_tokens', 'K'],
)
@triton.jit
def _fused_topk_softmax_bwd_kernel_2d(
    grad_w_ptr, w_ptr, idxs_ptr, grad_logits_ptr,
    num_tokens, E, K: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    
    # Row offsets
    offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens
    
    # Column offsets
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    
    # Load 2D matrices: shape [BLOCK_M, BLOCK_K]
    row_k_offs = offs_m[:, None] * K + offs_k[None, :]
    load_mask = m_mask[:, None] & k_mask[None, :]
    
    w = tl.load(w_ptr + row_k_offs, mask=load_mask, other=0.0).to(tl.float32)
    grad_w = tl.load(grad_w_ptr + row_k_offs, mask=load_mask, other=0.0).to(tl.float32)
    idxs = tl.load(idxs_ptr + row_k_offs, mask=load_mask, other=0).to(tl.int32)
    
    # Compute: dy = w * (grad_w - sum(grad_w * w, axis=1))
    sum_gw = tl.sum(grad_w * w, axis=1) # [BLOCK_M]
    dy = w * (grad_w - sum_gw[:, None]) # [BLOCK_M, BLOCK_K]
    dy_out = dy.to(grad_w_ptr.dtype.element_ty)
    
    # Scatter dy back to grad_logits [num_tokens, E]
    store_ptrs = grad_logits_ptr + offs_m[:, None] * E + idxs
    tl.store(store_ptrs, dy_out, mask=load_mask)


# Two-layer pattern: raw triton launches wrapped as custom_ops so the
# `weights`/`indices` outputs get proper FX provenance (no control_deps
# KeyError when fed into a functional collective under torch.compile).

@torch.library.custom_op(f"{_NS}::fused_topk_softmax_fwd", mutates_args=())
def _fused_topk_softmax_fwd(logits: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, E = logits.shape
    weights = torch.empty((num_tokens, K), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((num_tokens, K), device=logits.device, dtype=torch.int32)

    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_K = triton.next_power_of_2(K)

    # Grid is determined dynamically by the autotuned BLOCK_M config
    grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_M']),)

    _fused_topk_softmax_fwd_kernel_2d[grid](
        logits, weights, indices,
        num_tokens, E, K, BLOCK_E=BLOCK_E, BLOCK_K=BLOCK_K,
    )

    return weights, indices


@_fused_topk_softmax_fwd.register_fake
def _(logits: torch.Tensor, K: int):
    num_tokens = logits.shape[0]
    return (
        torch.empty((num_tokens, K), device=logits.device, dtype=logits.dtype),
        torch.empty((num_tokens, K), device=logits.device, dtype=torch.int32),
    )


@torch.library.custom_op(f"{_NS}::fused_topk_softmax_bwd", mutates_args=())
def _fused_topk_softmax_bwd(grad_weights: torch.Tensor, weights: torch.Tensor,
                            indices: torch.Tensor, num_tokens: int, E: int, K: int) -> torch.Tensor:
    grad_logits = torch.zeros((num_tokens, E), device=weights.device, dtype=weights.dtype)

    BLOCK_K = triton.next_power_of_2(K)

    # Grid is determined dynamically by the autotuned BLOCK_M config
    grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_M']),)

    _fused_topk_softmax_bwd_kernel_2d[grid](
        grad_weights, weights, indices, grad_logits,
        num_tokens, E, K, BLOCK_K=BLOCK_K,
    )

    return grad_logits


@_fused_topk_softmax_bwd.register_fake
def _(grad_weights: torch.Tensor, weights: torch.Tensor,
      indices: torch.Tensor, num_tokens: int, E: int, K: int):
    return torch.empty((num_tokens, E), device=weights.device, dtype=weights.dtype)


class FusedTopkSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, K):
        weights, indices = torch.ops.tg.fused_topk_softmax_fwd(logits, K)
        ctx.save_for_backward(weights, indices)
        ctx.E = logits.shape[1]
        return weights, indices

    @staticmethod
    def backward(ctx, grad_weights, grad_indices):
        weights, indices = ctx.saved_tensors
        E = ctx.E
        num_tokens = weights.shape[0]
        K = weights.shape[1]
        grad_logits = torch.ops.tg.fused_topk_softmax_bwd(
            grad_weights, weights, indices, num_tokens, E, K)
        return grad_logits, None


def fused_topk_softmax(logits: torch.Tensor, K: int, fp32_routing: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Top-K Softmax operator.
    
    Args:
        logits (Tensor): Input logits of shape [num_tokens, num_experts].
        K (int): Top-K value.
        fp32_routing (bool): If True, forces routing weights calculation to be done in float32.
        
    Returns:
        weights (Tensor): Softmax weights of shape [num_tokens, K].
        indices (Tensor): Selected expert indices of shape [num_tokens, K].
    """
    if fp32_routing and logits.dtype != torch.float32:
        logits = logits.float()
    return FusedTopkSoftmaxFunction.apply(logits, K)
