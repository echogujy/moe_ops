import torch
import triton
import triton.language as tl

@triton.jit
def _fused_topk_softmax_fwd_kernel(
    logits_ptr, weights_ptr, indices_ptr,
    num_tokens, E, K: tl.constexpr, BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= num_tokens:
        return
        
    # Load one row of logits
    offs_e = tl.arange(0, BLOCK_E)
    logits = tl.load(logits_ptr + t * E + offs_e, mask=offs_e < E, other=-float('inf'))
    
    # Initialize values and indices vectors of size K in registers
    w = tl.zeros((K,), dtype=tl.float32)
    idxs = tl.zeros((K,), dtype=tl.int32)
    
    temp_logits = logits
    
    # Loop to find top-K completely in registers
    for k in range(K):
        val = tl.max(temp_logits, axis=0)
        idx = tl.argmax(temp_logits, axis=0)
        
        # Insert at position k
        mask_k = tl.arange(0, K) == k
        w = tl.where(mask_k, val, w)
        idxs = tl.where(mask_k, idx.to(tl.int32), idxs)
        
        # Mask out this max element
        mask_e = offs_e == idx
        temp_logits = tl.where(mask_e, -float('inf'), temp_logits)
        
    # Compute softmax on the registers using Triton built-in
    w_softmax = tl.softmax(w)
    
    # Store the final result to global memory
    offs_k = tl.arange(0, K)
    tl.store(weights_ptr + t * K + offs_k, w_softmax)
    tl.store(indices_ptr + t * K + offs_k, idxs)


@triton.jit
def _fused_topk_softmax_bwd_kernel(
    grad_w_ptr, w_ptr, idxs_ptr, grad_logits_ptr,
    num_tokens, E, K: tl.constexpr, BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= num_tokens:
        return
        
    # Load w, grad_w and indices for this token
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    
    w = tl.load(w_ptr + t * K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    grad_w = tl.load(grad_w_ptr + t * K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    idxs = tl.load(idxs_ptr + t * K + offs_k, mask=k_mask, other=0).to(tl.int64)
    
    # Compute softmax backward: dy_k = w_k * (grad_w_k - sum(grad_w * w))
    sum_gw = tl.sum(grad_w * w, axis=0)
    dy = w * (grad_w - sum_gw)
    
    # Scatter dy back to grad_logits
    tl.store(grad_logits_ptr + t * E + idxs, dy, mask=k_mask)


class FusedTopkSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, K):
        num_tokens, E = logits.shape
        weights = torch.empty((num_tokens, K), device=logits.device, dtype=logits.dtype)
        indices = torch.empty((num_tokens, K), device=logits.device, dtype=torch.int32)
        
        BLOCK_E = triton.next_power_of_2(E)
        
        _fused_topk_softmax_fwd_kernel[(num_tokens,)](
            logits, weights, indices,
            num_tokens, E, K, BLOCK_E=BLOCK_E,
        )
        
        ctx.save_for_backward(weights, indices)
        ctx.E = E
        ctx.K = K
        return weights, indices

    @staticmethod
    def backward(ctx, grad_weights, grad_indices):
        weights, indices = ctx.saved_tensors
        E = ctx.E
        K = ctx.K
        num_tokens = weights.shape[0]
        
        grad_logits = torch.zeros((num_tokens, E), device=weights.device, dtype=weights.dtype)
        
        BLOCK_K = triton.next_power_of_2(K)
        
        _fused_topk_softmax_bwd_kernel[(num_tokens,)](
            grad_weights, weights, indices, grad_logits,
            num_tokens, E, K, BLOCK_K=BLOCK_K,
        )
        
        return grad_logits, None


def fused_topk_softmax(logits: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Top-K Softmax operator.
    
    Args:
        logits (Tensor): Input logits of shape [num_tokens, num_experts].
        K (int): Top-K value.
        
    Returns:
        weights (Tensor): Softmax weights of shape [num_tokens, K].
        indices (Tensor): Selected expert indices of shape [num_tokens, K].
    """
    return FusedTopkSoftmaxFunction.apply(logits, K)
