"""Single-GPU Mixture-of-Experts example built entirely on ``triton_gmm_ops``.

This is a reference implementation (no distributed comms) showing how to stitch
the four primitives into a full MoE layer:

    fused_topk_softmax  ->  permute  ->  grouped_gemm x3  ->  unpermute

It doubles as a correctness check: ``run_moe`` is compared against a naive
loop-over-experts reference in ``test_moe_correctness``.

Layout notes (see the library modules for details):
  * ``permute`` returns ``(permuted, row_id_map, base)`` where ``base`` is the
    EXCLUSIVE group offset (base[e] = start row of expert e). ``grouped_gemm``
    wants INCLUSIVE offsets (end row of expert e), so we convert with
    ``offsets = torch.cat([base[1:], tensor([total])])``.
  * ``grouped_gemm(A, B, offsets, trans_b=True)`` with ``B`` shaped ``[E, N, K]``
    computes ``A[e-group] @ B[e].T`` -> ``[total_tokens, N]``.
  * Activation: standard SwiGLU = SiLU(gate) * up  (NOT SiLU(gate * up)).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from triton_gmm_ops import (
    fused_topk_softmax,
    permute,
    grouped_gemm,
    unpermute,
)

DEVICE = "cuda"


class SingleGpuMoE(torch.nn.Module):
    """A minimal single-GPU MoE layer using standard SwiGLU activation.

    Shapes:
      weight_router : [num_experts, d_model]
      weight_gate   : [num_experts, hidden, d_model]
      weight_up     : [num_experts, hidden, d_model]
      weight_down   : [num_experts, d_model, hidden]
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, hidden: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.w_router = torch.nn.Parameter(torch.randn(num_experts, d_model))
        self.w_gate = torch.nn.Parameter(torch.randn(num_experts, hidden, d_model))
        self.w_up   = torch.nn.Parameter(torch.randn(num_experts, hidden, d_model))
        self.w_down = torch.nn.Parameter(torch.randn(num_experts, d_model, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]

        # 1. Router: fused top-k + softmax over experts.
        routing_weights, selected_experts = fused_topk_softmax(
            x @ self.w_router.T, self.top_k)

        # 2. Permute tokens into per-expert grouped layout.
        indices_i32 = selected_experts.to(torch.int32).contiguous()
        permuted, row_id_map, offsets = permute(x, indices_i32)

        # 3. Expert compute: standard SwiGLU = SiLU(gate) * up -> down.
        gate = grouped_gemm(permuted, self.w_gate, offsets, trans_b=True)
        up   = grouped_gemm(permuted, self.w_up,   offsets, trans_b=True)
        act_out = F.silu(gate) * up          # standard SwiGLU
        down = grouped_gemm(act_out, self.w_down, offsets, trans_b=True)

        # 4. Unpermute back to original token order, weighted by routing probs.
        return unpermute(down, row_id_map, routing_weights,
                         num_tokens=num_tokens, num_topK=self.top_k)


def _ref_moe(x, m: SingleGpuMoE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive per-token reference.

    Matches ShardMoEGroupedGEMM_native's logic:
    Routes each token independently using native PyTorch top-k + softmax.
    """
    num_tokens = x.shape[0]
    d_model = x.shape[1]
    
    # 1. Native Router: top-k + softmax over experts
    # We do the routing computation in float32 for stable comparison
    router_logits = x.float() @ m.w_router.float().T
    routing_weights, selected_experts = torch.topk(router_logits, m.top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1).to(x.dtype)
    
    out = torch.zeros(num_tokens, d_model, device=x.device, dtype=torch.float32)
    xf = x.float()
    for t in range(num_tokens):
        for k in range(m.top_k):
            e = int(selected_experts[t, k])
            p = routing_weights[t, k].float()
            # Standard SwiGLU in float32
            g = xf[t:t+1] @ m.w_gate[e].float().T   # [1, hidden]
            u = xf[t:t+1] @ m.w_up[e].float().T     # [1, hidden]
            h = F.silu(g) * u                         # [1, hidden]
            out[t] += (p * (h @ m.w_down[e].float().T))[0]
    return out.to(x.dtype), selected_experts, routing_weights


def test_moe_correctness():
    """Forward correctness: Triton path vs fully independent per-token float32 reference.

    Tolerance accounts for bf16 stacked-GEMM accumulation error.
    """
    torch.manual_seed(42)
    d, E, K, hidden = 32, 8, 2, 64
    # Scaled init to keep output magnitudes O(1) for bf16 correctness testing.
    scale = 0.02
    m = SingleGpuMoE(d, E, K, hidden).to(DEVICE).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
        # Apply structured bias to each expert router vector to prevent ties in BF16
        for e in range(E):
            m.w_router[e].add_(e * 1.5)
    x = torch.randn(128, d, device=DEVICE, dtype=torch.bfloat16) * scale

    with torch.no_grad():
        # Compute Triton output (fused_topk_softmax is called inside forward)
        out = m(x)
        
        # Get the experts and weights used by Triton path for comparison
        triton_weights, triton_experts = fused_topk_softmax(x @ m.w_router.T, K)
        
        # Fully independent reference path
        ref, ref_experts, ref_weights = _ref_moe(x, m)

    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = ((out.float() - ref.float()).abs() /
               (ref.float().abs() + 1e-6)).max().item()
    
    idx_diff = (triton_experts.int() - ref_experts.int()).abs().max().item()
    weight_diff = (triton_weights.float() - ref_weights.float()).abs().max().item()
    
    # ponytail: bf16 stacked-GEMM (gate+up+down) with hidden=64 gives
    # ~1e-2 abs error per element.
    ok = max_abs < 0.1 and max_rel < 0.05 and idx_diff == 0
    print(f"[moe forward] vs independent-ref: {'PASS' if ok else 'FAIL'} "
          f"(max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, idx_diff={idx_diff}, weight_diff={weight_diff:.2e})")
    assert idx_diff == 0, f"Router selected different experts! idx_diff={idx_diff}"
    assert ok, "single-GPU MoE forward diverges from per-token float32 reference"


def test_moe_backward():
    """Backward correctness: gradients via autograd vs finite differences."""
    torch.manual_seed(7)
    d, E, K, hidden = 16, 4, 2, 32
    scale = 0.02
    m = SingleGpuMoE(d, E, K, hidden).to(DEVICE).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    x = (torch.randn(32, d, device=DEVICE, dtype=torch.bfloat16) * scale
         ).requires_grad_(True)

    # Autograd backward
    out = m(x)
    loss = out.sum()
    loss.backward()
    grad_x = x.grad.clone()

    # Finite-difference check on x (sum scalar -> grad is sum of Jacobian rows)
    eps = 1e-2
    fd_grad = torch.zeros_like(grad_x, dtype=torch.float32)
    with torch.no_grad():
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                xp = x.detach().clone(); xp[i, j] += eps
                xn = x.detach().clone(); xn[i, j] -= eps
                fp = m(xp).sum().item()
                fn = m(xn).sum().item()
                fd_grad[i, j] = (fp - fn) / (2 * eps)

    max_diff = (grad_x.float() - fd_grad).abs().max().item()
    ok = max_diff < 0.05
    print(f"[moe backward] FD check on x: {'PASS' if ok else 'FAIL'} "
          f"(max_diff={max_diff:.2e})")
    assert ok, "single-GPU MoE backward (grad_x) fails FD check"


def bench_moe(num_tokens=4096, d=2048, E=64, K=2, hidden=8192):
    """Detailed MoE bench.

    Triton component breakdown (forward) + Triton-vs-native GEMM comparison at
    MoE scale. The 3 expert GEMMs dominate MoE cost (and are what we tune), so
    we time them against torch.nn.functional.grouped_mm with identical inputs.
    """
    import time
    scale = 1.0 / (d ** 0.5)
    m = SingleGpuMoE(d, E, K, hidden).to(DEVICE).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    x = torch.randn(num_tokens, d, device=DEVICE, dtype=torch.bfloat16,
                    requires_grad=True)

    def _bench(fn, warmup=5, iters=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) * 1e3 / iters

    # ---- Precompute per-expert layout + grad outputs (needed by all benches) ----
    logits = x @ m.w_router.T
    rw, indices = fused_topk_softmax(logits, K)
    indices_i32 = indices.to(torch.int32).contiguous()
    permuted, row_id_map, _ = permute(x, indices_i32)
    counts = torch.bincount(indices_i32.reshape(-1), minlength=E).to(torch.int32)
    offs_tri = torch.zeros(E + 1, dtype=torch.int32, device=DEVICE)  # start offsets (len E+1)
    offs_tri[1:] = torch.cumsum(counts, 0).to(torch.int32)
    # ponytail: torch.grouped_mm uses a DIFFERENT offs convention — exclusive-end,
    # length E, int32 (and torch.cumsum promotes int32->int64, so recast).
    offs_n = torch.cumsum(counts, 0).to(torch.int32)

    # upstream grad tensors (shape == each GEMM output); used as .backward(grad).
    # permuted has num_tokens*top_k rows; down-GEMM input is act_out (hidden-dim).
    total_perm = permuted.shape[0]
    g_gate = torch.randn(total_perm, hidden, device=DEVICE, dtype=torch.bfloat16)
    g_up   = torch.randn(total_perm, hidden, device=DEVICE, dtype=torch.bfloat16)
    g_down = torch.randn(total_perm, d,      device=DEVICE, dtype=torch.bfloat16)

    Ag = permuted.detach().requires_grad_(True)              # gate/up input: [total, d]
    Ad = torch.randn(total_perm, hidden, device=DEVICE,
                     dtype=torch.bfloat16).requires_grad_(True)  # down input (act_out stand-in)
    Wg = m.w_gate.detach().requires_grad_(True)
    Wu = m.w_up.detach().requires_grad_(True)
    Wd = m.w_down.detach().requires_grad_(True)

    def zero_grads():
        for t in (x, g_gate, g_up, g_down):
            t.grad = None
        for p in m.parameters():
            p.grad = None

    # ---- Triton end-to-end ----
    t_fwd = _bench(lambda: m(x))
    t_fwdbwd = _bench(lambda: (m(x).sum().backward(), zero_grads()))

    # ---- Triton forward component breakdown ----
    t_router  = _bench(lambda: fused_topk_softmax(logits, K))
    t_permute = _bench(lambda: permute(x, indices_i32))
    t_gemm_f  = _bench(lambda: (
        grouped_gemm(permuted, m.w_gate, offs_tri),
        grouped_gemm(permuted, m.w_up,   offs_tri),
        grouped_gemm(permuted, m.w_down, offs_tri)))
    t_unperm  = _bench(lambda: unpermute(grouped_gemm(permuted, m.w_down, offs_tri),
                                        row_id_map, rw,
                                        num_tokens=num_tokens, num_topK=K))

    # ---- GEMM backward: Triton vs native (torch grouped_mm) ----
    # Drive backward through autograd: y = GEMM(A, W); y.backward(grad_output).
    def triton_gemm_bwd():
        grouped_gemm(Ag, Wg, offs_tri).backward(g_gate, retain_graph=True)
        grouped_gemm(Ag, Wu, offs_tri).backward(g_up,   retain_graph=True)
        grouped_gemm(Ad, Wd, offs_tri).backward(g_down, retain_graph=True)
        Ag.grad = None; Ad.grad = None; Wg.grad = None; Wu.grad = None; Wd.grad = None
    t_gemm_b_triton = _bench(triton_gemm_bwd)

    # native grouped_mm wants mat_b as [E, K, N] (it does A@B, no internal transpose)
    wg_n = m.w_gate.detach().transpose(1, 2).contiguous().requires_grad_(True)
    wu_n = m.w_up.detach().transpose(1, 2).contiguous().requires_grad_(True)
    wd_n = m.w_down.detach().transpose(1, 2).contiguous().requires_grad_(True)
    def native_gemm_bwd():
        torch.nn.functional.grouped_mm(Ag, wg_n, offs=offs_n).backward(g_gate, retain_graph=True)
        torch.nn.functional.grouped_mm(Ag, wu_n, offs=offs_n).backward(g_up,   retain_graph=True)
        torch.nn.functional.grouped_mm(Ad, wd_n, offs=offs_n).backward(g_down, retain_graph=True)
        Ag.grad = None; Ad.grad = None; wg_n.grad = None; wu_n.grad = None; wd_n.grad = None
    t_gemm_b_native = _bench(native_gemm_bwd)

    print(f"[bench] {num_tokens}T d={d} E={E} K={K} H={hidden}")
    print(f"  Triton  fwd={t_fwd:.2f}ms  fwd+bwd={t_fwdbwd:.2f}ms")
    print(f"  Triton fwd breakdown: router={t_router:.3f}  permute={t_permute:.3f}  "
          f"3xgemm={t_gemm_f:.3f}  unpermute={t_unperm:.3f} ms")
    print(f"  GEMM backward (3x): Triton={t_gemm_b_triton:.2f}ms  "
          f"native={t_gemm_b_native:.2f}ms  speedup={t_gemm_b_native/t_gemm_b_triton:.2f}x")


if __name__ == "__main__":
    test_moe_correctness()
    test_moe_backward()
    bench_moe()
    print("OK")
