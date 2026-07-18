import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
if not hasattr(torch.nn.functional, "grouped_mm"):
    def grouped_mm_fallback(A, B, offs):
        E = B.shape[0]
        N = B.shape[2]
        C = torch.empty((A.shape[0], N), dtype=A.dtype, device=A.device)
        starts = torch.cat([torch.zeros(1, dtype=offs.dtype, device=offs.device), offs[:-1]])
        for g in range(E):
            s, e = int(starts[g]), int(offs[g])
            C[s:e] = A[s:e] @ B[g]
        return C
    torch.nn.functional.grouped_mm = grouped_mm_fallback
import time
from triton_gmm_ops import grouped_gemm

DEVICE = "cuda"

def _bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / iters


def run_correctness(dtype=torch.bfloat16, atol=1.5, rtol=0.02):
    # Shapes
    num_tokens = 8192
    K = 4096
    E = 16
    N = 4096
    
    A = torch.randn(num_tokens, K, device=DEVICE, dtype=dtype, requires_grad=True)
    B = torch.randn(E, N, K, device=DEVICE, dtype=dtype, requires_grad=True)
    w_native = B.transpose(1, 2).contiguous().detach().requires_grad_(True)
    
    indices = torch.randint(0, E, (num_tokens,), device=DEVICE, dtype=torch.int32)
    sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
    for idx in indices:
        sizes[idx] += 1
    offsets = torch.cumsum(sizes, dim=0).to(torch.int32)
    offsets_tri = torch.zeros(E + 1, dtype=torch.int32, device=DEVICE)
    offsets_tri[1:] = offsets
    
    # Triton
    C_tri = grouped_gemm(A, B, offsets_tri, trans_b=True)
    grad_output = torch.randn_like(C_tri)
    C_tri.backward(grad_output, retain_graph=True)
    grad_A_tri = A.grad.clone()
    grad_B_tri = B.grad.clone()
    
    # Native
    A_n = A.clone().detach().requires_grad_(True)
    w_native_n = w_native.clone().detach().requires_grad_(True)
    C_native = torch.nn.functional.grouped_mm(A_n, w_native_n, offs=offsets)
    
    # Map back native grad to trans_b style
    C_native.backward(grad_output, retain_graph=True)
    grad_A_nat = A_n.grad.clone()
    grad_B_nat = w_native_n.grad.transpose(1, 2).clone()

    # ── Float32 loop reference (ground truth) ────────────────────────────────
    A_f32 = A.detach().float()
    B_f32 = B.detach().float()
    go_f32 = grad_output.float()

    C_ref  = torch.zeros(num_tokens, N, device=DEVICE)
    gA_ref = torch.zeros_like(A_f32)
    gB_ref = torch.zeros(E, N, K, device=DEVICE)

    for g in range(E):
        s, e = offsets_tri[g].item(), offsets_tri[g + 1].item()
        C_ref[s:e]  = A_f32[s:e] @ B_f32[g].t()
        gA_ref[s:e] = go_f32[s:e] @ B_f32[g]
        gB_ref[g]   = go_f32[s:e].t() @ A_f32[s:e]

    def _check(x_tri, x_nat, ref):
        d_tri = (x_tri.float() - ref).abs().max().item()
        d_nat = (x_nat.float() - ref).abs().max().item()
        tol   = atol + rtol * ref.abs().max().item()
        return d_tri <= tol, d_nat <= tol, d_tri, d_nat

    ok_fwd_t, ok_fwd_n, abs_fwd_t, abs_fwd_n = _check(C_tri,      C_native,   C_ref)
    ok_ga_t,  ok_ga_n,  abs_ga_t,  abs_ga_n  = _check(grad_A_tri, grad_A_nat, gA_ref)
    ok_gb_t,  ok_gb_n,  abs_gb_t,  abs_gb_n  = _check(grad_B_tri, grad_B_nat, gB_ref)

    W = 72
    print("=" * W)
    print(f"Correctness vs float32 loop reference ({dtype}, K={K}):")
    print(f"  (atol={atol}, rtol={rtol})")
    print("=" * W)
    print(f"{'':12s}  {'Triton abs_max':>16s}  {'Native abs_max':>16s}")
    print(f"{'Forward':12s}  {abs_fwd_t:>14.4f}  {'✓' if ok_fwd_t else '✗'}  {abs_fwd_n:>14.4f}  {'✓' if ok_fwd_n else '✗'}")
    print(f"{'Grad_A':12s}  {abs_ga_t:>14.4f}  {'✓' if ok_ga_t else '✗'}  {abs_ga_n:>14.4f}  {'✓' if ok_ga_n else '✗'}")
    print(f"{'Grad_B':12s}  {abs_gb_t:>14.4f}  {'✓' if ok_gb_t else '✗'}  {abs_gb_n:>14.4f}  {'✓' if ok_gb_n else '✗'}")
    print("=" * W)

    assert ok_fwd_t, f"[{dtype}] Triton forward  abs diff too large vs f32 ref: {abs_fwd_t:.4f}"
    assert ok_ga_t,  f"[{dtype}] Triton grad_A   abs diff too large vs f32 ref: {abs_ga_t:.4f}"
    assert ok_gb_t,  f"[{dtype}] Triton grad_B   abs diff too large vs f32 ref: {abs_gb_t:.4f}"

def run_benchmark():
    shapes = [
        (4096, 4096, 16, 4096),
        (8192, 4096, 16, 4096),
        (16384, 4096, 16, 4096),
        (16384, 8192, 16, 8192),
    ]
    
    print("\n" + "=" * 80)
    print(f"{'Grouped GEMM - FORWARD Performance Comparison (ms)':^80}")
    print("=" * 80)
    print(f"{'Shape (Tokens x K x E x N)':<25} | {'Triton GMM':<15} | {'Torch Native':<15} | {'Speedup':<15}")
    print("-" * 80)
    
    for tokens, K, E, N in shapes:
        A = torch.randn(tokens, K, device=DEVICE, dtype=torch.bfloat16)
        B = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16)
        w_native = B.transpose(1, 2).contiguous()
        
        indices = torch.randint(0, E, (tokens,), device=DEVICE, dtype=torch.int32)
        sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
        for idx in indices:
            sizes[idx] += 1
        offsets = torch.cumsum(sizes, dim=0).to(torch.int32)
        offsets_tri = torch.zeros(E + 1, dtype=torch.int32, device=DEVICE)
        offsets_tri[1:] = offsets
        
        t_tri = _bench(lambda: grouped_gemm(A, B, offsets_tri, trans_b=True))
        t_nat = _bench(lambda: torch.nn.functional.grouped_mm(A, w_native, offs=offsets))
        
        shape_str = f"{tokens}x{K}x{E}x{N}"
        speedup_str = f"{t_nat / t_tri:.2f}x"
        
        print(f"{shape_str:<25} | {t_tri:7.3f} ms     | {t_nat:7.3f} ms     | {speedup_str:<15}")
    print("=" * 80)


    print("\n" + "=" * 80)
    print(f"{'Grouped GEMM - BACKWARD Performance Comparison (ms)':^80}")
    print("=" * 80)
    print(f"{'Shape (Tokens x K x E x N)':<25} | {'Triton GMM':<15} | {'Torch Native':<15} | {'Speedup':<15}")
    print("-" * 80)
    
    for tokens, K, E, N in shapes:
        A = torch.randn(tokens, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        B = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        w_native = B.transpose(1, 2).contiguous().detach().requires_grad_(True)
        
        indices = torch.randint(0, E, (tokens,), device=DEVICE, dtype=torch.int32)
        sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
        for idx in indices:
            sizes[idx] += 1
        offsets = torch.cumsum(sizes, dim=0).to(torch.int32)
        offsets_tri = torch.zeros(E + 1, dtype=torch.int32, device=DEVICE)
        offsets_tri[1:] = offsets
        
        C_tri = grouped_gemm(A, B, offsets_tri, trans_b=True)
        grad_out = torch.randn_like(C_tri)
        
        # backward triton bench
        def run_tri_bwd():
            A.grad = None
            B.grad = None
            C_tri.backward(grad_out, retain_graph=True)
            
        t_tri = _bench(run_tri_bwd)
        
        C_native = torch.nn.functional.grouped_mm(A, w_native, offs=offsets)
        
        # backward native bench
        def run_nat_bwd():
            A.grad = None
            w_native.grad = None
            C_native.backward(grad_out, retain_graph=True)
            
        t_nat = _bench(run_nat_bwd)
        
        shape_str = f"{tokens}x{K}x{E}x{N}"
        speedup_str = f"{t_nat / t_tri:.2f}x"
        print(f"{shape_str:<25} | {t_tri:7.3f} ms     | {t_nat:7.3f} ms     | {speedup_str:<15}")
    print("=" * 80)


if __name__ == "__main__":
    run_correctness(torch.bfloat16, atol=1.5, rtol=0.02)
    run_correctness(torch.float32,  atol=0.3, rtol=0.01)
    run_benchmark()
