import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import time
from triton_gmm_ops.grouped_gemm import grouped_gemm

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


def run_correctness():
    # Shapes
    num_tokens = 8192
    K = 4096
    E = 16
    N = 4096
    
    A = torch.randn(num_tokens, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, N, K, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    w_native = B.transpose(1, 2).contiguous().detach().requires_grad_(True)
    
    indices = torch.randint(0, E, (num_tokens,), device=DEVICE, dtype=torch.int32)
    sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
    for idx in indices:
        sizes[idx] += 1
    offsets = torch.cumsum(sizes, dim=0).to(torch.int32)
    
    # Triton
    C_tri = grouped_gemm(A, B, offsets, trans_b=True)
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
    
    max_diff_fwd = torch.max(torch.abs(C_tri - C_native)).item()
    max_diff_grad_a = torch.max(torch.abs(grad_A_tri - grad_A_nat)).item()
    max_diff_grad_b = torch.max(torch.abs(grad_B_tri - grad_B_nat)).item()
    
    print("================================================================")
    print("Correctness Check - Triton vs Torch Native:")
    print("================================================================")
    print(f"  Forward Diff:   {max_diff_fwd:.2e} -> {'PASS' if max_diff_fwd < 5.0 else 'FAIL'}")
    print(f"  Grad_A Diff:    {max_diff_grad_a:.2e} -> {'PASS' if max_diff_grad_a < 5.0 else 'FAIL'}")
    print(f"  Grad_B Diff:    {max_diff_grad_b:.2e} -> {'PASS' if max_diff_grad_b < 5.0 else 'FAIL'}")
    print("================================================================")


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
        
        t_tri = _bench(lambda: grouped_gemm(A, B, offsets, trans_b=True))
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
        
        C_tri = grouped_gemm(A, B, offsets, trans_b=True)
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
    run_correctness()
    run_benchmark()
