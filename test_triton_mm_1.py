import torch


import triton
import triton.language as tl
import sys
from typing import List


@triton.jit
def triton_mm(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 256
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 64
    matrix_instr_nonkdim : tl.constexpr = 16
    waves_per_eu : tl.constexpr = 0
    kpack : tl.constexpr = 2
    GROUP_M : tl.constexpr = 4
    ALLOW_TF32 : tl.constexpr = False
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 20120
    N = 512
    K = 1536
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1536
    stride_ak = 1
    stride_bk = 512
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 1536*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + (xindex))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n + 512*idx_m, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)


def run_triton_mm(m, n, k):
    dtype = torch.float16
    # Example usage (ensure inputs are on a GPU if using a CUDA backend)
    input = torch.randn(m, n, device='cuda', dtype=dtype)
    mat1 = torch.randn(m, k, device='cuda', dtype=dtype)
    mat2 = torch.randn(k, n, device='cuda', dtype=dtype)
    output = torch.empty_like(input)
    BLOCK_M = 256
    BLOCK_N = 256

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), )
    triton_mm[grid](input, mat1, mat2, output, num_stages=2, num_warps=1, waves_per_eu = 8, matrix_instr_nonkdim=16, kpack=2)

    output_ref = torch.addmm(input, mat1, mat2)

    torch.testing.assert_close(output, output_ref)


m = 20120
n = 512
k = 1536

run_triton_mm(m, n, k)
