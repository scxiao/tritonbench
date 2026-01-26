import torch
import torch._inductor.config as inductor_config

m = 20120
n = 512
k = 1536
dtype = torch.float16

def pt2_triton_matmul(a, mat1, mat2):
    torch._dynamo.reset()
    with inductor_config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    ):
        f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
        compiled = torch.compile(f, dynamic=False)
        print(f"pt2, loc2, compiled = {compiled}, a_shape = {a.shape}, mat1 = {mat1.shape}, mat2 = {mat2.shape}, type = {mat1.dtype}, o_type = {a.dtype}")
        compiled(a, mat1, mat2)
    return lambda: compiled(a, mat1, mat2)


# Example usage (ensure inputs are on a GPU if using a CUDA backend)
input = torch.randn(m, n, device='cuda', dtype=dtype)
mat1 = torch.randn(m, k, device='cuda', dtype=dtype)
mat2 = torch.randn(k, n, device='cuda', dtype=dtype)
output = pt2_triton_matmul(input, mat1, mat2)

