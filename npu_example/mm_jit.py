# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout
import torch

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=16384, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=16384, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=16384, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, K_L1, S1, S2, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    FLAG = {"M": {"MTE2": list(range(S1)), "MTE1": list(range(S2))}}

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Change into persistent mode.
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            with T.Scope("C"):
                T.use_swizzle(M, N, K, block_M, block_N, off=3)

                bx = cid // n_num
                by = cid % n_num

                A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
                B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

                T.annotate_layout({
                    A_L1: make_zn_layout(A_L1),
                    B_L1: make_zn_layout(B_L1),
                })

                A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
                B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
                C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
                # JIT
                T.init_flag(FLAG)
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    with T.rs("MTE2"):
                        T.wait_flag("M", k % S1)
                        T.copy(A[bx * block_M, k * K_L1], A_L1[k % S1, :, :])
                        T.copy(B[k * K_L1, by * block_N], B_L1[k % S1, :, :])
                        T.set_flag("MTE1", k % S1)

                    loop_kk = T.ceildiv(K_L1, block_K)

                    for kk in T.serial(loop_kk):
                        with T.rs("MTE1"):
                            if kk == 0:
                                T.wait_flag("MTE2", k % S1)
                            T.wait_flag("M", kk % S2)
                            T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                            T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                            T.set_flag("M", kk % S2)
                        with T.rs("M"):
                            T.wait_flag("MTE1", kk % S2)
                            if k == 0 and kk == 0:
                                T.gemm(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0, init=True)
                            else:
                                T.gemm(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0)

                            T.set_flag("MTE1", kk % S2)

                            if kk == loop_kk - 1:
                                T.set_flag("MTE2", k % S1)
                                if k == loop_k - 1:
                                    T.set_flag("FIX", 0)

                with T.rs("FIX"):
                    T.wait_flag("M", 0)
                    T.copy(C_L0, C[bx * block_M, by * block_N])

                T.clear_flag(FLAG)

    return main


func = matmul(M, N, K, 128, 256, 64, 256, 2, 2)

torch.manual_seed(0)

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")


c = func(a, b)
ref_c = a @ b

# print(c)
# print(ref_c)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")