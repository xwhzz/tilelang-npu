import argparse

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--heads", type=int, default=1, help="Number of heads")
parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
parser.add_argument("--dim", type=int, default=128, help="Dimension")


args = parser.parse_args()

def flashattn(batch, heads, seq_len, dim): 
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [seq_len, dim]
    dtype = "float16"
    accum_dtype = "float"


    def kernel_func(block_M, block_N):
        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
                workspace_1: T.Tensor(shape, dtype),
                workspace_2: T.Tensor(shape, dtype),
                workspace_3: T.Tensor(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), is_npu=True) as (cid, _):
                
                L1A = T.alloc_L1([block_M, dim], dtype)
                L1B = T.alloc_L1([block_N, dim], dtype)
                L0C = T.alloc_L0C([block_M, block_N], accum_dtype)


                logsum = T.alloc_ub([block_M, 1], accum_dtype)
                scores_max = T.alloc_ub([block_M, 1], accum_dtype)
                scores_max_prev = T.alloc_ub([block_M, 1], accum_dtype)
                scores_scale = T.alloc_ub([block_M, 1], accum_dtype)
                scores_sum = T.alloc_ub([block_M, 1], accum_dtype)

                acc_s = T.alloc_ub([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_ub([block_M, block_N], dtype)
                acc_o = T.alloc_ub([block_M, dim], accum_dtype)

                with T.Scope("C"):
                    # two copy
                    T.copy(Q[cid * block_M:(cid + 1) * block_M, :], L1A)

                    for i in T.serial(T.ceildiv(seq_len, block_N)):
                        T.copy(K[i * block_N:(i + 1) * block_N, :], L1B)

                        T.gemm(L1A[0], L1B, L0C, init=True, transpose_B=True)
                        T.gemm(L1A[1], L1B, L0C, init=False, transpose_B=True)

                        with T.rs("FIX"):
                            T.copy(L0C, workspace_1[cid * block_M:(cid + 1) * block_M, :])
                            T.set_flag("VEC", i)

                        T.copy(V[i * block_N:(i + 1) * block_N, :], L1B)

                        with T.rs("MTE2"):
                            T.wait_flag("VEC", 0)
                            T.copy(workspace_2[cid * block_M:(cid + 1) * block_M, :], L1A)

                        T.gemm(L1A, L1B, L0C, init=True)

                        with T.rs("FIX"):
                            T.copy(L0C, workspace_3[cid * block_M:(cid + 1) * block_M, :])
                            T.set_flag("VEC", 2)
                
                with T.Scope("V"):
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.fill(acc_o, 0)             
                    for i in T.serial(T.ceildiv(seq_len, block_N)):
                        T.copy(scores_max, scores_max_prev)
                        with T.rs("MTE2"):
                            T.wait_flag("CUBE", 0)
                            T.copy(workspace_1[cid * block_M:(cid + 1) * block_M, :], acc_s)
                        T.mul(acc_s, scale, acc_s)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        T.sub(scores_max_prev, scores_max, scores_scale)
                        T.exp(scores_scale, scores_scale)
                        for i in T.serial(block_M):
                            T.sub(acc_s[i, :], scores_scale[i], acc_s[i, :])
                        T.exp(acc_s, acc_s)

                        T.reduce_sum(acc_s, scores_sum, dim=1)

                        T.mul(logsum, scores_scale, logsum)
                        T.add(logsum, scores_sum, logsum)
                        for i in T.serial(block_M):
                            T.mul(acc_o[i, :], scores_scale[i], acc_o[i, :])
                        T.cast(acc_s, acc_s_cast)

                        with T.rs("MTE3"):
                            T.copy(workspace_2[cid * block_M:(cid + 1) * block_M, :], acc_s_cast)
                            T.set_flag("CUBE", 1)
                        
                        with T.rs("MTE2"):
                            T.wait_flag("CUBE", 2)
                            T.copy(workspace_3[cid * block_M:(cid + 1) * block_M, :], acc_s)

                        T.add(acc_o, acc_s, acc_o)
                        
                    for i in T.serial(block_M):
                        T.div(acc_o[i, :], logsum[i], acc_o[i, :])
                    
                    T.copy(acc_o, Output[cid * block_M:(cid + 1) * block_M, :])

        return main
    
    return kernel_func
