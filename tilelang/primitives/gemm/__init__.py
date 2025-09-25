# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import Optional
from tvm import tir
from tilelang.primitives.gemm.base import GemmWarpPolicy
from tilelang.primitives.gemm.gemm_mma import (
    GemmPrimitiveMMA,)


def gemm(
    A: tir.Buffer,
    B: tir.Buffer,
    C: tir.Buffer,
    transpose_A: bool = False,
    transpose_B: bool = False,
    block_row_warps: Optional[int] = None,
    block_col_warps: Optional[int] = None,
    warp_row_tiles: Optional[int] = None,
    warp_col_tiles: Optional[int] = None,
    chunk: Optional[int] = None,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    k_pack: int = 1,
):

    return GemmPrimitiveMMA(
        A=A,
        B=B,
        C=C,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        policy=policy,
        k_pack=k_pack,
    ).invoke()
