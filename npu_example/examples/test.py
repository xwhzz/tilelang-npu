lib_path = "./kernel_lib.so"

import torch
from typing import Callable, Optional, Literal, List, Union
from functools import partial

import argparse

parser = argparse.ArgumentParser(description="NPU Kernel Compilation")
parser.add_argument("--m", type=int, default=16384, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=16384, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=16384, help="Matrix K dimension")
args = parser.parse_args()

M = args.m
N = args.n
K = args.k


torch.manual_seed(0)

a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
ref_c = torch.empty(M, N).half().npu()
print("init successful!")
import ctypes

lib = ctypes.CDLL(lib_path)

stream = torch.npu.current_stream()._as_parameter_


def tl():
    return lib.call(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()),
        stream)


# bl = partial(torch.mm, a, b, out=ref_c)

tl()
# bl()


# ref_c = torch.mm(a, b)
print(c)
print(ref_c)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")


"""
tensor([[-221.6250,  181.7500,  269.0000,  ...,  111.3750,  -14.6484,
         -161.5000],
        [ -69.5625,   54.2500,   80.4375,  ...,  -48.2812,  256.0000,
          252.5000],
        [ 103.8125,  -96.7500, -163.6250,  ...,  -19.7344,  -28.2656,
          122.4375],
        ...,
        [ -59.6875, -180.7500,  -64.6875,  ...,  -29.8750,  120.0625,
           12.4609],
        [-156.1250,  105.5000, -276.2500,  ...,  152.7500,  142.6250,
          -70.6250],
        [-107.8125,  121.0625,  186.7500,  ...,  124.0625,  183.2500,
         -148.3750]], device='npu:0', dtype=torch.float16)

"""