# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

import torch
from ..base import BaseKernelAdapter
import ctypes
from typing import List, Optional, Union, Callable, Dict, Tuple, Any
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.relay import TensorType
from tvm import tir
from tilelang.jit.adapter.wrapper import TLWrapper
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_target
from tilelang.utils.language import retrieve_func_from_module
import os, tempfile, subprocess, shutil, textwrap

def _is_ascend_target(t) -> bool:
        # t可能是str或tvm Target，统一成str再比较
        return str(t).lower() == "ascend"

class CtypesKernelAdapter(BaseKernelAdapter):
    """Adapter class that converts TVM/TIR functions to callable CUDA kernels using ctypes.
    
    This adapter handles:
    1. Converting TIR functions to compiled CUDA libraries
    2. Managing dynamic shapes in tensor operations
    3. Wrapping C++ kernels for Python/PyTorch usage
    """

    # Class attributes to store compiled kernel information
    target = "cuda"
    ir_module: Optional[tvm.IRModule] = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    kernel_global_source: Optional[str] = None
    lib: Optional[ctypes.CDLL] = None  # Compiled library handle
    wrapped_source: Optional[str] = None  # Generated C++ wrapper code
    # Maps symbolic variables to their corresponding buffer and shape indices
    dynamic_symbolic_map: Optional[Dict[tir.Var, Tuple[int, int]]] = None
    # Pass configs for the compiler
    pass_configs: Optional[Dict[str, Any]] = None

    # Add new cache attributes
    param_dtypes: Optional[List[torch.dtype]] = None  # Cache for parameter dtypes
    param_shapes: Optional[List[List]] = None  # Cache for parameter shapes

    def __init__(self,
                 params: List[TensorType],
                 result_idx: List[int],
                 target: str,
                 func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                 host_mod: Optional[tvm.IRModule] = None,
                 device_mod: Optional[tvm.IRModule] = None,
                 kernel_global_source: Optional[str] = None,
                 verbose: bool = False,
                 pass_configs: Optional[Dict[str, Any]] = None):
        """Initialize the adapter with the given TIR function or module.
        
        Args:
            params: List of tensor types for inputs/outputs
            result_idx: Indices of output tensors
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_global_source = kernel_global_source
        self.pass_configs = pass_configs
        self.verbose = verbose

        if isinstance(func_or_mod, tir.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        # Cache parameter information during initialization
        self.param_dtypes = [param.dtype for param in params]
        self.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            self.param_shapes.append(native_shape)

        self.dynamic_symbolic_map = self._process_dynamic_symbolic()

        raw_target = determine_target(target)
        if _is_ascend_target(raw_target):
            self.target = "ascend"
        else:
            self.target = Target.canon_target(raw_target)

        if _is_ascend_target(self.target):
            self._ascend_build_and_load()
            self._post_init()
            return
        
        self.verbose = verbose
        self.wrapper = TLWrapper(self.target)
        self.lib_generator = LibraryGenerator(self.target)

        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        self.wrapped_source = self.wrapper.wrap(self.get_kernel_source(kernel_only=True))

        self.lib_generator.update_lib_code(self.wrapped_source)
        self.lib_generator.compile_lib()
        self.lib = self.lib_generator.load_lib()
        self.lib.init()

        self._post_init()

    @classmethod
    def from_database(cls,
                      params: List[TensorType],
                      result_idx: List[int],
                      target: str,
                      func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
                      kernel_global_source: str,
                      kernel_lib_path: str,
                      verbose: bool = False,
                      pass_configs: Optional[Dict[str, Any]] = None):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.kernel_global_source = kernel_global_source
        adapter.wrapped_source = kernel_global_source
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tir.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        # Cache parameter information during initialization
        adapter.param_dtypes = [param.dtype for param in params]
        adapter.param_shapes = []
        for param in params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tir.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tir.Var):
                    native_shape.append(dim)  # Keep tir.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            adapter.param_shapes.append(native_shape)

        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()

        raw_target = determine_target(target)
        if _is_ascend_target(raw_target):
            adapter.target = "ascend"
        else:
            adapter.target = Target.canon_target(raw_target)

        adapter.verbose = verbose

        if _is_ascend_target(adapter.target):
            # Ascend：直接调用ctypes加载，不调用lib_generator / init
            if not os.path.exists(kernel_lib_path):
                raise FileNotFoundError(kernel_lib_path)
            adapter.lib = ctypes.CDLL(kernel_lib_path)
            # 绑定符号
            if not hasattr(adapter.lib, "call"):
                raise AttributeError("[Ascend] symbol 'call' not found in the shared library.")
            adapter.lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            adapter.lib.call.restype  = None
            adapter._post_init()
            return adapter
        
        # 非Ascend：保持原有逻辑
        adapter.lib_generator = LibraryGenerator(adapter.target)
        adapter.lib = adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.lib.init()

        adapter._post_init()
        return adapter

    def _process_dynamic_symbolic(self):
        """Extract information about dynamic shapes from the TIR function.
        
        Maps symbolic variables to their corresponding (buffer_index, shape_dimension)
        for runtime shape resolution.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map = {}
        for i, param in enumerate(params):
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                    dynamic_symbolic_map[shape] = (i, j)
        return dynamic_symbolic_map

    def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
        """Low-level function to call the compiled CUDA kernel.
        
        Converts PyTorch tensor pointers to C void pointers for ctypes interface.
        """
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream if stream is not None else 0))
        self.lib.call(*ctypes_args)

    def _warp_forward_from_prebuild_lib(self,
                                        *ins: List[torch.Tensor],
                                        stream: Optional[int] = None):
        """High-level wrapper for kernel execution.
        
        Handles:
        1. Input validation
        2. Output tensor allocation
        3. Dynamic shape resolution
        4. CUDA stream management
        
        Args:
            ins: Input PyTorch tensors
            stream: Optional CUDA stream for asynchronous execution
        
        Returns:
            Single tensor or list of tensors containing the kernel results
        """
        if len(ins) + len(self.result_idx) != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
            )
        ins_idx = 0
        args = []

        # tensor pointers
        for i in range(len(self.params)):
            if i in self.result_idx:
                dtype = self.param_dtypes[i]
                shape = []
                # Now working with native Python list, no FFI calls needed
                for s in self.param_shapes[i]:
                    if isinstance(s, tir.Var):
                        ref_tensor_idx, ref_shape_idx = self.dynamic_symbolic_map[s]
                        shape.append(ins[ref_tensor_idx].shape[ref_shape_idx])
                    else:  # Already converted to Python int during initialization
                        shape.append(s)

                if len(ins) > 0:
                    device = ins[0].device
                else:
                    if _is_ascend_target(self.target):
                        device = torch.device("npu")
                    elif str(self.target).startswith("cuda") and torch.cuda.is_available():
                        device = torch.device("cuda", torch.cuda.current_device())
                    else:
                        device = torch.device("cpu")
                
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                tensor = ins[ins_idx]
                ins_idx += 1
            args.append(tensor)

        # dynamic symbolics
        if not _is_ascend_target(self.target):
            for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
                args.append(ins[buffer_idx].shape[shape_idx])

        # if stream is not None, we need to pass the stream to the library
        if stream is None:
            if _is_ascend_target(self.target):
                stream = torch.npu.current_stream()._as_parameter_
            elif str(self.target).startswith("cuda") and torch.cuda.is_available():
                stream = torch.cuda.current_stream().cuda_stream
            else:
                stream = 0

        self._forward_from_prebuild_lib(*args, stream=stream)

        if len(self.result_idx) == 1:
            return args[self.result_idx[0]]
        else:
            return [args[i] for i in self.result_idx]

    def _convert_torch_func(self) -> Callable:
        """Returns a PyTorch-compatible function wrapper for the kernel."""
        return self._warp_forward_from_prebuild_lib

    @property
    def prim_func(self) -> tir.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)

    @property
    def srcpath(self):
        """Returns the source path of the compiled library."""
        if hasattr(self, "lib_generator"):
            return self.lib_generator.srcpath
        return getattr(self, "_asc_workdir", None)

    @property
    def libpath(self):
        """Returns the path to the compiled library."""
        if hasattr(self, "lib_generator"):
            return self.lib_generator.libpath
        if hasattr(self, "_asc_workdir"):
            return os.path.join(self._asc_workdir, "kernel_lib.so")
        return None

    @property
    def lib_code(self):
        """Returns the code of the compiled library."""
        if hasattr(self, "lib_generator"):
            return self.lib_generator.lib_code
        return self.kernel_global_source # Ascend返回wrapped/source

    @property
    def is_dynamic(self):
        """Indicates whether the kernel handles dynamic shapes."""
        return (self.dynamic_symbolic_map is not None and len(self.dynamic_symbolic_map) > 0)

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        if kernel_only:
            return self.kernel_global_source
        else:
            assert self.wrapped_source is not None, "Wrapped source is not available"
            return self.wrapped_source
        
    def _ascend_build_and_load(self):
        assert self.kernel_global_source is not None, "Ascend path requires kernel_global_source (the generated Ascend C++)."

        # 1. 准备临时工作目录
        self._asc_workdir = tempfile.mkdtemp(prefix="tl_ascend_")
        cpp_name = "tl_kernel.cpp"
        so_name  = "kernel_lib.so"

        # 2. 写入common.h与kernel
        self._asc_write_file("common.h", self._ascend_common_h_text())
        self._asc_write_file(cpp_name, self.kernel_global_source)

        # 3. 生成build.sh
        self._asc_write_file("build.sh", self._ascend_build_sh_text(cpp_name, so_name))
        os.chmod(os.path.join(self._asc_workdir, "build.sh"), 0o755)

        # 4. 编译
        env = os.environ.copy()
        try:
            subprocess.check_call(["bash", "build.sh"], cwd=self._asc_workdir, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[Ascend] bisheng build failed: {e}") from e
        
        # 5. 加载.so
        so_path = os.path.join(self._asc_workdir, so_name)
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"[Ascend] built library not found: {so_path}")
        self.lib = ctypes.CDLL(so_path)

        # 6. 绑定符号
        if not hasattr(self.lib, "call"):
            raise AttributeError("[Ascend] symbol 'call' not found in the shared library.")
        # 签名默认设为：void* a, void* b, void* c, void* stream
        self.lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.call.restype  = None

    def _asc_write_file(self, name: str, content: str):
        path = os.path.join(self._asc_workdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path
    
    def _ascend_common_h_text(self) -> str:
        return textwrap.dedent(r"""
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"


#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace tl::ascend {
using namespace Catlass;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Gemm::Block;
using namespace Catlass::Epilogue::Tile;
using namespace AscendC;

using ArchTag = Arch::AtlasA2;
// using LayoutGM = layout::RowMajor;

using LayoutL0A = layout::zZ;
using LayoutL0B = layout::nZ;

template <typename T, typename LayoutGM, typename LayoutL1, uint32_t srcM, uint32_t srcN,
          uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_l1(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor) {
  auto layout = MakeLayoutFromTag(LayoutGM{srcM, srcN});
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::GM>(srcTensor, src_LAYOUT);

  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layoutInL1 = tla::MakeLayout<T, LayoutL1_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutInL1),
                             AscendC::TPosition::A1>(dstTensor, layoutInL1);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, typename LayoutL1, uint32_t srcM, uint32_t srcN,
          uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_l1_to_l0a(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto layout = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(dstM, dstN));

  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0A_ = Catlass::detail::TagToLayout_t<T, LayoutL0A>;
  constexpr auto layoutAInL0 = tla::MakeLayout<T, LayoutL0A_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutAInL0),
                             AscendC::TPosition::A2>(dstTensor, layoutAInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T, typename LayoutL1, uint32_t srcM, uint32_t srcN,
          uint32_t dstM, uint32_t dstN>
CATLASS_DEVICE void copy_l1_to_l0b(LocalTensor<T> dstTensor,
                                   LocalTensor<T> srcTensor) {
  using LayoutL1_ = Catlass::detail::TagToLayout_t<T, LayoutL1>;
  constexpr auto LAYOUT = tla::MakeLayout<T, LayoutL1_>(srcM, srcN);
  auto src_LAYOUT = MakeLayoutTile(LAYOUT, tla::MakeShape(dstM, dstN));
  ;

  auto src = MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                        AscendC::TPosition::A1>(srcTensor, src_LAYOUT);

  using LayoutL0B_ = Catlass::detail::TagToLayout_t<T, LayoutL0B>;
  constexpr auto layoutBInL0 = tla::MakeLayout<T, LayoutL0B_>(dstM, dstN);
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(layoutBInL0),
                             AscendC::TPosition::B2>(dstTensor, layoutBInL0);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src);
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool init>
CATLASS_DEVICE void mma(LocalTensor<T1> A, LocalTensor<T1> B, LocalTensor<T2> C,
                        uint8_t unitFlag = 0) {
  MmadParams mmadParams;
  mmadParams.m = M;
  mmadParams.n = N;
  mmadParams.k = K;
  mmadParams.cmatrixInitVal = init;
  // mmadParams.unitFlag = unitFlag;

  Mmad(C, A, B, mmadParams);

  constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
  if constexpr ((M / C0_NUM_PER_FRACTAL) * (N / C0_NUM_PER_FRACTAL) <
                PIPE_M_BARRIER_THRESHOLD) {
    PipeBarrier<PIPE_M>();
  }
}

template <typename T1, typename T2, typename LayoutGM, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_l0c_to_gm(GlobalTensor<T2> dstTensor,
                                   LocalTensor<T1> srcTensor,
                                   uint8_t unitFlag = 0) {
  auto layoutInL0C = tla::MakeLayoutL0C(srcM, srcN); // TODO (xwh): round up?
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(layoutInL0C),
                             AscendC::TPosition::CO1>(srcTensor, layoutInL0C);
  LayoutGM gm{dstM, dstN};
  auto layout = MakeLayoutFromTag(gm);
  auto dTensor = MakeTensor(dstTensor, layout, Arch::PositionGM{});
  auto layout_ = dTensor.layout();
  auto dst_LAYOUT = MakeLayoutTile(layout_, tla::MakeShape(srcM, srcN));
  auto dst = MakeTensor<decltype(dstTensor), decltype(dst_LAYOUT),
                        AscendC::TPosition::GM>(dstTensor, dst_LAYOUT);

  CopyL0CToGmTla<ArchTag, decltype(src), decltype(dst)> tileCopier;
  tileCopier(dst, src, unitFlag);
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t block_M,
          uint32_t block_N, uint32_t SwizzleOffset = 1,
          uint32_t SwizzleDirection = 0>
CATLASS_DEVICE auto thread_block_swizzle(uint64_t pid) {
  GemmCoord problem_shape = GemmCoord(M, N, K);
  MatrixCoord tile_shape = MatrixCoord(block_M, block_N);

  GemmIdentityBlockSwizzle swizzle =
      GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>(problem_shape,
                                                                tile_shape);

  auto cols = swizzle.loopsMN.column();

  auto coord = swizzle.GetBlockCoord(pid);

  return coord.m() * cols + coord.n();
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_ub(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor) {
  using LayoutSrc = layout::RowMajor;
  using LayoutDst = layout::RowMajor;

  using GType = Gemm::GemmType<T, layout::RowMajor>;

  auto copy_ = CopyGm2Ub<ArchTag, GType>();

  copy_(dstTensor, srcTensor, LayoutDst{dstM, dstN},
        LayoutSrc{dstM, dstN, srcN}); // row, col, ldm
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_ub_to_gm(GlobalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor) {
  using LayoutDst = layout::RowMajor;
  using LayoutSrc = layout::RowMajor;

  using GType = Gemm::GemmType<T, layout::RowMajor>;
  auto copy_ = CopyUb2Gm<ArchTag, GType>();
  copy_(dstTensor, srcTensor, LayoutDst{srcM, srcN, dstN},
        LayoutSrc{srcM, srcN});
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void tile_add(LocalTensor<T> const &ubIn0,
                             LocalTensor<T> const &ubIn1,
                             LocalTensor<T> const &ubOut) {
  AscendC::Add(ubOut, ubIn0, ubIn1, Len);
}

} // namespace tl::ascend
            """)
    
    def _ascend_build_sh_text(self, cpp_name: str, so_name: str) -> str:
        return textwrap.dedent(f"""\
        set -e
        bisheng --cce-soc-version=Ascend910B3 --cce-soc-core-type=CubeCore \
            -O2 -std=c++17 -xcce \
            -mllvm -cce-aicore-stack-size=0x8000 \
            -mllvm -cce-aicore-function-stack-size=0x8000 \
            -mllvm -cce-aicore-record-overflow=true \
            -mllvm -cce-aicore-addr-transform \
            -mllvm -cce-aicore-dcci-insert-for-scalar=false \
            -DL2_CACHE_HINT \
            -I$ASCEND_HOME_PATH/compiler/tikcpp \
            -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw \
            -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl \
            -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface \
            -I$ASCEND_HOME_PATH/include \
            -I$ASCEND_HOME_PATH/include/experiment/msprof \
            -I$TILELANG_PATH/3rdparty/catlass/include \
            -I./ \
            -L$ASCEND_HOME_PATH/lib64 -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl \
            -fPIC --shared {cpp_name} -o {so_name}
        """)
