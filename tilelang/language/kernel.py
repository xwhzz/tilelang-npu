# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Union, List, Tuple, Optional
from collections import deque
from tvm import tir
from tvm.tir import Var
from tvm.script.ir_builder.tir.frame import TIRFrame, BlockFrame
from tvm._ffi import register_object
from tilelang import _ffi_api
import threading


class FrameStack:
    """
    A simple stack-like wrapper around a deque that provides
    push, pop, and top methods for convenience.
    """

    def __init__(self):
        self._stack = deque()

    def push(self, item):
        """Pushes an item onto the top of the stack."""
        self._stack.append(item)

    def pop(self):
        """
        Pops and returns the top of the stack, or returns None
        if the stack is empty.
        """
        if self._stack:
            return self._stack.pop()
        raise IndexError(f"{self.__class__.__name__} is empty")

    def top(self):
        """
        Returns the item on the top of the stack without removing it,
        or None if the stack is empty.
        """
        if self._stack:
            return self._stack[-1]
        raise IndexError(f"{self.__class__.__name__} is empty")

    def size(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __len__(self):
        """Returns the number of items in the stack."""
        return len(self._stack)

    def __bool__(self):
        """
        Allows truthy checks on the stack object itself,
        e.g., 'if stack: ...'
        """
        return bool(self._stack)


# Use thread local to store the stack
# This is to avoid the cross-thread interference
_local = threading.local()


def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "kernel_launch_frame_stack"):
        _local.kernel_launch_frame_stack = FrameStack()
    return _local.kernel_launch_frame_stack


@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    """
    KernelLaunchFrame is a custom TIRFrame that manages block/thread indices
    and handles the entry and exit of the kernel launch scope.
    """

    def __enter__(self) -> Union[Var, List[Var]]:
        """
        Enters the KernelLaunchFrame scope and pushes this frame onto the stack.
        Returns one Var if we detect exactly 5 frames (meaning there is a single
        block dimension), or a list of Vars otherwise.
        """
        super().__enter__()
        _get_current_stack().push(self)
        # If we have exactly 5 frames, return the single iter_var.var.
        if len(self.frames) == 5:
            return self.frames[0].iter_var.var

        last_block_frame = self.frames[-1]
        assert isinstance(last_block_frame,
                          BlockFrame), f"Last frame must be a block frame, got {last_block_frame}"

        maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)
        maybe_npu = last_block_frame.annotations.get("tilelang.is_npu_kernel_frame", False)
        if maybe_cpu:
            # CPU kernel frame, return a list of for frame items.
            return [frame.vars[0] for frame in self.frames[0:-1]]
        elif maybe_npu:
            return [self.frames[i].iter_var.var for i in range(2)]
        else:
            # Otherwise, return a list of iter_var.var objects (excluding the last 4 frames).
            # As 4 frames for threadIdx.x, threadIdx.y, threadIdx.z and block frame with attributes
            return [frame.iter_var.var for frame in self.frames[0:-4]]

    def __exit__(self, ptype, value, trace):
        """
        Exits the KernelLaunchFrame scope and pops this frame from the stack,
        but only if it's indeed the topmost frame.
        """
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> Optional["KernelLaunchFrame"]:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def get_block_extent(self, dim: int) -> int:
        """
        Returns the block extent for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        iter_var = self.frames[dim].iter_var
        return int(iter_var.dom.extent)

    def get_block_extents(self) -> List[int]:
        """
        Returns the block extents for all three dimensions.
        """
        return [self.get_block_extent(dim) for dim in range(3)]

    def get_thread_extent(self, dim: int) -> int:
        """
        Returns the thread extent for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        iter_var = self.frames[-4 + dim].iter_var
        return int(iter_var.dom.extent)

    def get_thread_extents(self) -> List[int]:
        """
        Returns the thread extents for all three dimensions.
        """
        return [self.get_thread_extent(dim) for dim in range(3)]

    def get_thread_binding(self, dim: int = 0) -> Var:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return self.frames[-4 + dim].iter_var.var

    def get_thread_bindings(self) -> List[Var]:
        """
        Returns the thread binding for the given dimension.
        dim=0 corresponds to threadIdx.x, dim=1 to threadIdx.y, and dim=2 to threadIdx.z.
        """
        return [frame.iter_var.var for frame in self.frames[-4:-1]]

    def get_num_threads(self) -> int:
        """
        Returns the thread indices from the topmost frame.
        """
        num_threads: int = 1
        for thread_dim in range(3):
            num_threads *= self.get_thread_extent(thread_dim)
        return num_threads

    def get_block_binding(self, dim: int = 0) -> Var:
        """
        Returns the block binding for the given dimension.
        dim=0 corresponds to blockIdx.x, dim=1 to blockIdx.y, and dim=2 to blockIdx.z.
        """
        return self.frames[dim].iter_var.var

    def get_block_bindings(self) -> List[Var]:
        """
        Returns all three block bindings.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def blocks(self) -> List[Var]:
        """
        Returns the block indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[0:-4]]

    @property
    def threads(self) -> List[Var]:
        """
        Returns the thread indices from the topmost frame.
        """
        return [frame.iter_var.var for frame in self.frames[-4:]]

    @property
    def num_threads(self) -> int:
        """
        Returns the total number of threads.
        """
        return self.get_num_threads()


def Kernel(
    *blocks: List[tir.PrimExpr],
    threads: Optional[Union[int, List[int], Tuple]] = None,
    is_cpu: bool = False,
    prelude: Optional[str] = None,
    is_npu: bool = False,
    pipeline: bool = False,
):
    """Tools to quickly construct a GPU kernel launch frame.

    Parameters
    ----------
    blocks : List[int]
        A list of extent, can be 1-3 dimension, representing gridDim.(x|y|z)
    threads : int
        A integer representing blockDim.x
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    is_cpu : bool
        Whether the kernel is running on CPU.
        Thus we will not bind threadIdx.x, threadIdx.y, threadIdx.z.
        and blockIdx.x, blockIdx.y, blockIdx.z.
    prelude : str
        The import c code of the kernel,
        will be injected before the generated kernel code.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    """
    attrs: dict = {}
    if is_npu:
        assert len(blocks) == 1, "NPU kernel must have exactly one block dimension"
        attrs["tilelang.is_npu_kernel_frame"] = True
        return _ffi_api.KernelLaunch(blocks, threads, attrs)

    if not is_cpu and threads is None:
        threads = 128  # default thread number

    if isinstance(threads, int):
        threads = [threads, 1, 1]
    elif isinstance(threads, list):
        threads = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        threads = list(threads) + [1] * (3 - len(threads))
    else:
        assert is_cpu, "threads must be an integer or a list of integers"

    if is_cpu:
        attrs["tilelang.is_cpu_kernel_frame"] = True

    if prelude is not None:
        attrs["pragma_import_c"] = prelude

    return _ffi_api.KernelLaunch(blocks, threads, attrs)


def get_thread_binding(dim: int = 0) -> Var:
    """Returns the thread binding for the given dimension.
    """
    return KernelLaunchFrame.Current().get_thread_binding(dim)


def get_thread_bindings() -> List[Var]:
    """Returns all three thread bindings.
    """
    return KernelLaunchFrame.Current().get_thread_bindings()


def get_block_binding(dim: int = 0) -> Var:
    """Returns the block binding for the given dimension.
    """
    return KernelLaunchFrame.Current().get_block_binding(dim)


def get_block_bindings() -> List[Var]:
    """Returns all three block bindings.
    """
    return KernelLaunchFrame.Current().get_block_bindings()


def get_thread_extent(dim: int = 0) -> int:
    """Returns the thread extent for the given dimension.
    """
    return KernelLaunchFrame.Current().get_thread_extent(dim)


def get_thread_extents() -> List[int]:
    """Returns all three thread extents.
    """
    return KernelLaunchFrame.Current().get_thread_extents()


def get_block_extent(dim: int = 0) -> int:
    """Returns the block extent for the given dimension.
    """
    return KernelLaunchFrame.Current().get_block_extent(dim)


def get_block_extents() -> List[int]:
    """Returns all three block extents.
    """
    return KernelLaunchFrame.Current().get_block_extents()
