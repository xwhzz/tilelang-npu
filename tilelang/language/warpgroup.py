# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm.script.ir_builder.tir.frame import TIRFrame
from tvm._ffi import register_object
from tilelang import _ffi_api
from .kernel import get_thread_bindings, get_thread_extents, FrameStack
from typing import List, Optional
from tvm.tir import call_extern
import threading


@register_object("tl.WarpSpecializeFrame")
class WarpSpecializeFrame(TIRFrame):
    """
    WarpSpecializeFrame is a custom TIRFrame that manages warp group indices
    and handles the entry and exit of the kernel launch scope.
    """


def WarpSpecialize(*warp_group_idx):
    """Tools to construct a warp group frame.

    Parameters
    ----------
    warp_group_idx : int
        A integer representing warp group index
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    Examples:
        >>> T.ws(0) -> if tx < 128
        >>> T.ws(1) -> if tx >= 128 and tx < 256
        >>> T.ws(0, 1) -> if tx < 128 or (tx >= 128 and tx < 256)
    """
    id_x, id_y, id_z = get_thread_bindings()
    ex_x, ex_y, ex_z = get_thread_extents()
    tid = id_x
    if ex_y > 1:
        tid = id_y * ex_x + tid
    if ex_z > 1:
        tid = id_z * (ex_y * ex_x) + tid

    # only available for nvidia gpus.
    warp_group_size = 128

    warp_group_ids: List[int] = []
    for warp_group_id in warp_group_idx:
        warp_group_ids.append(warp_group_id)

    assert len(warp_group_ids) > 0, "warp_group_idx must be non-empty"

    return _ffi_api.WarpSpecialize(warp_group_ids, tid, warp_group_size)


# Alias for WarpSpecialize for more concise usage
ws = WarpSpecialize

_local = threading.local()


def _get_current_stack() -> FrameStack:
    if not hasattr(_local, "resource_specialize_frame_stack"):
        _local.resource_specialize_frame_stack = FrameStack()
    return _local.resource_specialize_frame_stack


@register_object("tl.ResourceSpecializeFrame")
class ResourceSpecializeFrame(TIRFrame):

    def __enter__(self):
        super().__enter__()
        _get_current_stack().push(self)
        self.name = self.frames[0].attr_key

    def __exit__(self, ptype, value, trace):
        stack = _get_current_stack()
        if stack.top() is self:
            stack.pop()
        super().__exit__(ptype, value, trace)

    @classmethod
    def Current(cls) -> Optional["ResourceSpecializeFrame"]:
        """
        Returns the topmost (current) KernelLaunchFrame from the stack if it exists,
        or None if the stack is empty.
        """
        stack = _get_current_stack()
        return stack.top() if stack else None

    def set(self, other, parity: int = 0):
        if other == "VEC" or other == "CUBE":
            return call_extern("handle", f"AscendC::CrossCoreSetFlag<2, PIPE_{self.name}>", parity)
        return call_extern("handle", f"AscendC::SetFlag<AscendC::HardEvent::{self.name}_{other}>",
                           parity)

    def wait(self, other, parity: int = 0):
        # TODO: need to check the cross core sync semantics.
        if other == "VEC" or other == "CUBE":
            return call_extern("handle", "AscendC::CrossCoreWaitFlag", parity)
        return call_extern("handle", f"AscendC::WaitFlag<AscendC::HardEvent::{other}_{self.name}>",
                           parity)


def ResourceSpecialize(resource: str):
    return _ffi_api.ResourceSpecialize(resource)


rs = ResourceSpecialize


def set_flag(other, parity: int = 0):
    return ResourceSpecializeFrame.Current().set(other, parity)


def wait_flag(other, parity: int = 0):
    return ResourceSpecializeFrame.Current().wait(other, parity)


@register_object("tl.ScopeFrame")
class ScopeFrame(TIRFrame):
    """
    WarpSpecializeFrame is a custom TIRFrame that manages warp group indices
    and handles the entry and exit of the kernel launch scope.
    """


def Scope(name):
    """Tools to construct a warp group frame.

    Parameters
    ----------
    warp_group_idx : int
        A integer representing warp group index
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    Examples:
        >>> T.ws(0) -> if tx < 128
        >>> T.ws(1) -> if tx >= 128 and tx < 256
        >>> T.ws(0, 1) -> if tx < 128 or (tx >= 128 and tx < 256)
    """

    return _ffi_api.Scope(name)
