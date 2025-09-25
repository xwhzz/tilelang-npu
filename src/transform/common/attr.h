// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

namespace tvm {
namespace tl {

constexpr const char *MainBlockName = "tilelang_root";

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame =
    "tilelang.is_npu_kernel_frame";

} // namespace tl
} // namespace tvm
