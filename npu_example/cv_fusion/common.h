#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

#include "catlass/detail/tag_to_layout.hpp"
// #include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
// #include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "lib/matmul_intf.h"

namespace tl::ascend {
using namespace Catlass;
using namespace tla;
using namespace Catlass::Gemm::Tile;
using namespace Catlass::Gemm::Block;
// using namespace Catlass::Epilogue::Tile;
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
  // if constexpr ((M / C0_NUM_PER_FRACTAL) * (N / C0_NUM_PER_FRACTAL) <
  //               PIPE_M_BARRIER_THRESHOLD) {
  //   PipeBarrier<PIPE_M>();
  // }
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


template <typename T, uint32_t M, uint32_t N>
CATLASS_DEVICE void copy_l0c_to_ub(LocalTensor<T> dstTensor, LocalTensor<T> srcTensor) {
  AscendC::FixpipeParamsM300 intriParams;

  intriParams.nSize = N;
  intriParams.mSize = M;

  intriParams.srcStride = M;
  intriParams.dstStride = N;

  intriParams.quantPre = QuantMode_t::NoQuant;
  intriParams.reluEn = false;
  intriParams.unitFlag = 0;

  AscendC::Fixpipe<T, T, AscendC::CFG_ROW_MAJOR>(
      dstTensor, srcTensor, intriParams);
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
  
  // return coord;
  return coord.m() * cols + coord.n();
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_gm_to_ub(LocalTensor<T> dstTensor,
                                  GlobalTensor<T> srcTensor) {
    AscendC::DataCopyExtParams dataCopyParams(
      dstM,
      dstN * sizeof(T),
      (srcN - dstN) * sizeof(T),
      0,
      0
  );
  AscendC::DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T, uint32_t srcM, uint32_t srcN, uint32_t dstM,
          uint32_t dstN>
CATLASS_DEVICE void copy_ub_to_gm(GlobalTensor<T> dstTensor,
                                  LocalTensor<T> srcTensor) {
  using LayoutGM = layout::RowMajor;
  auto layout = MakeLayoutFromTag(LayoutGM{dstM, dstN});
  auto dst_LAYOUT = MakeLayoutTile(layout, tla::MakeShape(srcM, srcN));
  auto dst = tla::MakeTensor<decltype(dstTensor), decltype(dst_LAYOUT),
                             AscendC::TPosition::GM>(dstTensor, dst_LAYOUT);

  auto src_LAYOUT = MakeLayoutFromTag(LayoutGM{srcM, srcN});
  auto src = tla::MakeTensor<decltype(srcTensor), decltype(src_LAYOUT),
                             AscendC::TPosition::VECCALC>(srcTensor, src_LAYOUT);

  TileCopyTla<ArchTag, decltype(src), decltype(dst)> tileCopier; 
  tileCopier(dst, src);
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void tile_add(LocalTensor<T> const &ubIn0,
                             LocalTensor<T> const &ubIn1,
                             LocalTensor<T> const &ubOut) {
  AscendC::Add(ubOut, ubIn0, ubIn1, Len);
}

template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_binary(LocalTensor<T> const &ubIn0,
                                LocalTensor<T> const &ubIn1,
                                LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    AscendC::Add(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 1) {
    AscendC::Sub(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 2) {
    AscendC::Mul(ubOut, ubIn0, ubIn1, Len);
  } else if constexpr (op == 3) {
    AscendC::Div(ubOut, ubIn0, ubIn1, Len);
  }
}


template <typename T, uint32_t Len, uint32_t op>
CATLASS_DEVICE void elementwise_unary(LocalTensor<T> const &ubIn,
                                LocalTensor<T> const &ubOut) {
  // AscendC::Elementwise(ubOut, ubIn0, ubIn1, op, Len);
  if constexpr (op == 0) {
    // TODO: Check layout, Len only has bug.
    AscendC::Exp(ubOut, ubIn, Len);
  } 
}

template <typename src, typename dst, uint32_t Len>
CATLASS_DEVICE void cast(LocalTensor<src> const &ubIn,
                        LocalTensor<dst> const &ubOut) {
  AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, Len);
}

template <typename T, uint32_t Len>
CATLASS_DEVICE void fill(LocalTensor<T> const &ubOut,
                         T value) {
  AscendC::Duplicate(ubOut, value, Len);
}

template <typename T, uint32_t M, uint32_t N, class pattern>
CATLASS_DEVICE void reduce_sum(
  LocalTensor<T> const &srcTensor,
  LocalTensor<T> const &dstTensor,
  LocalTensor<uint8_t> const &sharedTmpBuffer
) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceSum<T, pattern>(dstTensor, srcTensor, sharedTmpBuffer, shape, true);
}

template <typename T, uint32_t M, uint32_t N, class pattern>
CATLASS_DEVICE void reduce_max(
  LocalTensor<T> const &srcTensor,
  LocalTensor<T> const &dstTensor,
  LocalTensor<uint8_t> const &sharedTmpBuffer
) {
  uint32_t shape[] = {M, N};
  AscendC::ReduceMax<T, pattern>(dstTensor, srcTensor, sharedTmpBuffer, shape, true);
}





// NOTE: The following code is not correct.
template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool init, bool is_transpose_A = false, bool is_transpose_B = false>
CATLASS_DEVICE void gemm(LocalTensor<T1> const &A, LocalTensor<T1> const &B, LocalTensor<T2> const &C) {
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
  AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
  for (int kk = 0; kk < 4; ++kk) {
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((kk % 2));
    {
      auto A_L1_ = A_L1.Get<half>();
      auto A_L0_ = A_L0.Get<half>();
      tl::ascend::copy_l1_to_l0a<half, layout::zN, 128, 256, 128, 64>(A_L0_[((kk % 2) * 8192)], A_L1_[((kk * 8192))]);
    }
    {
      auto B_L1_ = B_L1.Get<half>();
      auto B_L0_ = B_L0.Get<half>();
      tl::ascend::copy_l1_to_l0b<half, layout::zN, 256, 256, 64, 256>(B_L0_[((kk % 2) * 16384)], B_L1_[((kk * 1024))]);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((kk % 2));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((kk % 2));
    if ((k == 0) && (kk == 0)) {
      {
        auto A_L0_ = A_L0.Get<half>();
        auto B_L0_ = B_L0.Get<half>();
        auto C_L0_ = C_L0.Get<float>();
        tl::ascend::mma<half, float, 128, 256, 64, true>(A_L0_[(kk * 8192)],B_L0_[(kk * 16384)],C_L0_[0]);
      }
    }
    if ((0 < k) || (0 < kk)) {
      {
        auto A_L0_ = A_L0.Get<half>();
        auto B_L0_ = B_L0.Get<half>();
        auto C_L0_ = C_L0.Get<float>();
        tl::ascend::mma<half, float, 128, 256, 64, false>(A_L0_[((kk % 2) * 8192)],B_L0_[((kk % 2) * 16384)],C_L0_[0]);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((kk % 2));
  }
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((k % 2));
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool init, bool is_transpose_A = false, bool is_transpose_B = false>
CATLASS_DEVICE void gemm_ub(LocalTensor<T1> const &A, LocalTensor<T1> const &B, LocalTensor<T2> const &C) {
  using A_TYPE = MatmulType<TPosition::A1, CubeFormat::NZ, T1, is_transpose_A>;
  using B_TYPE = MatmulType<TPosition::B1, CubeFormat::NZ, T1, is_transpose_B>;
  using C_TYPE = MatmulType<TPosition::VECIN, CubeFormat::ND, T2>;
  AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE> mm; 
  mm.SetTensorA(A, is_transpose_A);
  mm.SetTensorB(B, is_transpose_B);
  mm.SetSingleShape(M, N, K);
  mm.Iterate(init, C);
  mm.End();
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          bool init, bool is_transpose_A = false, bool is_transpose_B = false>
CATLASS_DEVICE void gemm_llg(LocalTensor<T1> const &A, LocalTensor<T1> const &B, GlobalTensor<T2> const &C) {
  using A_TYPE = MatmulType<TPosition::A1, CubeFormat::NZ, T1, is_transpose_A>;
  using B_TYPE = MatmulType<TPosition::B1, CubeFormat::NZ, T1, is_transpose_B>;
  using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, T2>;
  using Bias_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, T2>;

  if constexpr (!init) {
    AscendC::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, Bias_TYPE> mm; 
    mm.SetTensorA(A, is_transpose_A);
    mm.SetTensorB(B, is_transpose_B);
    mm.SetBias(C);
    mm.SetSingleShape(M, N, K);
    mm.Iterate();
    mm.GetTensorC(C); 
    // mm.End();
  }
  else {
    AscendC::MatmulImpl<A_TYPE, B_TYPE, C_TYPE> mm; 
    mm.SetTensorA(A, is_transpose_A);
    mm.SetTensorB(B, is_transpose_B);
    mm.SetSingleShape(M, N, K);
    mm.Iterate();
    mm.GetTensorC(C); 
  }
}

} // namespace tl::ascend