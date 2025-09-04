#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using namespace AscendC;

CATLASS_GLOBAL
void main_kernel(GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR Output_handle, GM_ADDR workspace_h, GM_ADDR workspace_2_h, GM_ADDR workspace_3_h, GM_ADDR logsum_handle, uint64_t fftsAddr) {
  AscendC::SetSyncBaseAddr(fftsAddr);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> Q;
  Q.SetGlobalBuffer((__gm__ half*)Q_handle);
  AscendC::GlobalTensor<half> K;
  K.SetGlobalBuffer((__gm__ half*)K_handle);
  AscendC::GlobalTensor<half> V;
  V.SetGlobalBuffer((__gm__ half*)V_handle);
  AscendC::GlobalTensor<float> Output;
  Output.SetGlobalBuffer((__gm__ float*)Output_handle);

  AscendC::GlobalTensor<float> workspace;
  workspace.SetGlobalBuffer((__gm__ float*)workspace_h);

  AscendC::GlobalTensor<half> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ half*)workspace_2_h);

  AscendC::GlobalTensor<float> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ float*)workspace_3_h);

  AscendC::GlobalTensor<float> logsum;
  logsum.SetGlobalBuffer((__gm__ float*)logsum_handle);

  auto cid = AscendC::GetBlockIdx();
  AscendC::TBuf<AscendC::TPosition::A1> Q_L1;
  // float or half
  pipe.InitBuffer(Q_L1, 16384 * 4);
  AscendC::TBuf<AscendC::TPosition::A1> K_L1;
  pipe.InitBuffer(K_L1, 16384 * 2);
  AscendC::TBuf<AscendC::TPosition::A1> V_L1;
  pipe.InitBuffer(V_L1, 16384 * 2);

  AscendC::TBuf<AscendC::TPosition::A2> l0a;
  pipe.InitBuffer(l0a, 16384 * 2);
  AscendC::TBuf<AscendC::TPosition::B2> l0b;
  pipe.InitBuffer(l0b, 16384 * 2);

  AscendC::TBuf<AscendC::TPosition::CO1> l0c;
  pipe.InitBuffer(l0c, 128 * 128 * 4);


  AscendC::TBuf<AscendC::TPosition::VECCALC> UB;
  pipe.InitBuffer(UB, 196352);

  auto logsum_ = UB.GetWithOffset<float>(128, 0);
  auto acc_o_ = UB.GetWithOffset<float>(16384, 512);
  auto scores_max_ = UB.GetWithOffset<float>(128, 66048);
  auto scores_max_prev_ = UB.GetWithOffset<float>(128, 66560);
  auto scores_scale_ = UB.GetWithOffset<float>(128, 67072); 
  auto scores_sum_ = UB.GetWithOffset<float>(128, 67584);
  auto acc_s_ = UB.GetWithOffset<float>(16384, 68096);
  auto acc_s_cast_ = UB.GetWithOffset<half>(16384, 68096);
  auto tmp_ = UB.GetWithOffset<uint8_t>(16 * 128 * 4 * 3, 133632);

  pipe.Destroy();

  if (g_coreType == AscendC::AIC) {
    auto Q_L1_ = Q_L1.Get<half>();
    auto l0c_ = l0c.Get<float>();
    auto l0a_ = l0a.Get<half>();
    auto l0b_ = l0b.Get<half>();

    tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(Q_L1_[0], Q[cid * 128 * 128]);
    for (int k = 0; k < 128; ++k) {
      auto K_L1_ = K_L1.Get<half>();
      tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(K_L1_[0], K[k * 128 * 128]);

      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);

      tl::ascend::copy_l1_to_l0a<half, layout::zN, 128, 128, 128, 128>(l0a_[0], Q_L1_[0]);
      tl::ascend::copy_l1_to_l0b<half, layout::nZ, 128, 128, 128, 128>(l0b_[0], K_L1_[0]);

      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);

      tl::ascend::mma<half, float, 128, 128, 128, true>(l0a_[0], l0b_[0], l0c_[0]);

      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

      // tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 16384, 16384>(workspace[cid * 128 * 16384 + k * 128], l0c_[0]);
      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 128, 128>(workspace[cid * 128 * 128], l0c_[0]);

      AscendC::PipeBarrier<PIPE_ALL>();
      // AscendC::PipeBarrier<PIPE_FIX>();
      AscendC::CrossCoreSetFlag<2, PIPE_FIX>(0);
      AscendC::PipeBarrier<PIPE_ALL>();

      // AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(0);
      // AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(0);

      auto V_L1_ = V_L1.Get<half>();
      tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 16384, 128, 128, 128>(V_L1_[0], V[k * 128 * 128]);
      AscendC::CrossCoreWaitFlag(1);
      AscendC::PipeBarrier<PIPE_ALL>();
      auto acc_s_cast_l1_ = K_L1.Get<half>();
      tl::ascend::copy_gm_to_l1<half, layout::RowMajor, layout::zN, 128, 128, 128, 128>(acc_s_cast_l1_[0], workspace_2[cid * 128 * 128]);

      // TODO: do gemm, acc_o

      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);

      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);

      // // AscendC::PipeBarrier<PIPE_MTE1>();

      tl::ascend::copy_l1_to_l0b<half, layout::zN, 128, 128, 128, 128>(l0b_[0], V_L1_[0]);
      tl::ascend::copy_l1_to_l0a<half, layout::zN, 128, 128, 128, 128>(l0a_[0], acc_s_cast_l1_[0]);

      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);

      tl::ascend::mma<half, float, 128, 128, 128, true>(l0a_[0], l0b_[0], l0c_[0]);

      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);

      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 128, 128>(workspace_3[cid * 128 * 128], l0c_[0]);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreWaitFlag(3);
      AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::PipeBarrier<PIPE_ALL>();
  }

  if (g_coreType == AscendC::AIV) {
    // if (cid == 1)
    //   return;
    tl::ascend::fill<float, 128>(logsum_[0], 0);
    tl::ascend::fill<float, 128>(scores_max_[0], -1.0f / 0.0f);
    tl::ascend::fill<float, 16384>(acc_o_[0], 0);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int k = 0; k < 128; ++k) {
        AscendC::DataCopy(scores_max_prev_[0], scores_max_[0], 128);
        AscendC::PipeBarrier<PIPE_ALL>();
        {
            AscendC::CrossCoreWaitFlag(0);
            AscendC::PipeBarrier<PIPE_ALL>();
            tl::ascend::copy_gm_to_ub<float, 128, 128, 128, 128>(acc_s_[0], workspace[cid * 128 * 128]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::PipeBarrier<PIPE_ALL>();
            // AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(acc_s_[0], acc_s_[0], 1.275174e-01f, 16384);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int i = 0; i < 128 / 16; ++i) {
                AscendC::PipeBarrier<PIPE_ALL>();
                tl::ascend::reduce_max<float, 16, 128, Pattern::Reduce::AR>(acc_s_[i * 128 * 16], scores_max_[i * 16], tmp_[0]);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

        // AscendC::Max(scores_max_[0], scores_max_[0], scores_max_prev_[0], 128);

        AscendC::PipeBarrier<PIPE_ALL>();
        {
            AscendC::Sub(scores_scale_[0], scores_max_prev_[0], scores_max_[0], 128);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(scores_scale_[0], scores_scale_[0], 128);
            AscendC::PipeBarrier<PIPE_ALL>();
            // AscendC::Muls(acc_s_[0], acc_s_[0], 1.275174e-01f, 16384);
            for (int i = 0; i < 128; ++i) {
                AscendC::PipeBarrier<PIPE_ALL>();
                auto score_max_ = 0 - scores_max_.GetValue(i);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Adds(acc_s_[i * 128], acc_s_[i * 128], score_max_, 128);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(acc_s_[0], acc_s_[0], 16384);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int i = 0; i < 128 / 16; ++i) {
                AscendC::PipeBarrier<PIPE_ALL>();
                tl::ascend::reduce_sum<float, 16, 128, Pattern::Reduce::AR>(acc_s_[i * 128 * 16], scores_sum_[i * 16], tmp_[0]);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Mul(logsum_[0], logsum_[0], scores_scale_[0], 128);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(logsum_[0], logsum_[0], scores_sum_[0], 128);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (int i = 0; i < 128; ++i) {
              auto score_scale_ = scores_scale_.GetValue(i);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              AscendC::Muls(acc_o_[i * 128], acc_o_[i * 128], score_scale_, 128);
            }

            tl::ascend::cast<float, half, 16384>(acc_s_[0], acc_s_cast_[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

            tl::ascend::copy_ub_to_gm<half, 128, 128, 128, 128>(workspace_2[cid * 128 * 128], acc_s_cast_[0]);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::CrossCoreWaitFlag(2);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);

        tl::ascend::copy_gm_to_ub<float, 128, 128, 128, 128>(acc_s_[0], workspace_3[cid * 128 * 128]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(acc_o_[0], acc_s_[0], acc_o_[0], 16384);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<2, PIPE_V>(3);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < 128; ++i) { 
      AscendC::PipeBarrier<PIPE_ALL>();
      auto logsum__ = 1.0f / logsum_.GetValue(i);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Muls(acc_o_[i * 128], acc_o_[i * 128], logsum__, 128);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    if (cid == 0) {
      tl::ascend::copy_ub_to_gm<float, 128, 128, 16384, 128>(Output[cid * 128 * 128], acc_o_[0]);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

// cann-ops: MatMulImpl

extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, uint8_t* D_handle, uint8_t* workspace, uint8_t* workspace_2, uint8_t* workspace_3, uint8_t* logsum_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel<<<1, nullptr, stream>>>(A_handle, B_handle, C_handle, D_handle, workspace, workspace_2, workspace_3, logsum_handle, fftsAddr);
}