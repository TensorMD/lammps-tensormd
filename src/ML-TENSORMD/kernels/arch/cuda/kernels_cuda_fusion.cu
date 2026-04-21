/* ----------------------------------------------------------------------
  MIT License
  Copyright (c) 2026
  [Chen Xin (chen_xin@iapcm.ac.cn), Ouyang Yucheng (ouyangyucheng@live.com)]
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ----------------------------------------------------------------------- */

#include <limits>

#include "../../../tensormd_constants.h"
#include "../kernels_activation.hpp"
#include "../kernels_internal_fusion.h"
#include "../kernels_internal_utils.h"
#include "cuda_math.cuh"
#include "kernels_cuda_fusion_helpers.cuh"
#include "kernels_cuda_fusion_types.cuh"
#include "kernels_cuda_head_helpers.cuh"

namespace TENSORMD::Kernels::Fusion::GPU {

using namespace Geometric;

/* ----------------------------------------------------------------------
  K=32, irreducible, FP32 unified fusion kernel family
  Supported specializations:
    - m = 4, hidden width = 64, NHidden = 2/3/4
    - m = 6, hidden width = 64, NHidden = 2/3/4
  Notes:
    - Hidden 64x64 layers are staged into shared memory as [64][65]
    - Layer-0 stays in global memory for flexibility
    - One warp handles one atom (k=32)
------------------------------------------------------------------------- */

constexpr int kWarpSize = 32;
constexpr int kHiddenWidth = 64;
constexpr int kHiddenStride = 65;  // bank-conflict-avoid padding
constexpr unsigned kFullMask = 0xFFFFFFFFu;

template <int NHidden>
struct MLPParams {
  float* arrays[NHidden + 1];
};

template <typename T>
__device__ __forceinline__ void warp_sum3(T& x, T& y, T& z) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(kFullMask, x, offset);
    y += __shfl_down_sync(kFullMask, y, offset);
    z += __shfl_down_sync(kFullMask, z, offset);
  }
}

template <int NHidden>
struct NetworkLayout {
  static_assert(NHidden >= 2 && NHidden <= 4, "NHidden must be 2/3/4");
  static constexpr int Num64Layers = NHidden - 1;  // hidden-hidden layers
};

template <int NHidden>
__device__ __forceinline__ void preload_hidden_biases_to_shared(
    int tid, float* __restrict__ s_B64, MLPParams<NHidden>& biases) {
  constexpr int Num64Layers = NetworkLayout<NHidden>::Num64Layers;

  for (int i = tid; i < Num64Layers * kHiddenWidth; i += blockDim.x) {
    const int layer = i / kHiddenWidth;
    const int col = i - layer * kHiddenWidth;
    s_B64[i] = biases.arrays[layer + 1][col];
  }
}

template <int NHidden>
__device__ __forceinline__ void preload_hidden_weights_to_shared(
    int tid, float* __restrict__ s_W64, MLPParams<NHidden>& weights) {
  constexpr int Num64Layers = NetworkLayout<NHidden>::Num64Layers;
  constexpr int LogicalElems = kHiddenWidth * kHiddenWidth;
  constexpr int PaddedElems = kHiddenWidth * kHiddenStride;

  for (int layer = 0; layer < Num64Layers; ++layer) {
    const float* __restrict__ src = weights.arrays[layer + 1];
    float* __restrict__ dst = s_W64 + layer * PaddedElems;

    for (int linear = tid; linear < LogicalElems; linear += blockDim.x) {
      const int row = linear >> 6;  // /64
      const int col = linear & 63;  // %64
      dst[row * kHiddenStride + col] = src[linear];
    }

    for (int row = tid; row < kHiddenWidth; row += blockDim.x) {
      dst[row * kHiddenStride + 64] = 0.0f;
    }
  }
}

template <int LayerIdx>
__device__ __forceinline__ const float* hidden_weight_ptr(
    const float* __restrict__ s_W64) {
  return s_W64 + LayerIdx * kHiddenWidth * kHiddenStride;
}

template <int Activation>
__device__ __forceinline__ void forward_hidden_layer_shared(
    int lane, const float* __restrict__ W, const float* __restrict__ B,
    const float A_prev[2], float A_out[2], float dA_out_act[2]) {
  A_out[0] = B[lane];
  A_out[1] = B[lane + 32];

#pragma unroll
  for (int in_idx = 0; in_idx < 64; ++in_idx) {
    const float a_in = (in_idx < 32) ? A_prev[0] : A_prev[1];
    const float val = __shfl_sync(kFullMask, a_in, in_idx & 31);
    const float* __restrict__ row = W + in_idx * kHiddenStride;
    A_out[0] = fmaf(val, row[lane], A_out[0]);
    A_out[1] = fmaf(val, row[lane + 32], A_out[1]);
  }

  compute_activation<float, Activation>(A_out[0], A_out[0], dA_out_act[0]);
  compute_activation<float, Activation>(A_out[1], A_out[1], dA_out_act[1]);
  A_out[0] += A_prev[0];
  A_out[1] += A_prev[1];
}

__device__ __forceinline__ void backward_hidden_layer_shared(
    int lane, const float* __restrict__ W, const float dA_out[2],
    const float dA_out_act[2], float dA_prev[2]) {
  const float dZ0 = dA_out[0] * dA_out_act[0];
  const float dZ1 = dA_out[1] * dA_out_act[1];

  const float* __restrict__ row0 = W + lane * kHiddenStride;
  const float* __restrict__ row1 = W + (lane + 32) * kHiddenStride;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

#pragma unroll
  for (int out_idx = 0; out_idx < 64; ++out_idx) {
    const float dz =
        __shfl_sync(kFullMask, (out_idx < 32) ? dZ0 : dZ1, out_idx & 31);
    acc0 = fmaf(dz, row0[out_idx], acc0);
    acc1 = fmaf(dz, row1[out_idx], acc1);
  }

  dA_prev[0] += acc0 + dA_out[0];  // residual
  dA_prev[1] += acc1 + dA_out[1];
}

template <int MVal, int NHidden, int Activation>
__device__ __forceinline__ void mlp_forward_k32_irred(
    int lane, int atom_type, const float (&G_reg)[MVal],
    const float* __restrict__ s_B64, const float* __restrict__ s_W64,
    MLPParams<NHidden> weights, MLPParams<NHidden> biases, float A1[2],
    float dA1_act[2], float A2[2], float dA2_act[2], float A3[2],
    float dA3_act[2], float A4[2], float dA4_act[2], float& E_i) {
  // static_assert(MVal == 4 || MVal == 6,
  //               "Only m=4 and m=6 are supported in this kernel family");
  constexpr int InputDim = MVal * kWarpSize;

  const float b0_0 = biases.arrays[0][atom_type * 64 + lane];
  const float b0_1 = biases.arrays[0][atom_type * 64 + lane + 32];
  const float out_bias = biases.arrays[NHidden][atom_type];

  A1[0] = b0_0;
  A1[1] = b0_1;

#pragma unroll
  for (int in_idx = 0; in_idx < InputDim; ++in_idx) {
    const int src_lane = in_idx & 31;
    const int reg_idx = in_idx >> 5;  // 0..MVal-1
    const float val = __shfl_sync(kFullMask, G_reg[reg_idx], src_lane);
    A1[0] = fmaf(val, weights.arrays[0][in_idx * 64 + lane], A1[0]);
    A1[1] = fmaf(val, weights.arrays[0][in_idx * 64 + lane + 32], A1[1]);
  }

  compute_activation<float, Activation>(A1[0], A1[0], dA1_act[0]);
  compute_activation<float, Activation>(A1[1], A1[1], dA1_act[1]);

  forward_hidden_layer_shared<Activation>(lane, hidden_weight_ptr<0>(s_W64),
                                          s_B64 + 0 * 64, A1, A2, dA2_act);

  if constexpr (NHidden == 2) {
    float my_E =
        A2[0] * weights.arrays[2][lane] + A2[1] * weights.arrays[2][lane + 32];
    my_E = Kernels::GPU::warpReduceSum(my_E);
    E_i = my_E + out_bias;
    return;
  }

  if constexpr (NHidden >= 3) {
    forward_hidden_layer_shared<Activation>(lane, hidden_weight_ptr<1>(s_W64),
                                            s_B64 + 1 * 64, A2, A3, dA3_act);

    if constexpr (NHidden == 3) {
      float my_E = A3[0] * weights.arrays[3][lane] +
                   A3[1] * weights.arrays[3][lane + 32];
      my_E = Kernels::GPU::warpReduceSum(my_E);
      E_i = my_E + out_bias;
      return;
    }
  }

  if constexpr (NHidden == 4) {
    forward_hidden_layer_shared<Activation>(lane, hidden_weight_ptr<2>(s_W64),
                                            s_B64 + 2 * 64, A3, A4, dA4_act);

    float my_E =
        A4[0] * weights.arrays[4][lane] + A4[1] * weights.arrays[4][lane + 32];
    my_E = Kernels::GPU::warpReduceSum(my_E);
    E_i = my_E + out_bias;
  }
}

template <int MVal, int NHidden>
__device__ __forceinline__ void mlp_backward_k32_irred(
    int lane, const float* __restrict__ s_W64, MLPParams<NHidden> weights,
    MLPParams<NHidden> weights_T, const float dA1_act[2],
    const float dA2_act[2], const float dA3_act[2], const float dA4_act[2],
    float (&dG_reg)[MVal]) {
  // static_assert(MVal == 4 || MVal == 6,
  //               "Only m=4 and m=6 are supported in this kernel family");
  constexpr int InputDim = MVal * kWarpSize;

  float dA1[2] = {0.f, 0.f};
  float dA2[2] = {0.f, 0.f};
  float dA3[2] = {0.f, 0.f};
  float dA4[2] = {0.f, 0.f};

  if constexpr (NHidden == 2) {
    dA2[0] = weights.arrays[2][lane];
    dA2[1] = weights.arrays[2][lane + 32];
  } else if constexpr (NHidden == 3) {
    dA3[0] = weights.arrays[3][lane];
    dA3[1] = weights.arrays[3][lane + 32];
  } else {
    dA4[0] = weights.arrays[4][lane];
    dA4[1] = weights.arrays[4][lane + 32];
  }

  if constexpr (NHidden == 4) {
    backward_hidden_layer_shared(lane, hidden_weight_ptr<2>(s_W64), dA4,
                                 dA4_act, dA3);
  }

  if constexpr (NHidden >= 3) {
    backward_hidden_layer_shared(lane, hidden_weight_ptr<1>(s_W64), dA3,
                                 dA3_act, dA2);
  }

  backward_hidden_layer_shared(lane, hidden_weight_ptr<0>(s_W64), dA2, dA2_act,
                               dA1);

  const float dZ1_0 = dA1[0] * dA1_act[0];
  const float dZ1_1 = dA1[1] * dA1_act[1];

#pragma unroll
  for (int i = 0; i < MVal; ++i) dG_reg[i] = 0.f;

#pragma unroll
  for (int out_idx = 0; out_idx < 64; ++out_idx) {
    const float dz =
        __shfl_sync(kFullMask, (out_idx < 32) ? dZ1_0 : dZ1_1, out_idx & 31);

#pragma unroll
    for (int p = 0; p < MVal; ++p) {
      const int in_idx = p * kWarpSize + lane;
      dG_reg[p] =
          fmaf(dz, weights_T.arrays[0][out_idx * InputDim + in_idx], dG_reg[p]);
    }
  }
}

template <class CFG, typename index_t, int NHidden, int Activation,
          int WarpsPerBlock>
__global__
__launch_bounds__(kWarpSize* WarpsPerBlock) void unified_fusion_kernel_k32_irred_fp32(
    int A, int C, const float* __restrict__ R, float* drdrx,
    const int* __restrict__ tau_list, const int* __restrict__ atom_types,
    MLPParams<NHidden> weights, MLPParams<NHidden> weights_T,
    MLPParams<NHidden> biases, const float* __restrict__ params_atlas,
    int grid_size, int atlas_stride, float x0, float dx, float xdx, float* Fr,
    float* __restrict__ etotal, float* __restrict__ eatoms) {
  static_assert(std::is_same_v<typename CFG::Scalar, float>, "FP32 only");
  static_assert(std::is_same_v<typename CFG::AtlasType, float>,
                "FP32 atlas only");
  static_assert(CFG::K == 32, "K must be 32");
  // static_assert(CFG::UseReducible == false, "irreducible only");
  // static_assert(CFG::m_val == 4 || CFG::m_val == 6,
  //               "This kernel supports only m=4 and m=6");

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int idx = blockIdx.x * WarpsPerBlock + warp;
  const bool active = (idx < A);
  const index_t C_t = static_cast<index_t>(C);

  constexpr int Num64Layers = NetworkLayout<NHidden>::Num64Layers;
  constexpr size_t HiddenBiasElems = Num64Layers * 64;
  constexpr size_t HiddenWeightElems = Num64Layers * 64 * 65;

  constexpr size_t ForwardGeomBytes =
      sizeof(float) * CFG::BatchSize * CFG::M_Stride +  // s_M
      sizeof(float) * CFG::BatchSize +                  // s_R
      sizeof(int) * CFG::BatchSize;                     // s_tau

  constexpr size_t BackwardGeomBytes =
      sizeof(float) * CFG::BatchSize +                   // s_R
      sizeof(float) * CFG::BatchSize * 3 +               // s_dr
      sizeof(float) * CFG::BatchSize * CFG::dM_Stride +  // s_dMdx
      sizeof(int) * CFG::BatchSize;                      // s_tau

  constexpr size_t PerWarpGeomBytes = (ForwardGeomBytes > BackwardGeomBytes)
                                          ? ForwardGeomBytes
                                          : BackwardGeomBytes;

  extern __shared__ char raw_shared_mem[];
  float* s_B64 = reinterpret_cast<float*>(raw_shared_mem);
  float* s_W64 = s_B64 + HiddenBiasElems;
  char* geom_base = reinterpret_cast<char*>(s_W64 + HiddenWeightElems);

  preload_hidden_biases_to_shared<NHidden>(tid, s_B64, biases);
  preload_hidden_weights_to_shared<NHidden>(tid, s_W64, weights);
  __syncthreads();

  if (!active) return;

  char* warp_geom = geom_base + warp * PerWarpGeomBytes;
  const int atom_type = atom_types[idx];

  // --------------------------------------------------------------------------
  // Phase 1: Forward descriptor, cached-M version (per warp scratch)
  // --------------------------------------------------------------------------
  float P_reg[CFG::D] = {0.f};
  calculate_P_cached<CFG, index_t>(C, idx, lane, tau_list, R, drdrx, warp_geom,
                                   params_atlas, grid_size, atlas_stride, x0,
                                   dx, xdx, P_reg);

  float G_reg[CFG::m_val] = {0.f};
  if constexpr (CFG::UseReducible) {
    Geometric::Reducible::contract_G_mkba_otf<float, CFG::m_val, false>(P_reg, G_reg);
  } else {
    Geometric::Irreducible::contract_G_mkba_otf<float, CFG::m_val>(P_reg, G_reg);
  }

  // --------------------------------------------------------------------------
  // Phase 2: Forward MLP
  // --------------------------------------------------------------------------
  float A1[2] = {0.f, 0.f}, dA1_act[2] = {1.f, 1.f};
  float A2[2] = {0.f, 0.f}, dA2_act[2] = {1.f, 1.f};
  float A3[2] = {0.f, 0.f}, dA3_act[2] = {1.f, 1.f};
  float A4[2] = {0.f, 0.f}, dA4_act[2] = {1.f, 1.f};
  float E_i = 0.f;

  mlp_forward_k32_irred<CFG::m_val, NHidden, Activation>(
      lane, atom_type, G_reg, s_B64, s_W64, weights, biases, A1, dA1_act, A2,
      dA2_act, A3, dA3_act, A4, dA4_act, E_i);

  if (lane == 0) {
    atomicAdd(etotal, E_i);
    if (eatoms) eatoms[idx] = E_i;
  }

  // --------------------------------------------------------------------------
  // Phase 3: Backward MLP -> dG (all in registers for G/dG)
  // --------------------------------------------------------------------------
  float dG_reg[CFG::m_val];
  mlp_backward_k32_irred<CFG::m_val, NHidden>(lane, s_W64, weights, weights_T,
                                              dA1_act, dA2_act, dA3_act,
                                              dA4_act, dG_reg);

  if constexpr (CFG::UseReducible) {
    Geometric::Reducible::contract_W_dkba_otf<float, CFG::m_val, false>(
        dG_reg, P_reg, P_reg);
  } else {
    Geometric::Irreducible::contract_W_dkba_otf<float, CFG::m_val>(
        dG_reg, P_reg, P_reg);
  }

  // --------------------------------------------------------------------------
  // Phase 4: Backward force projection
  // Recompute neighbor geometry for correctness-first version.
  // --------------------------------------------------------------------------
  constexpr int BS = CFG::BatchSize;
  constexpr int Stride = CFG::dM_Stride;

  float* s_R = reinterpret_cast<float*>(warp_geom);
  float* s_dr = s_R + BS;
  float* s_dMdx = s_dr + BS * 3;
  int* s_tau = reinterpret_cast<int*>(s_dMdx + BS * Stride);

  for (int c_base = 0; c_base < C; c_base += BS) {
    const int batch_limit = min(BS, C - c_base);

    if (lane < batch_limit) {
      const index_t c = static_cast<index_t>(c_base + lane);
      const index_t idx_c = static_cast<index_t>(idx) * C_t + c;
      const int t = tau_list[idx_c];
      s_tau[lane] = t;

      if (t >= 0) {
        const float r_val = R[idx_c];
        const float rijinv = 1.0f / r_val;
        s_R[lane] = r_val;

        const float* dr = drdrx + idx_c * 3;
        const float x = dr[0], y = dr[1], z = dr[2];
        s_dr[lane * 3 + 0] = x;
        s_dr[lane * 3 + 1] = y;
        s_dr[lane * 3 + 2] = z;

        calc_dM_to_shared<CFG>(x, y, z, rijinv, s_dMdx + lane * Stride);
      }
    }

    __syncwarp();

    for (int b = 0; b < batch_limit; ++b) {
      const int t = s_tau[b];
      const index_t c = static_cast<index_t>(c_base + b);
      const index_t idx_c = static_cast<index_t>(idx) * C_t + c;

      if (t < 0) {
        if (lane == 0) {
          Fr[idx_c * 3 + 0] = 0.f;
          Fr[idx_c * 3 + 1] = 0.f;
          Fr[idx_c * 3 + 2] = 0.f;
        }
        continue;
      }

      float h = 0.f, dh = 0.f;
      Kernels::Spline::GPU::eval_cubic_spline_single<float, float>(
          s_R[b], t, lane, 32, params_atlas, grid_size, atlas_stride, x0, dx,
          xdx, h, dh);

      float fa_x = 0.f, fa_y = 0.f, fa_z = 0.f, fr = 0.f;
      const float* dMdx_c = s_dMdx + b * Stride;
      accumulate_fa_from_dM<CFG>(P_reg, dMdx_c, fa_x, fa_y, fa_z);

      const float x = s_dr[b * 3 + 0];
      const float y = s_dr[b * 3 + 1];
      const float z = s_dr[b * 3 + 2];
      contract_W_M<CFG>(x, y, z, P_reg, fr);

      fa_x *= h;
      fa_y *= h;
      fa_z *= h;
      fr *= dh;

      warp_sum3(fa_x, fa_y, fa_z);
      fr = Kernels::GPU::warpReduceSum(fr);

      if (lane == 0) {
        Fr[idx_c * 3 + 0] = fr * x + fa_x;
        Fr[idx_c * 3 + 1] = fr * y + fa_y;
        Fr[idx_c * 3 + 2] = fr * z + fa_z;
      }
    }

    __syncwarp();
  }
}

template <int MVal, int NHidden, int Activation, int BatchSize,
          int WarpsPerBlock, bool UseReducible>
void launch_unified_fusion_kernel_k32_irred_fp32(DeviceArgs<float>& args,
                                                 HostArgs<float>& host_args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::TheOneKernel);

  using CFG =
      KernelConfig<float, float, MVal, 32, UseReducible, BatchSize, true, false>;

  const int A = args.a;
  const int C = args.c;
  const int D = CFG::D;
  const int K = CFG::K;
  const int blocks = (A + WarpsPerBlock - 1) / WarpsPerBlock;
  const int threads = kWarpSize * WarpsPerBlock;

  const int atlas_stride = args.interp_size * 4 * 32;
  const float x0 = args.interp_r0;
  const float dx = args.interp_dr;
  const float xdx = 1.0f / dx;

  constexpr int Num64Layers = NetworkLayout<NHidden>::Num64Layers;
  constexpr size_t HiddenBiasBytes = sizeof(float) * Num64Layers * 64;
  constexpr size_t HiddenWeightBytes = sizeof(float) * Num64Layers * 64 * 65;

  constexpr size_t ForwardGeomBytes =
      sizeof(float) * CFG::BatchSize * CFG::M_Stride +
      sizeof(float) * CFG::BatchSize + sizeof(int) * CFG::BatchSize;

  constexpr size_t BackwardGeomBytes =
      sizeof(float) * CFG::BatchSize + sizeof(float) * CFG::BatchSize * 3 +
      sizeof(float) * CFG::BatchSize * CFG::dM_Stride +
      sizeof(int) * CFG::BatchSize;

  constexpr size_t PerWarpGeomBytes = (ForwardGeomBytes > BackwardGeomBytes)
                                          ? ForwardGeomBytes
                                          : BackwardGeomBytes;

  const size_t shared_bytes =
      HiddenBiasBytes + HiddenWeightBytes + WarpsPerBlock * PerWarpGeomBytes;

  MLPParams<NHidden> weights, biases, weights_T;
  for (int i = 0; i < NHidden + 1; ++i) {
    weights.arrays[i] = args.weights[i];
    if (i < NHidden) weights_T.arrays[i] = args.weights_T[i];
    biases.arrays[i] = args.biases[i];
  }

  cudaMemset(args.etotal, 0, sizeof(float));

  const size_t A_t = static_cast<size_t>(A);
  const size_t C_t = static_cast<size_t>(C);
  constexpr size_t kMaxIntFlatIndex =
      static_cast<size_t>(std::numeric_limits<int>::max());
  const bool needs_wide_index = A_t * C_t > (kMaxIntFlatIndex / 3) -1;
  if (needs_wide_index) {
    cudaFuncSetAttribute(
        unified_fusion_kernel_k32_irred_fp32<CFG, size_t, NHidden, Activation,
                                             WarpsPerBlock>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    unified_fusion_kernel_k32_irred_fp32<CFG, size_t, NHidden, Activation,
                                         WarpsPerBlock>
        <<<blocks, threads, shared_bytes>>>(
            A, C, args.R, args.drdrx, args.tijlist, args.tlist, weights,
            weights_T, biases, args.params_atlas, args.interp_size,
            atlas_stride, x0, dx, xdx, args.Fr, args.etotal, args.eatoms);
  } else {
    cudaFuncSetAttribute(
        unified_fusion_kernel_k32_irred_fp32<CFG, int, NHidden, Activation,
                                             WarpsPerBlock>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    unified_fusion_kernel_k32_irred_fp32<CFG, int, NHidden, Activation,
                                         WarpsPerBlock>
        <<<blocks, threads, shared_bytes>>>(
            A, C, args.R, args.drdrx, args.tijlist, args.tlist, weights,
            weights_T, biases, args.params_atlas, args.interp_size,
            atlas_stride, x0, dx, xdx, args.Fr, args.etotal, args.eatoms);
  }

  // Use CUB::sum to ensure the total energy is numerically stable and
  // reproducible
  Head::GPU::compute_total_energy<float>(args.eatoms, args.etotal, A);
  float h_total = 0.0;
  cudaMemcpy(&h_total, args.etotal, sizeof(float), cudaMemcpyDeviceToHost);
  host_args.etotal[0] = static_cast<double>(h_total);

  double n = 1.0 * A;
  double FLOPs = 0.0;
  constexpr int InputDim = MVal * kWarpSize;
  constexpr int Hidden64Layers = NetworkLayout<NHidden>::Num64Layers;
  {
    double h_FLOPs = 10.0;
    double M_FLOPs = 0.0, G_FLOPs = 0.0;
    if constexpr (UseReducible) {
      M_FLOPs = Reducible::M_d_FLOPs[CFG::m_val];
      G_FLOPs = Reducible::G_km_FLOPs[CFG::m_val];
    } else {
      M_FLOPs = Irreducible::M_u_FLOPs[CFG::m_val];
      G_FLOPs = Irreducible::G_km_FLOPs[CFG::m_val];
    }
    FLOPs += n * C * (M_FLOPs + h_FLOPs * K + 2.0 * D * K);
    FLOPs += n * K * G_FLOPs;
  }

  {
    double FLOPs_act = activation_FLOPs[Activation];
    FLOPs += n * InputDim * 64 * 2;             // x * W0
    FLOPs += FLOPs_act * n * 64 * NHidden;      // all activations
    FLOPs += n * 64 * 64 * 2 * Hidden64Layers;  // hidden GEMV
    FLOPs += n * 64 * NHidden;                  // input bias + residual adds
    FLOPs += n * 64 * 2;                        // output layer dot + bias
    FLOPs += n * 64 * 64 * 2 * Hidden64Layers;  // hidden backward GEMV
    FLOPs += n * 64 * NHidden;                  // chain-rule muls
    FLOPs += n * 64 * InputDim * 2;             // W0^T * dA1
  }

  {
    double FLOPs_spline = 16.0;
    double FLOPs_dM = 0.0;
    double FLOPs_reduction = 0.0;
    double FLOPs_W = 0.0;
    if constexpr (UseReducible) {
      FLOPs_dM = Reducible::dM_d_FLOPs[CFG::m_val];
      FLOPs_reduction = 6.0 * D + 3.0 + Reducible::W_M_d_FLOPs[CFG::m_val];
      FLOPs_W = Reducible::W_km_FLOPs[CFG::m_val];
    } else {
      FLOPs_dM = Irreducible::dM_u_FLOPs[CFG::m_val];
      FLOPs_reduction = 6.0 * D + 3.0 + Irreducible::W_M_u_FLOPs[CFG::m_val];
      FLOPs_W = Irreducible::W_km_FLOPs[CFG::m_val];
    }
    FLOPs += n * FLOPs_W;
    FLOPs += n * C * (FLOPs_dM + 6.0 + (FLOPs_spline + FLOPs_reduction) * K);
  }

  Utils::GPU::timer_stop(Profiling::TimedKernel::TheOneKernel, FLOPs);
}

template <int Mval, int NHidden, bool UseReducible>
void dispatch_unified_fusion_kernel_k32_fp32(DeviceArgs<float>& args,
                                             HostArgs<float>& host_args) {
  const int actype = args.actype;
  constexpr int BS = UseReducible ? 8 : 16;
  switch (actype) {
    case 1:
      launch_unified_fusion_kernel_k32_irred_fp32<Mval, NHidden, 1, BS, 32,
                                                  UseReducible>(args,
                                                                host_args);
      break;
    case 4:
      launch_unified_fusion_kernel_k32_irred_fp32<Mval, NHidden, 4, BS, 32,
                                                  UseReducible>(args,
                                                                host_args);
      break;
    default:
      throw std::runtime_error("unknown activation type.");
  }
}

template <int NHidden>
void dispatch_unified_fusion_kernel_k32_m6_fp32(DeviceArgs<float>& args,
                                                HostArgs<float>& host_args) {
  const int actype = args.actype;
  switch (actype) {
    case 1:
      launch_unified_fusion_kernel_k32_irred_fp32<6, NHidden, 1, 4, 16, false>(
          args, host_args);
      break;
    case 4:
      launch_unified_fusion_kernel_k32_irred_fp32<6, NHidden, 4, 4, 16, false>(
          args, host_args);
      break;
    default:
      throw std::runtime_error("unknown activation type.");
  }
}

template <typename Scalar>
void unified_fusion_kernel(DeviceArgs<Scalar>& args,
                           HostArgs<Scalar>& host_args) {
  if constexpr (std::is_same_v<Scalar, float>) {
    if (args.k == 32 && args.n_layers <= 5 && args.n_layers >= 3) {
      if (args.m == 2) {
        // if (args.n_layers == 3) {
        //   dispatch_unified_fusion_kernel_k32_fp32<2, 2, false>(args, host_args);
        // } else if (args.n_layers == 4) {
        //   dispatch_unified_fusion_kernel_k32_fp32<2, 3, false>(args, host_args);
        // } else {
        //   dispatch_unified_fusion_kernel_k32_fp32<2, 4, false>(args, host_args);
        // }
      } else if (args.m == 3) {
        // if (args.n_layers == 3) {
        //   dispatch_unified_fusion_kernel_k32_fp32<3, 2, false>(args, host_args);
        // } else if (args.n_layers == 4) {
        //   dispatch_unified_fusion_kernel_k32_fp32<3, 3, false>(args, host_args);
        // } else {
        //   dispatch_unified_fusion_kernel_k32_fp32<3, 4, false>(args, host_args);
        // }
      } else if (args.m == 4) {
        if (args.n_layers == 3) {
          if (args.T_proj_algo < 2) {
            dispatch_unified_fusion_kernel_k32_fp32<4, 2, true>(args, host_args);
          } else {
            dispatch_unified_fusion_kernel_k32_fp32<4, 2, false>(args, host_args);
          }
        } else if (args.n_layers == 4) {
          if (args.T_proj_algo < 2) {
            dispatch_unified_fusion_kernel_k32_fp32<4, 3, true>(args, host_args);
          } else {
            dispatch_unified_fusion_kernel_k32_fp32<4, 3, false>(args, host_args);
          }
        } else {
          if (args.T_proj_algo < 2) {
            dispatch_unified_fusion_kernel_k32_fp32<4, 4, true>(args, host_args);
          } else {
            dispatch_unified_fusion_kernel_k32_fp32<4, 4, false>(args, host_args);
          }
        }
      } else if (args.m == 6) {
        if (args.n_layers == 3) {
          dispatch_unified_fusion_kernel_k32_m6_fp32<2>(args, host_args);
        } else if (args.n_layers == 4) {
          dispatch_unified_fusion_kernel_k32_m6_fp32<3>(args, host_args);
        } else {
          dispatch_unified_fusion_kernel_k32_m6_fp32<4>(args, host_args);
        }
      } else {
        throw std::runtime_error(
            "unified_fusion_kernel currently supports only K=32 with m=4 or "
            "m=6 and 3 <= n_layers<=5.");
      }
    } else {
      throw std::runtime_error(
          "unified_fusion_kernel currently supports only K=32 with m=4 or "
          "m=6 and 3 <= n_layers<=5.");
    }
  } else {
    throw std::runtime_error(
        "unified_fusion_kernel currently supports only float.");
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void unified_fusion_kernel<float>(DeviceArgs<float>& args,
                                           HostArgs<float>& host_args);
template void unified_fusion_kernel<double>(DeviceArgs<double>& args,
                                            HostArgs<double>& host_args);

}  // namespace TENSORMD::Kernels::Fusion::GPU