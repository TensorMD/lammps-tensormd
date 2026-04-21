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

#include <stdexcept>

#include "../../../tensormd_constants.h"
#include "../../../tensormd_strategy.h"
#include "../kernels_internal_fusion.h"
#include "../kernels_internal_utils.h"
#include "kernels_cuda_fusion_helpers.cuh"
#include "kernels_cuda_fusion_types.cuh"
#include "splines.cuh"

namespace TENSORMD::Kernels::Fusion::GPU {

// ============================================================================
// K=32 specialized kernel
// ============================================================================

template <class CFG, typename index_t, bool ReadP>
__global__ void fused_forces_kernel_k32_specialized(
    int A, int B, int C, const typename CFG::Scalar* __restrict__ dEdG,
    const typename CFG::AtlasType* __restrict__ P,
    const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    const int* __restrict__ tau_list,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    int atlas_stride, typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar xdx, typename CFG::Scalar* __restrict__ Fr) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int D = CFG::D;
  constexpr int K = CFG::K;
  constexpr int M = CFG::m_val;
  constexpr int BS = CFG::BatchSize;
  constexpr int Stride = CFG::dM_Stride;
  constexpr bool SaveM = CFG::BackwardSaveM;

  static_assert(K == 32, "This kernel is only for K=32.");

  const int idx_ab = blockIdx.x;
  if (idx_ab / B >= A) return;

  const int k = threadIdx.x;
  const int lane_id = k;
  const index_t idx_ab_t = static_cast<index_t>(idx_ab);
  const index_t C_t = static_cast<index_t>(C);

  extern __shared__ char raw_shared_mem[];
  AtlasType W_reg[D] = {0};

  if constexpr (ReadP) {
    build_W_reg<CFG>(dEdG, P, idx_ab, k, W_reg);
  } else {
    if constexpr (M < 5) {
      calculate_P_direct<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                       raw_shared_mem, params_atlas, grid_size,
                                       atlas_stride, x0, dx, xdx, W_reg);
    } else {
      calculate_P_cached<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                       raw_shared_mem, params_atlas, grid_size,
                                       atlas_stride, x0, dx, xdx, W_reg);
    }
    build_W_reg_inplace<CFG>(dEdG, W_reg, idx_ab, k);
  }

  Scalar* s_R = reinterpret_cast<Scalar*>(raw_shared_mem);
  AtlasType* s_dr = reinterpret_cast<AtlasType*>(s_R + BS);
  AtlasType* s_dMdx = reinterpret_cast<AtlasType*>(s_dr + BS * 3);
  int* s_tau = reinterpret_cast<int*>(s_dMdx + BS * Stride);

  for (int c_base = 0; c_base < C; c_base += BS) {
    const int batch_limit = min(BS, C - c_base);

    if (k < batch_limit) {
      const int c = c_base + k;
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c);
      const int t = tau_list[idx_c];
      s_tau[k] = t;

      if (t >= 0) {
        const Scalar r_val = R[idx_c];
        const AtlasType rijinv = static_cast<AtlasType>(Scalar(1.0) / r_val);
        s_R[k] = r_val;

        const Scalar* dr = drdrx + idx_c * 3;
        const AtlasType x = static_cast<AtlasType>(dr[0]);
        const AtlasType y = static_cast<AtlasType>(dr[1]);
        const AtlasType z = static_cast<AtlasType>(dr[2]);

        s_dr[k * 3 + 0] = x;
        s_dr[k * 3 + 1] = y;
        s_dr[k * 3 + 2] = z;

        if constexpr (SaveM) {
          calc_M_dM_to_shared<CFG>(x, y, z, rijinv, s_dMdx + k * Stride);
        } else {
          calc_dM_to_shared<CFG>(x, y, z, rijinv, s_dMdx + k * Stride);
        }
      }
    }

    __syncwarp();

    for (int b = 0; b < batch_limit; ++b) {
      const int t = s_tau[b];
      const int c = c_base + b;
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c);

      if (t < 0) {
        if (lane_id == 0) {
          Fr[idx_c * 3 + 0] = Scalar(0);
          Fr[idx_c * 3 + 1] = Scalar(0);
          Fr[idx_c * 3 + 2] = Scalar(0);
        }
        continue;
      }

      AtlasType h = 0;
      AtlasType dh = 0;
      Kernels::Spline::GPU::eval_cubic_spline_single<Scalar, AtlasType>(
          s_R[b], s_tau[b], k, K, params_atlas, grid_size, atlas_stride, x0, dx,
          xdx, h, dh);

      AtlasType fa_x = 0;
      AtlasType fa_y = 0;
      AtlasType fa_z = 0;
      AtlasType fr = 0;

      const AtlasType x = s_dr[b * 3 + 0];
      const AtlasType y = s_dr[b * 3 + 1];
      const AtlasType z = s_dr[b * 3 + 2];
      const AtlasType* dMdx_c = s_dMdx + b * Stride;

      if constexpr (SaveM) {
        accumulate_fr_fa_from_M_dM<CFG>(W_reg, dMdx_c, fr, fa_x, fa_y, fa_z);
      } else {
        accumulate_fa_from_dM<CFG>(W_reg, dMdx_c, fa_x, fa_y, fa_z);
        contract_W_M<CFG>(x, y, z, W_reg, fr);
      }
      fa_x *= h;
      fa_y *= h;
      fa_z *= h;
      fr *= dh;

      warp_reduce_sum4(fa_x, fa_y, fa_z, fr);

      if (lane_id == 0) {
        Fr[idx_c * 3 + 0] = static_cast<Scalar>(fr * x + fa_x);
        Fr[idx_c * 3 + 1] = static_cast<Scalar>(fr * y + fa_y);
        Fr[idx_c * 3 + 2] = static_cast<Scalar>(fr * z + fa_z);
      }
    }

    __syncwarp();
  }
}

// ============================================================================
// General K kernel
// ============================================================================

template <class CFG, typename index_t, bool ReadP>
__global__ void fused_forces_kernel(
    int A, int B, int C, const typename CFG::Scalar* __restrict__ dEdG,
    const typename CFG::AtlasType* __restrict__ P,
    const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    const int* __restrict__ tau_list,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    int atlas_stride, typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar xdx, typename CFG::Scalar* __restrict__ Fr) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int D = CFG::D;
  constexpr int K = CFG::K;
  constexpr int M = CFG::m_val;
  constexpr int BS = CFG::BatchSize;
  constexpr int Stride = CFG::dM_Stride;
  constexpr bool SaveM = CFG::BackwardSaveM;

  const int idx_ab = blockIdx.x;
  if (idx_ab / B >= A) return;

  const int k = threadIdx.x;
  const int warp_id = k / 32;
  const int lane_id = k % 32;
  constexpr int num_warps = K / 32;
  const index_t idx_ab_t = static_cast<index_t>(idx_ab);
  const index_t C_t = static_cast<index_t>(C);

  extern __shared__ char raw_shared_mem[];
  Scalar* s_R = reinterpret_cast<Scalar*>(raw_shared_mem);
  AtlasType* s_dr = reinterpret_cast<AtlasType*>(s_R + BS);
  AtlasType* s_dMdx = reinterpret_cast<AtlasType*>(s_dr + BS * 3);
  int* s_tau = reinterpret_cast<int*>(s_dMdx + BS * Stride);

  AtlasType* s_warp_reduce = reinterpret_cast<AtlasType*>(s_tau + BS);
  constexpr int w_stride = BS;
  constexpr int comp_stride = num_warps * BS;

  AtlasType W_reg[D] = {0.0};
  if constexpr (ReadP) {
    build_W_reg<CFG>(dEdG, P, idx_ab, k, W_reg);
  } else {
    if constexpr (M < 5) {
      calculate_P_direct<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                       raw_shared_mem, params_atlas, grid_size,
                                       atlas_stride, x0, dx, xdx, W_reg);
    } else {
      calculate_P_cached<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                       raw_shared_mem, params_atlas, grid_size,
                                       atlas_stride, x0, dx, xdx, W_reg);
    }

    build_W_reg_inplace<CFG>(dEdG, W_reg, idx_ab, k);
  }

  for (int c_base = 0; c_base < C; c_base += BS) {
    const int batch_limit = min(BS, C - c_base);

    if (k < batch_limit) {
      const int c = c_base + k;
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c);
      const int t = tau_list[idx_c];
      s_tau[k] = t;

      if (t >= 0) {
        const Scalar r_val = R[idx_c];
        const AtlasType rijinv = static_cast<AtlasType>(Scalar(1.0) / r_val);
        s_R[k] = r_val;

        const Scalar* dr = drdrx + idx_c * 3;
        const AtlasType x = static_cast<AtlasType>(dr[0]);
        const AtlasType y = static_cast<AtlasType>(dr[1]);
        const AtlasType z = static_cast<AtlasType>(dr[2]);

        s_dr[k * 3 + 0] = x;
        s_dr[k * 3 + 1] = y;
        s_dr[k * 3 + 2] = z;

        if constexpr (SaveM) {
          calc_M_dM_to_shared<CFG>(x, y, z, rijinv, s_dMdx + k * Stride);
        } else {
          calc_dM_to_shared<CFG>(x, y, z, rijinv, s_dMdx + k * Stride);
        }
      }
    }

    __syncthreads();

    for (int b = 0; b < batch_limit; ++b) {
      if (s_tau[b] < 0) continue;

      AtlasType h = 0;
      AtlasType dh = 0;
      Kernels::Spline::GPU::eval_cubic_spline_single<Scalar, AtlasType>(
          s_R[b], s_tau[b], k, K, params_atlas, grid_size, atlas_stride, x0, dx,
          xdx, h, dh);

      AtlasType fa_x = 0;
      AtlasType fa_y = 0;
      AtlasType fa_z = 0;
      AtlasType fr = 0;
      const AtlasType x = s_dr[b * 3 + 0];
      const AtlasType y = s_dr[b * 3 + 1];
      const AtlasType z = s_dr[b * 3 + 2];
      const AtlasType* dMdx_c = s_dMdx + b * Stride;
      if constexpr (SaveM) {
        accumulate_fr_fa_from_M_dM<CFG>(W_reg, dMdx_c, fr, fa_x, fa_y, fa_z);
      } else {
        accumulate_fa_from_dM<CFG>(W_reg, dMdx_c, fa_x, fa_y, fa_z);
        contract_W_M<CFG>(x, y, z, W_reg, fr);
      }
      fa_x *= h;
      fa_y *= h;
      fa_z *= h;
      fr *= dh;

      warp_reduce_sum4(fa_x, fa_y, fa_z, fr);

      if (lane_id == 0) {
        s_warp_reduce[0 * comp_stride + warp_id * w_stride + b] = fa_x;
        s_warp_reduce[1 * comp_stride + warp_id * w_stride + b] = fa_y;
        s_warp_reduce[2 * comp_stride + warp_id * w_stride + b] = fa_z;
        s_warp_reduce[3 * comp_stride + warp_id * w_stride + b] = fr;
      }
    }

    __syncthreads();

    if (k < batch_limit) {
      const int b = k;
      const int c = c_base + b;
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c);

      if (s_tau[b] < 0) {
        Fr[idx_c * 3 + 0] = Scalar(0);
        Fr[idx_c * 3 + 1] = Scalar(0);
        Fr[idx_c * 3 + 2] = Scalar(0);
      } else {
        AtlasType total_fa_x = 0;
        AtlasType total_fa_y = 0;
        AtlasType total_fa_z = 0;
        AtlasType total_fr = 0;

#pragma unroll
        for (int w = 0; w < num_warps; ++w) {
          total_fa_x += s_warp_reduce[0 * comp_stride + w * w_stride + b];
          total_fa_y += s_warp_reduce[1 * comp_stride + w * w_stride + b];
          total_fa_z += s_warp_reduce[2 * comp_stride + w * w_stride + b];
          total_fr += s_warp_reduce[3 * comp_stride + w * w_stride + b];
        }

        Fr[idx_c * 3 + 0] =
            static_cast<Scalar>(total_fr * s_dr[b * 3 + 0] + total_fa_x);
        Fr[idx_c * 3 + 1] =
            static_cast<Scalar>(total_fr * s_dr[b * 3 + 1] + total_fa_y);
        Fr[idx_c * 3 + 2] =
            static_cast<Scalar>(total_fr * s_dr[b * 3 + 2] + total_fa_z);
      }
    }

    __syncthreads();
  }
}

// ============================================================================
// Shared-memory helper
// ============================================================================

template <class CFG>
inline size_t get_fused_forces_smem_bytes() {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int K = CFG::K;
  constexpr int BS = CFG::BatchSize;
  constexpr int Stride = CFG::dM_Stride;
  constexpr int num_warps = K / 32;

  size_t bytes = 0;
  bytes += sizeof(Scalar) * BS;
  bytes += sizeof(AtlasType) * BS * 3;
  bytes += sizeof(AtlasType) * BS * Stride;
  bytes += sizeof(int) * BS;
  if constexpr (K > 32) {
    bytes += sizeof(AtlasType) * 4 * num_warps * BS;
  }
  return bytes;
}

// ============================================================================
// Launch helper
// ============================================================================

template <class CFG, typename index_t, bool ReadP>
inline void launch_fused_forces_kernel(
    int A, int B, int C, const typename CFG::Scalar* dEdG,
    const typename CFG::AtlasType* P, const typename CFG::Scalar* R,
    const typename CFG::Scalar* drdrx, const int* tau_list,
    const typename CFG::AtlasType* params_atlas, int grid_size,
    typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar* Fr) {
  using Scalar = typename CFG::Scalar;

  constexpr int K = CFG::K;
  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr bool UseReducible = CFG::UseReducible;
  constexpr bool SaveM = CFG::BackwardSaveM;

  Utils::GPU::timer_start(Profiling::TimedKernel::fused_forces);

  constexpr int threads = K;
  const int blocks = A * B;
  const int atlas_stride = 4 * K * grid_size;
  const Scalar xdx = Scalar(1.0) / dx;
  const size_t shared_bytes = get_fused_forces_smem_bytes<CFG>();

  if constexpr (K == 32) {
    cudaFuncSetAttribute(
        fused_forces_kernel_k32_specialized<CFG, index_t, ReadP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    fused_forces_kernel_k32_specialized<CFG, index_t, ReadP>
        <<<blocks, threads, shared_bytes>>>(A, B, C, dEdG, P, R, drdrx,
                                            tau_list, params_atlas, grid_size,
                                            atlas_stride, x0, dx, xdx, Fr);
  } else {
    cudaFuncSetAttribute(fused_forces_kernel<CFG, index_t, ReadP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
    fused_forces_kernel<CFG, index_t, ReadP><<<blocks, threads, shared_bytes>>>(
        A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size,
        atlas_stride, x0, dx, xdx, Fr);
  }

  double FLOPs_dM = 0.0;
  double FLOPs_spline = 16.0;
  double FLOPs_W = 0.0;
  double FLOPs_reduction = 0.0;

  if constexpr (UseReducible) {
    if constexpr (SaveM) {
      FLOPs_dM = Reducible::M_dM_d_FLOPs[M];
      FLOPs_reduction = 8.0 * D + 4.0;
    } else {
      FLOPs_dM = Reducible::dM_d_FLOPs[M];
      FLOPs_reduction = 6.0 * D + 3.0 + Reducible::W_M_d_FLOPs[M];
    }
    FLOPs_W = Reducible::W_km_FLOPs[M];
  } else {
    if constexpr (SaveM) {
      FLOPs_dM = Irreducible::M_dM_u_FLOPs[M];
      FLOPs_reduction = 8.0 * D + 4.0;
    } else {
      FLOPs_dM = Irreducible::dM_u_FLOPs[M];
      FLOPs_reduction = 6.0 * D + 3.0 + Irreducible::W_M_u_FLOPs[M];
    }
    FLOPs_W = Irreducible::W_km_FLOPs[M];
  }

  const double n_edges = static_cast<double>(A) * B * C;
  double FLOPs = 0.0;
  FLOPs += static_cast<double>(blocks) * FLOPs_W;
  FLOPs += n_edges * (FLOPs_dM + 6.0 + (FLOPs_spline + FLOPs_reduction) * K);

  double FLOPs_Recal = 0.0;
  if (!ReadP) {
    if constexpr (M < 5) {
      double M_FLOPs = 0.0, h_FLOPs = 10.0;
      if constexpr (CFG::UseReducible) {
        M_FLOPs = Geometric::Reducible::M_d_FLOPs[CFG::m_val];
      } else {
        M_FLOPs = Geometric::Irreducible::M_u_FLOPs[CFG::m_val];
      }
      FLOPs_Recal = 1.0 * A * B * C * (M_FLOPs + h_FLOPs * K + 2.0 * D * K);
    } else {
      double P_FLOPs = 0.0, h_FLOPs = 10.0;
      if constexpr (CFG::UseReducible) {
        P_FLOPs = Geometric::Reducible::P_kd_FLOPs[CFG::m_val];
      } else {
        P_FLOPs = Geometric::Irreducible::P_ku_FLOPs[CFG::m_val];
      }
      FLOPs_Recal = 1.0 * A * B * C * K * (P_FLOPs + h_FLOPs);
    }
  }
  FLOPs += FLOPs_Recal;

  Utils::GPU::timer_stop(Profiling::TimedKernel::fused_forces, FLOPs);
}

// ============================================================================
// m dispatch
// ============================================================================

template <typename Scalar, typename AtlasType, typename index_t, int K,
          int BatchSize, bool UseReducible, bool ReadP>
inline void dispatch_fused_forces_kernel_by_m(DeviceArgs<Scalar>& args,
                                              const AtlasType* P,
                                              const AtlasType* params_atlas) {
  const int A = args.a;
  const int B = args.b;
  const int C = args.c;
  const Scalar* dEdG = args.dEdG;
  const Scalar* R = args.R;
  const Scalar* drdrx = args.drdrx;
  const int* tau_list = args.tijlist;
  const int grid_size = args.interp_size;
  const Scalar x0 = args.interp_r0;
  const Scalar dx = args.interp_dr;
  Scalar* Fr = args.Fr;

  switch (args.m) {
    case 1: {
      using CFG = KernelConfig<Scalar, AtlasType, 1, K, UseReducible, BatchSize,
                               false, false>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    case 2: {
      using CFG = KernelConfig<Scalar, AtlasType, 2, K, UseReducible, BatchSize,
                               false, false>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    case 3: {
      using CFG = KernelConfig<Scalar, AtlasType, 3, K, UseReducible, BatchSize,
                               false, false>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    case 4: {
      // Tests suggest that for m=4, cache dMdrx in SMEM and recompute M in
      // registers is faster than caching M directly
      using CFG = KernelConfig<Scalar, AtlasType, 4, K, UseReducible, BatchSize,
                               false, false>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    case 5: {
      using CFG = KernelConfig<Scalar, AtlasType, 5, K, UseReducible, BatchSize,
                               false, true>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    case 6: {
      using CFG = KernelConfig<Scalar, AtlasType, 6, K, UseReducible, BatchSize,
                               false, true>;
      launch_fused_forces_kernel<CFG, index_t, ReadP>(
          A, B, C, dEdG, P, R, drdrx, tau_list, params_atlas, grid_size, x0, dx,
          Fr);
      break;
    }
    default:
      throw std::runtime_error(
          "Unsupported m value in calc_forces_fused_speed.");
  }
}

// ============================================================================
// P dispatch
// ============================================================================

template <typename Scalar, typename AtlasType, int K, int BatchSize,
          bool UseReducible>
inline void dispatch_fused_forces_kernel_by_P(DeviceArgs<Scalar>& args,
                                              const AtlasType* P,
                                              const AtlasType* params_atlas) {
  const size_t A = static_cast<size_t>(args.a);
  const size_t B = static_cast<size_t>(args.b);
  const size_t C = static_cast<size_t>(args.c);
  if (B != 1) {
    throw std::runtime_error("Batch size B > 1 is not supported when P is used.");
  }
  if (A * C < 1e9) {
    if (P != nullptr) {
      dispatch_fused_forces_kernel_by_m<Scalar, AtlasType, int, K, BatchSize,
                                        UseReducible, true>(args, P,
                                                            params_atlas);
    } else {
      dispatch_fused_forces_kernel_by_m<Scalar, AtlasType, int, K, BatchSize,
                                        UseReducible, false>(args, P,
                                                             params_atlas);
    }
  } else {
    if (P != nullptr) {
      dispatch_fused_forces_kernel_by_m<Scalar, AtlasType, size_t, K, BatchSize,
                                        UseReducible, true>(args, P,
                                                            params_atlas);
    } else {
      dispatch_fused_forces_kernel_by_m<Scalar, AtlasType, size_t, K, BatchSize,
                                        UseReducible, false>(args, P,
                                                             params_atlas);
    }
  }
}

// ============================================================================
// K dispatch
// ============================================================================

template <typename Scalar, typename AtlasType, int BatchSize, bool UseReducible>
inline void dispatch_fused_forces_kernel_by_k(DeviceArgs<Scalar>& args,
                                              const AtlasType* P,
                                              const AtlasType* params_atlas) {
  switch (args.k) {
    case 32:
      dispatch_fused_forces_kernel_by_P<Scalar, AtlasType, 32, BatchSize,
                                        UseReducible>(args, P, params_atlas);
      break;
    case 64:
      dispatch_fused_forces_kernel_by_P<Scalar, AtlasType, 64, BatchSize,
                                        UseReducible>(args, P, params_atlas);
      break;
    case 128:
      dispatch_fused_forces_kernel_by_P<Scalar, AtlasType, 128, BatchSize,
                                        UseReducible>(args, P, params_atlas);
      break;
    case 256:
      dispatch_fused_forces_kernel_by_P<Scalar, AtlasType, 256, BatchSize,
                                        UseReducible>(args, P, params_atlas);
      break;
    default:
      throw std::runtime_error("all_fused_forces requires K = 32/64/128/256");
  }
}

// ============================================================================
// Entry point
// ============================================================================

template <typename Scalar>
inline constexpr int get_BatchSize_for_backward() {
  if constexpr (sizeof(Scalar) == sizeof(float)) {
    return 32;
  } else {
    return 16;
  }
}

template <typename Scalar, bool UseReducible>
void calc_forces_fused_internal(DeviceArgs<Scalar>& args) {
  constexpr int BatchSize = get_BatchSize_for_backward<Scalar>();

  if (args.k != 32 && args.k != 64 && args.k != 128 && args.k != 256) {
    throw std::runtime_error("all_fused_forces requires K = 32/64/128/256");
  }

  if (TENSORMD_USE_FP32_ATLAS) {
    const float* P = reinterpret_cast<const float*>(args.P);
    dispatch_fused_forces_kernel_by_k<Scalar, float, BatchSize, UseReducible>(
        args, P, args.params_atlas_fp32);
  } else {
    const Scalar* P = args.P;
    dispatch_fused_forces_kernel_by_k<Scalar, Scalar, BatchSize, UseReducible>(
        args, P, args.params_atlas);
  }
}

template <typename Scalar>
void calc_forces_fused(DeviceArgs<Scalar>& args) {
  bool use_reducible = args.T_proj_algo < 2;
  if (use_reducible) {
    calc_forces_fused_internal<Scalar, true>(args);
  } else {
    calc_forces_fused_internal<Scalar, false>(args);
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_forces_fused<float>(DeviceArgs<float>& args);
template void calc_forces_fused<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Fusion::GPU
