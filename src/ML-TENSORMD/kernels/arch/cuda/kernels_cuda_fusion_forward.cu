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
#include <type_traits>

#include "../../../tensormd_constants.h"
#include "../../../tensormd_strategy.h"
#include "../kernels_internal_fusion.h"
#include "../kernels_internal_utils.h"
#include "kernels_cuda_fusion_helpers.cuh"
#include "kernels_cuda_fusion_types.cuh"
#include "splines.cuh"

namespace TENSORMD::Kernels::Fusion::GPU {

// ============================================================================
// Direct-moment kernel
// ============================================================================

template <class CFG, typename index_t>
__global__ void calc_P_G_fused_kernel(
    int A, int B, int C, const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    const int* __restrict__ tau_list,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    int atlas_stride, typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar xdx, bool write_p,
    typename CFG::Scalar* __restrict__ P,
    typename CFG::Scalar* __restrict__ G) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr int K = CFG::K;
  constexpr bool UseReducible = CFG::UseReducible;
  constexpr bool SaveM = CFG::ForwardSaveM;

  const int idx_ab = blockIdx.x;
  const int a = idx_ab / B;
  if (a >= A) return;

  const int k = threadIdx.x;
  extern __shared__ char shared_mem[];
  AtlasType P_reg[D] = {AtlasType(0)};

  if constexpr (SaveM) {
    calculate_P_cached<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                     shared_mem, params_atlas, grid_size,
                                     atlas_stride, x0, dx, xdx, P_reg);
  } else {
    calculate_P_direct<CFG, index_t>(C, idx_ab, k, tau_list, R, drdrx,
                                     shared_mem, params_atlas, grid_size,
                                     atlas_stride, x0, dx, xdx, P_reg);
  }

  AtlasType* P_cast = reinterpret_cast<AtlasType*>(P);
  store_P_if_needed<AtlasType, D>(write_p, P_cast, idx_ab, k, P_reg, K);

  Scalar* G_km = G + idx_ab * M * K;
  convert_P_to_G<Scalar, AtlasType, M, D, UseReducible>(P_reg, G_km, k, K);
}

// ============================================================================
// Launch helpers
// ============================================================================

template <class CFG, typename index_t>
void launch_calc_P_G_fused(
    int A, int B, int C, const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    const int* __restrict__ tijlist,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    typename CFG::Scalar x0, typename CFG::Scalar dx, bool write_p,
    typename CFG::Scalar* __restrict__ P,
    typename CFG::Scalar* __restrict__ G) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int K = CFG::K;
  constexpr int BS = CFG::BatchSize;
  constexpr bool SaveM = CFG::ForwardSaveM;
  constexpr int Strided = CFG::M_Stride;
  const int blocks = A * B;
  const int threads = K;
  const int atlas_stride = grid_size * 4 * K;
  const Scalar xdx = Scalar(1.0) / dx;

  size_t shared_bytes;
  if constexpr (SaveM) {
    shared_bytes =
        BS * (sizeof(AtlasType) * Strided + sizeof(Scalar) + sizeof(int));
    cudaFuncSetAttribute(calc_P_G_fused_kernel<CFG, index_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
  } else {
    shared_bytes = BS * (sizeof(AtlasType) * 3 + sizeof(Scalar) + sizeof(int));
  }

  Utils::GPU::timer_start(Profiling::TimedKernel::calc_P_fused);

  calc_P_G_fused_kernel<CFG, index_t><<<blocks, threads, shared_bytes>>>(
      A, B, C, R, drdrx, tijlist, params_atlas, grid_size, atlas_stride, x0, dx,
      xdx, write_p, P, G);

  double FLOPs = 0.0;
  if constexpr (SaveM) {
    double M_FLOPs = 0.0, h_FLOPs = 10.0;
    if constexpr (CFG::UseReducible) {
      M_FLOPs = Geometric::Reducible::M_d_FLOPs[CFG::m_val];
    } else {
      M_FLOPs = Geometric::Irreducible::M_u_FLOPs[CFG::m_val];
    }
    FLOPs = 1.0 * A * B * C * (M_FLOPs + h_FLOPs * K + 2.0 * CFG::D * K);
  } else {
    double P_FLOPs = 0.0, h_FLOPs = 10.0;
    if constexpr (CFG::UseReducible) {
      P_FLOPs = Geometric::Reducible::P_kd_FLOPs[CFG::m_val];
    } else {
      P_FLOPs = Geometric::Irreducible::P_ku_FLOPs[CFG::m_val];
    }
    FLOPs = 1.0 * A * B * C * K * (P_FLOPs + h_FLOPs);
  }

  if constexpr (CFG::UseReducible) {
    FLOPs += 1.0 * A * B * K * Geometric::Reducible::G_km_FLOPs[CFG::m_val];
  } else {
    FLOPs += 1.0 * A * B * K * Geometric::Irreducible::G_km_FLOPs[CFG::m_val];
  }

  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_P_fused, FLOPs);
}

// ============================================================================
// m-dispatch
// ============================================================================

template <typename Scalar, typename AtlasType, typename index_t, int K,
          bool UseReducible>
void launch_by_m_value(DeviceArgs<Scalar>& args, const AtlasType* params_atlas,
                       bool write_p) {
  const int A = args.a;
  const int B = args.b;
  const int C = args.c;
  const Scalar* __restrict__ R = args.R;
  const Scalar* __restrict__ drdrx = args.drdrx;
  const int* __restrict__ tijlist = args.tijlist;
  const Scalar dx = args.interp_dr;
  const Scalar x0 = args.interp_r0;
  const int grid_size = args.interp_size;
  Scalar* __restrict__ P = args.P;
  Scalar* __restrict__ G = args.G;
  const int m_val = args.m;
  constexpr int BS = 32;

  switch (m_val) {
    case 1: {
      using CFG =
          KernelConfig<Scalar, AtlasType, 1, K, UseReducible, BS, false>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    case 2: {
      using CFG =
          KernelConfig<Scalar, AtlasType, 2, K, UseReducible, BS, false>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    case 3: {
      using CFG =
          KernelConfig<Scalar, AtlasType, 3, K, UseReducible, BS, false>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    case 4: {
      // Tests suggest using the cached-M version for m=4 can be slightly
      // faster on A100.
      using CFG = KernelConfig<Scalar, AtlasType, 4, K, UseReducible, BS, true>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    case 5: {
      using CFG = KernelConfig<Scalar, AtlasType, 5, K, UseReducible, BS, true>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    case 6: {
      using CFG = KernelConfig<Scalar, AtlasType, 6, K, UseReducible, BS, true>;
      launch_calc_P_G_fused<CFG, index_t>(A, B, C, R, drdrx, tijlist,
                                          params_atlas, grid_size, x0, dx,
                                          write_p, P, G);
      break;
    }
    default:
      throw std::runtime_error(
          "Unsupported m_val for fused descriptor kernel.");
  }
}

// ============================================================================
// Atlas precision dispatch
// ============================================================================

template <typename Scalar, typename index_t, int K, bool UseReducible>
void dispatch_by_atlas_precision(DeviceArgs<Scalar>& args, bool write_p) {
  if (TENSORMD_USE_FP32_ATLAS) {
    launch_by_m_value<Scalar, float, index_t, K, UseReducible>(
        args, args.params_atlas_fp32, write_p);
  } else {
    launch_by_m_value<Scalar, Scalar, index_t, K, UseReducible>(
        args, args.params_atlas, write_p);
  }
}

// ============================================================================
// Public interface
// ============================================================================

template <typename Scalar, typename index_t>
void dispatch_calc_descriptors_fused(DeviceArgs<Scalar>& args) {
  const bool use_reducible = args.T_proj_algo < 2;
  const bool write_p = (STRATEGY != Strategy::Production);
  const int K = args.k;

  if (K != 32 && K != 64 && K != 128 && K != 256) {
    throw std::runtime_error(
        "GPU::calc_descriptors_fused requires K to be one of 32/64/128/256.");
  }

  if (use_reducible) {
    switch (K) {
      case 32:
        dispatch_by_atlas_precision<Scalar, index_t, 32, true>(args, write_p);
        break;
      case 64:
        dispatch_by_atlas_precision<Scalar, index_t, 64, true>(args, write_p);
        break;
      case 128:
        dispatch_by_atlas_precision<Scalar, index_t, 128, true>(args, write_p);
        break;
      case 256:
        dispatch_by_atlas_precision<Scalar, index_t, 256, true>(args, write_p);
        break;
    }
  } else {
    switch (K) {
      case 32:
        dispatch_by_atlas_precision<Scalar, index_t, 32, false>(args, write_p);
        break;
      case 64:
        dispatch_by_atlas_precision<Scalar, index_t, 64, false>(args, write_p);
        break;
      case 128:
        dispatch_by_atlas_precision<Scalar, index_t, 128, false>(args, write_p);
        break;
      case 256:
        dispatch_by_atlas_precision<Scalar, index_t, 256, false>(args, write_p);
        break;
    }
  }
}

template <typename Scalar>
void calc_descriptors_fused(DeviceArgs<Scalar>& args) {
  const int K = args.k;
  const size_t A = static_cast<size_t>(args.a);
  const size_t B = static_cast<size_t>(args.b);
  const size_t C = static_cast<size_t>(args.c);
  if (K != 32 && K != 64 && K != 128 && K != 256) {
    throw std::runtime_error(
        "GPU::calc_descriptors_fused requires K to be one of 32/64/128/256.");
  }
  if (B != 1) {
    throw std::runtime_error(
        "GPU::calc_descriptors_fused currently only supports B=1.");
  }
  if (A * C < 1e9) {
    dispatch_calc_descriptors_fused<Scalar, int>(args);
  } else {
    dispatch_calc_descriptors_fused<Scalar, size_t>(args);
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_descriptors_fused<float>(DeviceArgs<float>& args);
template void calc_descriptors_fused<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Fusion::GPU
