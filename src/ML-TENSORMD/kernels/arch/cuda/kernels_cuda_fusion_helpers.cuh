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

#ifndef TENSORMD_KERNELS_ARCH_CUDA_KERNELS_FUSION_HELPERS_CUH
#define TENSORMD_KERNELS_ARCH_CUDA_KERNELS_FUSION_HELPERS_CUH

#include "../../geometric/irreducible_PGW.hpp"
#include "../../geometric/irreducible_moments.hpp"
#include "../../geometric/reducible_PGW.hpp"
#include "../../geometric/reducible_moments.hpp"
#include "kernels_cuda_fusion_types.cuh"
#include "splines.cuh"

namespace TENSORMD::Kernels::Fusion::GPU {

using namespace Geometric;

// ============================================================================
// Forward helpers
// ============================================================================

template <int K>
__device__ __forceinline__ void block_sync_for_k() {
  if constexpr (K == 32) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

template <typename AtlasType, int D>
__device__ __forceinline__ void store_P_if_needed(
    bool write_p, AtlasType* __restrict__ P_cast, int idx_ab, int k,
    const AtlasType (&P_reg)[D], int K) {
  if (!write_p) return;

  AtlasType* P_dk = P_cast + idx_ab * D * K;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    P_dk[d * K + k] = P_reg[d];
  }
}

template <typename Scalar, typename AtlasType, int M, int D, bool UseReducible>
__device__ __forceinline__ void convert_P_to_G(AtlasType (&P_reg)[D],
                                               Scalar* __restrict__ G_km, int k,
                                               int K) {
  if constexpr (UseReducible) {
    Geometric::Reducible::P_d_to_G_kmba_otf<Scalar, AtlasType, M, false>(
        P_reg, G_km, k, K);
  } else {
    Geometric::Irreducible::P_u_to_G_kmba_otf<Scalar, AtlasType, M>(P_reg, G_km,
                                                                    k, K);
  }
}

template <class CFG, typename index_t>
__device__ __forceinline__ void calculate_P_direct(
    int C, int idx_ab, int k, const int* __restrict__ tau_list,
    const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    char* __restrict__ shared_mem,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    int atlas_stride, typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar xdx, typename CFG::AtlasType (&P_reg)[CFG::D]) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;
  constexpr int BS = CFG::BatchSize;
  constexpr int K = CFG::K;
  constexpr int M = CFG::m_val;
  constexpr bool UseReducible = CFG::UseReducible;

  const index_t C_t = static_cast<index_t>(C);

  AtlasType* s_dr = reinterpret_cast<AtlasType*>(shared_mem);  // [BS, 3]
  Scalar* s_R = reinterpret_cast<Scalar*>(s_dr + BS * 3);      // [BS]
  int* s_tau = reinterpret_cast<int*>(s_R + BS);               // [BS]

  for (int c_base = 0; c_base < C; c_base += BS) {
    const int batch_limit = min(BS, C - c_base);

    for (int i = k; i < batch_limit; i += K) {
      const int idx_ab_t = static_cast<index_t>(idx_ab);
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c_base + i);
      const int t = tau_list[idx_c];
      s_tau[i] = t;
      s_R[i] = R[idx_c];

      if (t >= 0) {
        const Scalar* dr = drdrx + idx_c * 3;
        s_dr[i * 3 + 0] = static_cast<AtlasType>(dr[0]);
        s_dr[i * 3 + 1] = static_cast<AtlasType>(dr[1]);
        s_dr[i * 3 + 2] = static_cast<AtlasType>(dr[2]);
      }
    }

    block_sync_for_k<K>();

    for (int b = 0; b < batch_limit; ++b) {
      const int t = s_tau[b];
      if (t < 0) continue;

      AtlasType hk = AtlasType(0);
      Kernels::Spline::GPU::eval_cubic_spline_single_value<Scalar, AtlasType>(
          s_R[b], t, k, K, params_atlas, grid_size, atlas_stride, x0, dx, xdx,
          hk);

      const AtlasType x = s_dr[b * 3 + 0];
      const AtlasType y = s_dr[b * 3 + 1];
      const AtlasType z = s_dr[b * 3 + 2];

      if constexpr (UseReducible) {
        Geometric::Reducible::calculate_P_kd_otf<AtlasType, M>(x, y, z, hk,
                                                               P_reg);
      } else {
        Geometric::Irreducible::calculate_P_ku_otf<AtlasType, M>(x, y, z, hk,
                                                                 P_reg);
      }
    }

    block_sync_for_k<K>();
  }
}

template <class CFG, typename index_t>
__device__ __forceinline__ void calculate_P_cached(
    int C, int idx_ab, int k, const int* __restrict__ tau_list,
    const typename CFG::Scalar* __restrict__ R,
    const typename CFG::Scalar* __restrict__ drdrx,
    char* __restrict__ shared_mem,
    const typename CFG::AtlasType* __restrict__ params_atlas, int grid_size,
    int atlas_stride, typename CFG::Scalar x0, typename CFG::Scalar dx,
    typename CFG::Scalar xdx, typename CFG::AtlasType (&P_reg)[CFG::D]) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;
  constexpr int BS = CFG::BatchSize;
  constexpr int K = CFG::K;
  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr bool UseReducible = CFG::UseReducible;
  constexpr int Strided = CFG::M_Stride;

  const index_t C_t = static_cast<index_t>(C);

  AtlasType* s_M = reinterpret_cast<AtlasType*>(shared_mem);    // [BS, strided]
  Scalar* s_R = reinterpret_cast<Scalar*>(s_M + BS * Strided);  // [BS]
  int* s_tau = reinterpret_cast<int*>(s_R + BS);                // [BS]

  for (int c_base = 0; c_base < C; c_base += BS) {
    const int batch_limit = min(BS, C - c_base);

    for (int i = k; i < batch_limit; i += K) {
      const int idx_ab_t = static_cast<index_t>(idx_ab);
      const index_t idx_c = idx_ab_t * C_t + static_cast<index_t>(c_base + i);
      const int t = tau_list[idx_c];
      s_tau[i] = t;
      s_R[i] = R[idx_c];

      if (t >= 0) {
        const Scalar* dr = drdrx + idx_c * 3;
        const AtlasType x = static_cast<AtlasType>(dr[0]);
        const AtlasType y = static_cast<AtlasType>(dr[1]);
        const AtlasType z = static_cast<AtlasType>(dr[2]);

        if constexpr (UseReducible) {
          Geometric::Reducible::calculate_M_d_otf<AtlasType, M>(
              x, y, z, s_M + i * Strided);
        } else {
          Geometric::Irreducible::calculate_M_u_otf<AtlasType, M>(
              x, y, z, s_M + i * Strided);
        }
      }
    }

    block_sync_for_k<K>();

    for (int b = 0; b < batch_limit; ++b) {
      const int t = s_tau[b];
      if (t < 0) continue;

      AtlasType hk = AtlasType(0);
      Kernels::Spline::GPU::eval_cubic_spline_single_value<Scalar, AtlasType>(
          s_R[b], t, k, K, params_atlas, grid_size, atlas_stride, x0, dx, xdx,
          hk);

#pragma unroll
      for (int d = 0; d < D; ++d) {
        P_reg[d] += hk * s_M[b * Strided + d];
      }
    }

    block_sync_for_k<K>();
  }
}

// ============================================================================
// Backward helpers
// ============================================================================

template <class CFG>
__device__ __forceinline__ void build_W_reg_inplace(
    const typename CFG::Scalar* __restrict__ dEdG,
    typename CFG::AtlasType (&W_reg)[CFG::D], int idx_ab, int k) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;
  constexpr int M = CFG::m_val;
  constexpr int K = CFG::K;
  constexpr bool UseReducible = CFG::UseReducible;

  const Scalar* __restrict__ G_km = dEdG + idx_ab * M * K;
  if constexpr (UseReducible) {
    Geometric::Reducible::P_d_Gp_km_to_W_d_inplace<Scalar, AtlasType, M>(
        G_km, W_reg, k, K);
  } else {
    Geometric::Irreducible::P_u_Gp_km_to_W_u_inplace<Scalar, AtlasType, M>(
        G_km, W_reg, k, K);
  }
}

template <class CFG>
__device__ __forceinline__ void build_W_reg(
    const typename CFG::Scalar* __restrict__ dEdG,
    const typename CFG::AtlasType* __restrict__ P, int idx_ab, int k,
    typename CFG::AtlasType (&W_reg)[CFG::D]) {
  using Scalar = typename CFG::Scalar;
  using AtlasType = typename CFG::AtlasType;

  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr int K = CFG::K;
  constexpr bool UseReducible = CFG::UseReducible;

  const Scalar* __restrict__ G_km = dEdG + idx_ab * M * K;
  const AtlasType* __restrict__ P_kd = P + idx_ab * D * K;

  if constexpr (UseReducible) {
    Reducible::P_kd_Gp_km_to_W_d<Scalar, AtlasType, M>(G_km, P_kd, W_reg, k, K);
  } else {
    Irreducible::P_ku_Gp_km_to_W_u<Scalar, AtlasType, M>(G_km, P_kd, W_reg, k,
                                                         K);
  }
}

template <class CFG>
__device__ __forceinline__ void calc_dM_to_shared(
    typename CFG::AtlasType x, typename CFG::AtlasType y,
    typename CFG::AtlasType z, typename CFG::AtlasType rijinv,
    typename CFG::AtlasType* __restrict__ out) {
  using AtlasType = typename CFG::AtlasType;
  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr bool UseReducible = CFG::UseReducible;

  if constexpr (UseReducible) {
    Reducible::calculate_dM_otf<AtlasType, M, D>(x, y, z, rijinv, out);
  } else {
    Irreducible::calculate_dM_otf<AtlasType, M, D>(x, y, z, rijinv, out);
  }
}

template <class CFG>
__device__ __forceinline__ void calc_M_dM_to_shared(
    typename CFG::AtlasType x, typename CFG::AtlasType y,
    typename CFG::AtlasType z, typename CFG::AtlasType rijinv,
    typename CFG::AtlasType* __restrict__ out) {
  using AtlasType = typename CFG::AtlasType;
  constexpr int M = CFG::m_val;
  constexpr int D = CFG::D;
  constexpr bool UseReducible = CFG::UseReducible;

  if constexpr (UseReducible) {
    Reducible::calculate_M_dM_otf<AtlasType, M, D>(x, y, z, rijinv, out);
  } else {
    Irreducible::calculate_M_dM_otf<AtlasType, M, D>(x, y, z, rijinv, out);
  }
}

template <class CFG>
__device__ __forceinline__ void contract_W_M(
    typename CFG::AtlasType x, typename CFG::AtlasType y,
    typename CFG::AtlasType z, const typename CFG::AtlasType (&W_reg)[CFG::D],
    typename CFG::AtlasType& fr) {
  using AtlasType = typename CFG::AtlasType;
  constexpr int M = CFG::m_val;
  constexpr bool UseReducible = CFG::UseReducible;

  if constexpr (UseReducible) {
    Reducible::contract_W_M_otf<AtlasType, M>(x, y, z, W_reg, fr);
  } else {
    Irreducible::contract_W_M_otf<AtlasType, M>(x, y, z, W_reg, fr);
  }
}

template <class CFG>
__device__ __forceinline__ void accumulate_fa_from_dM(
    const typename CFG::AtlasType (&W_reg)[CFG::D],
    const typename CFG::AtlasType* __restrict__ dMdx_c,
    typename CFG::AtlasType& fa_x, typename CFG::AtlasType& fa_y,
    typename CFG::AtlasType& fa_z) {
  using AtlasType = typename CFG::AtlasType;
  constexpr int D = CFG::D;

#pragma unroll
  for (int d = 0; d < D; ++d) {
    const AtlasType w = W_reg[d];
    fa_x += w * dMdx_c[0 * D + d];
    fa_y += w * dMdx_c[1 * D + d];
    fa_z += w * dMdx_c[2 * D + d];
  }
}

template <class CFG>
__device__ __forceinline__ void accumulate_fr_fa_from_M_dM(
    const typename CFG::AtlasType (&W_reg)[CFG::D],
    const typename CFG::AtlasType* __restrict__ dMdx_c,
    typename CFG::AtlasType& fr, typename CFG::AtlasType& fa_x,
    typename CFG::AtlasType& fa_y, typename CFG::AtlasType& fa_z) {
  using AtlasType = typename CFG::AtlasType;
  constexpr int D = CFG::D;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    const AtlasType w = W_reg[d];
    fr += w * dMdx_c[0 * D + d];
    fa_x += w * dMdx_c[1 * D + d];
    fa_y += w * dMdx_c[2 * D + d];
    fa_z += w * dMdx_c[3 * D + d];
  }
}

template <typename T>
__device__ __forceinline__ void warp_reduce_sum4(T& a, T& b, T& c, T& d) {
  constexpr unsigned mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_down_sync(mask, a, offset);
    b += __shfl_down_sync(mask, b, offset);
    c += __shfl_down_sync(mask, c, offset);
    d += __shfl_down_sync(mask, d, offset);
  }
}

}  // namespace TENSORMD::Kernels::Fusion::GPU

#endif
