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

#include "../../../tensormd_constants.h"
#include "../../../tensormd_strategy.h"
#include "../../geometric/irreducible_PGW.hpp"
#include "../../geometric/reducible_PGW.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "batch_matmul_ops.hpp"
#include "cppblas.hpp"

namespace TENSORMD::Kernels::CPU {

using namespace Geometric;

// -----------------------------------------------------------------------------
// Calculate P_dkba/P_kdba
// -----------------------------------------------------------------------------

template <typename Scalar>
void calc_P(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_P);
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int d = args.d;
  const int k = args.k;

  // P_kdba
  if (TENSORMD_PGW_K_CONTINUOUS) {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = static_cast<size_t>(a) * static_cast<size_t>(b);
    const int lda = k;
    const int ldb = d;
    const int ldc = k;
    const int stridea = c * k;
    const int strideb = c * d;
    const int stridec = d * k;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    const int incx = 1;
    const int incy = 1;

    Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasNoTrans, CblasTrans, k, d, c, alpha, args.H, lda,
        stridea, args.M, ldb, strideb, beta, args.P, ldc, stridec, a * b);
#else
    TensorMap<Tensor<Scalar, 4> > M_t(args.M, d, c, b, a);
    TensorMap<Tensor<Scalar, 4> > H_t(args.H, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > P_t(args.P, k, d, b, a);
    batch_matmul<Scalar>(H_t, M_t, false, true, &P_t);
#endif
  }
  // P_dkba
  else {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = static_cast<size_t>(a) * static_cast<size_t>(b);
    const int lda = d;
    const int ldb = k;
    const int ldc = d;
    const int stridea = d * c;
    const int strideb = c * k;
    const int stridec = d * k;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    const int incx = 1;
    const int incy = 1;

    Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasNoTrans, CblasTrans, d, k, c, alpha, args.M, lda,
        stridea, args.H, ldb, strideb, beta, args.P, ldc, stridec, a * b);
#else
    TensorMap<Tensor<Scalar, 4> > M_t(args.M, d, c, b, a);
    TensorMap<Tensor<Scalar, 4> > H_t(args.H, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > P_t(args.P, d, k, b, a);
    batch_matmul<Scalar>(M_t, H_t, false, true, &P_t);
#endif
  }

  const double FLOPs = a * b * c * d * k * 2.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_P, FLOPs);
}

// -----------------------------------------------------------------------------
// Calculate G_mkba/G_kmba
// -----------------------------------------------------------------------------

template <typename Scalar, int m_val, int D, bool use_legacy_T,
          bool UseReducible>
void calc_G_kernel(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_G);
  const int A = args.a;
  const int B = args.b;
  const int K = args.k;
  const Scalar* __restrict__ P = args.P;
  Scalar* G = args.G;

  // Thread-Level Parallelism
  // Strategy: G_kmba
  if (TENSORMD_PGW_K_CONTINUOUS) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (int a = 0; a < A; ++a) {
      for (int b = 0; b < B; ++b) {
        // Calculate the starting pointer for the current (a, b) slice.
        // P shape is [A, B, D, K], so slice size for one (a, b) is D * K
        const Scalar* P_ab = P + (a * B + b) * (D * K);

        // G shape is[A, B, m, K], so slice size for one (a, b) is m_val * K
        Scalar* G_ab = G + (a * B + b) * (m_val * K);

        // Data-Level Parallelism (Vectorization): The innermost K loop
        // The stride between different dimensions 'd' or 'm' is exactly K.
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (int k = 0; k < K; ++k) {
          if constexpr (UseReducible) {
            Reducible::contract_G_kmba_otf<Scalar, m_val, use_legacy_T>(
                P_ab, G_ab, k, K);
          } else {
            Irreducible::contract_G_kmba_otf<Scalar, m_val>(P_ab, G_ab, k, K);
          }
        }
      }
    }
  }
  // Strategy: G_mkba
  else {
    size_t size = static_cast<size_t>(A) * B * K;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      const Scalar* __restrict__ P_d = P + i * D;
      Scalar* __restrict__ G_m = G + i * m_val;
      if constexpr (UseReducible) {
        Reducible::contract_G_mkba_otf<Scalar, m_val, use_legacy_T>(P_d, G_m);
      } else {
        Irreducible::contract_G_mkba_otf<Scalar, m_val>(P_d, G_m);
      }
    }
  }

  const double FLOPs = 1.0 * A * B * K * (D + 2 * m_val * D + m_val);
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_G, FLOPs);
}

template <typename Scalar>
void calc_G(DeviceArgs<Scalar>& args) {
  if (args.T_proj_algo < 0 || args.T_proj_algo > 2) {
    throw std::runtime_error("Unknown P->G projection algorithm.");
  }
  if (args.T_proj_algo == 0) {
    if (args.m == 1) {
      calc_G_kernel<Scalar, 1, 1, true, true>(args);
    } else if (args.m == 2) {
      calc_G_kernel<Scalar, 2, 4, true, true>(args);
    } else if (args.m == 3) {
      calc_G_kernel<Scalar, 3, 10, true, true>(args);
    } else if (args.m == 4) {
      calc_G_kernel<Scalar, 4, 20, true, true>(args);
    } else if (args.m == 5) {
      calc_G_kernel<Scalar, 5, 35, true, true>(args);
    } else if (args.m == 6) {
      calc_G_kernel<Scalar, 6, 56, true, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_G kernel");
    }
  } else if (args.T_proj_algo == 1) {
    if (args.m == 1) {
      calc_G_kernel<Scalar, 1, 1, false, true>(args);
    } else if (args.m == 2) {
      calc_G_kernel<Scalar, 2, 4, false, true>(args);
    } else if (args.m == 3) {
      calc_G_kernel<Scalar, 3, 10, false, true>(args);
    } else if (args.m == 4) {
      calc_G_kernel<Scalar, 4, 20, false, true>(args);
    } else if (args.m == 5) {
      calc_G_kernel<Scalar, 5, 35, false, true>(args);
    } else if (args.m == 6) {
      calc_G_kernel<Scalar, 6, 56, false, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_G kernel");
    }
  } else {
    if (args.m == 1) {
      calc_G_kernel<Scalar, 1, 1, false, false>(args);
    } else if (args.m == 2) {
      calc_G_kernel<Scalar, 2, 4, false, false>(args);
    } else if (args.m == 3) {
      calc_G_kernel<Scalar, 3, 9, false, false>(args);
    } else if (args.m == 4) {
      calc_G_kernel<Scalar, 4, 16, false, false>(args);
    } else if (args.m == 5) {
      calc_G_kernel<Scalar, 5, 25, false, false>(args);
    } else if (args.m == 6) {
      calc_G_kernel<Scalar, 6, 36, false, false>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_G kernel");
    }
  }
}

// -----------------------------------------------------------------------------
// Calculate W_dkba/W_kdba
// -----------------------------------------------------------------------------

template <typename Scalar, int m_val, int D, bool use_legacy_T,
          bool UseReducible>
void calc_W_kernel(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_W);
  const int A = args.a;
  const int B = args.b;
  const int K = args.k;
  Scalar* P = args.P;
  Scalar* W = args.W;
  Scalar* __restrict__ dEdG = args.dEdG;

  if (TENSORMD_PGW_K_CONTINUOUS) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (int a = 0; a < A; ++a) {
      for (int b = 0; b < B; ++b) {
        // Calculate the starting pointer for the current (a, b) slice.
        // P/W shape is [A, B, D, K], so slice size for one (a, b) is D * K
        Scalar* P_base_ab = P + (a * B + b) * (D * K);
        Scalar* W_base_ab = W + (a * B + b) * (D * K);

        // G shape is[A, B, m, K], so slice size for one (a, b) is m_val * K
        const Scalar*__restrict__ G_base_ab = dEdG + (a * B + b) * (m_val * K);

        // Data-Level Parallelism (Vectorization): The innermost K loop
        // The stride between different dimensions 'd' or 'm' is exactly K.
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (int k = 0; k < K; ++k) {
          if (UseReducible) {
            Reducible::contract_W_kdba_otf<Scalar, m_val, use_legacy_T>(
                G_base_ab, P_base_ab, W_base_ab, k, K);
          } else {
            Irreducible::contract_W_kdba_otf<Scalar, m_val>(
                G_base_ab, P_base_ab, W_base_ab, k, K);
          }
        }
      }
    }
  } else {
    size_t size = static_cast<size_t>(A) * B * K;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      Scalar* P_d = P + i * D;
      Scalar* W_d = W + i * D;
      Scalar* dEdG_m = dEdG + i * m_val;
      if constexpr (UseReducible) {
        Reducible::contract_W_dkba_otf<Scalar, m_val, use_legacy_T>(dEdG_m, P_d,
                                                                    W_d);
      } else {
        Irreducible::contract_W_dkba_otf<Scalar, m_val>(dEdG_m, P_d, W_d);
      }
    }
  }

  const double FLOPs = 2.0 * A * B * K * (m_val * D + D + 2.0);
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_W, FLOPs);
}

template <typename Scalar>
void calc_W(DeviceArgs<Scalar>& args) {
  if (args.T_proj_algo < 0 || args.T_proj_algo > 2) {
    throw std::runtime_error("Unknown P->G projection algorithm.");
  }
  if (args.T_proj_algo == 0) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, true, true>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, true, true>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 10, true, true>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 20, true, true>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 35, true, true>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 56, true, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  } else if (args.T_proj_algo == 1) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, false, true>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, false, true>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 10, false, true>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 20, false, true>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 35, false, true>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 56, false, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  } else if (args.T_proj_algo == 2) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, false, false>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, false, false>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 9, false, false>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 16, false, false>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 25, false, false>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 36, false, false>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_P<float>(DeviceArgs<float>& args);
template void calc_P<double>(DeviceArgs<double>& args);
template void calc_G<float>(DeviceArgs<float>& args);
template void calc_G<double>(DeviceArgs<double>& args);
template void calc_W<float>(DeviceArgs<float>& args);
template void calc_W<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::CPU