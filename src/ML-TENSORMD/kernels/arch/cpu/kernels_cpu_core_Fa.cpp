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
#include "../../geometric/irreducible_moments.hpp"
#include "../../geometric/reducible_moments.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "batch_matmul_ops.hpp"
#include "cppblas.hpp"

namespace TENSORMD::Kernels::CPU {

// -----------------------------------------------------------------------------
// Kernel V: V_dcba = W_kdba/W_dkba * H_kcba
// -----------------------------------------------------------------------------
template <typename Scalar>
void calc_V(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_V);
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  const int d = args.d;

  // W_kdba
  if (TENSORMD_PGW_K_CONTINUOUS) {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = static_cast<size_t>(a) * static_cast<size_t>(b);
    const int lda = k;
    const int ldb = k;
    const int ldc = d;
    const int stridea = d * k;
    const int strideb = c * k;
    const int stridec = d * c;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    Kernels::Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasTrans, CblasNoTrans, d, c, k, alpha, args.W, lda,
        stridea, args.H, ldb, strideb, beta, args.V, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > H_t(args.H, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, k, d, b, a);
    TensorMap<Tensor<Scalar, 4> > V_t(args.V, d, c, b, a);
    batch_matmul<Scalar>(W_t, H_t, true, false, &V_t);
#endif
  }
  // W_dkba
  else {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = static_cast<size_t>(a) * static_cast<size_t>(b);
    const int lda = d;
    const int ldb = k;
    const int ldc = d;
    const int stridea = d * k;
    const int strideb = c * k;
    const int stridec = d * c;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    Kernels::Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasNoTrans, CblasNoTrans, d, c, k, alpha, args.W, lda,
        stridea, args.H, ldb, strideb, beta, args.V, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > H_t(args.H, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, d, k, b, a);
    TensorMap<Tensor<Scalar, 4> > V_t(args.V, d, c, b, a);
    batch_matmul<Scalar>(W_t, H_t, false, false, &V_t);
#endif
  }

  const double FLOPs = 2.0 * a * b * c * d * k;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_V, FLOPs);
}

// -----------------------------------------------------------------------------
// Fa_xcba = dMdrx_xdcba * V_dcba with dMdrx calculated on the fly
// -----------------------------------------------------------------------------

template <typename Scalar, const int m_val, const int D,
          const bool UseReducible>
void calc_Fa_with_OTF_dMdrx(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_Fa);
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
#if defined(_OPENMP)
#pragma omp for collapse(3) schedule(static)
#endif
    for (int a = 0; a < args.a; a++) {
      for (int b = 0; b < args.b; b++) {
        for (int c = 0; c < args.c; c++) {
          int stride1 = a * args.b * args.c + b * args.c + c;
          int stride3 = stride1 * 3;
          if (args.tijlist[stride1] < 0) {
            args.Fa[stride3 + 0] = 0.0;
            args.Fa[stride3 + 1] = 0.0;
            args.Fa[stride3 + 2] = 0.0;
            continue;
          }
          Scalar rij = args.R[stride1];
          Scalar rijinv = 1.0 / rij;
          Scalar* drdrx_c = args.drdrx + stride3;
          Scalar x = drdrx_c[0];
          Scalar y = drdrx_c[1];
          Scalar z = drdrx_c[2];
          Scalar fx = 0.0;
          Scalar fy = 0.0;
          Scalar fz = 0.0;
          auto V_d = args.V + stride1 * D;
          if constexpr (UseReducible) {
            Geometric::Reducible::contract_W_dM_otf<Scalar, m_val>(
                x, y, z, rijinv, V_d, fx, fy, fz);
          } else {
            Geometric::Irreducible::contract_W_dM_otf<Scalar, m_val>(
                x, y, z, rijinv, V_d, fx, fy, fz);
          }
          args.Fa[stride3 + 0] = fx;
          args.Fa[stride3 + 1] = fy;
          args.Fa[stride3 + 2] = fz;
        }
      }
    }
  }
  double size = static_cast<double>(args.a) * args.b * args.c;
  double factor = 0.0;
  if constexpr (UseReducible) {
    factor = Geometric::Reducible::W_dM_d_FLOPs[m_val];
  } else {
    factor = Geometric::Irreducible::W_dM_u_FLOPs[m_val];
  }
  const double FLOPs = size * factor;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_Fa, FLOPs);
}

// -----------------------------------------------------------------------------
// Fa_xcba = dMdrx_xdcba * V_dcba (Angular)
// -----------------------------------------------------------------------------
template <typename Scalar>
void calc_angular_forces(DeviceArgs<Scalar>& args) {
  calc_V(args);
  if (args.T_proj_algo < 0 || args.T_proj_algo > 2) {
    throw std::runtime_error("Unknown P->G projection algorithm.");
  }
  if (args.m == 1) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 1, 1, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 1, 1, true>(args);
    }
  } else if (args.m == 2) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 2, 4, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 2, 4, true>(args);
    }
  } else if (args.m == 3) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 3, 9, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 3, 10, true>(args);
    }
  } else if (args.m == 4) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 4, 16, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 4, 20, true>(args);
    }
  } else if (args.m == 5) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 5, 25, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 5, 35, true>(args);
    }
  } else if (args.m == 6) {
    if (args.T_proj_algo == 2) {
      calc_Fa_with_OTF_dMdrx<Scalar, 6, 36, false>(args);
    } else {
      calc_Fa_with_OTF_dMdrx<Scalar, 6, 56, true>(args);
    }
  } else {
    throw std::runtime_error("Unsupported m value in calc_angular_forces.");
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_angular_forces<float>(DeviceArgs<float>& data);
template void calc_angular_forces<double>(DeviceArgs<double>& data);

}  // namespace TENSORMD::Kernels::CPU