/* ----------------------------------------------------------------------
  MIT License
  Copyright (c) 2026
  [Chen Xin (chen_xin@iapcm.ac.cn), Ouyang Yucheng (ouyangyucheng@live.com)]
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without __restrict__ion, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
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
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"
#include "batch_matmul_ops.hpp"
#include "cppblas.hpp"
#include "splines.hpp"

namespace TENSORMD::Kernels::Encoder::CPU {

// -----------------------------------------------------------------------------
// compute U_dcba = W_kdba/W_dkba * dHdr_kcba
// compute Fr_xcba = U_dcba * M_dcba * drdrx_xcba
// -----------------------------------------------------------------------------

template <typename Scalar>
void calc_U_abcd(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_U);
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  const int d = args.d;

  // W_kdba
  if (TENSORMD_PGW_K_CONTINUOUS) {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = (size_t)a * (size_t)b;
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
        stridea, args.dHdr, ldb, strideb, beta, args.U, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > dHdr_t(args.dHdr, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, k, d, b, a);
    TensorMap<Tensor<Scalar, 4> > U_t(args.U, d, c, b, a);
    Kernels::CPU::batch_matmul<Scalar>(W_t, dHdr_t, true, false, &U_t);
#endif
  }
  // W_dkba
  else {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = (size_t)a * (size_t)b;
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
        stridea, args.dHdr, ldb, strideb, beta, args.U, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > dHdr_t(args.dHdr, k, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, d, k, b, a);
    TensorMap<Tensor<Scalar, 4> > U_t(args.U, d, c, b, a);
    Kernels::CPU::batch_matmul<Scalar>(W_t, dHdr_t, false, false, &U_t);
#endif
  }

  const double FLOPs = 2.0 * a * b * c * d * k;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_U, FLOPs);
}

template <typename Scalar>
void calc_Fr_with_U_abcd_M_abcd(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_Fr);
  Scalar* __restrict__ U = args.U;
  Scalar* __restrict__ M = args.M;
  Scalar* __restrict__ Fr = args.Fr;
  Scalar* __restrict__ drdrx = args.drdrx;
  int* __restrict__ tijlist = args.tijlist;

  int size = args.a * args.b * args.c;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int stride1 = 0; stride1 < size; stride1++) {
    int stride3 = stride1 * 3;
    if (tijlist[stride1] < 0) {
      Fr[stride3 + 0] = 0;
      Fr[stride3 + 1] = 0;
      Fr[stride3 + 2] = 0;
    } else {
      int strided = stride1 * args.d;
      Scalar* Up = U + strided;
      Scalar* Mp = M + strided;
      Scalar y = 0.0;
#if defined(_OPENMP)
#pragma simd reduction(+: y)
#endif
      for (int d = 0; d < args.d; d++) {
        y += Up[d] * Mp[d];
      }
      Fr[stride3 + 0] = drdrx[stride3 + 0] * y;
      Fr[stride3 + 1] = drdrx[stride3 + 1] * y;
      Fr[stride3 + 2] = drdrx[stride3 + 2] * y;
    }
  }
  const double FLOPs = 1.0 * args.a * args.b * args.c * (2.0 * args.d + 3.0);
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_Fr, FLOPs);
}

// -----------------------------------------------------------------------------
// Calculate radial forces Fr with different strategies
// -----------------------------------------------------------------------------

template <typename Scalar>
void interp_backward(DeviceArgs<Scalar>& args) {
  calc_U_abcd(args);
  calc_Fr_with_U_abcd_M_abcd(args);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void interp_backward<float>(DeviceArgs<float>& data);
template void interp_backward<double>(DeviceArgs<double>& data);

}  // namespace TENSORMD::Kernels::Encoder::CPU