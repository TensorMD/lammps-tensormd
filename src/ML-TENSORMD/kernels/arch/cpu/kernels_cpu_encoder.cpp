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
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"
#include "batch_matmul_ops.hpp"
#include "cppblas.hpp"

namespace TENSORMD::Kernels::Encoder::CPU {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void calc_U_abck(DeviceArgs<Scalar>& args) {
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
    const int ldb = d;
    const int ldc = k;
    const int stridea = k * d;
    const int strideb = c * d;
    const int stridec = k * c;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    Kernels::Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasNoTrans, CblasNoTrans, k, c, d, alpha, args.W, lda,
        stridea, args.M, ldb, strideb, beta, args.U, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > M_t(args.M, d, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, k, d, b, a);
    TensorMap<Tensor<Scalar, 4> > U_t(args.U, k, c, b, a);
    Kernels::CPU::batch_matmul<Scalar>(W_t, M_t, false, false, &U_t);
#endif
  }
  // W_dkba
  else {
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
    size_t limit = (size_t)a * (size_t)b;
    const int lda = d;
    const int ldb = d;
    const int ldc = k;
    const int stridea = d * k;
    const int strideb = c * d;
    const int stridec = k * c;
    const Scalar alpha = 1.0;
    const Scalar beta = 0.0;
    Kernels::Linalg::CPU::cppblas_gemm_batch_strided(
        CblasColMajor, CblasTrans, CblasNoTrans, k, c, d, alpha, args.W, lda,
        stridea, args.M, ldb, strideb, beta, args.U, ldc, stridec, limit);
#else
    TensorMap<Tensor<Scalar, 4> > M_t(args.M, d, c, b, a);
    TensorMap<Tensor<Scalar, 4> > W_t(args.W, d, k, b, a);
    TensorMap<Tensor<Scalar, 4> > U_t(args.U, k, c, b, a);
    Kernels::CPU::batch_matmul<Scalar>(W_t, M_t, true, false, &U_t);
#endif
  }
  const double FLOPs = 2.0 * a * b * c * d * k;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_U, FLOPs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void apply_cutoff(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::apply_sij);

  const int k = args.k;
  const int c = args.c;
  const int b = args.b;
  const int a = args.a;

  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < a * b * c * k; i++) {
    Scalar hi = args.H[i];
    args.H[i] *= args.sij[i / k];
    args.dHdr[i] = args.dsij[i / k] * hi;
  }

  const auto FLOPs = a * b * c * k * 2.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::apply_sij, FLOPs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void apply_cutoff_forward_only(int n, int k, const Scalar* __restrict__ sij,
                               Scalar* __restrict__ H) {
  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < n * k; i++) {
    H[i] *= sij[i / k];
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void calc_Up(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_Up);

  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < a * b * c * k; i++) {
    Scalar u = args.U[i];
    args.Up[i] = u * args.sij[i / k];
    args.dHdr[i] = u * args.dHdr[i];
  }

  const auto FLOPs = a * b * c * k * 2.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_Up, FLOPs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void calc_Fr(DeviceArgs<Scalar>& args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_Fr);

  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  const int* __restrict__ tijlist = args.tijlist;

  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < a * b * c; i++) {
    if (tijlist[i] < 0) {
      continue;
    }
    Scalar sum = args.dHdrp[i];
#if defined(_OPENMP)
#pragma omp simd reduction(+:sum)
#endif
    for (int ki = 0; ki < k; ki++) {
      sum += args.dHdr[i * k + ki];
    }
    args.Fr[i * 3 + 0] = sum * args.drdrx[i * 3 + 0];
    args.Fr[i * 3 + 1] = sum * args.drdrx[i * 3 + 1];
    args.Fr[i * 3 + 2] = sum * args.drdrx[i * 3 + 2];
  }

  const auto FLOPs = a * b * c * (k + 3);
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_Fr, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void apply_cutoff<float>(DeviceArgs<float>& args);
template void apply_cutoff<double>(DeviceArgs<double>& args);

template void apply_cutoff_forward_only<float>(int n, int k,
                                               const float* __restrict__ sij,
                                               float* __restrict__ H);
template void apply_cutoff_forward_only<double>(int n, int k,
                                                const double* __restrict__ sij,
                                                double* __restrict__ H);

template void calc_U_abck<float>(DeviceArgs<float>& args);
template void calc_U_abck<double>(DeviceArgs<double>& args);

template void calc_Up<float>(DeviceArgs<float>& args);
template void calc_Up<double>(DeviceArgs<double>& args);

template void calc_Fr<float>(DeviceArgs<float>& args);
template void calc_Fr<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Encoder::CPU
