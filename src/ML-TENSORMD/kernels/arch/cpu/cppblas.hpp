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

#ifndef TENSORMD_KERNELS_ARCH_CPU_CPPBLAS_HPP
#define TENSORMD_KERNELS_ARCH_CPU_CPPBLAS_HPP

#if defined(EIGEN_USE_MKL_ALL)
#include "mkl.h"
#elif defined(__APPLE__) && defined(AccelerateFramework)
#include <Accelerate/accelerate.h>
#else
#include <cblas.h>
#endif

namespace TENSORMD::Kernels::Linalg::CPU {

inline void cppblas_axpy(int n, float alpha, float* x, int incx, float* y,
                         int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

inline void cppblas_axpy(int n, double alpha, double* x, int incx, double* y,
                         int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

inline void cppblas_copy(int n, float* x, int incx, float* y, int incy) {
  cblas_scopy(n, x, incx, y, incy);
}

inline void cppblas_copy(int n, double* x, int incx, double* y, int incy) {
  cblas_dcopy(n, x, incx, y, incy);
}

inline void cppblas_gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, int M, int N, int K,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb, double beta, double* C,
                         int ldc) {
  cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}
inline void cppblas_gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, int M, int N, int K,
                         float alpha, const float* A, int lda, const float* B,
                         int ldb, float beta, float* C, int ldc) {
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

inline float cppblas_dot(int n, float* x, int incx, float* y, int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

inline double cppblas_dot(int n, double* x, int incx, double* y, int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

#ifndef MKL_INT
#define MKL_INT int
#endif

inline void cppblas_gemm_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n,
    const MKL_INT k, const double alpha, const double* a, const MKL_INT lda,
    const MKL_INT stridea, const double* b, const MKL_INT ldb,
    const MKL_INT strideb, const double beta, double* c, const MKL_INT ldc,
    const MKL_INT stridec, const MKL_INT batch_size) {
#if INTEL_MKL_VERSION >= 20210000
  cblas_dgemm_batch_strided(layout, transa, transb, m, n, k, alpha, a, lda,
                            stridea, b, ldb, strideb, beta, c, ldc, stridec,
                            batch_size);
#else
  for (MKL_INT batch = 0; batch < batch_size; ++batch) {
    const double* a_ptr = a + static_cast<std::ptrdiff_t>(batch) * stridea;
    const double* b_ptr = b + static_cast<std::ptrdiff_t>(batch) * strideb;
    double* c_ptr = c + static_cast<std::ptrdiff_t>(batch) * stridec;
    cblas_dgemm(layout, transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
                beta, c_ptr, ldc);
  }
#endif
}

inline void cppblas_gemm_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n,
    const MKL_INT k, const float alpha, const float* a, const MKL_INT lda,
    const MKL_INT stridea, const float* b, const MKL_INT ldb,
    const MKL_INT strideb, const float beta, float* c, const MKL_INT ldc,
    const MKL_INT stridec, const MKL_INT batch_size) {
#if INTEL_MKL_VERSION >= 20210000
  cblas_sgemm_batch_strided(layout, transa, transb, m, n, k, alpha, a, lda,
                            stridea, b, ldb, strideb, beta, c, ldc, stridec,
                            batch_size);
#else
  for (MKL_INT batch = 0; batch < batch_size; ++batch) {
    const float* a_ptr = a + static_cast<std::ptrdiff_t>(batch) * stridea;
    const float* b_ptr = b + static_cast<std::ptrdiff_t>(batch) * strideb;
    float* c_ptr = c + static_cast<std::ptrdiff_t>(batch) * stridec;
    cblas_sgemm(layout, transa, transb, m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
                beta, c_ptr, ldc);
  }
#endif
}

}  // namespace TENSORMD::Kernels::Linalg::CPU

#endif
