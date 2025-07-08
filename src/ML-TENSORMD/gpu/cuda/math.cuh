#ifndef LAMMPS_TENSORMD_GPU_MATH
#define LAMMPS_TENSORMD_GPU_MATH

#include "cublas_v2.h"

inline cublasStatus_t gpublasgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    long long strideA, const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC, int batchCount)
{
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline cublasStatus_t gpublasgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    long long strideA, const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC, int batchCount)
{
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline cublasStatus_t gpublasgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *A, int lda,
                                  const double *B, int ldb, const double *beta,
                                  double *C, int ldc)
{
  return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline cublasStatus_t gpublasgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *A, int lda,
                                  const float *B, int ldb, const float *beta,
                                  float *C, int ldc)
{
  return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline cublasStatus_t gpublasaxpy(cublasHandle_t handle, int n,
                                  const double *alpha, const double *x,
                                  int incx, double *y, int incy)
{
  return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t gpublasaxpy(cublasHandle_t handle, int n,
                                  const float *alpha, const float *x, int incx,
                                  float *y, int incy)
{
  return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t gpublasdot(cublasHandle_t handle, int n, const double *x,
                                 int incx, const double *y, int incy,
                                 double *result)
{
  return cublasDdot_v2(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t gpublasdot(cublasHandle_t handle, int n, const float *x,
                                 int incx, const float *y, int incy,
                                 float *result)
{
  return cublasSdot_v2(handle, n, x, incx, y, incy, result);
}

template <typename Scalar>
__global__ void vmul_device(Scalar *dz, Scalar *x, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) { dz[i] *= x[i]; }
}

template <typename Scalar> __global__ void ones_like_device(Scalar *x, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) { x[i] = 1.0; }
}

#endif