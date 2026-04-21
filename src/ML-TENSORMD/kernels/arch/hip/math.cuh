#include "hip/hip_runtime.h"
#ifndef LAMMPS_TENSORMD_GPU_MATH
#define LAMMPS_TENSORMD_GPU_MATH

#include "hipblas.h"

inline hipblasStatus_t gpublasgemmStridedBatched(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    long long strideA, const double *B, int ldb, long long strideB,
    const double *beta, double *C, int ldc, long long strideC, int batchCount)
{
  return hipblasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline hipblasStatus_t gpublasgemmStridedBatched(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    long long strideA, const float *B, int ldb, long long strideB,
    const float *beta, float *C, int ldc, long long strideC, int batchCount)
{
  return hipblasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline hipblasStatus_t gpublasgemm(hipblasHandle_t handle,
                                  hipblasOperation_t transa,
                                  hipblasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *A, int lda,
                                  const double *B, int ldb, const double *beta,
                                  double *C, int ldc)
{
  return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline hipblasStatus_t gpublasgemm(hipblasHandle_t handle,
                                  hipblasOperation_t transa,
                                  hipblasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *A, int lda,
                                  const float *B, int ldb, const float *beta,
                                  float *C, int ldc)
{
  return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline hipblasStatus_t gpublasaxpy(hipblasHandle_t handle, int n,
                                  const double *alpha, const double *x,
                                  int incx, double *y, int incy)
{
  return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline hipblasStatus_t gpublasaxpy(hipblasHandle_t handle, int n,
                                  const float *alpha, const float *x, int incx,
                                  float *y, int incy)
{
  return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline hipblasStatus_t gpublasdot(hipblasHandle_t handle, int n, const double *x,
                                 int incx, const double *y, int incy,
                                 double *result)
{
  return hipblasDdot(handle, n, x, incx, y, incy, result);
}

inline hipblasStatus_t gpublasdot(hipblasHandle_t handle, int n, const float *x,
                                 int incx, const float *y, int incy,
                                 float *result)
{
  return hipblasSdot(handle, n, x, incx, y, incy, result);
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