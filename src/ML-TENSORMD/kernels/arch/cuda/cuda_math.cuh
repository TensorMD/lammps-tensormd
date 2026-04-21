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

#ifndef TENSORMD_KERNELS_GPU_CUDA_MATH
#define TENSORMD_KERNELS_GPU_CUDA_MATH

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace TENSORMD::Kernels {

namespace Linalg::GPU {

// ----------------------------------------------------------------------
// CUBLAS Wrappers
// ----------------------------------------------------------------------

inline cublasStatus_t gpublasgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    long long strideA, const double* B, int ldb, long long strideB,
    const double* beta, double* C, int ldc, long long strideC, int batchCount) {
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline cublasStatus_t gpublasgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long strideA, const float* B, int ldb, long long strideB,
    const float* beta, float* C, int ldc, long long strideC, int batchCount) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

inline cublasStatus_t gpublasgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double* alpha, const double* A, int lda,
                                  const double* B, int ldb, const double* beta,
                                  double* C, int ldc) {
  return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline cublasStatus_t gpublasgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float* alpha, const float* A, int lda,
                                  const float* B, int ldb, const float* beta,
                                  float* C, int ldc) {
  return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                        beta, C, ldc);
}

inline cublasStatus_t gpublasaxpy(cublasHandle_t handle, int n,
                                  const double* alpha, const double* x,
                                  int incx, double* y, int incy) {
  return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t gpublasaxpy(cublasHandle_t handle, int n,
                                  const float* alpha, const float* x, int incx,
                                  float* y, int incy) {
  return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t gpublasdot(cublasHandle_t handle, int n, const double* x,
                                 int incx, const double* y, int incy,
                                 double* result) {
  return cublasDdot_v2(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t gpublasdot(cublasHandle_t handle, int n, const float* x,
                                 int incx, const float* y, int incy,
                                 float* result) {
  return cublasSdot_v2(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float* alpha, const float* A, int lda,
                                  const float* B, int ldb, const float* beta,
                                  float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double* alpha, const double* A, int lda,
                                  const double* B, int ldb, const double* beta,
                                  double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
inline cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const float* A, int lda, const float* W, int* lwork) {
  return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, (float*)A, lda,
                                     (float*)W, lwork);
}
inline cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double* A, int lda, const double* W, int* lwork) {
  return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, (double*)A, lda,
                                     (double*)W, lwork);
}
inline cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n, float* A,
                                         int lda, float* W, float* work,
                                         int lwork, int* info) {
  return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}
inline cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle,
                                         cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n,
                                         double* A, int lda, double* W,
                                         double* work, int lwork, int* info) {
  return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

}  // namespace Linalg::GPU

namespace GPU {

// ----------------------------------------------------------------------
// Kernel: fill (Fill an array with a constant value)
// ----------------------------------------------------------------------
template <typename T>
__global__ void fillKernel(T* __restrict__ arr, T value, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    arr[i] = value;
  }
}

// ----------------------------------------------------------------------
// Kernel: vMul (Element-wise Multiplication)
// C[i] = A[i] * B[i]
// ----------------------------------------------------------------------
template <typename T>
__global__ void vMulKernel(const T* A, T* B, T* C, size_t n) {
  // Calculate the global thread index
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate the total number of threads in the grid (stride)
  size_t stride = blockDim.x * gridDim.x;

  // Grid-Stride Loop
  // Handles N > Total Threads
  // Ensures coalesced memory access
  for (; i < n; i += stride) {
    C[i] = A[i] * B[i];
  }
}

// -------------------------------------------------------------------------
// Block Level Reduction for Max
// Reduces a value across a thread block to find the block-wide maximum.
// -------------------------------------------------------------------------
__device__ __forceinline__ int blockReduceMax(int val) {
  // Shared memory for 32 partial maximums (one per warp)
  static __shared__ int shared[32];

  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  // 1. Warp Reduce: Use shuffle to reduce within the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val = (other > val) ? other : val;
  }

  // 2. Write warp max to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // 3. First warp reduces the partial results from shared memory
  // (Assuming block size <= 1024, so max 32 warps)
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

  if (wid == 0) {
    for (int offset = 16; offset > 0; offset /= 2) {
      int other = __shfl_down_sync(0xFFFFFFFF, val, offset);
      val = (other > val) ? other : val;
    }
  }

  return val;
}

// ----------------------------------------------------------------------
// Device Helper: Warp Reduce
// Reduces a value across a warp (32 threads) using shuffle instructions.
// ----------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  // 0xFFFFFFFF means all threads in the warp participate
  // __shfl_down_sync shifts the variable 'val' from the thread (lane + delta)
  // to the current thread.
  val += __shfl_down_sync(0xFFFFFFFF, val, 16);
  val += __shfl_down_sync(0xFFFFFFFF, val, 8);
  val += __shfl_down_sync(0xFFFFFFFF, val, 4);
  val += __shfl_down_sync(0xFFFFFFFF, val, 2);
  val += __shfl_down_sync(0xFFFFFFFF, val, 1);
  return val;
}

// ----------------------------------------------------------------------
// Device Helper: Block Reduce
// Reduces a value across an entire thread block.
// ----------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
  // Shared memory to hold intermediate results from each warp
  // Assuming max block size of 1024, we need 32 elements (1024/32)
  static __shared__ T shared[32];

  int lane = threadIdx.x % 32;  // Lane ID within the warp
  int wid = threadIdx.x / 32;   // Warp ID

  // Reduce within the warp
  val = warpReduceSum(val);

  // The first thread of each warp writes the warp's result to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();  // Wait for all warps to write to shared memory

  // The first warp reads the results from shared memory and performs the
  // final reduction This assumes Block Size <= 1024 (max 32 warps)
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
}

// ----------------------------------------------------------------------
// The Reduce-Sum Kernel
// ----------------------------------------------------------------------
template <typename T>
__global__ void reduceSumKernel(const T* __restrict__ input,
                                T* __restrict__ output, size_t n) {
  T sum = 0;

  // Grid-Stride Loop
  // This allows the kernel to process arrays larger than the total number of
  // threads.
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

  while (i < n) {
    sum += input[i];
    i += gridSize;
  }

  // Reduce the grid-stride accumulated value within the block
  sum = blockReduceSum(sum);

  // The first thread of the block adds the block's partial sum to the global
  // result
  if (threadIdx.x == 0) {
    atomicAdd(output, sum);
  }
}

}  // namespace GPU

}  // namespace TENSORMD::Kernels

#endif