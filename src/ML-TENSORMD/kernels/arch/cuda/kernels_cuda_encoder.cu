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

#include <cuda_runtime.h>

#include "../../../tensormd_constants.h"
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::Encoder::GPU {

using Kernels::GPU::handle;

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void calc_U_abck(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_U);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.W);
  check_kernel_arg(args.M);
  check_kernel_arg(args.U);
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.d);
  check_kernel_arg(args.k);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int d = args.d;
  const int k = args.k;
  Scalar* W = args.W;
  Scalar* M = args.M;
  Scalar* U = args.U;
  const int lda = k;
  const int ldb = d;
  const int ldc = k;
  const int stridea = k * d;
  const int strideb = c * d;
  const int stridec = k * c;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  Linalg::GPU::gpublasgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, k, c, d, &alpha, W, lda, stridea, M,
      ldb, strideb, &beta, U, ldc, stridec, a * b);
  CHECK(cudaGetLastError());
  const double FLOPs = static_cast<double>(2.0f * a * b * c * d * k);
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_U, FLOPs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
__global__ void apply_cutoff_kernel(size_t total_elements, int k,
                                    Scalar* __restrict__ H,
                                    Scalar* __restrict__ dHdr,
                                    const Scalar* __restrict__ sij,
                                    const Scalar* __restrict__ dsij) {
  // Global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_elements) {
    // Map the index of H (size k*c*b*a) to index of sij (size c*b*a).
    // Assuming Eigen Column-Major layout where 'k' is the first dimension
    // (stride 1). Elements 0 to k-1 of H share index 0 of sij.
    size_t sij_idx = idx / k;

    // Load values
    // __restrict__ helps compiler optimize loads (LDG)
    Scalar s_val = sij[sij_idx];
    Scalar ds_val = dsij[sij_idx];
    Scalar h_val = H[idx];

    // Apply Chain Rule
    // dHdr = dsij * H (Must use the ORIGINAL H value)
    dHdr[idx] = ds_val * h_val;

    // H = sij * H
    H[idx] = s_val * h_val;
  }
}

template <typename Scalar>
void apply_cutoff(DeviceArgs<Scalar>& args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.k);
  check_kernel_arg(args.H);
  check_kernel_arg(args.dHdr);
  check_kernel_arg(args.sij);
  check_kernel_arg(args.dsij);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  size_t total_elements = (size_t)k * c * b * a;

  // Safety check
  if (total_elements == 0) return;

  int blockSize = 256;
  int gridSize = (total_elements + blockSize - 1) / blockSize;

  // Launch Kernel
  // Note: data->H, data->dHdr, etc., must be device pointers.
  apply_cutoff_kernel<<<gridSize, blockSize>>>(
      total_elements, k, args.H, args.dHdr, args.sij, args.dsij);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
__global__ void apply_cutoff_forward_only_kernel(
    size_t total_elements, int k, Scalar* __restrict__ H,
    const Scalar* __restrict__ sij) {
  // Global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_elements) {
    // Map the index of H (size k*c*b*a) to index of sij (size c*b*a).
    // Assuming Eigen Column-Major layout where 'k' is the first dimension
    // (stride 1). Elements 0 to k-1 of H share index 0 of sij.
    size_t sij_idx = idx / k;

    // Load values
    // __restrict__ helps compiler optimize loads (LDG)
    Scalar s_val = sij[sij_idx];
    Scalar h_val = H[idx];

    // H = sij * H
    H[idx] = s_val * h_val;
  }
}

template <typename Scalar>
void apply_cutoff_forward_only(int n, int k, const Scalar* __restrict__ sij,
                               Scalar* __restrict__ H) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(n);
  check_kernel_arg(k);
  check_kernel_arg(sij);
  check_kernel_arg(H);
#endif
  size_t total_elements = (size_t)n * k;
  if (total_elements == 0) return;
  int blockSize = 256;
  int gridSize = (total_elements + blockSize - 1) / blockSize;
  apply_cutoff_forward_only_kernel<<<gridSize, blockSize>>>(total_elements, k,
                                                            H, sij);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
__global__ void calc_Up_kernel(size_t total_elements, int k,
                               const Scalar* __restrict__ sij,
                               const Scalar* __restrict__ U,
                               Scalar* __restrict__ Up,
                               Scalar* __restrict__ dHdr) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_elements) {
    // Map global index 'i' (of U) to index of 'sij'.
    // U layout: [k_0, k_1...k_n, k_0, k_1...]
    // dHdr layout: same as U
    // sij layout: [val_0,       val_1...     ]
    // Every 'k' elements in U share 1 element in sij.
    size_t sij_idx = i / k;
    Up[i] = U[i] * sij[sij_idx];
    dHdr[i] = U[i] * dHdr[i];
  }
}

template <typename Scalar>
__global__ void calc_Up_kernel(size_t total_elements, int k,
                               const Scalar* __restrict__ sij,
                               Scalar* __restrict__ U,
                               Scalar* __restrict__ dHdr) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_elements) {
    size_t sij_idx = i / k;
    Scalar u_val = U[i];
    U[i] = u_val * sij[sij_idx];
    dHdr[i] = u_val * dHdr[i];
  }
}

template <typename Scalar>
void calc_Up(DeviceArgs<Scalar>& args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.k);
  check_kernel_arg(args.U);
  check_kernel_arg(args.sij);
  check_kernel_arg(args.Up);
  check_kernel_arg(args.dHdr);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  size_t total_elements = (size_t)k * c * b * a;

  int blockSize = 256;
  int gridSize = (total_elements + blockSize - 1) / blockSize;

  // Launch Kernel
#if defined(TENSORMD_DEBUG)
  // In the debug mode, every tensor has its own memory.
  calc_Up_kernel<<<gridSize, blockSize>>>(total_elements, k, args.sij, args.U,
                                          args.Up, args.dHdr);
#else
  // In the release mode, H, U and Up shares the same memory.
  // We only pass U to CUDA and modify it in-place.
  calc_Up_kernel<<<gridSize, blockSize>>>(total_elements, k, args.sij, args.U,
                                          args.dHdr);
#endif
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
__global__ void calc_Fr_kernel(size_t num_tuples,  // Size of c * b * a
                               int k, const Scalar* __restrict__ dHdr,
                               const Scalar* __restrict__ dHdrp,
                               const Scalar* __restrict__ drdrx,
                               Scalar* __restrict__ Fr) {
  // One thread handles one (c, b, a) tuple
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_tuples) {
    // Calculate Dot Product (Sum over k)
    size_t k_offset = idx * k;

    // dHdr is already scaled by U in calc_Up, so we can directly sum it.
    Scalar sum_val = 0;
    for (int i = 0; i < k; ++i) {
      sum_val += dHdr[k_offset + i];
    }

    // Add dHdrp
    // dHdrp has size (1, c, b, a), so it maps 1:1 with idx
    sum_val += dHdrp[idx];

    // Broadcast and Multiply for Fr
    // Fr and drdrx have size (3, c, b, a).
    // 3 is the fastest dimension. Start index is idx * 3.
    size_t vec_offset = idx * 3;

    Fr[vec_offset + 0] = sum_val * drdrx[vec_offset + 0];
    Fr[vec_offset + 1] = sum_val * drdrx[vec_offset + 1];
    Fr[vec_offset + 2] = sum_val * drdrx[vec_offset + 2];
  }
}

template <typename Scalar>
void calc_Fr(DeviceArgs<Scalar>& args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.k);
  check_kernel_arg(args.dHdr);
  check_kernel_arg(args.dHdrp);
  check_kernel_arg(args.drdrx);
  check_kernel_arg(args.Fr);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int k = args.k;
  // Number of (c, b, a) tuples
  size_t num_tuples = (size_t)c * b * a;
  if (num_tuples == 0) return;

  int blockSize = 256;
  int gridSize = (num_tuples + blockSize - 1) / blockSize;

  calc_Fr_kernel<<<gridSize, blockSize>>>(num_tuples, k, args.dHdr, args.dHdrp,
                                          args.drdrx, args.Fr);
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

}  // namespace TENSORMD::Kernels::Encoder::GPU
