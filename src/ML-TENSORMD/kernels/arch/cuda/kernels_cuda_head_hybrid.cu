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

#include <cuda_pipeline_primitives.h>

#include <map>
#include <stdexcept>

#include "../../../tensormd_constants.h"
#include "../kernels_activation.hpp"
#include "../kernels_internal_head.h"
#include "../kernels_internal_utils.h"
#include "kernels_cuda_head_helpers.cuh"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::Head::GPU {

using Kernels::GPU::handle;

// ============================================================================
// Forward & Backward Inline Kernels
// ============================================================================

template <int Activation>
__device__ __forceinline__ void forward_layer(
    const float (*W)[65],         // [64][65] padded row-major in shared
    const float* __restrict__ B,  // [64]
    float (*s_Act)[65],           // [64][65], each row = one atom, padded
    float dy_out[8][2], const int r_base, const int lane) {
  float next_A[8][2];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    next_A[i][0] = B[lane];
    next_A[i][1] = B[lane + 32];
  }

#pragma unroll 8
  for (int step = 0; step < 64; ++step) {
    // Forward GEMV: next = A * W + b
    const float w1 = W[step][lane];
    const float w2 = W[step][lane + 32];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a_val = s_Act[r_base + i][step];
      next_A[i][0] += a_val * w1;
      next_A[i][1] += a_val * w2;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    compute_activation<float, Activation>(next_A[i][0], next_A[i][0],
                                          dy_out[i][0]);
    compute_activation<float, Activation>(next_A[i][1], next_A[i][1],
                                          dy_out[i][1]);

    // Residual: y = act(Wx+b) + x
    next_A[i][0] += s_Act[r_base + i][lane];
    next_A[i][1] += s_Act[r_base + i][lane + 32];

    s_Act[r_base + i][lane] = next_A[i][0];
    s_Act[r_base + i][lane + 32] = next_A[i][1];
  }
}

__device__ __forceinline__ void backward_layer(
    const float (*W)[65],       // [64][65] padded row-major in shared
    const float dy_curr[8][2],  // derivative of current layer activation
    const float my_dA_curr[8][2], float dA_prev[8][2], float my_dZ[8][2],
    float (*s_dZ)[65],  // current dZ buffer in shared, [64][65]
    const int r_base, const int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Residual path contributes identity
    dA_prev[i][0] = my_dA_curr[i][0];
    dA_prev[i][1] = my_dA_curr[i][1];
  }

#pragma unroll 8
  for (int step = 0; step < 64; ++step) {
    // dA_prev += dZ_curr * W^T
    // For each thread's two output features (lane, lane+32), read down a column
    const float w1 = W[lane][step];
    const float w2 = W[lane + 32][step];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float dz_val = s_dZ[r_base + i][step];
      dA_prev[i][0] += dz_val * w1;
      dA_prev[i][1] += dz_val * w2;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dZ[i][0] = dA_prev[i][0] * dy_curr[i][0];
    my_dZ[i][1] = dA_prev[i][1] * dy_curr[i][1];

    s_dZ[r_base + i][lane] = my_dZ[i][0];
    s_dZ[r_base + i][lane + 32] = my_dZ[i][1];
  }
}

// ============================================================================
// Block-Tiled Fused MLP Kernel (FP32 SIMT, width=64, 3/4 hidden layers total)
// - 1 block = 256 threads = 8 warps
// - 1 block processes 64 atoms
// - 1 warp processes 8 atoms
// - each lane owns 2 features
//
// x_in and dEdx are allowed to alias in-place because:
// 1. each block owns a disjoint atom tile,
// 2. all x_in reads occur before any dEdx writes,
// 3. x_in is never reread after forward initialization.
// ============================================================================

template <int N_TYPES, int NHiddenLayers, int Activation>
__global__ __launch_bounds__(256, 2) void fused_mlp_block_tiled_fp32(
    int A,
    float* x_in,                         // [A, 64]
    const int* __restrict__ atom_types,  // [A]
    const float* __restrict__ W1,        // [64, 64]
    const float* __restrict__ W2,        // [64, 64]
    const float* __restrict__ W3,        // [64, 64] or [64] output vector
    const float* __restrict__ W4,        // [64], only used when NHiddenLayers=4
    const float* __restrict__ B0,        // [N_TYPES, 64]
    const float* __restrict__ B1,        // [64]
    const float* __restrict__ B2,        // [64]
    const float* __restrict__ B3,        // [64] or [N_TYPES] output bias
    const float* __restrict__ B4,        // [N_TYPES], only used when NHiddenLayers=4
    float* dEdx,                         // [A, 64]
    float* __restrict__ eatoms) {        // [A, 1]
  static_assert(NHiddenLayers == 3 || NHiddenLayers == 4,
                "Hybrid fused kernel supports only 3 or 4 hidden layers.");

  const int tid = threadIdx.x;
  const int wid = tid >> 5;    // warp 0..7
  const int lane = tid & 31;   // lane 0..31
  const int r_base = wid * 8;  // this warp's 8 atoms
  const int block_atom_base = blockIdx.x * 64;

  // --------------------------------------------------------------------------
  // Shared memory layout
  // --------------------------------------------------------------------------
  extern __shared__ float s_data[];
  float* smem = s_data;

  float (*s_W1)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;
  float (*s_W2)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;
  float (*s_W3)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;

  float (*s_Act)[65] = reinterpret_cast<float (*)[65]>(smem);  // [64][65]
  smem += 64 * 65;

  float* s_W4 = smem;
  smem += 64;

  float (*s_B0)[65] = reinterpret_cast<float (*)[65]>(smem);  // [N_TYPES][65]
  smem += N_TYPES * 65;

  float* s_B1 = smem;
  smem += 64;
  float* s_B2 = smem;
  smem += 64;
  float* s_B3 = smem;
  smem += 64;
  float* s_B4 = smem;
  smem += N_TYPES;

  // --------------------------------------------------------------------------
  // Copy weights/biases to shared memory
  // --------------------------------------------------------------------------
  for (int i = tid; i < 64 * 64; i += 256) {
    __pipeline_memcpy_async(&s_W1[i / 64][i % 64], &W1[i], sizeof(float));
    __pipeline_memcpy_async(&s_W2[i / 64][i % 64], &W2[i], sizeof(float));
    if constexpr (NHiddenLayers == 4) {
      __pipeline_memcpy_async(&s_W3[i / 64][i % 64], &W3[i], sizeof(float));
    }
  }

  for (int i = tid; i < N_TYPES * 64; i += 256) {
    __pipeline_memcpy_async(&s_B0[i / 64][i % 64], &B0[i], sizeof(float));
  }

  if (tid < 64) {
    __pipeline_memcpy_async(&s_B1[tid], &B1[tid], sizeof(float));
    __pipeline_memcpy_async(&s_B2[tid], &B2[tid], sizeof(float));
    if constexpr (NHiddenLayers == 4) {
      __pipeline_memcpy_async(&s_W4[tid], &W4[tid], sizeof(float));
      __pipeline_memcpy_async(&s_B3[tid], &B3[tid], sizeof(float));
    } else {
      __pipeline_memcpy_async(&s_W4[tid], &W3[tid], sizeof(float));
    }
  }

  if (tid < N_TYPES) {
    if constexpr (NHiddenLayers == 4) {
      __pipeline_memcpy_async(&s_B4[tid], &B4[tid], sizeof(float));
    } else {
      __pipeline_memcpy_async(&s_B4[tid], &B3[tid], sizeof(float));
    }
  }

  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // --------------------------------------------------------------------------
  // Forward caches
  // --------------------------------------------------------------------------
  float dy1[8][2];
  float dy2[8][2];
  float dy3[8][2];
  float dy4[8][2];

  // --------------------------------------------------------------------------
  // Layer 0: A1 = act(x_in + B0[type])
  // --------------------------------------------------------------------------
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int a_idx = block_atom_base + r_base + i;

    float v0 = 0.0f;
    float v1 = 0.0f;

    if (a_idx < A) {
      const int t = atom_types[a_idx];
      const float* x_ptr = x_in + a_idx * 64;

      v0 = x_ptr[lane] + s_B0[t][lane];
      v1 = x_ptr[lane + 32] + s_B0[t][lane + 32];
    }

    compute_activation<float, Activation>(v0, v0, dy1[i][0]);
    compute_activation<float, Activation>(v1, v1, dy1[i][1]);

    s_Act[r_base + i][lane] = v0;
    s_Act[r_base + i][lane + 32] = v1;
  }
  __syncwarp();

  // --------------------------------------------------------------------------
  // Hidden residual stack
  // --------------------------------------------------------------------------
  forward_layer<Activation>(s_W1, s_B1, s_Act, dy2, r_base, lane);
  __syncwarp();

  forward_layer<Activation>(s_W2, s_B2, s_Act, dy3, r_base, lane);
  __syncwarp();

  if constexpr (NHiddenLayers == 4) {
    forward_layer<Activation>(s_W3, s_B3, s_Act, dy4, r_base, lane);
    __syncwarp();
  }

  // --------------------------------------------------------------------------
  // Output layer: E = A_hidden dot W_out + B_out[type]
  // --------------------------------------------------------------------------
  float my_E[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_E[i] = s_Act[r_base + i][lane] * s_W4[lane] +
              s_Act[r_base + i][lane + 32] * s_W4[lane + 32];
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      my_E[i] += __shfl_down_sync(FULL_MASK, my_E[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int a_idx = block_atom_base + r_base + i;
      if (a_idx < A) {
        const float e_val = my_E[i] + s_B4[atom_types[a_idx]];
        if (eatoms != nullptr) eatoms[a_idx] = e_val;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Backward base:
  // dA_out = W_out
  // dZ_out = dA_out * dy_out
  // --------------------------------------------------------------------------
  float my_dA_curr[8][2];
  float my_dZ[8][2];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dA_curr[i][0] = s_W4[lane];
    my_dA_curr[i][1] = s_W4[lane + 32];

    if constexpr (NHiddenLayers == 4) {
      my_dZ[i][0] = my_dA_curr[i][0] * dy4[i][0];
      my_dZ[i][1] = my_dA_curr[i][1] * dy4[i][1];
    } else {
      my_dZ[i][0] = my_dA_curr[i][0] * dy3[i][0];
      my_dZ[i][1] = my_dA_curr[i][1] * dy3[i][1];
    }

    s_Act[r_base + i][lane] = my_dZ[i][0];
    s_Act[r_base + i][lane + 32] = my_dZ[i][1];
  }
  __syncwarp();

  // --------------------------------------------------------------------------
  // Backward through residual stack
  // --------------------------------------------------------------------------
  float dA_prev[8][2];

  if constexpr (NHiddenLayers == 4) {
    backward_layer(s_W3, dy3, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);
    __syncwarp();

  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      my_dA_curr[i][0] = dA_prev[i][0];
      my_dA_curr[i][1] = dA_prev[i][1];
    }
  }

  backward_layer(s_W2, dy2, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);
  __syncwarp();

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dA_curr[i][0] = dA_prev[i][0];
    my_dA_curr[i][1] = dA_prev[i][1];
  }

  backward_layer(s_W1, dy1, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);
  __syncwarp();

  // --------------------------------------------------------------------------
  // Final writeback: dEdx = dX = dA1 * dy1, already stored in my_dZ
  // --------------------------------------------------------------------------
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int a_idx = block_atom_base + r_base + i;
    if (a_idx < A) {
      float* dx_ptr = dEdx + a_idx * 64;
      dx_ptr[lane] = my_dZ[i][0];
      dx_ptr[lane + 32] = my_dZ[i][1];
    }
  }
}

// d_x_in is allowed to alias with grad_out for in-place backward.
template <typename Scalar>
void nnp_hybrid(int M, int n_layers, std::vector<int>& full_sizes, int act,
                bool use_resnet_dt, bool apply_output_bias, bool sum_output,
                Scalar** d_weights, Scalar** d_weights_T, Scalar** d_biases,
                Scalar* d_x_in, int eff_ndim_in, const int* d_tlist,
                Scalar* d_total, Scalar* d_outputs, Scalar& h_total,
                Scalar* h_outputs, Scalar* grad_out, Scalar* d_ping,
                bool use_W_T) {
  Utils::GPU::timer_start(Profiling::TimedKernel::head_hybrid);
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  double FLOPs = 0.0;
  const double Mf = static_cast<double>(M);  // Avoid overflow

  // --- The first GEMM call ---

  int K = eff_ndim_in;    // Input dim
  int N = full_sizes[1];  // Output dim
  if (n_layers != 4 && n_layers != 5) {
    throw std::runtime_error(
        "Hybrid CUDA head kernel supports only 3 or 4 hidden layers "
        "(n_layers=4 or 5).");
  }
  if (N != 64 || full_sizes[n_layers] != 1) {
    throw std::runtime_error(
        "Hybrid CUDA head kernel currently requires hidden width 64 and "
        "output width 1.");
  }
  for (int i = 1; i < n_layers; ++i) {
    if (full_sizes[i] != 64) {
      throw std::runtime_error(
          "Hybrid CUDA head kernel currently requires all hidden layers to "
          "have width 64.");
    }
  }
  Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           d_weights[0], N, d_x_in, K, &beta, d_ping, N);
  FLOPs += 2.0 * Mf * N * K;

  // --- The fusion kernel ---
  if constexpr (std::is_same_v<Scalar, float>) {
    const int N_TYPES = 16;  // Assuming max 16 atom types for bias lookup
    const int shared_mem_bytes =
        (64 * 65 * 3 + 64 * 65 + 64 + N_TYPES * 65 + 64 * 3 + N_TYPES) *
        sizeof(float);
    int grid_size = (M + 63) / 64;
    int block_size = 256;
#define LAUNCH_FUSED_KERNEL(N_HIDDEN, ACTIVATION)                                 \
  cudaFuncSetAttribute(                                                           \
      fused_mlp_block_tiled_fp32<N_TYPES, N_HIDDEN, ACTIVATION>,                 \
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);            \
  fused_mlp_block_tiled_fp32<N_TYPES, N_HIDDEN, ACTIVATION>                      \
      <<<grid_size, block_size, shared_mem_bytes>>>(                             \
          M, d_ping, d_tlist, d_weights[1], d_weights[2], d_weights[3],          \
          (n_layers == 5) ? d_weights[4] : nullptr, d_biases[0], d_biases[1],    \
          d_biases[2], d_biases[3], (n_layers == 5) ? d_biases[4] : nullptr,     \
          d_ping, d_outputs);
    switch (n_layers) {
      case 4:
        switch (act) {
          case 1:
            LAUNCH_FUSED_KERNEL(3, 1);
            break;
          case 4:
            LAUNCH_FUSED_KERNEL(3, 4);
            break;
          default:
            throw std::runtime_error(
                "Unsupported activation for fused head kernel");
        }
        break;
      case 5:
        switch (act) {
          case 1:
            LAUNCH_FUSED_KERNEL(4, 1);
            break;
          case 4:
            LAUNCH_FUSED_KERNEL(4, 4);
            break;
          default:
            throw std::runtime_error(
                "Unsupported activation for fused head kernel");
        }
        break;
      default:
        throw std::runtime_error(
            "Hybrid CUDA head kernel supports only n_layers=4 or 5.");
    }
    compute_total_energy(d_outputs, d_total, M);
    cudaMemcpy(&h_total, d_total, sizeof(Scalar), cudaMemcpyDeviceToHost);
    if (h_outputs != nullptr) {
      cudaMemcpy(h_outputs, d_outputs, M * sizeof(Scalar),
                 cudaMemcpyDeviceToHost);
    }
#undef LAUNCH_FUSED_KERNEL
  } else {
    throw std::runtime_error("Unsupported data type for fused head kernel");
  }

  const int n_hidden_layers = n_layers - 1;
  const int n_hidden_gemv = n_hidden_layers - 1;
  FLOPs += activation_FLOPs[act] * Mf * 64 * n_hidden_layers;
  FLOPs += Mf * 64 * 64 * 2 * n_hidden_gemv;
  FLOPs += Mf * 64 * n_hidden_layers;
  FLOPs += Mf * 64 * 2;           // output layer dot product + bias add

  // Backward FLOPs (dEdx only, no weight/bias gradients):
  FLOPs += Mf * 64 * 64 * 2 * n_hidden_gemv;
  FLOPs += Mf * 64 * n_hidden_layers;

  // --- The final GEMM call ---
  if (use_W_T) {
    Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha,
                             d_weights_T[0], K, d_ping, N, &beta, grad_out, K);
  } else {
    Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha,
                             d_weights[0], N, d_ping, N, &beta, grad_out, K);
  }
  FLOPs += 2.0 * Mf * N * K;

  Utils::GPU::timer_stop(Profiling::TimedKernel::head_hybrid, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void nnp_hybrid<float>(int M, int n_layers,
                                std::vector<int>& full_sizes, int act,
                                bool use_resnet_dt, bool apply_output_bias,
                                bool sum_output, float** weights,
                                float** weights_T, float** biases, float* x_in,
                                int eff_ndim_in, const int* tlist,
                                float* d_total, float* d_outputs,
                                float& h_total, float* h_outputs,
                                float* grad_out, float* d_ping, bool use_W_T);
template void nnp_hybrid<double>(
    int n_atoms, int n_layers, std::vector<int>& full_sizes, int act,
    bool use_resnet_dt, bool apply_output_bias, bool sum_output,
    double** weights, double** weights_T, double** biases, double* x_in,
    int eff_ndim_in, const int* tlist, double* d_total, double* d_outputs,
    double& h_total, double* h_outputs, double* grad_out, double* d_ping,
    bool use_W_T);

}  // namespace TENSORMD::Kernels::Head::GPU
