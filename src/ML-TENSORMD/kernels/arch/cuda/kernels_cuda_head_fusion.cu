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

#include "../../../tensormd_constants.h"
#include "../kernels_internal_head.h"
#include "../kernels_internal_types.h"
#include "../kernels_internal_utils.h"
#include "../kernels_activation.hpp"

namespace TENSORMD::Kernels::Head::GPU {

#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

// ============================================================================
// Forward & Backward Inline Kernels
// ============================================================================
__device__ __forceinline__ void forward_layer(
    const float (*W)[65], const float* B, float (*s_Act)[65],
    float dy_out[8][2], const int r_base, const int lane) {
  float next_A[8][2];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    next_A[i][0] = B[lane];
    next_A[i][1] = B[lane + 32];
  }

  #pragma unroll 8
  for (int step = 0; step < 64; ++step) {
    // Stride-1 read: Adjacent threads read adjacent floats. Zero conflicts.
    float b1 = W[step][lane];
    float b2 = W[step][lane + 32];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // Broadcast read across the warp. Zero conflicts.
      float a_val = s_Act[r_base + i][step];
      next_A[i][0] += a_val * b1;
      next_A[i][1] += a_val * b2;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    compute_activation<float, 1>(next_A[i][0], next_A[i][0], dy_out[i][0]);
    compute_activation<float, 1>(next_A[i][1], next_A[i][1], dy_out[i][1]);

    // Add Residual from previous activation state
    next_A[i][0] += s_Act[r_base + i][lane];
    next_A[i][1] += s_Act[r_base + i][lane + 32];

    // Write the new output to the Activation Buffer
    s_Act[r_base + i][lane] = next_A[i][0];
    s_Act[r_base + i][lane + 32] = next_A[i][1];
  }
}

__device__ __forceinline__ void backward_layer(
    const float (*W)[65], const float dy_curr[8][2],
    const float my_dA_curr[8][2], float dA_prev[8][2], float my_dZ[8][2],
    float (*s_Act)[65], const int r_base, const int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Chain rule base: Pass the residual gradient directly backwards
    dA_prev[i][0] = my_dA_curr[i][0];
    dA_prev[i][1] = my_dA_curr[i][1];
  }

#pragma unroll 8
  for (int step = 0; step < 64; ++step) {
    // Transposed Matrix Multiply W^T.
    // Reading down columns. Padding of 65 guarantees zero bank conflicts!
    float w1 = W[lane][step];
    float w2 = W[lane + 32][step];

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float dz_val = s_Act[r_base + i][step];
      dA_prev[i][0] += dz_val * w1;
      dA_prev[i][1] += dz_val * w2;
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // Apply activation derivative
    my_dZ[i][0] = dA_prev[i][0] * dy_curr[i][0];
    my_dZ[i][1] = dA_prev[i][1] * dy_curr[i][1];

    s_Act[r_base + i][lane] = my_dZ[i][0];
    s_Act[r_base + i][lane + 32] = my_dZ[i][1];
  }
}

// ============================================================================
// Block-Tiled Fused MLP Kernel (FP32 SIMT)
// 1 Block = 256 threads, processing a 64-Atom Tile
// Each thread computes a 4x4 matrix tile
// ============================================================================
template <int N_TYPES>
__global__ __launch_bounds__(256, 2) void fused_mlp_block_tiled_fp32(
    int A,
    float* G,                            // [A, 256] (Will be overwritten by dG)
    const int* __restrict__ atom_types,  // [A]
    const float* __restrict__ W0,        // [256, 64]
    const float* __restrict__ W1,        // [64, 64]
    const float* __restrict__ W2,        // [64, 64]
    const float* __restrict__ W3,        // [64, 64]
    const float* __restrict__ W4,        //[64]
    const float* __restrict__ B0,        // [N_TYPES, 64]
    const float* __restrict__ B1,        // [64]
    const float* __restrict__ B2,        // [64]
    const float* __restrict__ B3,        //[64]
    const float* __restrict__ B4,        // [N_TYPES]
    const float* __restrict__ W0T,       // [64, 256]
    float* dG,                           // [A, 256] (Usually aliased to G)
    float* __restrict__ eatoms,          // [A]
    float* __restrict__ etotal) {        // scalar or nullptr

  const int tid = threadIdx.x;

  // 8x2 Thread Mapping: 8 Warps per block
  const int wid = tid / 32;    // Warp ID: 0..7
  const int lane = tid % 32;   // Lane ID: 0..31
  const int r_base = wid * 8;  // Each warp handles 8 atoms
  const int block_atom_base = blockIdx.x * 64;

  // --------------------------------------------------------------------------
  // Shared Memory Layout Mapping
  // --------------------------------------------------------------------------
  extern __shared__ float s_data[];
  float* smem = s_data;

  float (*s_W1)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;
  float (*s_W2)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;
  float (*s_W3)[65] = reinterpret_cast<float (*)[65]>(smem);
  smem += 64 * 65;

  // Union memory: Used for L0 Global Loads AND later as Activation Buffer
  float* s_scratch = smem;
  smem += 4192;
  float (*s_Act)[65] =
      reinterpret_cast<float (*)[65]>(s_scratch);  // [64][65] = 4160
  float (*s_G_chunk)[33] =
      reinterpret_cast<float (*)[33]>(s_scratch);  // [64][33] = 2112
  float (*s_W0_chunk)[65] =
      reinterpret_cast<float (*)[65]>(s_scratch + 64 * 33);  // [32][65] = 2080

  float* s_W4 = smem;
  smem += 64;
  float (*s_B0)[65] = reinterpret_cast<float (*)[65]>(smem);
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
  // Async Copy Weights & Biases to SMEM
  // --------------------------------------------------------------------------
  for (int i = tid; i < 64 * 64; i += 256) {
    __pipeline_memcpy_async(&s_W1[i / 64][i % 64], &W1[i], sizeof(float));
    __pipeline_memcpy_async(&s_W2[i / 64][i % 64], &W2[i], sizeof(float));
    __pipeline_memcpy_async(&s_W3[i / 64][i % 64], &W3[i], sizeof(float));
  }
  for (int i = tid; i < N_TYPES * 64; i += 256) {
    __pipeline_memcpy_async(&s_B0[i / 64][i % 64], &B0[i], sizeof(float));
  }
  if (tid < 64) {
    __pipeline_memcpy_async(&s_W4[tid], &W4[tid], sizeof(float));
    __pipeline_memcpy_async(&s_B1[tid], &B1[tid], sizeof(float));
    __pipeline_memcpy_async(&s_B2[tid], &B2[tid], sizeof(float));
    __pipeline_memcpy_async(&s_B3[tid], &B3[tid], sizeof(float));
  }
  if (tid < N_TYPES) {
    __pipeline_memcpy_async(&s_B4[tid], &B4[tid], sizeof(float));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // --------------------------------------------------------------------------
  // Forward Pass Registers
  // --------------------------------------------------------------------------

  // Chain-rule caches to survive until backward pass
  float dy1[8][2], dy2[8][2], dy3[8][2], dy4[8][2];

  // --------------------------------------------------------------------------
  // Layer 0 Forward: G(64x256) * W0(256x64) + B0 -> A1
  // --------------------------------------------------------------------------
  float my_A[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int a_idx = block_atom_base + r_base + i;
    int type = (a_idx < A) ? atom_types[a_idx] : 0;
    my_A[i][0] = s_B0[type][lane];
    my_A[i][1] = s_B0[type][lane + 32];
  }

  for (int k = 0; k < 256; k += 32) {
#pragma unroll
    for (int idx = tid; idx < 64 * 32; idx += 256) {
      int gr = idx / 32;
      int gc = idx % 32;
      int a_idx = block_atom_base + gr;
      s_G_chunk[gr][gc] = (a_idx < A) ? G[a_idx * 256 + k + gc] : 0.0f;
    }
#pragma unroll
    for (int idx = tid; idx < 32 * 64; idx += 256) {
      int wr = idx / 64;
      int wc = idx % 64;
      s_W0_chunk[wr][wc] = W0[(k + wr) * 64 + wc];
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < 32; ++step) {
      float b1 = s_W0_chunk[step][lane];
      float b2 = s_W0_chunk[step][lane + 32];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float g_val = s_G_chunk[r_base + i][step];
        my_A[i][0] += g_val * b1;
        my_A[i][1] += g_val * b2;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    compute_activation<float, 1>(my_A[i][0], my_A[i][0], dy1[i][0]);
    compute_activation<float, 1>(my_A[i][1], my_A[i][1], dy1[i][1]);
    s_Act[r_base + i][lane] = my_A[i][0];
    s_Act[r_base + i][lane + 32] = my_A[i][1];
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Forward
  // --------------------------------------------------------------------------
  forward_layer(s_W1, s_B1, s_Act, dy2, r_base, lane);
  forward_layer(s_W2, s_B2, s_Act, dy3, r_base, lane);
  forward_layer(s_W3, s_B3, s_Act, dy4, r_base, lane);

  // --------------------------------------------------------------------------
  // Layer 4 Output (A4 -> E)
  // --------------------------------------------------------------------------
  float my_E[8] = {0.0f};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_E[i] += s_Act[r_base + i][lane] * s_W4[lane];
    my_E[i] += s_Act[r_base + i][lane + 32] * s_W4[lane + 32];
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      my_E[i] += __shfl_down_sync(FULL_MASK, my_E[i], offset);
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int a_idx = block_atom_base + r_base + i;
      if (a_idx < A) {
        float e_val = my_E[i] + s_B4[atom_types[a_idx]];
        if (eatoms != nullptr) eatoms[a_idx] = e_val;
        if (etotal != nullptr) atomicAdd(etotal, e_val);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Backward Pass Base
  // dA4 is just W4. Initialize dZ4.
  // --------------------------------------------------------------------------
  float my_dZ[8][2];
  float my_dA_curr[8][2];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dA_curr[i][0] = s_W4[lane];
    my_dA_curr[i][1] = s_W4[lane + 32];

    my_dZ[i][0] = my_dA_curr[i][0] * dy4[i][0];
    my_dZ[i][1] = my_dA_curr[i][1] * dy4[i][1];

    s_Act[r_base + i][lane] = my_dZ[i][0];
    s_Act[r_base + i][lane + 32] = my_dZ[i][1];
  }
  __syncthreads();

  // --------------------------------------------------------------------------
  // Backward Propagation
  // --------------------------------------------------------------------------
  float dA_prev[8][2];

  backward_layer(s_W3, dy3, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dA_curr[i][0] = dA_prev[i][0];
    my_dA_curr[i][1] = dA_prev[i][1];
  }
  backward_layer(s_W2, dy2, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    my_dA_curr[i][0] = dA_prev[i][0];
    my_dA_curr[i][1] = dA_prev[i][1];
  }
  backward_layer(s_W1, dy1, my_dA_curr, dA_prev, my_dZ, s_Act, r_base, lane);

  // --------------------------------------------------------------------------
  // Layer 0 Backward: dG = dZ1 * W0^T
  // dZ1 is currently safely held in s_Act.
  // --------------------------------------------------------------------------
  // We no longer need s_W1, so we map the W0T memory chunk loader over it.
  float (*s_W0T_chunk)[33] = reinterpret_cast<float (*)[33]>(s_W1);

  for (int k = 0; k < 256; k += 32) {
#pragma unroll
    for (int idx = tid; idx < 64 * 32; idx += 256) {
      int wr = idx / 32;
      int wc = idx % 32;
      s_W0T_chunk[wr][wc] = W0T[wr * 256 + k + wc];
    }
    __syncthreads();

    float my_dG[8] = {0.0f};

#pragma unroll
    for (int step = 0; step < 64; ++step) {
      float w_val =
          s_W0T_chunk[step]
                     [lane];  // LANE naturally handles the 32 chunks cleanly!
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float dz_val = s_Act[r_base + i][step];
        my_dG[i] += dz_val * w_val;
      }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int a_idx = block_atom_base + r_base + i;
      if (a_idx < A) {
        dG[a_idx * 256 + k + lane] = my_dG[i];
      }
    }
    __syncthreads();
  }
}

// ============================================================================
// Host Launcher
// ============================================================================
template <int N_TYPES>
void launch_fused_mlp_fw_bw_block_tiled_fp32(
    int A, float* G, const int* atom_types, const float* W0, const float* W1,
    const float* W2, const float* W3, const float* W4, const float* B0,
    const float* B1, const float* B2, const float* B3, const float* B4,
    const float* W0T, float* dG, float* eatoms, float* etotal) {
  const int block_threads = 256;
  const int grid_blocks = (A + 64 - 1) / 64;  // 64 Atoms per Block

  // Precise SMEM calculation
  const size_t smem_floats =
      3 * 64 * 65 +   // s_W1, s_W2, s_W3
      4192 +          // Union scratch (s_Act, s_G_chunk, s_W0_chunk)
      64 +            // s_W4
      N_TYPES * 65 +  // s_B0
      64 * 3 +        // s_B1, s_B2, s_B3
      N_TYPES;        // s_B4

  const size_t smem_bytes = smem_floats * sizeof(float);
  cudaFuncSetAttribute(fused_mlp_block_tiled_fp32<N_TYPES>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_bytes));

  fused_mlp_block_tiled_fp32<N_TYPES>
      <<<grid_blocks, block_threads, smem_bytes>>>(A, G, atom_types, W0, W1, W2,
                                                   W3, W4, B0, B1, B2, B3, B4,
                                                   W0T, dG, eatoms, etotal);
}

template <typename Scalar>
void nnp_fusion_kernel(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::head_fused);
  const int A = args.a;
  Scalar* G = args.G;
  const int* atom_types = args.tlist;
  Scalar** weights = args.weights;
  Scalar** weights_T = args.weights_T;
  Scalar** biases = args.biases;

  if constexpr (std::is_same_v<Scalar, float>) {
    cudaMemset(args.etotal, 0, sizeof(float));
    launch_fused_mlp_fw_bw_block_tiled_fp32<3>(
        A, G, atom_types, weights[0], weights[1], weights[2], weights[3],
        weights[4], biases[0], biases[1], biases[2], biases[3], biases[4],
        weights_T[0], args.G, args.eatoms, args.etotal);
  }
  Utils::GPU::timer_stop(Profiling::TimedKernel::head_fused, 0.0);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void nnp_fusion_kernel<float>(DeviceArgs<float>& args);
template void nnp_fusion_kernel<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Head::GPU
