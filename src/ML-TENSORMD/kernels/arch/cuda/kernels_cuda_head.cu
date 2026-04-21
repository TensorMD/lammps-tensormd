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

// ----------------------------------------------------------------------
// Fused Forward Kernel
// Performs: y = Act(x + bias)
// Stores:   y (output), dy (derivative w.r.t pre-activation)
// Handles:  Residual connection (y += res)
// ----------------------------------------------------------------------

template <typename Scalar, const int Activation>
__global__ void kernel_nnp_fused_forward(
    Scalar* val,  // [Input/Output] In: GEMM result (x), Out: Activated (y)
    const Scalar* bias,  // [Input] Bias vector of size N
    Scalar* deriv,       // [Output] Stored derivative for backward
    const Scalar* res,   // [Input] Residual input (optional)
    int M, int N,        // M = n_atoms, N = layer_width
    bool use_residual) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = M * N;

  for (int i = idx; i < total; i += stride) {
    int col = i % N;  // Bias index

    Scalar x = val[i] + bias[col];  // Apply Bias
    Scalar y, dy;

    compute_activation<Scalar, Activation>(x, y, dy);

    if (use_residual) {
      y += res[i];
    }

    val[i] = y;     // Store Output
    deriv[i] = dy;  // Store Derivative
  }
}

// ----------------------------------------------------------------------
// Output Layer Kernel (No Activation, Just Bias)
// ----------------------------------------------------------------------

template <typename Scalar>
__global__ void kernel_nnp_bias_only(Scalar* val, const Scalar* bias, int M,
                                     int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < M * N; i += stride) {
    val[i] += bias[i % N];
  }
}

template <typename Scalar>
__global__ void kernel_nnp_Ebias_only(Scalar* val, const Scalar* E_bias,
                                      const int* zi, int M) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  int atom_type = zi[row];
  val[row] += E_bias[atom_type];
}

// ----------------------------------------------------------------------
// Fused Forward Kernel with atom-type based bias
// Performs: y = Act(x + bias[t])
// Stores:   y (output), dy (derivative w.r.t pre-activation)
// ----------------------------------------------------------------------

template <typename Scalar, const int Activation>
__global__ void kernel_nnp_fused_forward_Ebias(
    Scalar* val,  // [Input/Output] In: GEMM result (x), Out: Activated (y)
    const Scalar* E_bias,  // [Input] Precomputed bias[n_types * N]
    const int* zi,         // [Input] Atom types [M]
    Scalar* deriv,         // [Output] Stored derivative (can be nullptr)
    int M, int N           // M = n_atoms, N = layer_width
) {
  // One Block processes exactly One Atom
  int row = blockIdx.x;
  if (row >= M) return;

  int atom_type = zi[row];

  // Cache the starting pointer for this atom's bias
  const Scalar* E_bias_row = &E_bias[atom_type * N];

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    int i = row * N + col;

    Scalar x = val[i] + E_bias_row[col];
    Scalar y, dy;

    // Zero divergence, heavily optimized inline math
    compute_activation<Scalar, Activation>(x, y, dy);

    // Store Output
    val[i] = y;

    // Store Derivative conditionally
    // (Safe to pass nullptr for the final output layer)
    if (deriv != nullptr) {
      deriv[i] = dy;
    }
  }
}

// ----------------------------------------------------------------------
// NNP Forward Implementation
// ----------------------------------------------------------------------

inline int get_grid_size(int block_size, int total_size) {
  return (total_size + block_size - 1) / block_size;
}

template <typename Scalar>
void nnp_forward(int n_atoms, int n_layers, std::vector<int>& full_sizes,
                 int act, bool use_resnet_dt, bool apply_output_bias,
                 bool sum_output, Scalar** d_weights, Scalar** d_biases,
                 const Scalar* d_x_in, int eff_ndim_in, const int* tlist,
                 Scalar& h_total, Scalar* h_outputs, Scalar** d_actvals,
                 Scalar* d_buf1, Scalar* d_buf2, TimedKernel kernel) {
  Utils::GPU::timer_start(kernel);
  Scalar* C = d_buf1;
  Scalar* x = const_cast<Scalar*>(d_x_in);
  int M = n_atoms;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  double FLOPs = 0.0;
  int grid_size, total_size, block_size;
  bool use_atom_type_based_bias = false;
  if (eff_ndim_in > 0 && eff_ndim_in < full_sizes[0]) {
    use_atom_type_based_bias = true;
  }

  for (int i = 0; i < n_layers; i++) {
    int K = full_sizes[i];      // Input dim
    int N = full_sizes[i + 1];  // Output dim
    if (i == 0 && use_atom_type_based_bias) {
      K = eff_ndim_in;
    }

    // GEMM: C = alpha * A * B + beta * C
    // Note that cuBLAS uses column-major order, so we treat our row-major
    // data as transposed.
    // d_weights[i] is KxN row-major (NxK col-major)
    // x is MxK row-major (KxM col-major)
    // output C is MxN row-major (NxM col-major)
    Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_weights[i], N, x, K, &beta, C, N);
    FLOPs += 2.0 * M * N * K;

    // Fused Activation + Bias + Residual + Derivative Storage
    if (i < n_layers - 1) {
      bool apply_resnet_dt = i > 0 && use_resnet_dt && K == N;
      if (i == 0 && use_atom_type_based_bias) {
        grid_size = M;
        if (N >= 256) {
          block_size = 256;
        } else {
          block_size = ((N + 31) / 32) * 32;
        }
#define LAUNCH_ACT_KERNEL(act)                \
  kernel_nnp_fused_forward_Ebias<Scalar, act> \
      <<<grid_size, block_size>>>(C, d_biases[i], tlist, d_actvals[i], M, N);
        if (act == 0) {
          LAUNCH_ACT_KERNEL(0);
        } else if (act == 1) {
          LAUNCH_ACT_KERNEL(1);
        } else if (act == 2) {
          LAUNCH_ACT_KERNEL(2);
        } else if (act == 3) {
          LAUNCH_ACT_KERNEL(3);
        } else if (act == 4) {
          LAUNCH_ACT_KERNEL(4);
        } else if (act == 5) {
          LAUNCH_ACT_KERNEL(5);
        }
#undef LAUNCH_ACT_KERNEL
      } else {
        total_size = M * N;
        block_size = 256;
        grid_size = get_grid_size(block_size, total_size);
#define LAUNCH_ACT_KERNEL(act)                                             \
  kernel_nnp_fused_forward<Scalar, act><<<grid_size, block_size>>>(        \
      C, d_biases[i], d_actvals[i], (apply_resnet_dt ? x : nullptr), M, N, \
      apply_resnet_dt);
        if (act == 0) {
          LAUNCH_ACT_KERNEL(0);
        } else if (act == 1) {
          LAUNCH_ACT_KERNEL(1);
        } else if (act == 2) {
          LAUNCH_ACT_KERNEL(2);
        } else if (act == 3) {
          LAUNCH_ACT_KERNEL(3);
        } else if (act == 4) {
          LAUNCH_ACT_KERNEL(4);
        } else if (act == 5) {
          LAUNCH_ACT_KERNEL(5);
        }
#undef LAUNCH_ACT_KERNEL
      }
      // Launch kernel
      // Grid size based on total elements M*N
      FLOPs += (activation_FLOPs[act] + 1.0) * M * N;  // Activation + Bias
      if (apply_resnet_dt) {
        FLOPs += M * N;  // For residual addition in forward pass
      }
    }
    // Output Layer: Bias only (if enabled)
    else if (apply_output_bias) {
      block_size = 256;
      grid_size = get_grid_size(block_size, M * N);
      if (use_atom_type_based_bias) {
        kernel_nnp_Ebias_only<<<grid_size, block_size>>>(C, d_biases[i], tlist,
                                                         M);
      } else {
        kernel_nnp_bias_only<<<grid_size, block_size>>>(C, d_biases[i], M, N);
      }
      FLOPs += M * N;
    }

    // Ping Pong
    if (C == d_buf1) {
      x = d_buf1;
      C = d_buf2;
    } else {
      x = d_buf2;
      C = d_buf1;
    }
  }

  // Store Results
  // Note that 'toten' and 'h_eatom' are host pointers, so we need to copy data
  // back to host.
  total_size = M * full_sizes[n_layers];
  if (h_outputs) {
    CHECK(cudaMemcpy(h_outputs, x, sizeof(Scalar) * total_size,
                     cudaMemcpyDeviceToHost));
  }
  h_total = static_cast<Scalar>(0.0);
  if (sum_output) {
    Scalar* d_toten = nullptr;
    CHECK(cudaMalloc((void**)&d_toten, sizeof(Scalar)));
    CHECK(cudaMemset(d_toten, 0, sizeof(Scalar)));
    compute_total_energy(x, d_toten, M);
    CHECK(
        cudaMemcpy(&h_total, d_toten, sizeof(Scalar), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_toten));
    FLOPs += total_size;  // For reduction
  }
  CHECK(cudaGetLastError());
  Utils::GPU::timer_stop(kernel, FLOPs);
}

// ----------------------------------------------------------------------
// NNP Backward Implementation
// ----------------------------------------------------------------------

template <typename Scalar>
void nnp_backward(int M, int n_layers, std::vector<int>& full_sizes,
                  bool use_resnet_dt, Scalar** weights, const Scalar* grad_in,
                  int eff_ndim_in, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T) {
  Utils::GPU::timer_start(kernel);
  if (eff_ndim_in > 0 && eff_ndim_in > full_sizes[0]) {
    throw std::invalid_argument(
        "eff_ndim_in cannot be greater than input n_in (full_sizes[0] = " +
        std::to_string(full_sizes[0]) + ")");
  }
  Scalar alpha = 1.0;
  double FLOPs = 0.0;
  const int block_size = 256;
  int grid_size, total_size;

  // `curr_delta` is the current upstream gradient for the current layer output.
  // It may alias an external buffer (`grad_in`) or one of our internal
  // ping-pong buffers.
  const Scalar* curr_delta;

  // Track which internal buffer currently holds `curr_delta` when it is
  // internal. 0 = none/external, 1 = d_ping, 2 = d_pong
  int curr_buf_id = 0;

  // Top-level delta:
  // - if grad_in == nullptr, dL/dy = 1 for all outputs
  // - otherwise, use grad_in directly and avoid the initial copy
  {
    const int K_top = full_sizes[n_layers];
    if (grad_in == nullptr) {
      grid_size = get_grid_size(block_size, M * K_top);
      Kernels::GPU::fillKernel<<<grid_size, block_size>>>(
          d_ping, static_cast<Scalar>(1.0), static_cast<size_t>(M * K_top));
      curr_delta = d_ping;
      curr_buf_id = 1;
    } else {
      curr_delta = grad_in;
      curr_buf_id = 0;  // external pointer
    }
  }

  for (int i = n_layers; i > 0; i--) {
    const int N = full_sizes[i - 1];
    const int K = full_sizes[i];

    // For hidden layers, apply activation derivative in-place:
    // d_hidden[i-1] := activation_grad * curr_delta
    const Scalar* this_delta;
    if (i < n_layers) {
      total_size = M * K;
      grid_size = get_grid_size(block_size, total_size);
      Kernels::GPU::vMulKernel<<<grid_size, block_size>>>(
          curr_delta, d_hidden[i - 1], d_hidden[i - 1], total_size);
      this_delta = d_hidden[i - 1];
      FLOPs += 1.0 * M * K;
    } else {
      // Top layer: no activation-gradient multiply here
      this_delta = curr_delta;
    }

    // For the first backprop step into inputs, optionally restrict output width
    // to avoid computing unused embedding grads.
    const int Np = (i == 1 && eff_ndim_in > 0) ? eff_ndim_in : N;

    const bool is_residual =
        use_resnet_dt && (Np == K) && (i > 1) && (i < n_layers);

    Scalar* next_delta = nullptr;
    auto beta = Scalar(0.0);
    int next_buf_id = 0;

    if (i == 1) {
      // Final result goes directly to grad_out
      next_delta = grad_out;
      beta = Scalar(0.0);
      next_buf_id = 0;
    } else if (is_residual) {
      // Residual case: next_delta = this_delta * W^T + curr_delta
      // Reuse the current internal buffer when possible so BLAS does GEMM +
      // AXPY via beta=1.
      //
      // Note: for any residual layer here, curr_delta is guaranteed to come
      // from an internal buffer, because residual is only possible for i <
      // n_layers, and after the first GEMM all deltas are internal.
      if (curr_buf_id == 1) {
        next_delta = d_ping;
        next_buf_id = 1;
      } else if (curr_buf_id == 2) {
        next_delta = d_pong;
        next_buf_id = 2;
      } else {
        throw std::logic_error("GPU::nnp_backward: unexpected curr_buf_id = " +
                               std::to_string(curr_buf_id));
      }
      beta = Scalar(1.0);
    } else {
      // Normal ping-pong
      if (curr_buf_id == 1) {
        next_delta = d_pong;
        next_buf_id = 2;
      } else {
        next_delta = d_ping;
        next_buf_id = 1;
      }
      beta = Scalar(0.0);
    }

    if (use_W_T) {
      // Attention: EnergyHeadV2 only transpose the feature part of W0, so the
      // ldb should be Np (eff_ndim_in) but not the full dimension.
      Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Np, M, K,
                               &alpha, weights[i - 1], Np, this_delta, K, &beta,
                               next_delta, Np);
    } else {
      Linalg::GPU::gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Np, M, K,
                               &alpha, weights[i - 1], K, this_delta, K, &beta,
                               next_delta, Np);
    }

    FLOPs += 2.0 * M * Np * K;
    if (is_residual) {
      FLOPs += 1.0 * M * Np;  // fused accumulation via beta=1
    }

    curr_delta = next_delta;
    curr_buf_id = next_buf_id;
  }
  Utils::GPU::timer_stop(kernel, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void nnp_forward<float>(
    int n_atoms, int n_layers, std::vector<int>& full_sizes, int act,
    bool use_resnet_dt, bool apply_output_bias, bool sum_output,
    float** weights, float** biases, const float* x_in, int eff_ndim_in,
    const int* tlist, float& h_total, float* h_outputs, float** actvals,
    float* buf1, float* buf2, TimedKernel kernel);
template void nnp_forward<double>(
    int n_atoms, int n_layers, std::vector<int>& full_sizes, int act,
    bool use_resnet_dt, bool apply_output_bias, bool sum_output,
    double** weights, double** biases, const double* x_in, int eff_ndim_in,
    const int* tlist, double& h_total, double* h_outputs, double** actvals,
    double* buf1, double* buf2, TimedKernel kernel);

template void nnp_backward<float>(int n_atoms, int n_layers,
                                  std::vector<int>& full_sizes,
                                  bool use_resnet_dt, float** weights,
                                  const float* grad_in, int grad_nout,
                                  float* grad_out, float** actvals, float* buf1,
                                  float* buf2, TimedKernel kernel,
                                  bool use_W_T);
template void nnp_backward<double>(int n_atoms, int n_layers,
                                   std::vector<int>& full_sizes,
                                   bool use_resnet_dt, double** weights,
                                   const double* grad_in, int grad_nout,
                                   double* grad_out, double** actvals,
                                   double* buf1, double* buf2,
                                   TimedKernel kernel, bool use_W_T);

}  // namespace TENSORMD::Kernels::Head::GPU
