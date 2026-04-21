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
#include "../kernels_activation.hpp"
#include "../kernels_internal_head.h"
#include "../kernels_internal_utils.h"
#include "cppblas.hpp"

namespace TENSORMD::Kernels::Head::CPU {

/* ---------------------------------------------------------------------- */

template <typename Scalar, int Activation>
void internal_nnp_forward(int M, int n_layers, std::vector<int>& full_sizes,
                          bool use_resnet_dt, bool apply_output_bias,
                          bool sum_output, Scalar** weights, Scalar** biases,
                          const Scalar* d_x, int eff_ndim_in,
                          const int* d_tlist, Scalar& h_total,
                          Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                          Scalar* d_pong, double& FLOPs) {
  int j;
  Scalar* C = d_ping;
  bool C_is_ping = true;
  auto* x = const_cast<Scalar*>(d_x);
  static const double act_FLOPs[6] = {1, 4, 2, 7, 7, 5};
  FLOPs = 0.0;
  bool use_atom_type_based_bias = false;
  if (eff_ndim_in > 0 && eff_ndim_in < full_sizes[0]) {
    use_atom_type_based_bias = true;
  }

  for (int i = 0; i < n_layers; i++) {
    int K = full_sizes[i];
    int N = full_sizes[i + 1];

    // Special logic to handle for the first x*W
    if (i == 0 && use_atom_type_based_bias) {
      K = eff_ndim_in;
    }
    Linalg::CPU::cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N,
                              K, 1.0, x, K, weights[i], N, 0.0, C, N);
    FLOPs += 2.0 * M * N * K;

    if (i < n_layers - 1 || apply_output_bias) {
      // Special logic to handle the first and last bias_add
      if ((i == 0 || i == n_layers - 1) && use_atom_type_based_bias) {
        // biases[i] length is n_types * full_sizes[i + 1]
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (j = 0; j < M; j++) {
          for (int d = 0; d < N; d++) {
            C[j * N + d] += biases[i][d_tlist[j] * N + d];
          }
        }
      } else {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (j = 0; j < M; j++) {
          Linalg::CPU::cppblas_axpy(N, 1.0, biases[i], 1, &C[j * N], 1);
        }
      }
      FLOPs += M * N;
    }
    if (i < n_layers - 1) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (j = 0; j < M * N; j++) {
        compute_activation<Scalar, Activation>(C[j], C[j], d_hidden[i][j]);
      }
      FLOPs += act_FLOPs[Activation] * M * N;
    } else {
      break;
    }
    if (i > 0 && use_resnet_dt && K == N) {
      Linalg::CPU::cppblas_axpy(M * N, 1.0, x, 1.0, C, 1);
      FLOPs += M * N;
    }
    if (C_is_ping) {
      x = d_ping;
      C = d_pong;
    } else {
      x = d_pong;
      C = d_ping;
    }
    if (i == n_layers - 2 && h_outputs) C = h_outputs;
    C_is_ping = !C_is_ping;
  }
  h_total = 0.0;
  if (sum_output) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (j = 0; j < M * full_sizes[n_layers]; j++) {
      h_total += C[j];
    }
    FLOPs += M * full_sizes[n_layers];
  }
}

template <typename Scalar>
void nnp_forward(int M, int n_layers, std::vector<int>& full_sizes, int act,
                 bool use_resnet_dt, bool apply_output_bias, bool sum_output,
                 Scalar** weights, Scalar** biases, const Scalar* d_x,
                 int eff_ndim_in, const int* d_tlist, Scalar& h_total,
                 Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                 Scalar* d_pong, TimedKernel kernel) {
  Utils::CPU::timer_start(kernel);
  double FLOPs = 0.0;
#define LAUNCH_KERNEL(Activation)                                            \
  internal_nnp_forward<Scalar, Activation>(                                  \
      M, n_layers, full_sizes, use_resnet_dt, apply_output_bias, sum_output, \
      weights, biases, d_x, eff_ndim_in, d_tlist, h_total, h_outputs,        \
      d_hidden, d_ping, d_pong, FLOPs);
  if (act == 0) {
    LAUNCH_KERNEL(0);
  } else if (act == 1) {
    LAUNCH_KERNEL(1);
  } else if (act == 2) {
    LAUNCH_KERNEL(2);
  } else if (act == 3) {
    LAUNCH_KERNEL(3);
  } else if (act == 4) {
    LAUNCH_KERNEL(4);
  } else if (act == 5) {
    LAUNCH_KERNEL(5);
  } else {
    throw std::invalid_argument("Unsupported activation type: " +
                                std::to_string(act));
  }
#undef LAUNCH_KERNEL
  Utils::CPU::timer_stop(kernel, FLOPs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void nnp_backward(int M, int n_layers, std::vector<int>& full_sizes,
                  bool use_resnet_dt, Scalar** weights, const Scalar* grad_in,
                  int eff_ndim_in, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T) {
  if (eff_ndim_in > 0 && eff_ndim_in > full_sizes[0]) {
    throw std::invalid_argument(
        "eff_ndim_in cannot be greater than input n_in (full_sizes[0] = " +
        std::to_string(full_sizes[0]) + ")");
  }

  Utils::CPU::timer_start(kernel);

  const auto alpha = Scalar(1.0);
  double FLOPs = 0.0;

  // `curr_delta` is the current upstream gradient for the current layer output.
  // It may alias an external buffer (`grad_in`) or one of our internal
  // ping-pong buffers.
  const Scalar* curr_delta = nullptr;

  // Track which internal buffer currently holds `curr_delta` when it is
  // internal. 0 = none/external, 1 = d_ping, 2 = d_pong
  int curr_buf_id = 0;

  // Top-level delta:
  // - if grad_in == nullptr, dL/dy = 1 for all outputs
  // - otherwise, use grad_in directly and avoid the initial copy
  {
    const int K_top = full_sizes[n_layers];
    if (grad_in == nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int j = 0; j < M * K_top; ++j) {
        d_ping[j] = Scalar(1.0);
      }
      curr_delta = d_ping;
      curr_buf_id = 1;
    } else {
      curr_delta = grad_in;
      curr_buf_id = 0;  // external pointer
    }
  }

  for (int i = n_layers; i > 0; --i) {
    const int N = full_sizes[i - 1];
    const int K = full_sizes[i];

    // For hidden layers, apply activation derivative in-place:
    // d_hidden[i-1] := activation_grad * curr_delta
    const Scalar* this_delta;
    if (i < n_layers) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int j = 0; j < M * K; ++j) {
        d_hidden[i - 1][j] *= curr_delta[j];
      }
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
        // Defensive fallback: should not normally happen
        next_delta = d_ping;
        next_buf_id = 1;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int j = 0; j < M * Np; ++j) {
          next_delta[j] = curr_delta[j];
        }
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

    // GEMM: next_delta = this_delta * W^T + beta * next_delta
    //
    // Shapes:
    //   this_delta : [M, K]
    //   weights    : [Np, K]   (row-major)
    //   next_delta : [M, Np]

    if (use_W_T) {
      // Attention: EnergyHeadV2 only transpose the feature part of W0, so the
      // ldb should be Np (eff_ndim_in) but not the full dimension.
      Linalg::CPU::cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M,
                                Np, K, alpha, const_cast<Scalar*>(this_delta),
                                K, weights[i - 1], Np, beta, next_delta, Np);
    } else {
      Linalg::CPU::cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, Np,
                                K, alpha, const_cast<Scalar*>(this_delta), K,
                                weights[i - 1], K, beta, next_delta, Np);
    }

    FLOPs += 2.0 * M * Np * K;
    if (is_residual) {
      FLOPs += 1.0 * M * Np;  // fused accumulation via beta=1
    }

    curr_delta = next_delta;
    curr_buf_id = next_buf_id;
  }
  Utils::CPU::timer_stop(kernel, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void nnp_forward<float>(
    int M, int n_layers, std::vector<int>& full_sizes, int act,
    bool use_resnet_dt, bool apply_output_bias, bool sum_output,
    float** weights, float** biases, const float* d_x, int eff_ndim_in,
    const int* d_tlist, float& h_total, float* h_outputs, float** d_hidden,
    float* d_ping, float* d_pong, TimedKernel kernel);
template void nnp_forward<double>(
    int M, int n_layers, std::vector<int>& full_sizes, int act,
    bool use_resnet_dt, bool apply_output_bias, bool sum_output,
    double** weights, double** biases, const double* d_x, int eff_ndim_in,
    const int* d_tlist, double& h_total, double* h_outputs, double** d_hidden,
    double* d_ping, double* d_pong, TimedKernel kernel);

template void nnp_backward<float>(int n_atoms, int n_layers,
                                  std::vector<int>& full_sizes,
                                  bool use_resnet_dt, float** weights,
                                  const float* grad_in, int grad_nout,
                                  float* grad_out, float** d_hidden,
                                  float* d_ping, float* d_pong,
                                  TimedKernel kernel, bool use_W_T);
template void nnp_backward<double>(int n_atoms, int n_layers,
                                   std::vector<int>& full_sizes,
                                   bool use_resnet_dt, double** weights,
                                   const double* grad_in, int grad_nout,
                                   double* grad_out, double** d_hidden,
                                   double* d_ping, double* d_pong,
                                   TimedKernel kernel, bool use_W_T);

}  // namespace TENSORMD::Kernels::Head::CPU
