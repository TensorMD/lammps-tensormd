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

#include "../../../tensormd_constants.h"
#include "../kernels_internal_utils.h"
#include "../kernels_internal_v2_ops.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::V2::GPU {

// ----------------------------------------------------------------------
// Encoder Expansion
// ----------------------------------------------------------------------
template <typename Scalar>
__global__ void kernel_expand_encoder_input(
    int n, int n_rbf, int embed_dim, const Scalar* __restrict__ R,
    const int* __restrict__ zij, const Scalar* __restrict__ centers,
    Scalar gamma, const Scalar* __restrict__ embeddings,
    Scalar* __restrict__ x_out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

  int row_dim = n_rbf + 2 * embed_dim;
  Scalar* row = x_out + k * row_dim;
  Scalar r_val = R[k];

  for (int i = 0; i < n_rbf; ++i) {
    Scalar diff = r_val - centers[i];
    row[i] = exp(-gamma * diff * diff);
  }

  int Zi = zij[k * 2 + 0];
  int Zj = zij[k * 2 + 1];

  const Scalar* emb_i = embeddings + Zi * embed_dim;
  const Scalar* emb_j = embeddings + Zj * embed_dim;

  for (int i = 0; i < embed_dim; ++i) {
    row[n_rbf + i] = emb_i[i];
    row[n_rbf + embed_dim + i] = emb_j[i];
  }
}

template <typename Scalar>
void expand_encoder_input(int n, int n_rbf, int embed_dim, const Scalar* R,
                          const int* zij, const Scalar* rbf_centers,
                          Scalar rbf_gamma, const Scalar* embeddings,
                          Scalar* x_out) {
  Utils::GPU::timer_start(Profiling::TimedKernel::v2_embed_input);
  double FLOPs = 0.0;
  if (n > 0) {
    int grid = (n + 255) / 256;
    kernel_expand_encoder_input<<<grid, 256>>>(
        n, n_rbf, embed_dim, R, zij, rbf_centers, rbf_gamma, embeddings, x_out);
    CHECK(cudaGetLastError());
    FLOPs = 5.0 * n * n_rbf;
  }
  Utils::GPU::timer_stop(Profiling::TimedKernel::v2_embed_input, FLOPs);
}

// ----------------------------------------------------------------------
// Encoder Backward
// ----------------------------------------------------------------------
template <typename Scalar>
__global__ void kernel_expand_encoder_grad(int n, int n_rbf, int n_in_mlp,
                                           const Scalar* __restrict__ grad_in,
                                           const Scalar* __restrict__ R,
                                           const Scalar* __restrict__ centers,
                                           Scalar gamma,
                                           Scalar* __restrict__ grad_R) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

  Scalar r_val = R[k];
  Scalar sum_grad = 0.0;
  const Scalar* grad_row = grad_in + k * n_in_mlp;

  for (int i = 0; i < n_rbf; ++i) {
    Scalar diff = r_val - centers[i];
    // d(exp(-g * (r-c)^2))/dr = exp(...) * (-2 * g * (r-c))
    // We recompute exp here. Alternatively, we could cache the forward RBF
    // output. Recomputing is usually less memory bandwidth bound.
    Scalar rbf = exp(-gamma * diff * diff);
    Scalar d_rbf = -2.0 * rbf * gamma * diff;

    sum_grad += d_rbf * grad_row[i];
  }
  // Gradients w.r.t embeddings are 0 for forces (embeddings are constant)
  grad_R[k] = sum_grad;
}

template <typename Scalar>
void expand_encoder_grad(int n, int n_rbf, int n_in_mlp, const Scalar* grad_in,
                         const Scalar* R, const Scalar* rbf_centers,
                         Scalar rbf_gamma, Scalar* grad_R) {
  Utils::GPU::timer_start(Profiling::TimedKernel::v2_embed_grad);
  double FLOPs = 0.0;
  if (n > 0) {
    int grid = (n + 255) / 256;
    kernel_expand_encoder_grad<<<grid, 256>>>(n, n_rbf, n_in_mlp, grad_in, R,
                                              rbf_centers, rbf_gamma, grad_R);
    CHECK(cudaGetLastError());
    FLOPs = 9.0 * n * n_rbf;
  }
  Utils::GPU::timer_stop(Profiling::TimedKernel::v2_embed_grad, FLOPs);
}

// ----------------------------------------------------------------------
// Energy Head First Layer Embedded Bias
// ----------------------------------------------------------------------

template <typename Scalar>
__global__ void compute_first_embedding_kernel(
    int n_types, int h_dim, int embed_dim, int N, const Scalar* embeddings,
    const Scalar* weights0, const Scalar* bias0, Scalar* embed_biases) {
  // Total number of elements we need to compute is (n_types * N)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = n_types * N;

  if (idx < total_elements) {
    int t = idx / N;  // Atom type index
    int j = idx % N;  // Neuron index in the first hidden layer

    // Initialize with the standard bias for this neuron
    Scalar val = bias0[j];

    // Add the dot product of the type embedding and the corresponding weights
    for (int e = 0; e < embed_dim; ++e) {
      // weights0 is row-major: shape[(h_dim + embed_dim), N]
      // We skip the first h_dim rows and index into the embedding rows.
      val += embeddings[t * embed_dim + e] * weights0[(h_dim + e) * N + j];
    }

    // Write the precomputed bias for atom type `t` and neuron `j`
    embed_biases[idx] = val;
  }
}

template <typename Scalar>
void compute_first_embedding(int n_types, int h_dim, int embed_dim, int N,
                             const Scalar* d_embeddings,
                             const Scalar* d_weights0, const Scalar* d_bias0,
                             Scalar* d_embed_biases) {
  int total_elements = n_types * N;
  int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  // Launch the kernel on the specified stream
  compute_first_embedding_kernel<Scalar><<<blocks, threads_per_block>>>(
      n_types, h_dim, embed_dim, N, d_embeddings, d_weights0, d_bias0,
      d_embed_biases);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void expand_encoder_input<float>(int n, int n_rbf, int embed_dim,
                                          const float* R, const int* zij,
                                          const float* rbf_centers,
                                          float rbf_gamma,
                                          const float* embeddings,
                                          float* x_out);
template void expand_encoder_input<double>(int n, int n_rbf, int embed_dim,
                                           const double* R, const int* zij,
                                           const double* rbf_centers,
                                           double rbf_gamma,
                                           const double* embeddings,
                                           double* x_out);

template void expand_encoder_grad<float>(int n, int n_rbf, int n_in_mlp,
                                         const float* grad_in, const float* R,
                                         const float* rbf_centers,
                                         float rbf_gamma, float* grad_R);
template void expand_encoder_grad<double>(int n, int n_rbf, int n_in_mlp,
                                          const double* grad_in,
                                          const double* R,
                                          const double* rbf_centers,
                                          double rbf_gamma, double* grad_R);

template void compute_first_embedding<float>(
    int n_types, int h_dim, int embed_dim, int N, const float* embeddings,
    const float* weights0, const float* bias0, float* embed_biases);
template void compute_first_embedding<double>(
    int n_types, int h_dim, int embed_dim, int N, const double* embeddings,
    const double* weights0, const double* bias0, double* embed_biases);

}  // namespace TENSORMD::Kernels::V2::GPU
