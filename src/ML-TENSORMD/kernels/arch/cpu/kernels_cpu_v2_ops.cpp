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

#include <cmath>
#include <cstring>

#include "../../../tensormd_constants.h"
#include "../kernels_internal_utils.h"
#include "../kernels_internal_v2_ops.h"

namespace TENSORMD::Kernels::V2::CPU {

// ----------------------------------------------------------------------
// Encoder Expansion
// ----------------------------------------------------------------------

template <typename Scalar>
void expand_encoder_input(int n, int n_rbf, int embed_dim, const Scalar* R,
                          const int* zijlist, const Scalar* rbf_centers,
                          Scalar rbf_gamma, const Scalar* embeddings,
                          Scalar* x_out) {
  Utils::CPU::timer_start(Profiling::TimedKernel::v2_embed_input);
  int row_dim = n_rbf + 2 * embed_dim;
  size_t embed_bytes = embed_dim * sizeof(Scalar);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int k = 0; k < n; ++k) {
    Scalar* row = x_out + k * row_dim;

    // 1. RBF
    Scalar r = R[k];
#pragma omp simd
    for (int i = 0; i < n_rbf; ++i) {
      Scalar delta = r - rbf_centers[i];
      row[i] = std::exp(-rbf_gamma * (delta * delta));
    }

    // 2. Embeddings
    int Zi = zijlist[k * 2 + 0];
    int Zj = zijlist[k * 2 + 1];
    const Scalar* emb_i = embeddings + Zi * embed_dim;
    const Scalar* emb_j = embeddings + Zj * embed_dim;
    std::memcpy(row + n_rbf, emb_i, embed_bytes);
    std::memcpy(row + n_rbf + embed_dim, emb_j, embed_bytes);
  }
  const double FLOPs = 5.0 * n * n_rbf;  // Rough estimate for RBF expansion
  Utils::CPU::timer_stop(Profiling::TimedKernel::v2_embed_input, FLOPs);
}

// ----------------------------------------------------------------------
// Encoder Backward
// ----------------------------------------------------------------------

template <typename Scalar>
void expand_encoder_grad(int n, int n_rbf, int n_in_mlp, const Scalar* grad_in,
                         const Scalar* R, const Scalar* rbf_centers,
                         Scalar rbf_gamma, Scalar* grad_R) {
  Utils::CPU::timer_start(Profiling::TimedKernel::v2_embed_grad);
  // Precompute constants outside the loop
  Scalar two_gamma = -2.0 * rbf_gamma;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int k = 0; k < n; ++k) {
    Scalar r = R[k];
    const Scalar* grad_row = grad_in + k * n_in_mlp;
    Scalar sum_grad = 0.0;

// Vectorize the reduction loop
#pragma omp simd reduction(+ : sum_grad)
    for (int i = 0; i < n_rbf; ++i) {
      Scalar delta = r - rbf_centers[i];
      // Recompute RBF value: exp(-gamma * delta^2)
      Scalar r_expanded_i = std::exp(-rbf_gamma * (delta * delta));

      // df/dr = -2 * gamma * (r - center) * f(r)
      Scalar df = two_gamma * delta * r_expanded_i;
      sum_grad += df * grad_row[i];
    }
    grad_R[k] = sum_grad;
  }
  const double FLOPs = 9.0 * n * n_rbf;  // Rough estimate for RBF gradient
  Utils::CPU::timer_stop(Profiling::TimedKernel::v2_embed_grad, FLOPs);
}

// ----------------------------------------------------------------------
// Atom-type based embedding (bias) for the first layer
// ----------------------------------------------------------------------

template <typename Scalar>
void compute_first_embedding(int n_types, int h_dim, int embed_dim, int N,
                             const Scalar* embeddings, const Scalar* weights,
                             const Scalar* biases, Scalar* out_bias) {
  for (int t = 0; t < n_types; ++t) {
    for (int j = 0; j < N; ++j) {
      Scalar val = biases[j];
      for (int e = 0; e < embed_dim; ++e) {
        // weights is row-major: shape (h_dim + embed_dim) x N
        val += embeddings[t * embed_dim + e] * weights[(h_dim + e) * N + j];
      }
      out_bias[t * N + j] = val;
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void expand_encoder_input<float>(int n, int n_rbf, int embed_dim,
                                          const float* R, const int* zijlist,
                                          const float* rbf_centers,
                                          float rbf_gamma,
                                          const float* embeddings,
                                          float* x_out);
template void expand_encoder_input<double>(int n, int n_rbf, int embed_dim,
                                           const double* R, const int* zijlist,
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
    const float* weights, const float* biases, float* out_bias);
template void compute_first_embedding<double>(
    int n_types, int h_dim, int embed_dim, int N, const double* embeddings,
    const double* weights, const double* biases, double* out_bias);

}  // namespace TENSORMD::Kernels::V2::CPU
