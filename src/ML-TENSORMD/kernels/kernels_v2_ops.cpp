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

#include "kernels_v2_ops.h"

#include "arch/kernels_internal_v2_ops.h"

namespace TENSORMD::Kernels::V2 {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void expand_encoder_input(Device dev, int n, int n_rbf, int embed_dim,
                          const Scalar* R, const int* zij,
                          const Scalar* rbf_centers, Scalar rbf_gamma,
                          const Scalar* embeddings, Scalar* x_out) {
  if (dev == Device::CPU) {
    CPU::expand_encoder_input(n, n_rbf, embed_dim, R, zij, rbf_centers,
                              rbf_gamma, embeddings, x_out);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::expand_encoder_input(n, n_rbf, embed_dim, R, zij, rbf_centers,
                              rbf_gamma, embeddings, x_out);
#endif
  }
}

/* ---------------------------------------------------------------------- */

// Backward: dL/dx_out -> dL/dR
template <typename Scalar>
void expand_encoder_grad(Device dev, int n, int n_rbf, int n_in_mlp,
                         const Scalar* grad_in, const Scalar* R,
                         const Scalar* rbf_centers, Scalar rbf_gamma,
                         Scalar* grad_R) {
  if (dev == Device::CPU) {
    CPU::expand_encoder_grad(n, n_rbf, n_in_mlp, grad_in, R, rbf_centers,
                             rbf_gamma, grad_R);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::expand_encoder_grad(n, n_rbf, n_in_mlp, grad_in, R, rbf_centers,
                             rbf_gamma, grad_R);
#endif
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void compute_first_embedding(Device dev, int n_types, int h_dim, int embed_dim,
                             int N, const Scalar* embeddings,
                             const Scalar* weights, const Scalar* biases,
                             Scalar* out_bias) {
  if (dev == Device::CPU) {
    CPU::compute_first_embedding(n_types, h_dim, embed_dim, N, embeddings,
                                 weights, biases, out_bias);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::compute_first_embedding(n_types, h_dim, embed_dim, N, embeddings,
                                 weights, biases, out_bias);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void expand_encoder_input<float>(
    Device dev, int n, int n_rbf, int embed_dim, const float* R, const int* zij,
    const float* rbf_centers, float rbf_gamma, const float* embeddings,
    float* x_out);
template void expand_encoder_input<double>(
    Device dev, int n, int n_rbf, int embed_dim, const double* R,
    const int* zij, const double* rbf_centers, double rbf_gamma,
    const double* embeddings, double* x_out);

template void expand_encoder_grad<float>(Device dev, int n, int n_rbf,
                                         int n_in_mlp, const float* grad_in,
                                         const float* R,
                                         const float* rbf_centers,
                                         float rbf_gamma, float* grad_R);
template void expand_encoder_grad<double>(Device dev, int n, int n_rbf,
                                          int n_in_mlp, const double* grad_in,
                                          const double* R,
                                          const double* rbf_centers,
                                          double rbf_gamma, double* grad_R);

template void compute_first_embedding<float>(Device dev, int n_types, int h_dim,
                                             int embed_dim, int N,
                                             const float* embeddings,
                                             const float* weights,
                                             const float* biases,
                                             float* out_bias);
template void compute_first_embedding<double>(Device dev, int n_types,
                                              int h_dim, int embed_dim, int N,
                                              const double* embeddings,
                                              const double* weights,
                                              const double* biases,
                                              double* out_bias);

}  // namespace TENSORMD::Kernels::V2
