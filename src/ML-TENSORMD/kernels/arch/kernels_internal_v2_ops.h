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

#ifndef TENSORMD_KERNELS_INTERNAL_V2_OPS_H
#define TENSORMD_KERNELS_INTERNAL_V2_OPS_H

namespace TENSORMD::Kernels::V2 {

namespace CPU {

template <typename Scalar>
void expand_encoder_input(int n, int n_rbf, int embed_dim, const Scalar* R,
                          const int* zij, const Scalar* rbf_centers,
                          Scalar rbf_gamma, const Scalar* embeddings,
                          Scalar* x_out);

template <typename Scalar>
void expand_encoder_grad(int n, int n_rbf, int n_in_mlp, const Scalar* grad_in,
                         const Scalar* R, const Scalar* rbf_centers,
                         Scalar rbf_gamma, Scalar* grad_R);

template <typename Scalar>
void compute_first_embedding(int n_types, int h_dim, int embed_dim, int N,
                             const Scalar* embeddings, const Scalar* weights,
                             const Scalar* biases, Scalar* out_bias);

}  // namespace CPU

namespace GPU {
#ifdef TENSORMD_GPU

template <typename Scalar>
void expand_encoder_input(int n, int n_rbf, int embed_dim, const Scalar* R,
                          const int* zij, const Scalar* rbf_centers,
                          Scalar rbf_gamma, const Scalar* embeddings,
                          Scalar* x_out);

template <typename Scalar>
void expand_encoder_grad(int n, int n_rbf, int n_in_mlp, const Scalar* grad_in,
                         const Scalar* R, const Scalar* rbf_centers,
                         Scalar rbf_gamma, Scalar* grad_R);

template <typename Scalar>
void compute_first_embedding(int n_types, int h_dim, int embed_dim, int N,
                             const Scalar* embeddings, const Scalar* weights,
                             const Scalar* biases, Scalar* out_bias);

#endif
}  // namespace GPU
}  // namespace TENSORMD::Kernels::V2

#endif
