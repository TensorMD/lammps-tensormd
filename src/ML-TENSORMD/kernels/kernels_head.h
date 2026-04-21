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

#ifndef TENSORMD_KERNELS_HEAD_H
#define TENSORMD_KERNELS_HEAD_H

#include "../heads/neural_head_context.h"
#include "../tensormd_constants.h"

namespace TENSORMD::Kernels::Head {

using Profiling::TimedKernel;

// ----------------------------------------------------------------------
// MLP based Neural Network Potential
// ----------------------------------------------------------------------

// Forward pass of a MLP
// elti: the i-th set of weights and biases, corresponding to the i-th atom type
// d_x: the input matrix [n, n_features]
// d_tot: sum(d_y)
// d_y: the output matrix [n, n_out]
template <typename Scalar>
void nnp_forward(NeuralHeadContext<Scalar>* ctx, int elti, int n,
                 const Scalar* d_x, const int* d_tlist, Scalar& h_total,
                 Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                 Scalar* d_pong, TimedKernel kernel);

// Backward pass of a MLP
template <typename Scalar>
void nnp_backward(NeuralHeadContext<Scalar>* ctx, int elti, int n,
                  const Scalar* grad_in, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T);

// The hybrid fusion kernel for the MLP: BLAS + Fusion
// BLAS is used to calculate the first and the last GEMM.
// The rest calculations are fused into one kernel.
// x -> (BLAS) -> A0 -> (Fusion) -> E,Eatom -> (Fusion) -> A0' -> (BLAS) -> grad
// Only applicable to V2 models on GPU devices (NVIDIA, DCU).
template <typename Scalar>
void nnp_hybrid(NeuralHeadContext<Scalar>* ctx, int n, Scalar* d_x,
                const int* d_tlist, Scalar& h_total, Scalar* h_outputs,
                Scalar *grad_out, Scalar* d_ping, bool use_W_T);

// The fusion kernel for the MLP: forward + backward
// Only applicable to V2 models on GPU devices (NVIDIA, DCU).
template <typename Scalar>
void nnp_fusion(NeuralHeadContext<Scalar>* ctx, int n, const Scalar* d_x,
                const int* d_tlist, Scalar& h_total, Scalar* h_outputs);

}  // namespace TENSORMD::Kernels::Head

#endif  // TENSORMD_KERNELS_HEAD_H