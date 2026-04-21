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

#ifndef TENSORMD_KERNELS_INTERNAL_HEAD_H
#define TENSORMD_KERNELS_INTERNAL_HEAD_H

#include <vector>

#include "../../tensormd_constants.h"
#include "kernels_internal_types.h"

namespace TENSORMD::Kernels::Head {

using Profiling::TimedKernel;

namespace CPU {

template <typename Scalar>
void nnp_forward(int M, int n_layers, std::vector<int>& full_sizes, int act,
                 bool use_resnet_dt, bool apply_output_bias, bool sum_output,
                 Scalar** weights, Scalar** biases, const Scalar* d_x,
                 int eff_ndim_in, const int* d_tlist, Scalar& h_total,
                 Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                 Scalar* d_pong, TimedKernel kernel);
template <typename Scalar>
void nnp_backward(int M, int n_layers, std::vector<int>& full_sizes,
                  bool use_resnet_dt, Scalar** weights, const Scalar* grad_in,
                  int eff_ndim_in, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T);

}  // namespace CPU

namespace GPU {
#ifdef TENSORMD_GPU

template <typename Scalar>
void nnp_forward(int M, int n_layers, std::vector<int>& full_sizes, int act,
                 bool use_resnet_dt, bool apply_output_bias, bool sum_output,
                 Scalar** weights, Scalar** biases, const Scalar* d_x,
                 int eff_ndim_in, const int* d_tlist, Scalar& h_total,
                 Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                 Scalar* d_pong, TimedKernel kernel);

template <typename Scalar>
void nnp_backward(int M, int n_layers, std::vector<int>& full_sizes,
                  bool use_resnet_dt, Scalar** weights, const Scalar* grad_in,
                  int grad_nout, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T);

template <typename Scalar>
void nnp_hybrid(int M, int n_layers, std::vector<int>& full_sizes, int act,
                bool use_resnet_dt, bool apply_output_bias, bool sum_output,
                Scalar** d_weights, Scalar** d_weights_T, Scalar** d_biases,
                Scalar* d_x_in, int eff_ndim_in, const int* tlist,
                Scalar* d_total, Scalar* d_outpus, Scalar& h_total,
                Scalar* h_outputs, Scalar* grad_out, Scalar* d_ping,
                bool use_W_T);

template <typename Scalar>
void nnp_fusion_kernel(DeviceArgs<Scalar>& args);

#endif
}  // namespace GPU
}  // namespace TENSORMD::Kernels::Head

#endif
