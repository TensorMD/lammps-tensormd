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

#include "kernels_head.h"

#include "arch/kernels_internal_head.h"

namespace TENSORMD::Kernels::Head {

// ----------------------------------------------------------------------
// Neural Network Potential
// ----------------------------------------------------------------------

template <typename Scalar>
void nnp_forward(NeuralHeadContext<Scalar>* ctx, int elti, int n,
                 const Scalar* d_x, const int* d_tlist, Scalar& h_total,
                 Scalar* h_outputs, Scalar** d_hidden, Scalar* d_ping,
                 Scalar* d_pong, TimedKernel kernel) {
  if (ctx->device == Device::CPU) {
    CPU::nnp_forward(
        n, ctx->n_layers, ctx->full_sizes, static_cast<int>(ctx->actype),
        ctx->use_resnet_dt, ctx->apply_output_bias, ctx->sum_output,
        ctx->d_weights[elti], ctx->d_biases[elti], d_x, ctx->eff_ndim_in,
        d_tlist, h_total, h_outputs, d_hidden, d_ping, d_pong, kernel);
  } else if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::nnp_forward(
        n, ctx->n_layers, ctx->full_sizes, static_cast<int>(ctx->actype),
        ctx->use_resnet_dt, ctx->apply_output_bias, ctx->sum_output,
        ctx->d_weights[elti], ctx->d_biases[elti], d_x, ctx->eff_ndim_in,
        d_tlist, h_total, h_outputs, d_hidden, d_ping, d_pong, kernel);
#endif
  }
}

template <typename Scalar>
void nnp_backward(NeuralHeadContext<Scalar>* ctx, int elti, int n,
                  const Scalar* grad_in, Scalar* grad_out, Scalar** d_hidden,
                  Scalar* d_ping, Scalar* d_pong, TimedKernel kernel,
                  bool use_W_T) {
  Scalar** d_weights = nullptr;
  if (use_W_T) {
    d_weights = ctx->d_weights_T[elti];
  } else {
    d_weights = ctx->d_weights[elti];
  }
  if (ctx->device == Device::CPU) {
    CPU::nnp_backward(n, ctx->n_layers, ctx->full_sizes, ctx->use_resnet_dt,
                      d_weights, grad_in, ctx->eff_ndim_in, grad_out, d_hidden,
                      d_ping, d_pong, kernel, use_W_T);
  } else if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::nnp_backward(n, ctx->n_layers, ctx->full_sizes, ctx->use_resnet_dt,
                      d_weights, grad_in, ctx->eff_ndim_in, grad_out, d_hidden,
                      d_ping, d_pong, kernel, use_W_T);
#endif
  }
}

template <typename Scalar>
void nnp_hybrid(NeuralHeadContext<Scalar>* ctx, int n, Scalar* d_x,
                const int* d_tlist, Scalar& h_total, Scalar* h_outputs,
                Scalar* grad_out, Scalar* d_ping, bool use_W_T) {
  if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::nnp_hybrid(n, ctx->n_layers, ctx->full_sizes,
                    static_cast<int>(ctx->actype), ctx->use_resnet_dt,
                    ctx->apply_output_bias, ctx->sum_output, ctx->d_weights[0],
                    ctx->d_weights_T[0], ctx->d_biases[0], d_x,
                    ctx->eff_ndim_in, d_tlist, ctx->d_output_sum, ctx->d_eatom,
                    h_total, h_outputs, grad_out, d_ping, use_W_T);
#endif
  }
}

template <typename Scalar>
void nnp_fusion(NeuralHeadContext<Scalar>* ctx, int n, const Scalar* d_x,
                const int* d_tlist, Scalar& d_total, Scalar* d_outputs) {
  if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    DeviceArgs<Scalar> args{};
    args.a = n;
    args.weights = ctx->d_weights[0];
    args.weights_T = ctx->d_weights_T[0];
    args.biases = ctx->d_biases[0];
    args.tlist = const_cast<int*>(d_tlist);
    args.G = const_cast<Scalar*>(d_x);
    args.eatoms = d_outputs;
    args.etotal = &d_total;
    GPU::nnp_fusion_kernel(args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void nnp_forward<float>(NeuralHeadContext<float>* ctx, int elti, int n,
                                 const float* d_x, const int* d_tlist,
                                 float& h_total, float* h_outputs,
                                 float** d_hidden, float* d_ping, float* d_pong,
                                 TimedKernel kernel);
template void nnp_forward<double>(NeuralHeadContext<double>* ctx, int elti,
                                  int n, const double* d_x, const int* d_tlist,
                                  double& h_total, double* h_outputs,
                                  double** d_hidden, double* d_ping,
                                  double* d_pong, TimedKernel kernel);

template void nnp_backward<float>(NeuralHeadContext<float>* ctx, int elti,
                                  int n, const float* grad_in, float* grad_out,
                                  float** d_hidden, float* d_ping,
                                  float* d_pong, TimedKernel kernel,
                                  bool use_W_T);
template void nnp_backward<double>(NeuralHeadContext<double>* ctx, int elti,
                                   int n, const double* grad_in,
                                   double* grad_out, double** d_hidden,
                                   double* d_ping, double* d_pong,
                                   TimedKernel kernel, bool use_W_T);

template void nnp_hybrid<float>(NeuralHeadContext<float>* ctx, int n,
                                float* d_x, const int* d_tlist, float& h_total,
                                float* h_outputs, float* grad_out,
                                float* d_ping, bool use_W_T);
template void nnp_hybrid<double>(NeuralHeadContext<double>* ctx, int n,
                                 double* d_x, const int* d_tlist,
                                 double& h_total, double* h_outputs,
                                 double* grad_out, double* d_ping,
                                 bool use_W_T);

template void nnp_fusion<float>(NeuralHeadContext<float>* ctx, int n,
                                const float* d_x, const int* d_tlist,
                                float& h_total, float* h_outputs);
template void nnp_fusion<double>(NeuralHeadContext<double>* ctx, int n,
                                 const double* d_x, const int* d_tlist,
                                 double& h_total, double* h_outputs);

}  // namespace TENSORMD::Kernels::Head
