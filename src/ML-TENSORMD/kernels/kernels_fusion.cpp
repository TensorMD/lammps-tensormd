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

#include "kernels_fusion.h"

#include "arch/kernels_internal_fusion.h"
#include "arch/kernels_internal_types.h"

namespace TENSORMD::Kernels::Fusion {

// Calculate P_kdba with both M and H calculated on-the-fly.
template <typename Scalar, const bool WriteP>
void calc_descriptors_fused(TensorMDContext<Scalar>* ctx) {
  DeviceArgs<Scalar> args{};
  args.a = ctx->data->dims.a;
  args.b = ctx->data->dims.b;
  args.c = ctx->data->dims.c;
  args.k = ctx->data->dims.k;
  args.d = ctx->data->dims.d;
  args.m = ctx->data->dims.m;
  args.R = ctx->data->R;
  args.P = ctx->data->P;
  args.G = ctx->data->G;
  args.T_proj_algo = ctx->data->T_proj_algo;
  args.drdrx = ctx->data->drdrx;
  args.tijlist = ctx->data->tijlist;
  args.params_atlas = const_cast<Scalar*>(ctx->encoder->d_params_atlas);
  args.interp_size = ctx->encoder->grid_size;
  args.interp_r0 = ctx->encoder->grid_rmin;
  args.interp_dr = ctx->encoder->grid_dr;
  args.params_atlas_fp32 =
      const_cast<float*>(ctx->encoder->d_params_atlas_fp32);
  if (ctx->device == Device::CPU) {
    CPU::calc_descriptors_fused(args);
  } else if (ctx->device == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::calc_descriptors_fused(args);
#endif
  }
}

// The fusion kernel to calculate radial and angular forces simultaneously.
// Requires W_kdba, R_cba, drdrx_xcba, tijlist_cba and spline param atlas.
template <typename Scalar, const bool ReadP>
void calc_forces_fused(TensorMDContext<Scalar>* ctx) {
  DeviceArgs<Scalar> args{};
  args.a = ctx->data->dims.a;
  args.b = ctx->data->dims.b;
  args.c = ctx->data->dims.c;
  args.k = ctx->data->dims.k;
  args.d = ctx->data->dims.d;
  args.m = ctx->data->dims.m;
  args.R = ctx->data->R;
  args.W = ctx->data->W;
  args.P = ctx->data->P;
  args.T_proj_algo = ctx->data->T_proj_algo;
  args.dEdG = ctx->data->dEdG;
  args.drdrx = ctx->data->drdrx;
  args.tijlist = ctx->data->tijlist;
  args.params_atlas = const_cast<Scalar*>(ctx->encoder->d_params_atlas);
  args.interp_size = ctx->encoder->grid_size;
  args.interp_r0 = ctx->encoder->grid_rmin;
  args.interp_dr = ctx->encoder->grid_dr;
  args.Fr = ctx->data->Fr;
  args.Fa = ctx->data->Fa;
  args.params_atlas_fp32 =
      const_cast<float*>(ctx->encoder->d_params_atlas_fp32);
  if (ctx->data->device == Device::CPU) {
    CPU::calc_forces_fused(args);
  } else if (ctx->data->device == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::calc_forces_fused(args);
#endif
  }
}

template <typename Scalar>
void calc_all_fused(TensorMDContext<Scalar>* ctx, double& h_etotal,
                    Scalar* d_eatoms) {
  if (ctx->device == Device::GPU) {
#ifdef TENSORMD_GPU
    DeviceArgs<Scalar> args{};
    args.a = ctx->data->dims.a;
    args.b = ctx->data->dims.b;
    args.c = ctx->data->dims.c;
    args.k = ctx->data->dims.k;
    args.m = ctx->data->dims.m;
    args.R = ctx->data->R;
    args.drdrx = ctx->data->drdrx;
    args.Fr = ctx->data->Fr;
    args.tijlist = ctx->data->tijlist;
    args.tlist = ctx->data->tlist;
    args.T_proj_algo = ctx->data->T_proj_algo;
    args.params_atlas = const_cast<Scalar*>(ctx->encoder->d_params_atlas);
    args.interp_dr = ctx->encoder->grid_dr;
    args.interp_r0 = ctx->encoder->grid_rmin;
    args.interp_size = ctx->encoder->grid_size;
    args.weights = const_cast<Scalar**>(ctx->head->d_weights[0]);
    args.weights_T = ctx->head->d_weights_T[0];
    args.biases = const_cast<Scalar**>(ctx->head->d_biases[0]);
    args.etotal = ctx->head->d_output_sum;
    args.eatoms = d_eatoms;
    args.n_layers = ctx->head->n_layers;
    args.actype = static_cast<int>(ctx->head->actype);

    HostArgs<Scalar> host_args{};
    host_args.etotal = &h_etotal;
    host_args.n_numbers = ctx->lmpdata.ntypes;

    GPU::unified_fusion_kernel(args, host_args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_descriptors_fused<float, true>(TensorMDContext<float>* ctx);
template void calc_descriptors_fused<float, false>(TensorMDContext<float>* ctx);
template void calc_descriptors_fused<double, true>(
    TensorMDContext<double>* ctx);
template void calc_descriptors_fused<double, false>(
    TensorMDContext<double>* ctx);

template void calc_forces_fused<float, true>(TensorMDContext<float>* ctx);
template void calc_forces_fused<float, false>(TensorMDContext<float>* ctx);
template void calc_forces_fused<double, true>(TensorMDContext<double>* ctx);
template void calc_forces_fused<double, false>(TensorMDContext<double>* ctx);

template void calc_all_fused<float>(TensorMDContext<float>* ctx,
                                    double& h_etotal, float* d_eatoms);
template void calc_all_fused<double>(TensorMDContext<double>* ctx,
                                     double& h_etotal, double* d_eatoms);

}  // namespace TENSORMD::Kernels::Fusion
