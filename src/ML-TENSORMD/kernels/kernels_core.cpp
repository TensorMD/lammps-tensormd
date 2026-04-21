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

#include "arch/kernels_internal.h"
#include "kernels.h"

namespace TENSORMD::Kernels {

// ----------------------------------------------------------------------
// Descriptors
// ----------------------------------------------------------------------

// P_dkba = M_dcba * H_kcba
template <typename Scalar>
void calc_P(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.d = data->dims.d;
  args.k = data->dims.k;
  args.M = data->M;
  args.H = data->H;
  args.P = data->P;
  if (data->device == Device::CPU) {
    CPU::calc_P(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_P(args);
#endif
  }
}

// Q_mkba = T_md * P_dkba
// G_mkba[1:,:,:,:] = Q_mkba[1:,:,:,:]
// G_mkba[0,:,:,:] = sqrt(Q_mkba[0,:,:,:]) * sign(P_dkba[0,:,:,:])
template <typename Scalar>
void calc_G(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.d = data->dims.d;
  args.k = data->dims.k;
  args.m = data->dims.m;
  args.P = data->P;
  args.G = data->G;
  args.T_proj_algo = data->T_proj_algo;
  if (data->device == Device::CPU) {
    CPU::calc_G(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_G(args);
#endif
  }
}

// ----------------------------------------------------------------------
// Gradients
// ----------------------------------------------------------------------

template <typename Scalar>
void calc_W(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.d = data->dims.d;
  args.k = data->dims.k;
  args.m = data->dims.m;
  args.G = data->G;
  args.dEdG = data->dEdG;
  args.W = data->W;
  args.P = data->P;
  args.T_proj_algo = data->T_proj_algo;
  if (data->device == Device::CPU) {
    CPU::calc_W(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_W(args);
#endif
  }
}

// Fa_xcba = W_dkba * H_kcba * dMdrx_xdcba = V_dcba * dMdrx_xdcba
// dMdrx is calculated on-the-fly with R, drdrx and m
template <typename Scalar>
void calc_angular_forces(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.d = data->dims.d;
  args.k = data->dims.k;
  args.m = data->dims.m;
  args.V = data->V;
  args.H = data->H;
  args.W = data->W;
  args.drdrx = data->drdrx;
  args.Fa = data->Fa;
  args.tijlist = data->tijlist;
  args.R = data->R;
  args.T_proj_algo = data->T_proj_algo;
  if (data->device == Device::CPU) {
    CPU::calc_angular_forces(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_angular_forces(args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_P<float>(TensorData<float>* data);
template void calc_P<double>(TensorData<double>* data);

template void calc_G<float>(TensorData<float>* data);
template void calc_G<double>(TensorData<double>* data);

template void calc_W<float>(TensorData<float>* data);
template void calc_W<double>(TensorData<double>* data);

template void calc_angular_forces<float>(TensorData<float>* data);
template void calc_angular_forces<double>(TensorData<double>* data);

}  // namespace TENSORMD::Kernels
