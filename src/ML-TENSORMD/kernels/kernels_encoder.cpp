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

#include "kernels_encoder.h"

#include "arch/kernels_internal_encoder.h"

namespace TENSORMD::Kernels::Encoder {

// ----------------------------------------------------------------------
// Calculate radial forces with tensor contraction:
// Fr_xcba = W_dkba * M_dcba * dHdr_kcba * drdrx_xcba
// ----------------------------------------------------------------------

template <typename Scalar>
void interp_backward(TensorData<Scalar>* data, const Scalar* params_atlas,
                     int size, Scalar x0, Scalar dx) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.k = data->dims.k;
  args.d = data->dims.d;
  args.m = data->dims.m;
  args.U = data->U;
  args.dHdr = data->dHdr;
  args.drdrx = data->drdrx;
  args.Fr = data->Fr;
  args.R = data->R;
  args.W = data->W;
  args.M = data->M;
  args.tijlist = data->tijlist;
  args.T_proj_algo = data->T_proj_algo;
  args.params_atlas = const_cast<Scalar*>(params_atlas);
  args.interp_dr = dx;
  args.interp_r0 = x0;
  args.interp_size = size;

  if (data->device == Device::CPU) {
    CPU::interp_backward(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::interp_backward(args);
#endif
  }
}

// ----------------------------------------------------------------------
// Encoder and Cutoff Kernels
// ----------------------------------------------------------------------

template <typename Scalar>
void cosine_cutoff(Device dev, Scalar rcut, int n, Scalar* x, Scalar* f,
                   Scalar* df, int* tijlist) {
  DeviceArgs<Scalar> args{};
  args.a = n;
  args.tijlist = tijlist;
  args.R = x;
  args.sij = f;
  args.dsij = df;
  args.cutforcesq = rcut;
  if (dev == Device::CPU) {
    CPU::cosine_cutoff(args);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::cosine_cutoff(args);
#endif
  }
}

template <typename Scalar>
void polynomial_cutoff(Device dev, Scalar rcut, int n, Scalar* x, Scalar* f,
                       Scalar* df, int* tijlist, Scalar gamma) {
  DeviceArgs<Scalar> args{};
  args.a = n;
  args.tijlist = tijlist;
  args.R = x;
  args.sij = f;
  args.dsij = df;
  args.cutforcesq = rcut;
  if (dev == Device::CPU) {
    CPU::polynomial_cutoff(args, gamma);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::polynomial_cutoff(args, gamma);
#endif
  }
}

template <typename Scalar>
void apply_cutoff(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.k = data->dims.k;
  args.sij = data->sij;
  args.dsij = data->dsij;
  args.H = data->H;
  args.dHdr = data->dHdr;
  if (data->device == Device::CPU) {
    CPU::apply_cutoff(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::apply_cutoff(args);
#endif
  }
}

template <typename Scalar>
void apply_cutoff_forward_only(Device dev, int n, int k, const Scalar* sij,
                               Scalar* H) {
  if (dev == Device::CPU) {
    CPU::apply_cutoff_forward_only(n, k, sij, H);
  } else if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::apply_cutoff_forward_only(n, k, sij, H);
#endif
  }
}

template <typename Scalar>
void calc_U_abck(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.d = data->dims.d;
  args.k = data->dims.k;
  args.W = data->W;
  args.U = data->U;
  args.M = data->M;
  if (data->device == Device::CPU) {
    CPU::calc_U_abck(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_U_abck(args);
#endif
  }
}

template <typename Scalar>
void calc_Up(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.k = data->dims.k;
  args.sij = data->sij;
  args.U = data->U;
  args.Up = data->Up;
  args.dHdr = data->dHdr;
  if (data->device == Device::CPU) {
    CPU::calc_Up(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_Up(args);
#endif
  }
}

template <typename Scalar>
void calc_Fr(TensorData<Scalar>* data) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.k = data->dims.k;
  args.dHdr = data->dHdr;
  args.drdrx = data->drdrx;
  args.tijlist = data->tijlist;
  args.Fr = data->Fr;
  args.dHdrp = data->dHdrp;
  if (data->device == Device::CPU) {
    CPU::calc_Fr(args);
  } else if (data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::calc_Fr(args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void interp_backward<float>(TensorData<float>* data,
                                     const float* params_atlas, int size,
                                     float x0, float dx);
template void interp_backward<double>(TensorData<double>* data,
                                      const double* params_atlas, int size,
                                      double x0, double dx);

template void cosine_cutoff<float>(Device dev, float rcut, int n, float* x,
                                   float* f, float* df, int* tijlist);
template void cosine_cutoff<double>(Device dev, double rcut, int n, double* x,
                                    double* f, double* df, int* tijlist);

template void polynomial_cutoff<float>(Device dev, float rcut, int n, float* x,
                                       float* f, float* df, int* tijlist,
                                       float gamma);
template void polynomial_cutoff<double>(Device dev, double rcut, int n,
                                        double* x, double* f, double* df,
                                        int* tijlist, double gamma);

template void apply_cutoff<float>(TensorData<float>* data);
template void apply_cutoff<double>(TensorData<double>* data);

template void apply_cutoff_forward_only<float>(Device dev, int n, int k,
                                               const float* sij, float* H);
template void apply_cutoff_forward_only<double>(Device dev, int n, int k,
                                                const double* sij, double* H);

template void calc_U_abck<float>(TensorData<float>* data);
template void calc_U_abck<double>(TensorData<double>* data);

template void calc_Up<float>(TensorData<float>* data);
template void calc_Up<double>(TensorData<double>* data);

template void calc_Fr<float>(TensorData<float>* data);
template void calc_Fr<double>(TensorData<double>* data);

}  // namespace TENSORMD::Kernels::Encoder
