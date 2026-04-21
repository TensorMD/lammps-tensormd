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
// Forces Accumulation
// ----------------------------------------------------------------------

template <typename Scalar>
void sum_forces(double** f, double** vatom, double scale,
                TensorMDContext<Scalar>* ctx) {
  HostArgs<Scalar> host_args{};
  host_args.f = f;
  host_args.vatom = vatom;
  host_args.nlocal = ctx->lmpdata.nlocal;
  host_args.nghost = ctx->lmpdata.nghost;

  DeviceArgs<Scalar> device_args{};
  device_args.a = ctx->data->dims.a;
  device_args.b = ctx->data->dims.b;
  device_args.c = ctx->data->dims.c;
  device_args.tijlist = ctx->data->tijlist;
  device_args.ijlist = ctx->data->ijlist;
  device_args.R = ctx->data->R;
  device_args.drdrx = ctx->data->drdrx;
  device_args.Fr = ctx->data->Fr;
  device_args.Fa = ctx->data->Fa;
  device_args.scale = scale;
  device_args.imax = ctx->lmpdata.imax;
  device_args.f = ctx->lmpdata.f;
  device_args.vatom = ctx->lmpdata.vatom;

  if (ctx->data->device == Device::CPU) {
    CPU::sum_forces(host_args, device_args);
  } else if (ctx->data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::sum_forces(host_args, device_args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void sum_forces<float>(double** f, double** vatom, double scale,
                                TensorMDContext<float>* context);
template void sum_forces<double>(double** f, double** vatom, double scale,
                                 TensorMDContext<double>* context);

}  // namespace TENSORMD::Kernels
