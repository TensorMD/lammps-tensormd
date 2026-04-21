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

#include "kernels_spline.h"

#include "arch/kernels_internal_spline.h"

namespace TENSORMD::Kernels::Spline {

// ----------------------------------------------------------------------
// Cubic Interpolation
// ----------------------------------------------------------------------

template <typename Scalar>
void cubic_setup(Device dev, int n, int n_out, Scalar delta,
                 const Scalar* f_vals, Scalar* params) {
  if (dev == Device::CPU) {
    CPU::cubic_setup(n, n_out, delta, f_vals, params);
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::cubic_setup(n, n_out, delta, f_vals, params);
#endif
  }
}

template <typename Scalar>
void cubic_interpolate(TensorData<Scalar>* data, const Scalar* params_atlas,
                       int size, Scalar x0, Scalar dx) {
  DeviceArgs<Scalar> args{};
  args.a = data->dims.a;
  args.b = data->dims.b;
  args.c = data->dims.c;
  args.k = data->dims.k;
  args.H = data->H;
  args.R = data->R;
  args.dHdr = data->dHdr;
  args.tijlist = data->tijlist;
  args.params_atlas = const_cast<Scalar*>(params_atlas);
  args.interp_dr = dx;
  args.interp_size = size;
  args.interp_r0 = x0;
  if (data->device == Device::CPU) {
    CPU::cubic_interpolate(args);
  } else if (data->device == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::cubic_interpolate(args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void cubic_setup<float>(Device dev, int n, int n_out, float delta,
                                 const float* f_vals, float* params);
template void cubic_setup<double>(Device dev, int n, int n_out, double delta,
                                  const double* f_vals, double* params);

template void cubic_interpolate<float>(TensorData<float>* data,
                                       const float* params_atlas, int size,
                                       float x0, float dx);
template void cubic_interpolate<double>(TensorData<double>* data,
                                        const double* params_atlas, int size,
                                        double x0, double dx);

}  // namespace TENSORMD::Kernels::Spline
