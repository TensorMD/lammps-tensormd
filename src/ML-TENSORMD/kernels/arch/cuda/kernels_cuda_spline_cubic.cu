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

#include <cstdio>
#include <stdexcept>
#include "../../../tensormd_constants.h"
#include "../kernels_internal_spline.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "splines.cuh"

namespace TENSORMD::Kernels::Spline::GPU {

// -------------------------------------------------------------------------
// Setup cubic interpolation
// -------------------------------------------------------------------------

template <typename Scalar>
__global__ void cubic_setup_kernel(int size, int n_out, Scalar dx,
                                   const Scalar* __restrict__ f_vals,
                                   Scalar* __restrict__ out_params) {
  // One thread per (grid_interval, feature)
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_intervals = size - 1;
  if (idx >= num_intervals * n_out) return;

  int j = idx % n_out;  // Feature index
  int i = idx / n_out;  // Grid index

  Scalar xdx = 1.0f / dx;

  auto get_m = [&](int grid_i) -> Scalar {
    if (grid_i == 0) {
      return (f_vals[1 * n_out + j] - f_vals[0 * n_out + j]) * xdx;
    } else if (grid_i == size - 1) {
      return (f_vals[(size - 1) * n_out + j] - f_vals[(size - 2) * n_out + j]) *
             xdx;
    } else {
      return (f_vals[(grid_i + 1) * n_out + j] -
              f_vals[(grid_i - 1) * n_out + j]) *
             0.5f * xdx;
    }
  };

  Scalar y0 = f_vals[i * n_out + j];
  Scalar y1 = f_vals[(i + 1) * n_out + j];
  Scalar m0 = get_m(i);
  Scalar m1 = get_m(i + 1);

  // Write Layout:[size, n_out, 4]
  Scalar* coeff = out_params + i * (n_out * 4) + j * 4;
  coeff[0] = y0;
  coeff[1] = m0;
  coeff[2] = (3.0f * (y1 - y0) * xdx * xdx) - (2.0f * m0 + m1) * xdx;
  coeff[3] = (2.0f * (y0 - y1) * xdx * xdx * xdx) + (m0 + m1) * xdx * xdx;
}

// -------------------------------------------------------------------------
// The Setup Dispatcher
// -------------------------------------------------------------------------

template <typename Scalar>
void cubic_setup(int size, int n_out, Scalar dx, const Scalar* d_f_vals,
                 Scalar* d_out_params) {
  int total = (size - 1) * n_out;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cubic_setup_kernel<<<blocks, threads>>>(size, n_out, dx, d_f_vals,
                                          d_out_params);
}

// -------------------------------------------------------------------------
// The Interpolation Dispatcher
// -------------------------------------------------------------------------

template <typename Scalar>
void cubic_interpolate(DeviceArgs<Scalar>& args) {
  const int n = args.a * args.b * args.c;
  Scalar* __restrict__ R = args.R;
  int* __restrict__ tijlist = args.tijlist;
  Scalar* __restrict__ H = args.H;
  Scalar* __restrict__ dHdr = args.dHdr;
  const Scalar* __restrict__ params_atlas = args.params_atlas;
  Scalar dx = args.interp_dr;
  Scalar x0 = args.interp_r0;
  const int size = args.interp_size;
  const int n_out = args.k;
  Utils::GPU::timer_start(Profiling::TimedKernel::interpolate);

  int total_elements = n * n_out;
  int blockSize = 256;
  int numBlocks = (total_elements + blockSize - 1) / blockSize;
  int atlas_stride = size * n_out * 4;
  Scalar xdx = Scalar(1.0) / dx;

  // CalcVal   = bit 1 (value 2)
  // CalcDeriv = bit 0 (value 1)
  int mode = ((H != nullptr) << 1) | ((dHdr != nullptr) << 0);
  double FLOPs = 0.0;

#define LAUNCH_KERNEL(CalcVal, CalcDeriv)                                    \
  cubic_spline_kernel<Scalar, CalcVal, CalcDeriv><<<numBlocks, blockSize>>>( \
      total_elements, n_out, R, tijlist, H, dHdr, params_atlas, size,        \
      atlas_stride, x0, dx, xdx);
  switch (mode) {
    case 1: LAUNCH_KERNEL(false, true); FLOPs = 10.0; break;
    case 2: LAUNCH_KERNEL(true, false); FLOPs = 10.0; break;
    case 3: LAUNCH_KERNEL(true, true);  FLOPs = 16.0; break;
    default: throw std::runtime_error("Invalid mode for cubic interpolation kernel");
  }
#undef LAUNCH_KERNEL
  CHECK(cudaGetLastError());

  FLOPs *= 1.0 * n * n_out;
  Utils::GPU::timer_stop(Profiling::TimedKernel::interpolate, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void cubic_setup<float>(int ndim, int n, float delta,
                                 const float* f_vals, float* out_params);
template void cubic_setup<double>(int ndim, int n, double delta,
                                  const double* f_vals, double* out_params);

template void cubic_interpolate<float>(DeviceArgs<float>& args);
template void cubic_interpolate<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Spline::GPU
