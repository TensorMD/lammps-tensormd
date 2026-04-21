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

#include "../../../tensormd_constants.h"
#include "../kernels_internal_spline.h"
#include "../kernels_internal_utils.h"
#include "splines.hpp"

namespace TENSORMD::Kernels::Spline::CPU {

#include <vector>

template <typename Scalar>
void cubic_setup(int n, int n_out, Scalar dx, const Scalar* f_vals,
                 Scalar* out_params) {
  Scalar xdx = static_cast<Scalar>(1.0) / dx;
  std::vector<Scalar> m(n * n_out, 0.0);  // Temporary storage for slopes

  // Calculate derivatives (slopes) using Central Difference
  for (int i = 1; i < n - 1; ++i) {
    for (int j = 0; j < n_out; ++j) {
      m[i * n_out + j] =
          (f_vals[(i + 1) * n_out + j] - f_vals[(i - 1) * n_out + j]) * 0.5f *
          xdx;
    }
  }
  // Handle Endpoints (Forward/Backward difference)
  for (int j = 0; j < n_out; ++j) {
    m[0 * n_out + j] = (f_vals[1 * n_out + j] - f_vals[0 * n_out + j]) * xdx;
    m[(n - 1) * n_out + j] =
        (f_vals[(n - 1) * n_out + j] - f_vals[(n - 2) * n_out + j]) * xdx;
  }

  // Set the 4 coefficients: H(q) = c0 + c1*q + c2*q^2 + c3*q^3
  // Layout: [size, n_out, 4]
  for (int i = 0; i < n - 1; ++i) {
    for (int j = 0; j < n_out; ++j) {
      Scalar y0 = f_vals[i * n_out + j];
      Scalar y1 = f_vals[(i + 1) * n_out + j];
      Scalar m0 = m[i * n_out + j];
      Scalar m1 = m[(i + 1) * n_out + j];

      // Base index for the 4 coefficients
      int out_idx = (i * n_out + j) * 4;

      // c0 (Value)
      out_params[out_idx + 0] = y0;

      // c1 (First Derivative)
      out_params[out_idx + 1] = m0;

      // c2
      out_params[out_idx + 2] =
          (static_cast<Scalar>(3.0) * (y1 - y0) * xdx * xdx) -
          (static_cast<Scalar>(2.0) * m0 + m1) * xdx;

      // c3
      out_params[out_idx + 3] =
          (static_cast<Scalar>(2.0) * (y0 - y1) * xdx * xdx * xdx) +
          (m0 + m1) * xdx * xdx;
    }
  }
}

template <typename Scalar>
void cubic_interpolate_scatter(int n, const Scalar* R,
                               const int* __restrict__ tijlist,
                               Scalar* __restrict__ H,
                               Scalar* __restrict__ dHdr,
                               const Scalar* __restrict__ params_atlas,
                               int grid_size, int atlas_stride, Scalar x0,
                               Scalar dx, Scalar xdx, int n_out) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int k = 0; k < n; ++k) {
    int pair_idx = tijlist[k];
    if (pair_idx < 0) continue;
    Scalar* __restrict__ out_H = H + k * n_out;
    Scalar* __restrict__ out_dH = dHdr ? dHdr + k * n_out : nullptr;
    eval_cubic_spline_otf(R[k], pair_idx, n_out, params_atlas, grid_size,
                          atlas_stride, x0, dx, xdx, out_H, out_dH);
  }
}

template <typename Scalar>
void cubic_interpolate(DeviceArgs<Scalar>& args) {
  const int n = args.a * args.b * args.c;
  Scalar* R = args.R;
  int* tijlist = args.tijlist;
  Scalar* H = args.H;
  Scalar* dHdr = args.dHdr;
  const Scalar* params_atlas = args.params_atlas;
  Scalar dx = args.interp_dr;
  Scalar x0 = args.interp_r0;
  Scalar xdx = Scalar(1.0) / dx;
  const int grid_size = args.interp_size;
  const int atlas_stride = grid_size * 4 * args.k;
  const int n_out = args.k;

  Utils::CPU::timer_start(Profiling::TimedKernel::interpolate);
  cubic_interpolate_scatter(n, R, tijlist, H, dHdr, params_atlas, grid_size,
                            atlas_stride, x0, dx, xdx, n_out);

  const bool CalcVal = (H != nullptr);
  const bool CalcDeriv = (dHdr != nullptr);
  double FLOPs = 1.0 * n * n_out;
  if (CalcVal && CalcDeriv)
    FLOPs *= 12.0;
  else if (CalcVal)
    FLOPs *= 6.0;
  else if (CalcDeriv)
    FLOPs *= 6.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::interpolate, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void cubic_setup<float>(int n, int n_out, float dx,
                                 const float* f_vals, float* out_params);
template void cubic_setup<double>(int n, int n_out, double dx,
                                  const double* f_vals, double* out_params);

template void cubic_interpolate<float>(DeviceArgs<float>& args);
template void cubic_interpolate<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::Spline::CPU
