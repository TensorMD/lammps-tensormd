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

#ifndef TENSORMD_KERNELS_ARCH_CPU_SPLINES_HPP
#define TENSORMD_KERNELS_ARCH_CPU_SPLINES_HPP

#include <cmath>

namespace TENSORMD::Kernels::Spline::CPU {

template <typename Scalar>
void eval_cubic_spline_otf(Scalar r_val, int t, int n_out,
                           const Scalar* __restrict__ params_atlas,
                           int grid_size, int atlas_stride, Scalar x0,
                           Scalar dx, Scalar xdx, Scalar* __restrict__ H_u,
                           Scalar* __restrict__ dH_u) {
  // Find the correct atlas block for pair type 't'
  const Scalar* __restrict__ params = params_atlas + t * atlas_stride;

  // Grid lookup
  Scalar dr_x0 = r_val - x0;
  int i = static_cast<int>(dr_x0 * xdx);
  i = std::max(0, std::min(grid_size - 2, i));
  Scalar q = std::fma(-static_cast<Scalar>(i), dx, dr_x0);

  // Base pointer for the 4 coefficients of all U features
  const Scalar* __restrict__ base = params + (i * 4) * n_out;

  // ---------------------------------------------
  // Condition 1: calculate values and derivatives
  // ---------------------------------------------
  if (H_u && dH_u) {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int u = 0; u < n_out; ++u) {
      Scalar c0 = base[4 * u + 0];
      Scalar c1 = base[4 * u + 1];
      Scalar c2 = base[4 * u + 2];
      Scalar c3 = base[4 * u + 3];
      // Horner's Method
      H_u[u] = c0 + q * (c1 + q * (c2 + q * c3));
      dH_u[u] = c1 + q * (2.0f * c2 + 3.0f * c3 * q);
    }
  }
  // ---------------------------------------------
  // Condition 2: calculate values only
  // ---------------------------------------------
  else if (H_u) {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int u = 0; u < n_out; ++u) {
      Scalar c0 = base[4 * u + 0];
      Scalar c1 = base[4 * u + 1];
      Scalar c2 = base[4 * u + 2];
      Scalar c3 = base[4 * u + 3];
      H_u[u] = c0 + q * (c1 + q * (c2 + q * c3));
    }
  }
  // ---------------------------------------------
  // Condition 3: calculate derivatives only
  // ---------------------------------------------
  else if (dH_u) {
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int u = 0; u < n_out; ++u) {
      Scalar c1 = base[4 * u + 1];
      Scalar c2 = base[4 * u + 2];
      Scalar c3 = base[4 * u + 3];
      dH_u[u] = c1 + q * (2.0f * c2 + 3.0f * c3 * q);
    }
  }
}

}  // namespace TENSORMD::Kernels::Spline::CPU

#endif  // TENSORMD_KERNELS_ARCH_CPU_SPLINES_HPP
