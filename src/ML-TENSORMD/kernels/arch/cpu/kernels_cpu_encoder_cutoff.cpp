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

#include <algorithm>
#include <cmath>

#include "../../../tensormd_constants.h"
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"

#ifndef MY_PI
#define MY_PI 3.14159265358979323846
#endif

namespace TENSORMD::Kernels::Encoder::CPU {

template <typename Scalar>
void cosine_cutoff(DeviceArgs<Scalar> &args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_sij);
  const int size = args.a;
  int i;
  int *tijlist = args.tijlist;
  Scalar *sij = args.sij;
  Scalar *R = args.R;
  Scalar *dsij = args.dsij;
  Scalar rcut = args.cutforcesq;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    Scalar x = std::min(R[i] / rcut, Scalar(1.0));
    Scalar m = tijlist ? (tijlist[i] >= 0 ? 1.0f : 0.0f) : 1.0f;
    sij[i] = m * (0.5 + 0.5 * std::cos(x * MY_PI));
    if (dsij) {
      dsij[i] = m * (-0.5 * MY_PI * std::sin(x * MY_PI) / rcut);
    }
  }
  const double FLOPs = size * 12.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_sij, FLOPs);
}

template <typename Scalar>
void polynomial_cutoff(DeviceArgs<Scalar> &args, Scalar gamma) {
  Utils::CPU::timer_start(Profiling::TimedKernel::calc_sij);
  const int size = args.a;
  int i;
  int *tijlist = args.tijlist;
  Scalar *sij = args.sij;
  Scalar *R = args.R;
  Scalar *dsij = args.dsij;
  Scalar rcut = args.cutforcesq;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    Scalar x = std::min(R[i] / rcut, Scalar(1.0));
    Scalar m = tijlist ? (tijlist[i] >= 0 ? 1.0f : 0.0f) : 1.0f;
    Scalar x4 = std::pow(x, 4);
    Scalar x5 = x4 * x;
    sij[i] = m * (1 + gamma * x5 * x - (gamma + 1) * x5);
    if (dsij) {
      dsij[i] = m * ((gamma + 1) * gamma / rcut * (x5 - x4));
    }
  }
  const double FLOPs = size * 17.0;
  Utils::CPU::timer_stop(Profiling::TimedKernel::calc_sij, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void cosine_cutoff<float>(DeviceArgs<float> &args);
template void cosine_cutoff<double>(DeviceArgs<double> &args);

template void polynomial_cutoff<float>(DeviceArgs<float> &args, float gamma);
template void polynomial_cutoff<double>(DeviceArgs<double> &args, double gamma);

}  // namespace TENSORMD::Kernels::Encoder::CPU
