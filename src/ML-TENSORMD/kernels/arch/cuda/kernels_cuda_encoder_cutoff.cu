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

#include <cuda_runtime.h>

#include <cmath>

#include "../../../tensormd_constants.h"
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"

namespace TENSORMD::Kernels::Encoder::GPU {

#ifndef MY_PI
#define MY_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------------------
// Kernel: Cosine Cutoff
// -------------------------------------------------------------------------
template <typename Scalar>
__global__ void cosine_cutoff_thread(int size, const Scalar* __restrict__ R,
                                     const int* __restrict__ tijlist,
                                     Scalar rcut, Scalar* __restrict__ sij,
                                     Scalar* __restrict__ dsij) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    // Handle Masking: If pointer is null, assume 1.0 (unmasked)
    Scalar mask_val = 1.0;
    if (tijlist != nullptr) {
      mask_val = static_cast<Scalar>(tijlist[i] >= 0);
    }

    Scalar r_val = R[i];
    Scalar my_pi = static_cast<Scalar>(MY_PI);

    // Math: Z = min(R / rcut, 1.0) * PI
    Scalar ratio = r_val / rcut;
    Scalar x = (ratio < 1.0) ? ratio : 1.0;
    Scalar z = x * my_pi;

    // sij = 0.5 + 0.5 * cos(z)
    sij[i] = mask_val * (0.5 + 0.5 * cos(z));

    // dsij = -0.5 * PI * sin(z) / rcut
    if (dsij != nullptr) {
      dsij[i] = mask_val * (-0.5 * my_pi / rcut * sin(z));
    }
  }
}

// -------------------------------------------------------------------------
// Kernel: Polynomial Cutoff
// -------------------------------------------------------------------------
template <typename Scalar>
__global__ void polynomial_cutoff_thread(int size, const Scalar* __restrict__ R,
                                         const int* __restrict__ tijlist,
                                         Scalar rcut, Scalar gamma,
                                         Scalar* __restrict__ sij,
                                         Scalar* __restrict__ dsij) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    Scalar mask_val = 1.0;
    if (tijlist != nullptr) {
      mask_val = static_cast<Scalar>(tijlist[i] >= 0);
    }

    // CPU Logic: Z = min(R / rcut, 1.0)
    Scalar val = R[i] / rcut;
    Scalar one = static_cast<Scalar>(1.0);
    Scalar Z = (val < one) ? val : one;

    // Powers of Z
    Scalar Zg = pow(Z, gamma);
    Scalar Zg1 = Zg * Z;               // Z^(gamma+1)
    Scalar Zgi = pow(Z, gamma - 1.0);  // Z^(gamma-1)

    // Sij = 1 + gamma * Z^(gamma+1) - (gamma + 1) * Z^gamma
    sij[i] = mask_val * (1.0 + gamma * Zg1 - (gamma + 1.0) * Zg);

    if (dsij != nullptr) {
      Scalar df1 = gamma * (gamma + 1.0) * Zg / rcut;
      Scalar df2 = gamma * (gamma + 1.0) * Zgi / rcut;
      dsij[i] = mask_val * (df1 - df2);
    }
  }
}

// -------------------------------------------------------------------------
// Host Wrappers
// -------------------------------------------------------------------------

template <typename Scalar>
void cosine_cutoff(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_sij);
  int size = args.a;
  if (size <= 0) return;

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  // Pass args.mask directly. If it is nullptr (unmasked),
  // the kernel handles it efficiently without needing separate kernels.
  cosine_cutoff_thread<<<gridSize, blockSize>>>(
      size, args.R, args.tijlist, static_cast<Scalar>(args.cutforcesq),
      args.sij, args.dsij);
  const double FLOPs = size * 12.0;
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_sij, FLOPs);
}

template <typename Scalar>
void polynomial_cutoff(DeviceArgs<Scalar>& args, Scalar gamma) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_sij);
  int size = args.a;
  if (size <= 0) return;

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  polynomial_cutoff_thread<<<gridSize, blockSize>>>(
      size, args.R, args.tijlist, static_cast<Scalar>(args.cutforcesq), gamma,
      args.sij, args.dsij);
  const double FLOPs = size * 17.0;
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_sij, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void cosine_cutoff<float>(DeviceArgs<float>& args);
template void cosine_cutoff<double>(DeviceArgs<double>& args);

template void polynomial_cutoff<float>(DeviceArgs<float>& args, float gamma);
template void polynomial_cutoff<double>(DeviceArgs<double>& args, double gamma);

}  // namespace TENSORMD::Kernels::Encoder::GPU
