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

#include <cstdlib>

#include "../../../tensormd_constants.h"
#include "../../geometric/irreducible_moments.hpp"
#include "../../geometric/reducible_moments.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::GPU {

template <typename Scalar, int m_val, int D, bool UseReducible>
__global__ void fused_Fa_kernel(const Scalar* __restrict__ V,
                                const Scalar* __restrict__ R,
                                const Scalar* __restrict__ drdrx,
                                const int* __restrict__ tijlist,
                                Scalar* __restrict__ Fa, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    const int tau = tijlist[i];
    Scalar rij = R[i];
    if (tau >= 0) {
      Scalar rijinv = Scalar(1.0) / rij;
      Scalar x = drdrx[i * 3 + 0];
      Scalar y = drdrx[i * 3 + 1];
      Scalar z = drdrx[i * 3 + 2];

      const Scalar* V_d = V + i * D;
      Scalar fa_x = 0.0, fa_y = 0.0, fa_z = 0.0;
      if constexpr (UseReducible) {
        Geometric::Reducible::contract_W_dM_otf<Scalar, m_val>(
            x, y, z, rijinv, V_d, fa_x, fa_y, fa_z);
      } else {
        Geometric::Irreducible::contract_W_dM_otf<Scalar, m_val>(
            x, y, z, rijinv, V_d, fa_x, fa_y, fa_z);
      }

      Fa[i * 3 + 0] = fa_x;
      Fa[i * 3 + 1] = fa_y;
      Fa[i * 3 + 2] = fa_z;
    } else {
      Fa[i * 3 + 0] = 0.0;
      Fa[i * 3 + 1] = 0.0;
      Fa[i * 3 + 2] = 0.0;
    }
  }
}

// -----------------------------------------------------------------------------
// Fa_xcba = dMdrx_xdcba * V_dcba
// -----------------------------------------------------------------------------

template <typename Scalar>
void calc_Fa_with_OTF_dMdrx(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_Fa);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.V);
  check_kernel_arg(args.R);
  check_kernel_arg(args.drdrx);
  check_kernel_arg(args.Fa);
#endif
  const int n = args.a * args.b * args.c;
  Scalar* __restrict__ V = args.V;
  Scalar* __restrict__ R = args.R;
  Scalar* __restrict__ drdrx = args.drdrx;
  Scalar* __restrict__ Fa = args.Fa;
  const int* __restrict__ tijlist = args.tijlist;
  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;

#define LAUNCH_KERNEL(MVAL, DVAL, REDUCIBLE)     \
  fused_Fa_kernel<Scalar, MVAL, DVAL, REDUCIBLE> \
      <<<gridSize, blockSize>>>(V, R, drdrx, tijlist, Fa, n);
  if (n > 0) {
    // clang-format off
    if (args.T_proj_algo == 2) {
      switch (args.m) {
        case 1: LAUNCH_KERNEL(1,  1, false); break;
        case 2: LAUNCH_KERNEL(2,  4, false); break;
        case 3: LAUNCH_KERNEL(3,  9, false); break;
        case 4: LAUNCH_KERNEL(4, 16, false); break;
        case 5: LAUNCH_KERNEL(5, 25, false); break;
        case 6: LAUNCH_KERNEL(6, 36, false); break;
      } 
    } else {
      switch (args.m) {
        case 1: LAUNCH_KERNEL(1,  1, true); break;
        case 2: LAUNCH_KERNEL(2,  4, true); break;
        case 3: LAUNCH_KERNEL(3, 10, true); break;
        case 4: LAUNCH_KERNEL(4, 20, true); break;
        case 5: LAUNCH_KERNEL(5, 35, true); break;
        case 6: LAUNCH_KERNEL(6, 56, true); break;
      } 
    }
    // clang-format on
  }
#undef LAUNCH_KERNEL
  CHECK(cudaGetLastError());
  const double size = static_cast<double>(args.a) * args.b * args.c;
  double FLOPs_dM = 0.0;
  if (args.T_proj_algo == 2) {
    FLOPs_dM = Geometric::Reducible::W_dM_d_FLOPs[args.m];
  } else {
    FLOPs_dM = Geometric::Irreducible::W_dM_u_FLOPs[args.m];
  }
  const double FLOPs = size * (1 + FLOPs_dM);
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_Fa, FLOPs);
}

// V_dcba = W_kdba * H_kcba
template <typename Scalar>
void calc_V(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_V);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.d);
  check_kernel_arg(args.k);
  check_kernel_arg(args.W);
  check_kernel_arg(args.H);
  check_kernel_arg(args.V);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int d = args.d;
  const int k = args.k;
  const int lda = k;
  const int ldb = k;
  const int ldc = d;
  const int stridea = k * d;
  const int strideb = k * c;
  const int stridec = d * c;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  Scalar* W = args.W;
  Scalar* H = args.H;
  Scalar* V = args.V;
  Linalg::GPU::gpublasgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, d, c, k, &alpha, W, lda, stridea, H,
      ldb, strideb, &beta, V, ldc, stridec, a * b);
  CHECK(cudaGetLastError());
  const double FLOPs = 2.0 * a * b * c * d * k;
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_V, FLOPs);
}

// -----------------------------------------------------------------------------
// Fa_xcba = dMdrx_xdcba * V_dcba (Angular)
// -----------------------------------------------------------------------------
template <typename Scalar>
void calc_angular_forces(DeviceArgs<Scalar>& args) {
  calc_V(args);
  calc_Fa_with_OTF_dMdrx(args);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_angular_forces<float>(DeviceArgs<float>& data);
template void calc_angular_forces<double>(DeviceArgs<double>& data);

}  // namespace TENSORMD::Kernels::GPU