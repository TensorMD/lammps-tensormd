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
#include "../../../tensormd_strategy.h"
#include "../kernels_internal_encoder.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::Encoder::GPU {

using Kernels::GPU::handle;

// -----------------------------------------------------------------------------
// compute U_dcba = W_kdba * dHdr_kcba
// compute Fr_xcba = U_dcba * M_dcba * drdrx_xcba
// -----------------------------------------------------------------------------

template <typename Scalar>
void calc_U_abcd(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_U);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.W);
  check_kernel_arg(args.dHdr);
  check_kernel_arg(args.U);
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.d);
  check_kernel_arg(args.k);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int d = args.d;
  const int k = args.k;
  Scalar* W = args.W;
  Scalar* dHdr = args.dHdr;
  Scalar* U = args.U;
  const int lda = k;
  const int ldb = k;
  const int ldc = d;
  const int stridea = k * d;
  const int strideb = k * c;
  const int stridec = d * c;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  Linalg::GPU::gpublasgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, d, c, k, &alpha, W, lda, stridea, dHdr,
      ldb, strideb, &beta, U, ldc, stridec, a * b);
  CHECK(cudaGetLastError());
  const double FLOPs = static_cast<double>(a) * b * c * d * k * 2.0;
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_U, FLOPs);
}

// -----------------------------------------------------------------------------
// The baseline CUDA kernel with one thread per 'i' (no reduction, no shuffles)
// -----------------------------------------------------------------------------

template <typename Scalar, int D>
__global__ void kernel_calc_Fr(const Scalar* __restrict__ U,
                               const Scalar* __restrict__ M,
                               const int* __restrict__ tijlist,
                               const Scalar* __restrict__ drdrx,
                               Scalar* __restrict__ Fr, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    int stride3 = i * 3;

    if (tijlist[i] < 0) {
      Fr[stride3 + 0] = Scalar(0.0);
      Fr[stride3 + 1] = Scalar(0.0);
      Fr[stride3 + 2] = Scalar(0.0);
    } else {
      int strided = i * D;
      const Scalar* __restrict__ Up = U + strided;
      const Scalar* __restrict__ Mp = M + strided;
      Scalar y = Scalar(0.0);

#pragma unroll
      for (int j = 0; j < D; ++j) {
        y += Up[j] * Mp[j];
      }

      Fr[stride3 + 0] = drdrx[stride3 + 0] * y;
      Fr[stride3 + 1] = drdrx[stride3 + 1] * y;
      Fr[stride3 + 2] = drdrx[stride3 + 2] * y;
    }
  }
}

template <typename Scalar, int D>
void calc_Fr_with_U_abcd_M_abcd(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_Fr);

  const int n = args.a * args.b * args.c;
  Scalar* U = args.U;
  Scalar* M = args.M;
  Scalar* drdrx = args.drdrx;
  Scalar* Fr = args.Fr;
  int* tijlist = args.tijlist;
  const int blockDim = 256;
  int gridDim = (n + blockDim - 1) / blockDim;
  kernel_calc_Fr<Scalar, D><<<gridDim, blockDim>>>(U, M, tijlist, drdrx, Fr, n);
  CHECK(cudaGetLastError());
  const double FLOPs = 1.0 * n * (3.0 + D * 2.0);
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_Fr, FLOPs);
}

// -----------------------------------------------------------------------------
// Calculate radial forces when interpolation is used
// -----------------------------------------------------------------------------

template <typename Scalar>
void interp_backward(DeviceArgs<Scalar>& args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(args.U);
  check_kernel_arg(args.M);
  check_kernel_arg(args.W);
  check_kernel_arg(args.dHdr);
  check_kernel_arg(args.tijlist);
  check_kernel_arg(args.drdrx);
  check_kernel_arg(args.Fr);
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.k);
  check_kernel_arg(args.m);
#endif
  // Call the batch GEMM to compute U_dcba = W_kdba * dHdr_kcba
  calc_U_abcd(args);

  // Call the kernel to compute Fr_xcba = U_dcba * M_dcba * drdrx_xcba
  bool use_reducible = args.T_proj_algo < 2;
  if (use_reducible) {
    switch (args.m) {
      case 1:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 1>(args); break;
      case 2:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 4>(args); break;
      case 3:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 10>(args); break;
      case 4:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 20>(args); break;
      case 5:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 35>(args); break;
      case 6:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 56>(args); break;
      default:
        break;
    }
  } else {
    switch (args.m) {
      case 1:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 1>(args); break;
      case 2:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 4>(args); break;
      case 3:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 9>(args); break;
      case 4:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 16>(args); break;
      case 5:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 25>(args); break;
      case 6:
        calc_Fr_with_U_abcd_M_abcd<Scalar, 36>(args); break;
      default:
        break;
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void interp_backward<float>(DeviceArgs<float>& data);
template void interp_backward<double>(DeviceArgs<double>& data);

}  // namespace TENSORMD::Kernels::Encoder::GPU
