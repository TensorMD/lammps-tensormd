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
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::GPU {

/* ---------------------------------------------------------------------- */


template <typename Scalar>
void calc_P(DeviceArgs<Scalar>& args) {
  #if defined(DEBUG)
  check_kernel_arg(args.a);
  check_kernel_arg(args.b);
  check_kernel_arg(args.c);
  check_kernel_arg(args.d);
  check_kernel_arg(args.k);
  check_kernel_arg(args.M);
  check_kernel_arg(args.H);
  check_kernel_arg(args.P);
#endif
  const int a = args.a;
  const int b = args.b;
  const int c = args.c;
  const int d = args.d;
  const int k = args.k;
  const int lda = k;
  const int ldb = d;
  const int ldc = k;
  const int stridea = c * k;
  const int strideb = c * d;
  const int stridec = d * k;
  Scalar* M = args.M;
  Scalar* H = args.H;
  Scalar* P = args.P;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;

  // On GPU, the memory layout of P is always K-continuous
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_P);
  Linalg::GPU::gpublasgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, k, d, c, &alpha, H, lda, stridea, M,
      ldb, strideb, &beta, P, ldc, stridec, a * b);
  double FLOPs = 2.0 * a * b * c * d * k;
  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_P, FLOPs);
  CHECK(cudaGetLastError());
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_P<float>(DeviceArgs<float>& args);
template void calc_P<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::GPU
