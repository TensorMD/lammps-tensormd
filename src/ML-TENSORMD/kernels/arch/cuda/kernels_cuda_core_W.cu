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

#include <stdexcept>

#include "../../../tensormd_constants.h"
#include "../../../tensormd_strategy.h"
#include "../../geometric/irreducible_PGW.hpp"
#include "../../geometric/reducible_PGW.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::GPU {

using namespace Geometric;

/* ---------------------------------------------------------------------- */

template <typename Scalar, int m_val, int D, bool use_legacy_T,
          bool UseReducible>
__global__ void contract_W_kdba(const Scalar* __restrict__ Gp,
                                Scalar* __restrict__ P, Scalar* __restrict__ W,
                                int K, int total_threads) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_threads) return;

  int k = tid % K;
  int ab = tid / K;

  Scalar* P_base_ab = P + ab * (D * K);
  Scalar* W_base_ab = W + ab * (D * K);
  const Scalar* G_base_ab = Gp + ab * (m_val * K);

  if (UseReducible) {
    Reducible::contract_W_kdba_otf<Scalar, m_val, use_legacy_T>(
        G_base_ab, P_base_ab, W_base_ab, k, K);
  } else {
    Irreducible::contract_W_kdba_otf<Scalar, m_val>(G_base_ab, P_base_ab,
                                                    W_base_ab, k, K);
  }
}

template <typename Scalar, int m_val, int D, bool use_legacy_T,
          bool UseReducible>
void calc_W_kernel(DeviceArgs<Scalar>& args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::calc_W);

  const int A = args.a;
  const int B = args.b;
  const int K = args.k;
  const Scalar* __restrict__ dEdG = args.dEdG;

  // Flatten A, B, and K into a single 1D workload
  int total_threads = A * B * K;
  int threads_per_block = 256;
  int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  Scalar* P = args.P;
  Scalar* W = args.W;
  contract_W_kdba<Scalar, m_val, D, use_legacy_T, UseReducible>
      <<<num_blocks, threads_per_block>>>(dEdG, P, W, K, total_threads);

  Utils::GPU::timer_stop(Profiling::TimedKernel::calc_W,
                         A * B * K * D * (2.0 * m_val + 1.0) + A * B * K * 2.0);
}

template <typename Scalar>
void calc_W(DeviceArgs<Scalar>& args) {
  if (args.T_proj_algo == 0) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, true, true>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, true, true>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 10, true, true>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 20, true, true>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 35, true, true>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 56, true, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  } else if (args.T_proj_algo == 1) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, false, true>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, false, true>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 10, false, true>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 20, false, true>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 35, false, true>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 56, false, true>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  } else if (args.T_proj_algo == 2) {
    if (args.m == 1) {
      calc_W_kernel<Scalar, 1, 1, false, false>(args);
    } else if (args.m == 2) {
      calc_W_kernel<Scalar, 2, 4, false, false>(args);
    } else if (args.m == 3) {
      calc_W_kernel<Scalar, 3, 9, false, false>(args);
    } else if (args.m == 4) {
      calc_W_kernel<Scalar, 4, 16, false, false>(args);
    } else if (args.m == 5) {
      calc_W_kernel<Scalar, 5, 25, false, false>(args);
    } else if (args.m == 6) {
      calc_W_kernel<Scalar, 6, 36, false, false>(args);
    } else {
      throw std::runtime_error("Invalid m value for calc_W kernel");
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void calc_W<float>(DeviceArgs<float>& args);
template void calc_W<double>(DeviceArgs<double>& args);

}  // namespace TENSORMD::Kernels::GPU
