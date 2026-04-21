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
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cub/cub.cuh"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::GPU {

// -----------------------------------------------------------------------------
// Helper Structs for Vectorized Reduction
// -----------------------------------------------------------------------------
struct Vec3 {
  double x, y, z;
  __host__ __device__ __forceinline__ Vec3 operator+(const Vec3& other) const {
    return {x + other.x, y + other.y, z + other.z};
  }
};

struct Vec6 {
  double v0, v1, v2, v3, v4, v5;
  __host__ __device__ __forceinline__ Vec6 operator+(const Vec6& other) const {
    return {v0 + other.v0, v1 + other.v1, v2 + other.v2,
            v3 + other.v3, v4 + other.v4, v5 + other.v5};
  }
};

// -----------------------------------------------------------------------------
// Kernel: Sum Forces with CUB Block Reduction
// -----------------------------------------------------------------------------
template <typename Scalar, typename index_t, bool ComputeVirial, bool UseFa,
          int BlockSize>
__global__ void sum_forces_cub_kernel(
    int a, int b, int c, Scalar* __restrict__ Fr, Scalar* __restrict__ Fa,
    Scalar* __restrict__ R, Scalar* __restrict__ drdrx,
    int* __restrict__ ijlist, int* __restrict__ tijlist, double scale,
    double* __restrict__ f, double* __restrict__ vatom) {
  // blockIdx.x represents the central atom index (or the 'stride1' block index)
  // Each block handles [blockIdx.x * stride_size] to [(blockIdx.x+1) *
  // stride_size]
  int i_block = blockIdx.x;

  if (i_block >= a) return;

  // Number of neighbors/features per central atom
  int stride_size = b * c;

  // Base index for this block in the flat arrays
  index_t base_idx =
      static_cast<index_t>(i_block) * static_cast<index_t>(stride_size);

  // Local accumulators for atom i
  Vec3 local_f_i = {0.0, 0.0, 0.0};
  Vec6 local_vir_i = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // Grid-Stride Loop within the block (in case stride_size > blockDim.x)
  for (int k = threadIdx.x; k < stride_size; k += blockDim.x) {
    index_t ii = base_idx + static_cast<index_t>(k);  // The global index

    if (tijlist[ii] >= 0) {
      // Load indices
      // We assume i is constant for this block, but we load j per thread
      // For safety, we can load i here, or rely on blockIdx logic.
      int j = ijlist[ii * 2 + 1];

      // Calculate Force Vector
      double df[3];
#pragma unroll
      for (int x = 0; x < 3; x++) {
        if constexpr (UseFa) {
          df[x] = static_cast<double>(Fr[ii * 3 + x] + Fa[ii * 3 + x]) * scale;
        } else {
          df[x] = static_cast<double>(Fr[ii * 3 + x]) * scale;
        }
      }

      // 1. Accumulate Force on i (Local Register)
      local_f_i.x += df[0];
      local_f_i.y += df[1];
      local_f_i.z += df[2];

      // 2. Update Force on j (Global Atomic - Random Access)
      if (j >= 0) {
        atomicAdd(f + j * 3 + 0, -df[0]);
        atomicAdd(f + j * 3 + 1, -df[1]);
        atomicAdd(f + j * 3 + 2, -df[2]);
      }

      // 3. Calculate Virial
      if constexpr (ComputeVirial) {
        Scalar r = R[ii];
        Scalar dr[3];
        dr[0] = drdrx[ii * 3 + 0];
        dr[1] = drdrx[ii * 3 + 1];
        dr[2] = drdrx[ii * 3 + 2];

        double virial[6];
        virial[0] = -dr[0] * df[0] * r;
        virial[1] = -dr[1] * df[1] * r;
        virial[2] = -dr[2] * df[2] * r;
        virial[3] = -dr[0] * df[1] * r;
        virial[4] = -dr[0] * df[2] * r;
        virial[5] = -dr[1] * df[2] * r;

        // Accumulate Virial for i (Local Register)
        local_vir_i.v0 += virial[0];
        local_vir_i.v1 += virial[1];
        local_vir_i.v2 += virial[2];
        local_vir_i.v3 += virial[3];
        local_vir_i.v4 += virial[4];
        local_vir_i.v5 += virial[5];

        // Update Virial for j (Global Atomic)
        if (j >= 0) {
          atomicAdd(vatom + j * 6 + 0, 0.5 * virial[0]);
          atomicAdd(vatom + j * 6 + 1, 0.5 * virial[1]);
          atomicAdd(vatom + j * 6 + 2, 0.5 * virial[2]);
          atomicAdd(vatom + j * 6 + 3, 0.5 * virial[3]);
          atomicAdd(vatom + j * 6 + 4, 0.5 * virial[4]);
          atomicAdd(vatom + j * 6 + 5, 0.5 * virial[5]);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Block Reductions
  // ---------------------------------------------------------------------------

  typedef cub::BlockReduce<Vec3, BlockSize> BlockReduceForce;
  typedef cub::BlockReduce<Vec6, BlockSize> BlockReduceVirial;

  // Shared Memory Storage (Union to save space if reductions were sequential,
  // but here we do them conceptually in parallel flow, though disjoint types
  // needed)
  __shared__ typename BlockReduceForce::TempStorage temp_force;
  __shared__ typename BlockReduceVirial::TempStorage temp_virial;

  // 1. Reduce Force on i
  Vec3 block_f_i = BlockReduceForce(temp_force).Sum(local_f_i);

  // 2. Reduce Virial on i (if needed)
  Vec6 block_vir_i;
  if constexpr (ComputeVirial) {
    block_vir_i = BlockReduceVirial(temp_virial).Sum(local_vir_i);
  }

  // ---------------------------------------------------------------------------
  // Global Update for Atom i (Thread 0 only)
  // ---------------------------------------------------------------------------
  if (threadIdx.x == 0) {
    // Determine 'i'. We check the first valid interaction in the block.
    // Assuming the block corresponds to a single 'i'.
    // Safe lookup: ijlist[base_idx * 2]
    int i = ijlist[base_idx * 2];

    if (i >= 0) {
      atomicAdd(f + i * 3 + 0, block_f_i.x);
      atomicAdd(f + i * 3 + 1, block_f_i.y);
      atomicAdd(f + i * 3 + 2, block_f_i.z);

      if constexpr (ComputeVirial) {
        atomicAdd(vatom + i * 6 + 0, 0.5 * block_vir_i.v0);
        atomicAdd(vatom + i * 6 + 1, 0.5 * block_vir_i.v1);
        atomicAdd(vatom + i * 6 + 2, 0.5 * block_vir_i.v2);
        atomicAdd(vatom + i * 6 + 3, 0.5 * block_vir_i.v3);
        atomicAdd(vatom + i * 6 + 4, 0.5 * block_vir_i.v4);
        atomicAdd(vatom + i * 6 + 5, 0.5 * block_vir_i.v5);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar, typename index_t, int BlockSize>
void dispatch_sum_forces_kernel(HostArgs<Scalar>& host_args,
                                DeviceArgs<Scalar>& device_args) {
  const int a = device_args.a;
  const int b = device_args.b;
  const int c = device_args.c;
  Scalar* R = device_args.R;
  Scalar* drdrx = device_args.drdrx;
  int* tijlist = device_args.tijlist;
  int* ijlist = device_args.ijlist;
  Scalar* Fr = device_args.Fr;
  Scalar* Fa = device_args.Fa;
  Scalar scale = device_args.scale;
  const int gridSize = a;

  bool compute_virial =
      (host_args.vatom != nullptr && device_args.vatom != nullptr);
  bool use_Fa = (Fa != nullptr);

  if (compute_virial) {
    if (use_Fa) {
      sum_forces_cub_kernel<Scalar, index_t, true, true, BlockSize>
          <<<gridSize, BlockSize>>>(a, b, c, Fr, Fa, R, drdrx, ijlist, tijlist,
                                    scale, device_args.f, device_args.vatom);
    } else {
      sum_forces_cub_kernel<Scalar, index_t, true, false, BlockSize>
          <<<gridSize, BlockSize>>>(a, b, c, Fr, Fa, R, drdrx, ijlist, tijlist,
                                    scale, device_args.f, device_args.vatom);
    }
  } else if (use_Fa) {
    sum_forces_cub_kernel<Scalar, index_t, false, true, BlockSize>
        <<<gridSize, BlockSize>>>(a, b, c, Fr, Fa, R, drdrx, ijlist, tijlist,
                                  scale, device_args.f, nullptr);
  } else {
    sum_forces_cub_kernel<Scalar, index_t, false, false, BlockSize>
        <<<gridSize, BlockSize>>>(a, b, c, Fr, Fa, R, drdrx, ijlist, tijlist,
                                  scale, device_args.f, nullptr);
  }
}

template <typename Scalar>
void sum_forces(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::SumForces);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(device_args.f);
  check_kernel_arg(device_args.Fr);
  check_kernel_arg(device_args.R);
  check_kernel_arg(device_args.drdrx);
  check_kernel_arg(device_args.tijlist);
  check_kernel_arg(device_args.ijlist);
  check_kernel_arg(device_args.a);
  check_kernel_arg(device_args.b);
  check_kernel_arg(device_args.c);
  check_kernel_arg(device_args.scale);
  check_kernel_arg(host_args.f);
  check_kernel_arg(host_args.nlocal);
  if (host_args.vatom != nullptr) {
    check_kernel_arg(device_args.vatom);
  }
#endif
  const size_t nall = static_cast<size_t>(host_args.nlocal) +
                      static_cast<size_t>(host_args.nghost);

  // Set force array and virial array to zero
  CHECK(cudaMemset(device_args.f, 0, sizeof(double) * nall * 3));
  if (host_args.vatom && device_args.vatom) {
    CHECK(cudaMemset(device_args.vatom, 0, sizeof(double) * nall * 6));
  }

  const size_t A = static_cast<size_t>(device_args.a);
  const size_t B = static_cast<size_t>(device_args.b);
  const size_t C = static_cast<size_t>(device_args.c);
  constexpr size_t kMaxIntFlatIndex =
      static_cast<size_t>(std::numeric_limits<int>::max());
  const bool needs_wide_index = A * B * C > (kMaxIntFlatIndex / 3) - 1;

  if (needs_wide_index) {
    dispatch_sum_forces_kernel<Scalar, size_t, 256>(host_args, device_args);
  } else {
    dispatch_sum_forces_kernel<Scalar, int, 256>(host_args, device_args);
  }

  CHECK(cudaGetLastError());

  double FLOPs = A * B * C * 12.0;
  if (host_args.vatom) {
    FLOPs += 42.0 * A * B * C;
  }

  // Copy back force array and virial array back to host
  CHECK(cudaMemcpy(host_args.f[0], device_args.f, sizeof(double) * nall * 3,
                   cudaMemcpyDeviceToHost));
  if (host_args.vatom && device_args.vatom) {
    CHECK(cudaMemcpy(host_args.vatom[0], device_args.vatom,
                     sizeof(double) * nall * 6, cudaMemcpyDeviceToHost));
  }
  Utils::GPU::timer_stop(Profiling::TimedKernel::SumForces, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void sum_forces<float>(HostArgs<float>& host_args,
                                DeviceArgs<float>& device_args);
template void sum_forces<double>(HostArgs<double>& host_args,
                                 DeviceArgs<double>& device_args);

}  // namespace TENSORMD::Kernels::GPU
