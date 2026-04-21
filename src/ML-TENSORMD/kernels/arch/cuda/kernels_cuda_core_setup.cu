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

#include <cstddef>
#include <stdexcept>

#include "../../../tensormd_constants.h"
#include "../../../tensormd_strategy.h"
#include "../../geometric/irreducible_moments.hpp"
#include "../../geometric/reducible_moments.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"
#include "cuda_math.cuh"

namespace TENSORMD::Kernels::GPU {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void initialize(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(device_args.d);
  check_kernel_arg(device_args.eltij_pos);
  check_kernel_arg(device_args.fmap);
  check_kernel_arg(host_args.fmap);
  check_kernel_arg(host_args.lmp_ntypes);
  check_kernel_arg(host_args.eltij_pos);
#endif

  const int b = device_args.b;

  // Copy eltij_pos and fmap to device
  // These (and numbers) are used for get_exact_cmax and setup_tensors
  CHECK(cudaMemcpy(device_args.eltij_pos, host_args.eltij_pos,
                   sizeof(int) * b * b, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device_args.fmap, host_args.fmap,
                   sizeof(int) * (host_args.lmp_ntypes + 1),
                   cudaMemcpyHostToDevice));

  if (host_args.numbers != nullptr) {
    // For TensorMDV2, we need to copy the static atomic number mapping to
    // device as well.
    CHECK(cudaMemcpy(device_args.numbers, host_args.numbers,
                     sizeof(int) * host_args.n_numbers,
                     cudaMemcpyHostToDevice));
  }
  CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------- */

template <typename index_t>
__global__ void get_exact_cmax_kernel(
    int inum, const int* __restrict__ ilist, const int* __restrict__ numneigh,
    const int* __restrict__ firstneigh_flat, int numneigh_max,
    const double* __restrict__ x, const int* __restrict__ type,
    const int* __restrict__ fmap, const int* __restrict__ eltij_pos,
    int num_blocks, double cutforcesq, int* d_global_cmax) {
  // 1 Warp handles exactly 1 atom `i`
  int ii = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  // Utilize shared memory for B > 1 to drastically lower register footprint
  extern __shared__ int shared_neigh_cmax[];
  int* neigh = &shared_neigh_cmax[(threadIdx.x / 32) * num_blocks];

  int my_cmax = 0;

  if (ii < inum) {
    int i = ilist[ii];
    const index_t firstneigh_offset =
        static_cast<index_t>(i) * static_cast<index_t>(numneigh_max);
    const int* firstneigh = firstneigh_flat + firstneigh_offset;
    int elti = fmap[type[i]];

    double3 tmp;
    tmp.x = x[i * 3 + 0];
    tmp.y = x[i * 3 + 1];
    tmp.z = x[i * 3 + 2];

    int nn = numneigh[i];

    if (num_blocks == 1) {
      // Extremely fast register path for B=1
      int current_c = 0;
      for (int jn_base = 0; jn_base < nn; jn_base += 32) {
        int jn = jn_base + lane;
        bool valid = false;
        if (jn < nn) {
          int j = firstneigh[jn] & NEIGHMASK;
          double dx = x[j * 3 + 0] - tmp.x;
          double dy = x[j * 3 + 1] - tmp.y;
          double dz = x[j * 3 + 2] - tmp.z;
          double rij2 = dx * dx + dy * dy + dz * dz;
          if (rij2 < cutforcesq && rij2 > 1e-10) {
            valid = true;
          }
        }
        unsigned mask = __ballot_sync(0xffffffff, valid);
        if (lane == 0) {
          current_c += __popc(mask);
        }
      }
      if (lane == 0) {
        my_cmax = current_c;
      }
    } else {
      // Cooperative tracking for B > 1
      for (int b = lane; b < num_blocks; b += 32) {
        neigh[b] = 0;
      }
      __syncwarp();

      for (int jn_base = 0; jn_base < nn; jn_base += 32) {
        int jn = jn_base + lane;
        bool valid = false;
        int b = 0;

        if (jn < nn) {
          int j = firstneigh[jn] & NEIGHMASK;
          int eltj = fmap[type[j]];
          b = eltij_pos[elti * num_blocks + eltj];

          double dx = x[j * 3 + 0] - tmp.x;
          double dy = x[j * 3 + 1] - tmp.y;
          double dz = x[j * 3 + 2] - tmp.z;
          double rij2 = dx * dx + dy * dy + dz * dz;

          if (rij2 < cutforcesq && rij2 > 1e-10) {
            valid = true;
          }
        }

        int match_val = valid ? b : -1;
        unsigned mask_b = __match_any_sync(0xffffffff, match_val);

        if (valid) {
          int count = __popc(mask_b);
          int leader = __ffs(mask_b) - 1;
          if (lane == leader) {
            neigh[b] += count;
          }
        }
        __syncwarp();
      }

      // Inter-thread max discovery for `b` arrays mapped to this atom
      int local_max = 0;
      for (int b = lane; b < num_blocks; b += 32) {
        local_max = max(local_max, neigh[b]);
      }

      // Fast warp reduction
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        local_max =
            max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
      }
      if (lane == 0) {
        my_cmax = local_max;
      }
    }
  }

  // Block-Level Reduction to prevent global atomic clashes
  __shared__ int block_max[32];  // Accommodates up to 1024 threads
  int warp_id = threadIdx.x / 32;

  if (lane == 0) {
    block_max[warp_id] = my_cmax;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_max = 0;
    int num_warps = blockDim.x / 32;
    for (int w = 0; w < num_warps; ++w) {
      final_max = max(final_max, block_max[w]);
    }
    if (final_max > 0) {
      atomicMax(d_global_cmax, final_max);
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int get_exact_cmax(DeviceArgs<Scalar>& device_args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::get_exact_cmax);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(device_args.inum);
  check_kernel_arg(device_args.b);
  check_kernel_arg(device_args.ilist);
  check_kernel_arg(device_args.numneigh);
  check_kernel_arg(device_args.x);
  check_kernel_arg(device_args.type);
  check_kernel_arg(device_args.eltij_pos);
  check_kernel_arg(device_args.cutforcesq);
#endif
  int h_cmax = 0;
  int* d_global_cmax = nullptr;

  // Allocate device memory for the result
  cudaMalloc((void**)&d_global_cmax, sizeof(int));
  cudaMemset(d_global_cmax, 0, sizeof(int));

  // Utilize larger thread count since arrays are cooperatively shared now
  int blockSize = 128;
  int warpsPerBlock = blockSize / 32;
  int gridSize = (device_args.inum + warpsPerBlock - 1) / warpsPerBlock;

  // Allocate shared memory only when needed (b > 1)
  int sharedMemSize =
      (device_args.b > 1) ? warpsPerBlock * device_args.b * sizeof(int) : 0;

  // Launch
  const int* __restrict__ ilist = device_args.ilist;
  const int* __restrict__ numneigh = device_args.numneigh;
  const int* __restrict__ firstneigh_flat = device_args.firstneigh_flat;
  const int numneigh_max = device_args.numneigh_max;
  const double* __restrict__ x = device_args.x;
  const int* __restrict__ type = device_args.type;
  const int* __restrict__ fmap = device_args.fmap;
  const int* __restrict__ eltij_pos = device_args.eltij_pos;
  const int inum = device_args.inum;
  const int b = device_args.b;
  const double cutforcesq = device_args.cutforcesq;

  if (device_args.inum > 1e7) {
    get_exact_cmax_kernel<size_t><<<gridSize, blockSize, sharedMemSize>>>(
        inum, ilist, numneigh, firstneigh_flat, numneigh_max, x, type, fmap,
        eltij_pos, b, cutforcesq, d_global_cmax);
  } else {
    get_exact_cmax_kernel<int><<<gridSize, blockSize, sharedMemSize>>>(
        inum, ilist, numneigh, firstneigh_flat, numneigh_max, x, type, fmap,
        eltij_pos, b, cutforcesq, d_global_cmax);
  }

  // Copy result back
  cudaMemcpy(&h_cmax, d_global_cmax, sizeof(int), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_global_cmax);

  CHECK(cudaGetLastError());
  Utils::GPU::timer_stop(Profiling::TimedKernel::get_exact_cmax, 0.0);
  return h_cmax;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar, typename index_t, int m_val, int D,
          bool UseReducible>
__global__ void setup_tensors_kernel_opt(
    const int* __restrict__ ilist, const int* __restrict__ i2a,
    const int* __restrict__ type, const int* __restrict__ fmap,
    const int* __restrict__ numneigh, const int* __restrict__ firstneigh_flat,
    const double* __restrict__ pos, double cutforcesq, Scalar* __restrict__ R,
    Scalar* __restrict__ M, Scalar* __restrict__ drdrx,
    int* __restrict__ ijlist, int* __restrict__ eltij_pos, int numneigh_max,
    int* __restrict__ numbers, int* __restrict__ tlist,
    int* __restrict__ zijlist, int* __restrict__ tijlist,
    const int* __restrict__ pair_map, int A, int B, int C) {
  // 1 Warp handles exactly 1 atom `i`
  int ii = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  extern __shared__ int shared_neigh[];
  int* neigh = &shared_neigh[(threadIdx.x / 32) * B];
  index_t A_t = static_cast<index_t>(A);
  index_t B_t = static_cast<index_t>(B);
  index_t C_t = static_cast<index_t>(C);
  index_t D_t = static_cast<index_t>(D);

  if (ii < A) {
    int i = ilist[ii];
    int a = (B == 1) ? i : i2a[i];
    int elti = fmap[type[i]];

    if (lane == 0 && tlist) {
      tlist[i] = elti;
    }

    // Initialize shared dynamic neigh array collaboratively
    for (int b = lane; b < B; b += 32) {
      neigh[b] = 0;
    }
    __syncwarp();

    const index_t firstneigh_offset =
        static_cast<index_t>(i) * static_cast<index_t>(numneigh_max);
    const int* firstneigh = firstneigh_flat + firstneigh_offset;

    double3 tmp;
    tmp.x = pos[i * 3 + 0];
    tmp.y = pos[i * 3 + 1];
    tmp.z = pos[i * 3 + 2];

    int nn = numneigh[i];

    // Process neighbors inside warp-level stripmining
    for (int jn_base = 0; jn_base < nn; jn_base += 32) {
      int jn = jn_base + lane;
      bool valid = false;
      int j, eltj, b;
      double rij2;
      double3 dij;

      if (jn < nn) {
        j = firstneigh[jn];
        j &= NEIGHMASK;
        eltj = fmap[type[j]];
        b = eltij_pos[elti * B + eltj];

        dij.x = pos[j * 3 + 0] - tmp.x;
        dij.y = pos[j * 3 + 1] - tmp.y;
        dij.z = pos[j * 3 + 2] - tmp.z;
        rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;

        if (rij2 < cutforcesq && rij2 > 1e-10) {
          valid = true;
        }
      }

      // Hardware-accelerated matching to compute uniform thread clusters that
      // share `b`
      int match_val = valid ? b : -1;
      unsigned mask_b = __match_any_sync(0xffffffff, match_val);

      if (valid) {
        int count = __popc(mask_b);
        int leader = __ffs(mask_b) - 1;
        if (lane == leader) {
          neigh[b] += count;  // Single point of accumulation eliminates any
                              // potential contention
        }
      }

      // Ensure neigh[b] update is visible to the rest of the threads before
      // index assignment
      __syncwarp();

      if (valid) {
        int count = __popc(mask_b);
        int rank = __popc(mask_b & ((1u << lane) - 1));
        int c =
            neigh[b] - count + rank;  // Extract safe offset keeping consecutive
                                      // exact bounds deterministically

        if (c < C) {
          double rij_exact = sqrt(rij2);
          double rijinv_exact = 1.0 / rij_exact;
          double x_exact = dij.x * rijinv_exact;
          double y_exact = dij.y * rijinv_exact;
          double z_exact = dij.z * rijinv_exact;
          Scalar rij = static_cast<Scalar>(rij_exact);
          index_t stride1 = static_cast<index_t>(a) * B_t * C_t +
                            static_cast<index_t>(b) * C_t +
                            static_cast<index_t>(c);
          index_t stride2 = stride1 * 2;
          index_t stride3 = stride1 * 3;

          R[stride1] = rij;

          Scalar x = static_cast<Scalar>(x_exact);
          Scalar y = static_cast<Scalar>(y_exact);
          Scalar z = static_cast<Scalar>(z_exact);
          drdrx[stride3 + 0] = x;
          drdrx[stride3 + 1] = y;
          drdrx[stride3 + 2] = z;

          ijlist[stride2 + 0] = i;
          ijlist[stride2 + 1] = j;

          if (pair_map) {
            const int pair_index = PAIR_KEY(numbers[elti], numbers[eltj]);
            tijlist[stride1] = pair_map[pair_index];
          } else {
            tijlist[stride1] = 0;
          }

          if (zijlist) {
            zijlist[stride2 + 0] = numbers[elti];
            zijlist[stride2 + 1] = numbers[eltj];
          }

          if (M) {
            index_t strided = stride1 * D_t;
            if constexpr (UseReducible) {
              Geometric::Reducible::calculate_M_d_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            } else {
              Geometric::Irreducible::calculate_M_u_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            }
          }
        }
      }
      __syncwarp();
    }

    // Flat-parallelized warp-level padding loop directly replacing nested logic
    // Massively outperforms generic loop assignments achieving perfect
    // coalescing pattern.
    for (int b = 0; b < B; b++) {
      int start_c = neigh[b];
      index_t pad_c = C_t - static_cast<index_t>(start_c);
      if (pad_c > 0) {
        index_t stride1_base = static_cast<index_t>(a) * B_t * C_t +
                               static_cast<index_t>(b) * C_t +
                               static_cast<index_t>(start_c);
        for (index_t idx = lane; idx < pad_c; idx += 32) {
          tijlist[stride1_base + idx] = -1;
        }
        for (index_t idx = lane; idx < pad_c; idx += 32) {
          R[stride1_base + idx] = 0.0;
        }
        for (index_t idx = lane; idx < pad_c * 2; idx += 32) {
          ijlist[stride1_base * 2 + idx] = -1;
        }
        if (zijlist) {
          for (index_t idx = lane; idx < pad_c * 2; idx += 32) {
            zijlist[stride1_base * 2 + idx] = 0;
          }
        }
        for (index_t idx = lane; idx < pad_c * 3; idx += 32) {
          drdrx[stride1_base * 3 + idx] = 0.0;
        }
        if (M) {
          for (index_t idx = lane; idx < pad_c * D; idx += 32) {
            M[stride1_base * D_t + idx] = 0.0;
          }
        }
      }
    }
  }
}

template <typename Scalar, typename index_t, int m_val, int D,
          bool UseReducible>
__global__ void setup_tensors_kernel_unitary(
    const int* __restrict__ ilist, const int* __restrict__ type,
    const int* __restrict__ fmap, const int* __restrict__ numneigh,
    const int* __restrict__ firstneigh_flat, const double* __restrict__ pos,
    double cutforcesq, Scalar* __restrict__ R, Scalar* __restrict__ M,
    Scalar* __restrict__ drdrx, int* __restrict__ ijlist, int numneigh_max,
    int* __restrict__ numbers, int* __restrict__ tlist,
    int* __restrict__ zijlist, int* __restrict__ tijlist,
    const int* __restrict__ pair_map, int A, int C) {
  // 1 Warp handles exactly 1 atom `i`
  int ii = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  index_t A_t = static_cast<index_t>(A);
  index_t C_t = static_cast<index_t>(C);
  index_t D_t = static_cast<index_t>(D);

  const double3* d_pos = reinterpret_cast<const double3*>(pos);

  if (ii < A) {
    int i = ilist[ii];
    int elti = fmap[type[i]];

    if (lane == 0 && tlist) {
      tlist[i] = elti;
    }

    const index_t firstneigh_offset =
        static_cast<index_t>(i) * static_cast<index_t>(numneigh_max);
    const int* firstneigh = firstneigh_flat + firstneigh_offset;

    double3 tmp = d_pos[i];

    int nn = numneigh[i];
    int current_c = 0;  // Replace shared memory with a fast register variable

    for (int jn_base = 0; jn_base < nn; jn_base += 32) {
      int jn = jn_base + lane;
      bool valid = false;
      int j, eltj;
      double rij2;
      double3 dij;

      if (jn < nn) {
        j = firstneigh[jn] & NEIGHMASK;
        eltj = fmap[type[j]];

        dij.x = pos[j * 3 + 0] - tmp.x;
        dij.y = pos[j * 3 + 1] - tmp.y;
        dij.z = pos[j * 3 + 2] - tmp.z;
        rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;

        if (rij2 < cutforcesq && rij2 > 1e-10) {
          valid = true;
        }
      }

      // Fast single-category matching via ballot
      unsigned mask = __ballot_sync(0xffffffff, valid);

      if (valid) {
        int rank = __popc(mask & ((1u << lane) - 1));
        int c = current_c + rank;  // Deterministic allocation

        if (c < C) {
          double rij_exact = sqrt(rij2);
          double rijinv_exact = 1.0 / rij_exact;
          double x_exact = dij.x * rijinv_exact;
          double y_exact = dij.y * rijinv_exact;
          double z_exact = dij.z * rijinv_exact;
          Scalar rij = static_cast<Scalar>(rij_exact);
          index_t stride1 =
              static_cast<index_t>(i) * C_t + static_cast<index_t>(c);
          index_t stride2 = stride1 * 2;
          index_t stride3 = stride1 * 3;

          R[stride1] = rij;
          Scalar x = static_cast<Scalar>(x_exact);
          Scalar y = static_cast<Scalar>(y_exact);
          Scalar z = static_cast<Scalar>(z_exact);
          drdrx[stride3 + 0] = x;
          drdrx[stride3 + 1] = y;
          drdrx[stride3 + 2] = z;

          ijlist[stride2 + 0] = i;
          ijlist[stride2 + 1] = j;

          if (pair_map) {
            const int pair_index = PAIR_KEY(numbers[elti], numbers[eltj]);
            tijlist[stride1] = pair_map[pair_index];
          } else {
            tijlist[stride1] = 0;
          }

          if (zijlist) {
            zijlist[stride2 + 0] = numbers[elti];
            zijlist[stride2 + 1] = numbers[eltj];
          }

          if (M) {
            index_t strided = stride1 * D_t;
            if constexpr (UseReducible) {
              Geometric::Reducible::calculate_M_d_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            } else {
              Geometric::Irreducible::calculate_M_u_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            }
          }
        }
      }
      current_c += __popc(mask);  // Advance warp counter
    }

    // Flat-parallelized warp padding for single block
    index_t pad_c = C - current_c;
    if (pad_c > 0) {
      index_t stride1_base =
          static_cast<index_t>(i) * C_t + static_cast<index_t>(current_c);

      for (index_t idx = lane; idx < pad_c; idx += 32) {
        tijlist[stride1_base + idx] = -1;
        R[stride1_base + idx] = 0.0;
      }
      for (index_t idx = lane; idx < pad_c * 2; idx += 32) {
        ijlist[stride1_base * 2 + idx] = -1;
      }
      if (zijlist) {
        for (index_t idx = lane; idx < pad_c * 2; idx += 32) {
          zijlist[stride1_base * 2 + idx] = 0;
        }
      }
      for (index_t idx = lane; idx < pad_c * 3; idx += 32) {
        drdrx[stride1_base * 3 + idx] = 0.0;
      }
      if (M) {
        for (index_t idx = lane; idx < pad_c * D_t; idx += 32) {
          M[stride1_base * D_t + idx] = 0.0;
        }
      }
    }
  }
}

template <typename Scalar, typename index_t, int m_val, int D,
          bool UseReducible, int MaxNTypes = 16>
__global__ void setup_tensors_kernel_unitary_sorted(
    const int* __restrict__ ilist, const int* __restrict__ type,
    const int* __restrict__ fmap, const int* __restrict__ numneigh,
    const int* __restrict__ firstneigh_flat, const double* __restrict__ pos,
    Scalar cutforcesq, Scalar* __restrict__ R, Scalar* __restrict__ M,
    Scalar* __restrict__ drdrx, int* __restrict__ ijlist, int numneigh_max,
    int* __restrict__ numbers, int* __restrict__ tlist,
    int* __restrict__ zijlist, int* __restrict__ tijlist,
    const int* __restrict__ pair_map, int A, int C) {
  // 1 Warp handles exactly 1 atom `i`
  int ii = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  // Warp-specific shared memory counters for bucket sort
  extern __shared__ int shared_sort_buffer[];
  int* warp_offsets = &shared_sort_buffer[(threadIdx.x / 32) * MaxNTypes];

  if (ii < A) {
    const index_t A_t = static_cast<index_t>(A);
    const index_t C_t = static_cast<index_t>(C);
    const index_t D_t = static_cast<index_t>(D);
    int i = ilist[ii];
    int elti = fmap[type[i]];

    if (lane == 0 && tlist) tlist[i] = elti;

    const index_t firstneigh_offset =
        static_cast<index_t>(i) * static_cast<index_t>(numneigh_max);
    const int* firstneigh = firstneigh_flat + firstneigh_offset;

    double3 tmp;
    tmp.x = pos[i * 3 + 0];
    tmp.y = pos[i * 3 + 1];
    tmp.z = pos[i * 3 + 2];

    int nn = numneigh[i];

    // Initialize bucket counters
    for (int t = lane; t < MaxNTypes; t += 32) {
      warp_offsets[t] = 0;
    }
    __syncwarp();

    // ==========================================
    // PASS 1: Count occurrences of each pair type
    // ==========================================
    for (int jn_base = 0; jn_base < nn; jn_base += 32) {
      int jn = jn_base + lane;
      bool valid = false;
      int eltj = -1;

      if (jn < nn) {
        int j = firstneigh[jn] & NEIGHMASK;
        double3 dij;
        dij.x = pos[j * 3 + 0] - tmp.x;
        dij.y = pos[j * 3 + 1] - tmp.y;
        dij.z = pos[j * 3 + 2] - tmp.z;
        Scalar rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;

        if (rij2 < cutforcesq && rij2 > 1e-10) {
          valid = true;
          eltj = fmap[type[j]];
        }
      }

      // Hardware match on the neighbor species instead of global pair type
      unsigned mask_eltj = __match_any_sync(0xffffffff, valid ? eltj : -1);
      if (valid && eltj < MaxNTypes) {
        int count = __popc(mask_eltj);
        int leader = __ffs(mask_eltj) - 1;
        if (lane == leader) {
          warp_offsets[eltj] += count;
        }
      }
    }
    __syncwarp();

    // ==========================================
    // INTERLUDE: Prefix Sum to generate offsets
    // ==========================================
    int total_valid = 0;
    if (lane == 0) {
      int current_offset = 0;
      for (int t = 0; t < MaxNTypes; t++) {
        int count = warp_offsets[t];
        warp_offsets[t] = current_offset;
        current_offset += count;
      }
      total_valid = current_offset;
    }
    total_valid =
        __shfl_sync(0xffffffff, total_valid, 0);  // Broadcast to all threads
    __syncwarp();

    // ==========================================
    // PASS 2: Write arrays perfectly grouped by `t`
    // ==========================================
    for (int jn_base = 0; jn_base < nn; jn_base += 32) {
      int jn = jn_base + lane;
      bool valid = false;
      int j, eltj = -1;
      Scalar rij2;
      double3 dij;

      if (jn < nn) {
        j = firstneigh[jn] & NEIGHMASK;
        eltj = fmap[type[j]];
        dij.x = pos[j * 3 + 0] - tmp.x;
        dij.y = pos[j * 3 + 1] - tmp.y;
        dij.z = pos[j * 3 + 2] - tmp.z;
        rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;

        if (rij2 < cutforcesq && rij2 > 1e-10) {
          valid = true;
          eltj = fmap[type[j]];
        }
      }

      unsigned mask_eltj = __match_any_sync(0xffffffff, valid ? eltj : -1);
      if (valid && eltj < MaxNTypes) {
        int count = __popc(mask_eltj);
        int rank = __popc(mask_eltj & ((1u << lane) - 1));
        int leader = __ffs(mask_eltj) - 1;
        int base_idx;
        if (lane == leader) {
          base_idx = warp_offsets[eltj];
          warp_offsets[eltj] += count;
        }
        base_idx = __shfl_sync(mask_eltj, base_idx, leader);

        int c = base_idx + rank;
        if (c < C) {
          int t = pair_map[PAIR_KEY(numbers[elti], numbers[eltj])];
          Scalar rij = sqrt(rij2);
          index_t stride1 =
              static_cast<index_t>(i) * C_t + static_cast<index_t>(c);
          index_t stride2 = stride1 * 2;
          index_t stride3 = stride1 * 3;

          R[stride1] = rij;
          Scalar rijinv = 1.0 / rij;
          Scalar x = dij.x * rijinv;
          Scalar y = dij.y * rijinv;
          Scalar z = dij.z * rijinv;
          drdrx[stride3 + 0] = x;
          drdrx[stride3 + 1] = y;
          drdrx[stride3 + 2] = z;

          ijlist[stride2 + 0] = i;
          ijlist[stride2 + 1] = j;
          tijlist[stride1] = t;

          if (zijlist) {
            zijlist[stride2 + 0] = numbers[elti];
            zijlist[stride2 + 1] = numbers[eltj];
          }
          if (M) {
            index_t strided = stride1 * D_t;
            if constexpr (UseReducible) {
              Geometric::Reducible::calculate_M_d_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            } else {
              Geometric::Irreducible::calculate_M_u_otf<Scalar, m_val>(
                  x, y, z, M + strided);
            }
          }
        }
      }
    }

    // ==========================================
    // PADDING: Fill the rest up to C
    // ==========================================
    int pad_count = C - total_valid;
    if (pad_count > 0) {
      index_t base =
          static_cast<index_t>(i) * C_t + static_cast<index_t>(total_valid);
      for (index_t p = lane; p < pad_count; p += 32) {
        tijlist[base + p] = -1;
        R[base + p] = 0.0;
        ijlist[(base + p) * 2 + 0] = -1;
        ijlist[(base + p) * 2 + 1] = -1;
        if (zijlist) {
          zijlist[(base + p) * 2 + 0] = 0;
          zijlist[(base + p) * 2 + 1] = 0;
        }
        drdrx[(base + p) * 3 + 0] = 0.0;
        drdrx[(base + p) * 3 + 1] = 0.0;
        drdrx[(base + p) * 3 + 2] = 0.0;
        if (M) {
          for (index_t d = 0; d < D_t; d++)
            M[(base + p) * D_t + d] = 0.0;  // Keep loop unrolled for D
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar, typename index_t, int m_val, int D,
          bool UseReducible>
void dispatch_setup_tensors(DeviceArgs<Scalar>& device_args) {
  const int A = device_args.a;
  const int B = device_args.b;
  const int C = device_args.c;

  const int* __restrict__ ilist = device_args.ilist;
  const int inum = device_args.inum;
  const int* __restrict__ type = device_args.type;
  const int* __restrict__ fmap = device_args.fmap;
  int* __restrict__ elt_a_offset = device_args.elt_a_offset;
  int* __restrict__ i2a = device_args.i2a;
  int* __restrict__ a2i = device_args.a2i;
  int* __restrict__ numneigh = device_args.numneigh;
  const int* __restrict__ firstneigh_flat = device_args.firstneigh_flat;
  const double* __restrict__ x = device_args.x;
  double cutforcesq = device_args.cutforcesq;
  Scalar* __restrict__ R = device_args.R;
  Scalar* __restrict__ M = device_args.M;
  Scalar* __restrict__ drdrx = device_args.drdrx;
  int* __restrict__ ijlist = device_args.ijlist;
  int* __restrict__ eltij_pos = device_args.eltij_pos;
  int numneigh_max = device_args.numneigh_max;
  int* __restrict__ numbers = device_args.numbers;
  int* __restrict__ tlist = device_args.tlist;
  int* __restrict__ zijlist = device_args.zijlist;
  int* __restrict__ tijlist = device_args.tijlist;
  const int* __restrict__ pair_map = device_args.pair_map;

  // Utilize 1 warp (32 threads) per element layout matching.
  // Launching 128 threads perfectly enables reaching max SM allocation ceilings
  int blockSize = 128;
  int warpsPerBlock = blockSize / 32;
  int numBlocks = (A + warpsPerBlock - 1) / warpsPerBlock;
  int sharedMemSize = warpsPerBlock * B * sizeof(int);
  const int MaxNTypes = 16;  // For the sorted kernel variant, adjust as needed

  if (B == 1) {
    if (TENSORMD_GROUP_BY_TYPE) {
      sharedMemSize *= MaxNTypes;
      setup_tensors_kernel_unitary_sorted<Scalar, index_t, m_val, D,
                                          UseReducible, MaxNTypes>
          <<<numBlocks, blockSize>>>(ilist, type, fmap, numneigh,
                                     firstneigh_flat, x, cutforcesq, R, M,
                                     drdrx, ijlist, numneigh_max, numbers,
                                     tlist, zijlist, tijlist, pair_map, A, C);
    } else {
      setup_tensors_kernel_unitary<Scalar, index_t, m_val, D, UseReducible>
          <<<numBlocks, blockSize>>>(
              ilist, type, fmap, numneigh, firstneigh_flat, x, cutforcesq, R, M,
              drdrx, ijlist, numneigh_max, numbers, tlist, zijlist, tijlist,
              pair_map, A, C);  // clang-format on
    }
  } else {
    setup_tensors_kernel_opt<Scalar, index_t, m_val, D, UseReducible>
        <<<numBlocks, blockSize, sharedMemSize>>>(
            ilist, i2a, type, fmap, numneigh, firstneigh_flat, x, cutforcesq, R,
            M, drdrx, ijlist, eltij_pos, numneigh_max, numbers, tlist, zijlist,
            tijlist, pair_map, A, B, C);
  }
}

template <typename Scalar, int m_val, int D, bool UseReducible>
void dispatch_setup_tensors_by_index_type(DeviceArgs<Scalar>& device_args) {
  if (device_args.inum < 1e7)
    dispatch_setup_tensors<Scalar, int, m_val, D, UseReducible>(device_args);
  else
    dispatch_setup_tensors<Scalar, size_t, m_val, D, UseReducible>(device_args);
}

template <typename Scalar, bool UseReducible>
void dispatch_setup_tensors_by_moments(DeviceArgs<Scalar>& device_args) {
  if (UseReducible) {
    switch (device_args.m) {
      case 1:
        dispatch_setup_tensors_by_index_type<Scalar, 1, 1, true>(device_args);
        break;
      case 2:
        dispatch_setup_tensors_by_index_type<Scalar, 2, 4, true>(device_args);
        break;
      case 3:
        dispatch_setup_tensors_by_index_type<Scalar, 3, 10, true>(device_args);
        break;
      case 4:
        dispatch_setup_tensors_by_index_type<Scalar, 4, 20, true>(device_args);
        break;
      case 5:
        dispatch_setup_tensors_by_index_type<Scalar, 5, 35, true>(device_args);
        break;
      case 6:
        dispatch_setup_tensors_by_index_type<Scalar, 6, 56, true>(device_args);
        break;
      default:
        throw std::runtime_error("Unsupported m value in setup_tensors");
    }
  } else {
    switch (device_args.m) {
      case 1:
        dispatch_setup_tensors_by_index_type<Scalar, 1, 1, false>(device_args);
        break;
      case 2:
        dispatch_setup_tensors_by_index_type<Scalar, 2, 4, false>(device_args);
        break;
      case 3:
        dispatch_setup_tensors_by_index_type<Scalar, 3, 9, false>(device_args);
        break;
      case 4:
        dispatch_setup_tensors_by_index_type<Scalar, 4, 16, false>(device_args);
        break;
      case 5:
        dispatch_setup_tensors_by_index_type<Scalar, 5, 25, false>(device_args);
        break;
      case 6:
        dispatch_setup_tensors_by_index_type<Scalar, 6, 36, false>(device_args);
        break;
      default:
        throw std::runtime_error("Unsupported m value in setup_tensors");
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void setup_tensors(DeviceArgs<Scalar>& device_args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::setup_tensors);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(device_args.a);
  check_kernel_arg(device_args.b);
  check_kernel_arg(device_args.c);
  check_kernel_arg(device_args.d);
  check_kernel_arg(device_args.m);
  check_kernel_arg(device_args.ilist);
  check_kernel_arg(device_args.numneigh);
  check_kernel_arg(device_args.eltij_pos);
  check_kernel_arg(device_args.x);
  check_kernel_arg(device_args.type);
  check_kernel_arg(device_args.cutforcesq);
  check_kernel_arg(device_args.tijlist);
  check_kernel_arg(device_args.ijlist);
  check_kernel_arg(device_args.R);
  check_kernel_arg(device_args.drdrx);
  if (device_args.b > 1) {
    check_kernel_arg(device_args.i2a);
  }
#endif
  const int A = device_args.a;
  const int B = device_args.b;
  const int C = device_args.c;

  if (device_args.T_proj_algo < 2) {
    dispatch_setup_tensors_by_moments<Scalar, true>(device_args);
  } else {
    dispatch_setup_tensors_by_moments<Scalar, false>(device_args);
  }

  CHECK(cudaGetLastError());
  double FLOPs = 0.0;
  if (device_args.M) {
    if (device_args.T_proj_algo < 2) {
      FLOPs = Geometric::Reducible::M_d_FLOPs[device_args.m];
    } else {
      FLOPs = Geometric::Irreducible::M_u_FLOPs[device_args.m];
    }
    FLOPs *= 1.0 * A * B * C;
  }
  Utils::GPU::timer_stop(Profiling::TimedKernel::setup_tensors, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void initialize<float>(HostArgs<float>& host_args,
                                DeviceArgs<float>& device_args);
template void initialize<double>(HostArgs<double>& host_args,
                                 DeviceArgs<double>& device_args);

template int get_exact_cmax<float>(DeviceArgs<float>& device_args);
template int get_exact_cmax<double>(DeviceArgs<double>& device_args);

template void setup_tensors<float>(DeviceArgs<float>& device_args);
template void setup_tensors<double>(DeviceArgs<double>& device_args);

}  // namespace TENSORMD::Kernels::GPU
