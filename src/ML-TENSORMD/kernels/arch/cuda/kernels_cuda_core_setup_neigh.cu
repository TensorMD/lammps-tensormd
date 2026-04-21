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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cub/cub.cuh>

#include "../../../tensormd_constants.h"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::GPU {

struct GpuNeighborWorkspace {
  // binning
  int* d_atom_bin = nullptr;
  int* d_cell_counts = nullptr;
  int* d_cell_offsets = nullptr;
  int* d_cell_fill = nullptr;
  int* d_cell_atoms = nullptr;

  // final list
  int* d_neighbors_flat = nullptr;
  int* d_overflow_flag = nullptr;

  // cub temp
  void* d_scan_temp = nullptr;
  size_t scan_temp_bytes = 0;

  void* d_reduce_temp = nullptr;
  size_t reduce_temp_bytes = 0;

  // reduction outputs
  int* d_reduce_out_int = nullptr;

  int cap_nall = 0;
  int cap_nbins = 0;
  size_t cap_neighbor_ints = 0;

  void clear() {
    if (d_atom_bin) {
      CHECK(cudaFree(d_atom_bin));
      d_atom_bin = nullptr;
    }
    if (d_cell_counts) {
      CHECK(cudaFree(d_cell_counts));
      d_cell_counts = nullptr;
    }
    if (d_cell_offsets) {
      CHECK(cudaFree(d_cell_offsets));
      d_cell_offsets = nullptr;
    }
    if (d_cell_fill) {
      CHECK(cudaFree(d_cell_fill));
      d_cell_fill = nullptr;
    }
    if (d_cell_atoms) {
      CHECK(cudaFree(d_cell_atoms));
      d_cell_atoms = nullptr;
    }
    if (d_neighbors_flat) {
      CHECK(cudaFree(d_neighbors_flat));
      d_neighbors_flat = nullptr;
    }
    if (d_overflow_flag) {
      CHECK(cudaFree(d_overflow_flag));
      d_overflow_flag = nullptr;
    }
    if (d_scan_temp) {
      CHECK(cudaFree(d_scan_temp));
      d_scan_temp = nullptr;
    }
    if (d_reduce_temp) {
      CHECK(cudaFree(d_reduce_temp));
      d_reduce_temp = nullptr;
    }
    if (d_reduce_out_int) {
      CHECK(cudaFree(d_reduce_out_int));
      d_reduce_out_int = nullptr;
    }

    cap_nall = 0;
    cap_nbins = 0;
    cap_neighbor_ints = 0;
    scan_temp_bytes = 0;
    reduce_temp_bytes = 0;
  }

  ~GpuNeighborWorkspace() { clear(); }
};

static GpuNeighborWorkspace neighlist_ws;
static double neighlist_ws_mem = 0.0;

// -------------------------------
// Clean up
// -------------------------------

void clear_neighbor_list_workspace() { neighlist_ws.clear(); }

// -------------------------------
// Memory usage estimation
// -------------------------------

void update_neighbor_list_memory_usage() {
  size_t total_bytes = 0;
  if (neighlist_ws.d_atom_bin) {
    total_bytes += static_cast<size_t>(neighlist_ws.cap_nall) * sizeof(int);
  }
  if (neighlist_ws.d_cell_counts) {
    total_bytes += static_cast<size_t>(neighlist_ws.cap_nbins) * sizeof(int);
  }
  if (neighlist_ws.d_cell_offsets) {
    total_bytes += static_cast<size_t>(neighlist_ws.cap_nbins) * sizeof(int);
  }
  if (neighlist_ws.d_cell_fill) {
    total_bytes += static_cast<size_t>(neighlist_ws.cap_nbins) * sizeof(int);
  }
  if (neighlist_ws.d_cell_atoms) {
    total_bytes += static_cast<size_t>(neighlist_ws.cap_nall) * sizeof(int);
  }
  if (neighlist_ws.d_neighbors_flat) {
    total_bytes += neighlist_ws.cap_neighbor_ints * sizeof(int);
  }
  if (neighlist_ws.d_overflow_flag) {
    total_bytes += sizeof(int);
  }
  if (neighlist_ws.d_scan_temp) {
    total_bytes += neighlist_ws.scan_temp_bytes;
  }
  if (neighlist_ws.d_reduce_temp) {
    total_bytes += neighlist_ws.reduce_temp_bytes;
  }
  if (neighlist_ws.d_reduce_out_int) {
    total_bytes += sizeof(int);
  }
  neighlist_ws_mem = static_cast<double>(total_bytes);
}

double get_neighbor_list_memory_usage() { return neighlist_ws_mem; }

// -------------------------------
// Device helpers
// -------------------------------

static inline int round_up_int(int x, int pad) {
  return ((x + pad - 1) / pad) * pad;
}

__device__ __forceinline__ double rsq_cart(const double* __restrict__ x, int i,
                                           int j) {
  const double dx = x[3 * j + 0] - x[3 * i + 0];
  const double dy = x[3 * j + 1] - x[3 * i + 1];
  const double dz = x[3 * j + 2] - x[3 * i + 2];
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ int flatten_bin(int ix, int iy, int iz, int nbinx,
                                           int nbiny, int /*nbinz*/) {
  return ix + nbinx * (iy + nbiny * iz);
}

// ============================================================
// Device kernels: binning
// ============================================================

__global__ void kernel_assign_bins_and_histogram(
    int nall, const double* __restrict__ x, double xmin, double ymin,
    double zmin, double inv_binsize, int nbinx, int nbiny, int nbinz,
    int* __restrict__ atom_bin, int* __restrict__ cell_counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nall) return;

  const double rx = x[3 * i + 0];
  const double ry = x[3 * i + 1];
  const double rz = x[3 * i + 2];

  int ix = (int)((rx - xmin) * inv_binsize);
  int iy = (int)((ry - ymin) * inv_binsize);
  int iz = (int)((rz - zmin) * inv_binsize);

  // Clamp for safety
  if (ix < 0)
    ix = 0;
  else if (ix >= nbinx)
    ix = nbinx - 1;
  if (iy < 0)
    iy = 0;
  else if (iy >= nbiny)
    iy = nbiny - 1;
  if (iz < 0)
    iz = 0;
  else if (iz >= nbinz)
    iz = nbinz - 1;

  const int b = flatten_bin(ix, iy, iz, nbinx, nbiny, nbinz);
  atom_bin[i] = b;
  atomicAdd(&cell_counts[b], 1);
}

__global__ void kernel_scatter_atoms_to_bins(
    int nall, const int* __restrict__ atom_bin,
    const int* __restrict__ cell_offsets, int* __restrict__ cell_fill,
    int* __restrict__ cell_atoms) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nall) return;

  const int b = atom_bin[i];
  const int pos = atomicAdd(&cell_fill[b], 1);
  cell_atoms[pos] = i;
}

// ============================================================
// Device kernels: count neighbors
// ============================================================

__global__ void kernel_count_neighbors_uniform(
    int inum, const int* __restrict__ ilist, const double* __restrict__ x,
    const int* __restrict__ type,  // unused, kept for signature symmetry
    double xmin, double ymin, double zmin, double inv_binsize, int nbinx,
    int nbiny, int nbinz, int sx, int sy, int sz,
    const int* __restrict__ cell_offsets, const int* __restrict__ cell_counts,
    const int* __restrict__ cell_atoms, double cutsq_uniform,
    int* __restrict__ numneigh) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= inum) return;
  const int i = ilist[idx];

  const double rx = x[3 * i + 0];
  const double ry = x[3 * i + 1];
  const double rz = x[3 * i + 2];

  int bix = (int)((rx - xmin) * inv_binsize);
  int biy = (int)((ry - ymin) * inv_binsize);
  int biz = (int)((rz - zmin) * inv_binsize);

  if (bix < 0)
    bix = 0;
  else if (bix >= nbinx)
    bix = nbinx - 1;
  if (biy < 0)
    biy = 0;
  else if (biy >= nbiny)
    biy = nbiny - 1;
  if (biz < 0)
    biz = 0;
  else if (biz >= nbinz)
    biz = nbinz - 1;

  int count = 0;

  const int ix0 = max(0, bix - sx);
  const int ix1 = min(nbinx - 1, bix + sx);
  const int iy0 = max(0, biy - sy);
  const int iy1 = min(nbiny - 1, biy + sy);
  const int iz0 = max(0, biz - sz);
  const int iz1 = min(nbinz - 1, biz + sz);

  for (int iz = iz0; iz <= iz1; iz++) {
    for (int iy = iy0; iy <= iy1; iy++) {
      for (int ix = ix0; ix <= ix1; ix++) {
        const int bin = flatten_bin(ix, iy, iz, nbinx, nbiny, nbinz);
        const int start = cell_offsets[bin];
        const int n = cell_counts[bin];

        for (int t = 0; t < n; t++) {
          const int j = cell_atoms[start + t];
          if (j == i) continue;

          const double rsq = rsq_cart(x, i, j);
          if (rsq < cutsq_uniform) {
            count++;
          }
        }
      }
    }
  }

  numneigh[i] = count;
}

__global__ void kernel_count_neighbors_typed(
    int inum, const int* __restrict__ ilist, const double* __restrict__ x,
    const int* __restrict__ type, double xmin, double ymin, double zmin,
    double inv_binsize, int nbinx, int nbiny, int nbinz, int sx, int sy, int sz,
    const int* __restrict__ cell_offsets, const int* __restrict__ cell_counts,
    const int* __restrict__ cell_atoms,
    const double* __restrict__ cutsq_table,  // [ld * ld], 1-based types
    int ld, int* __restrict__ numneigh) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= inum) return;
  const int i = ilist[idx];

  const double rx = x[3 * i + 0];
  const double ry = x[3 * i + 1];
  const double rz = x[3 * i + 2];
  const int ti = type[i];

  int bix = (int)((rx - xmin) * inv_binsize);
  int biy = (int)((ry - ymin) * inv_binsize);
  int biz = (int)((rz - zmin) * inv_binsize);

  if (bix < 0)
    bix = 0;
  else if (bix >= nbinx)
    bix = nbinx - 1;
  if (biy < 0)
    biy = 0;
  else if (biy >= nbiny)
    biy = nbiny - 1;
  if (biz < 0)
    biz = 0;
  else if (biz >= nbinz)
    biz = nbinz - 1;

  int count = 0;

  const int ix0 = max(0, bix - sx);
  const int ix1 = min(nbinx - 1, bix + sx);
  const int iy0 = max(0, biy - sy);
  const int iy1 = min(nbiny - 1, biy + sy);
  const int iz0 = max(0, biz - sz);
  const int iz1 = min(nbinz - 1, biz + sz);

  for (int iz = iz0; iz <= iz1; iz++) {
    for (int iy = iy0; iy <= iy1; iy++) {
      for (int ix = ix0; ix <= ix1; ix++) {
        const int bin = flatten_bin(ix, iy, iz, nbinx, nbiny, nbinz);
        const int start = cell_offsets[bin];
        const int n = cell_counts[bin];

        for (int t = 0; t < n; t++) {
          const int j = cell_atoms[start + t];
          if (j == i) continue;

          const double rsq = rsq_cart(x, i, j);
          const double cutsq = cutsq_table[ti * ld + type[j]];
          if (rsq < cutsq) count++;
        }
      }
    }
  }

  numneigh[i] = count;
}

// ============================================================
// Device kernels: fill neighbors
// ============================================================

__global__ void kernel_fill_neighbors_uniform(
    int inum, const int* __restrict__ ilist, const double* __restrict__ x,
    const int* __restrict__ type,  // unused, kept for signature symmetry
    double xmin, double ymin, double zmin, double inv_binsize, int nbinx,
    int nbiny, int nbinz, int sx, int sy, int sz,
    const int* __restrict__ cell_offsets, const int* __restrict__ cell_counts,
    const int* __restrict__ cell_atoms, double cutsq_uniform, int neighstride,
    int* __restrict__ numneigh, int* __restrict__ neighbors_flat,
    int* __restrict__ overflow_flag) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= inum) return;
  const int i = ilist[idx];

  const double rx = x[3 * i + 0];
  const double ry = x[3 * i + 1];
  const double rz = x[3 * i + 2];

  int bix = (int)((rx - xmin) * inv_binsize);
  int biy = (int)((ry - ymin) * inv_binsize);
  int biz = (int)((rz - zmin) * inv_binsize);

  if (bix < 0)
    bix = 0;
  else if (bix >= nbinx)
    bix = nbinx - 1;
  if (biy < 0)
    biy = 0;
  else if (biy >= nbiny)
    biy = nbiny - 1;
  if (biz < 0)
    biz = 0;
  else if (biz >= nbinz)
    biz = nbinz - 1;

  int* row = neighbors_flat + (size_t)i * (size_t)neighstride;
  int k = 0;

  const int ix0 = max(0, bix - sx);
  const int ix1 = min(nbinx - 1, bix + sx);
  const int iy0 = max(0, biy - sy);
  const int iy1 = min(nbiny - 1, biy + sy);
  const int iz0 = max(0, biz - sz);
  const int iz1 = min(nbinz - 1, biz + sz);

  for (int iz = iz0; iz <= iz1; iz++) {
    for (int iy = iy0; iy <= iy1; iy++) {
      for (int ix = ix0; ix <= ix1; ix++) {
        const int bin = flatten_bin(ix, iy, iz, nbinx, nbiny, nbinz);
        const int start = cell_offsets[bin];
        const int n = cell_counts[bin];

        for (int t = 0; t < n; t++) {
          const int j = cell_atoms[start + t];
          if (j == i) continue;

          const double rsq = rsq_cart(x, i, j);
          if (rsq < cutsq_uniform) {
            if (k < neighstride) {
              row[k] = j;
            } else {
              atomicExch(overflow_flag, 1);
            }
            k++;
          }
        }
      }
    }
  }

  numneigh[i] = k;
}

__global__ void kernel_fill_neighbors_typed(
    int inum, const int* __restrict__ ilist, const double* __restrict__ x,
    const int* __restrict__ type, double xmin, double ymin, double zmin,
    double inv_binsize, int nbinx, int nbiny, int nbinz, int sx, int sy, int sz,
    const int* __restrict__ cell_offsets, const int* __restrict__ cell_counts,
    const int* __restrict__ cell_atoms, const double* __restrict__ cutsq_table,
    int ld, int neighstride, int* __restrict__ numneigh,
    int* __restrict__ neighbors_flat, int* __restrict__ overflow_flag) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= inum) return;
  const int i = ilist[idx];

  const double rx = x[3 * i + 0];
  const double ry = x[3 * i + 1];
  const double rz = x[3 * i + 2];
  const int ti = type[i];

  int bix = (int)((rx - xmin) * inv_binsize);
  int biy = (int)((ry - ymin) * inv_binsize);
  int biz = (int)((rz - zmin) * inv_binsize);

  if (bix < 0)
    bix = 0;
  else if (bix >= nbinx)
    bix = nbinx - 1;
  if (biy < 0)
    biy = 0;
  else if (biy >= nbiny)
    biy = nbiny - 1;
  if (biz < 0)
    biz = 0;
  else if (biz >= nbinz)
    biz = nbinz - 1;

  int* row = neighbors_flat + (size_t)i * (size_t)neighstride;
  int k = 0;

  const int ix0 = max(0, bix - sx);
  const int ix1 = min(nbinx - 1, bix + sx);
  const int iy0 = max(0, biy - sy);
  const int iy1 = min(nbiny - 1, biy + sy);
  const int iz0 = max(0, biz - sz);
  const int iz1 = min(nbinz - 1, biz + sz);

  for (int iz = iz0; iz <= iz1; iz++) {
    for (int iy = iy0; iy <= iy1; iy++) {
      for (int ix = ix0; ix <= ix1; ix++) {
        const int bin = flatten_bin(ix, iy, iz, nbinx, nbiny, nbinz);
        const int start = cell_offsets[bin];
        const int n = cell_counts[bin];

        for (int t = 0; t < n; t++) {
          const int j = cell_atoms[start + t];
          if (j == i) continue;

          const double rsq = rsq_cart(x, i, j);
          const double cutsq = cutsq_table[ti * ld + type[j]];
          if (rsq < cutsq) {
            if (k < neighstride) {
              row[k] = j;
            } else {
              atomicExch(overflow_flag, 1);
            }
            k++;
          }
        }
      }
    }
  }

  numneigh[i] = k;
}

// ============================================================
// Workspace
// ============================================================

void reserve_basic_workspace(GpuNeighborWorkspace& ws, int nall, int nbins) {
  if (nall > ws.cap_nall) {
    cudaFree(ws.d_atom_bin);
    cudaFree(ws.d_cell_atoms);

    CHECK(cudaMalloc(&ws.d_atom_bin, sizeof(int) * (size_t)nall));
    CHECK(cudaMalloc(&ws.d_cell_atoms, sizeof(int) * (size_t)nall));

    ws.cap_nall = nall;
  }

  if (nbins > ws.cap_nbins) {
    cudaFree(ws.d_cell_counts);
    cudaFree(ws.d_cell_offsets);
    cudaFree(ws.d_cell_fill);

    CHECK(cudaMalloc(&ws.d_cell_counts, sizeof(int) * (size_t)nbins));
    CHECK(cudaMalloc(&ws.d_cell_offsets, sizeof(int) * (size_t)nbins));
    CHECK(cudaMalloc(&ws.d_cell_fill, sizeof(int) * (size_t)nbins));

    ws.scan_temp_bytes = 0;
    CHECK(cub::DeviceScan::ExclusiveSum(nullptr, ws.scan_temp_bytes,
                                        ws.d_cell_counts, ws.d_cell_offsets,
                                        nbins));
    cudaFree(ws.d_scan_temp);
    CHECK(cudaMalloc(&ws.d_scan_temp, ws.scan_temp_bytes));

    ws.cap_nbins = nbins;
  }

  if (!ws.d_overflow_flag) {
    CHECK(cudaMalloc(&ws.d_overflow_flag, sizeof(int)));
  }
  if (!ws.d_reduce_out_int) {
    CHECK(cudaMalloc(&ws.d_reduce_out_int, sizeof(int)));
  }
}

void reserve_neighbor_storage(GpuNeighborWorkspace& ws, size_t neighbor_ints) {
  if (neighbor_ints > ws.cap_neighbor_ints) {
    CHECK(cudaFree(ws.d_neighbors_flat));
    CHECK(cudaMalloc(&ws.d_neighbors_flat, sizeof(int) * neighbor_ints));
    ws.cap_neighbor_ints = neighbor_ints;
  }
}

static void ensure_reduce_temp_for_ints(GpuNeighborWorkspace& ws, int n,
                                        const int* d_in) {
  size_t bytes = 0;
  cub::DeviceReduce::Max(nullptr, bytes, d_in, ws.d_reduce_out_int, n);
  if (bytes > ws.reduce_temp_bytes) {
    cudaFree(ws.d_reduce_temp);
    CHECK(cudaMalloc(&ws.d_reduce_temp, bytes));
    ws.reduce_temp_bytes = bytes;
  }
}

static int device_reduce_max_int(void* temp, size_t bytes, const int* d_in,
                                 int* d_out, int n, cudaStream_t stream) {
  CHECK(cub::DeviceReduce::Max(temp, bytes, d_in, d_out, n, stream));
  int h = 0;
  CHECK(
      cudaMemcpyAsync(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));
  return h;
}

// -------------------------------
// Main builder
// -------------------------------
//
// Inputs:
//   x_d : device pointer to positions, packed as x[3*i + 0..2],
//   type_d : device pointer to atom types, 1-based typical LAMMPS style
//   nlocal : number of local atoms
//   nghost : number of ghost atoms; total atoms = nlocal + nghost
//   neighbor_cut  : neighbor-list cutoff (typically force cutoff + skin)
//   subdomain     : domain->sublo/hi
//   cutsq_table_d : optional device table [(ntypes+1)*(ntypes+1)]
//   ntypes        : number of atom types; ignored if cutsq_table_d == nullptr
//
// Outputs:
//   numneigh_d        : device pointer [nlocal] already allocated by caller
//   neighstride       : stride of firstneigh_flat for each center atom
//   firstneigh_flat_d : returned device pointer owned by workspace
//
void build_neighbor_list(const double* x_d, const int* type_d,
                         const int* ilist_d, int inum, int nlocal, int nghost,
                         double neighbor_cut, double subdomain[6],
                         const double* cutsq_table_d, int ntypes,
                         int* numneigh_d, int*& firstneigh_flat_d,
                         int& neighstride) {
  Utils::GPU::timer_start(Profiling::TimedKernel::neighbor_list);
  assert(inum >= 0);
  assert(nlocal >= 0);
  assert(nghost >= 0);

  const int nall = nlocal + nghost;
  if (nlocal == 0 || inum == 0 || nall == 0) {
    if (numneigh_d && nlocal > 0) {
      CHECK(cudaMemsetAsync(numneigh_d, 0, sizeof(int) * (size_t)nlocal,
                            stream));
    }
    firstneigh_flat_d = nullptr;
    neighstride = 1;
    update_neighbor_list_memory_usage();
    Utils::GPU::timer_stop(Profiling::TimedKernel::neighbor_list, 0.0);
    return;
  }

  if (inum > nlocal) {
    fprintf(stderr,
            "ERROR: invalid neighbor-list sizes: inum (%d) > nlocal (%d).\n",
            inum, nlocal);
    std::abort();
  }

  if (!(neighbor_cut > 0.0) || !std::isfinite(neighbor_cut)) {
    fprintf(stderr,
            "ERROR: invalid neighbor cutoff in build_neighbor_list: %g\n",
            neighbor_cut);
    std::abort();
  }

  const int block = 256;
  const int grid_all = (nall + block - 1) / block;
  const int grid_inum = (inum + block - 1) / block;

  // ----------------------------------------------------------
  // 1. Extract subdomain ranges
  // ----------------------------------------------------------
  const double xmin = subdomain[0];
  const double xmax = subdomain[1];
  const double ymin = subdomain[2];
  const double ymax = subdomain[3];
  const double zmin = subdomain[4];
  const double zmax = subdomain[5];

  // ----------------------------------------------------------
  // 2. Choose bins from actual ghosted cloud
  // ----------------------------------------------------------
  //
  // Use binsize = neighbor_cut
  // This is robust and simple.
  //
  const double binsize = neighbor_cut;
  const double eps = 1e-12;

  const double lx = std::max(xmax - xmin, eps);
  const double ly = std::max(ymax - ymin, eps);
  const double lz = std::max(zmax - zmin, eps);

  int nbinx = std::max(1, (int)std::floor(lx / binsize) + 1);
  int nbiny = std::max(1, (int)std::floor(ly / binsize) + 1);
  int nbinz = std::max(1, (int)std::floor(lz / binsize) + 1);

  // Candidate stencil. Because binsize == neighbor_cut,
  // sx=sy=sz=1 is sufficient and robust for Cartesian cloud bins.
  const int sx = 1;
  const int sy = 1;
  const int sz = 1;

  const size_t nbins_size = static_cast<size_t>(nbinx) *
                            static_cast<size_t>(nbiny) *
                            static_cast<size_t>(nbinz);
  if (nbins_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    fprintf(stderr,
            "ERROR: too many neighbor-list bins (%zu) for 32-bit indexing.\n",
            nbins_size);
    std::abort();
  }
  const int nbins = static_cast<int>(nbins_size);
  const double inv_binsize = 1.0 / binsize;

  reserve_basic_workspace(neighlist_ws, nall, nbins);

  // ----------------------------------------------------------
  // 3. Histogram atoms into bins
  // ----------------------------------------------------------
  CHECK(cudaMemsetAsync(neighlist_ws.d_cell_counts, 0,
                        sizeof(int) * (size_t)nbins, stream));

  kernel_assign_bins_and_histogram<<<grid_all, block, 0, stream>>>(
      nall, x_d, xmin, ymin, zmin, inv_binsize, nbinx, nbiny, nbinz,
      neighlist_ws.d_atom_bin, neighlist_ws.d_cell_counts);
  CHECK(cudaGetLastError());

  // ----------------------------------------------------------
  // 4. Exclusive scan counts -> offsets
  // ----------------------------------------------------------
  CHECK(cub::DeviceScan::ExclusiveSum(
      neighlist_ws.d_scan_temp, neighlist_ws.scan_temp_bytes,
      neighlist_ws.d_cell_counts, neighlist_ws.d_cell_offsets, nbins, stream));

  // ----------------------------------------------------------
  // 5. Scatter atoms into bins
  // ----------------------------------------------------------
  CHECK(cudaMemcpyAsync(neighlist_ws.d_cell_fill, neighlist_ws.d_cell_offsets,
                        sizeof(int) * (size_t)nbins, cudaMemcpyDeviceToDevice,
                        stream));

  kernel_scatter_atoms_to_bins<<<grid_all, block, 0, stream>>>(
      nall, neighlist_ws.d_atom_bin, neighlist_ws.d_cell_offsets,
      neighlist_ws.d_cell_fill, neighlist_ws.d_cell_atoms);
  CHECK(cudaGetLastError());

  // ----------------------------------------------------------
  // 6. Count neighbors
  // ----------------------------------------------------------
  CHECK(cudaMemsetAsync(numneigh_d, 0, sizeof(int) * (size_t)nlocal, stream));

  if (cutsq_table_d) {
    const int ld = ntypes + 1;
    kernel_count_neighbors_typed<<<grid_inum, block, 0, stream>>>(
        inum, ilist_d, x_d, type_d, xmin, ymin, zmin, inv_binsize, nbinx, nbiny,
        nbinz, sx, sy, sz, neighlist_ws.d_cell_offsets,
        neighlist_ws.d_cell_counts, neighlist_ws.d_cell_atoms, cutsq_table_d,
        ld, numneigh_d);
  } else {
    const double cutsq_uniform = neighbor_cut * neighbor_cut;
    kernel_count_neighbors_uniform<<<grid_inum, block, 0, stream>>>(
        inum, ilist_d, x_d, type_d, xmin, ymin, zmin, inv_binsize, nbinx, nbiny,
        nbinz, sx, sy, sz, neighlist_ws.d_cell_offsets,
        neighlist_ws.d_cell_counts, neighlist_ws.d_cell_atoms, cutsq_uniform,
        numneigh_d);
  }
  CHECK(cudaGetLastError());

  // ----------------------------------------------------------
  // 7. Reduce max(numneigh) -> neighstride
  // ----------------------------------------------------------
  ensure_reduce_temp_for_ints(neighlist_ws, nlocal, numneigh_d);
  const int maxneigh = device_reduce_max_int(
      neighlist_ws.d_reduce_temp, neighlist_ws.reduce_temp_bytes, numneigh_d,
      neighlist_ws.d_reduce_out_int, nlocal, stream);

  neighstride = round_up_int(std::max(maxneigh, 1), 32);

  const size_t neighbor_ints = (size_t)nlocal * (size_t)neighstride;
  reserve_neighbor_storage(neighlist_ws, neighbor_ints);
  firstneigh_flat_d = neighlist_ws.d_neighbors_flat;

  CHECK(cudaMemsetAsync(neighlist_ws.d_overflow_flag, 0, sizeof(int), stream));

  // ----------------------------------------------------------
  // 8. Fill neighbors
  // ----------------------------------------------------------
  if (cutsq_table_d) {
    const int ld = ntypes + 1;
    kernel_fill_neighbors_typed<<<grid_inum, block, 0, stream>>>(
        inum, ilist_d, x_d, type_d, xmin, ymin, zmin, inv_binsize, nbinx, nbiny,
        nbinz, sx, sy, sz, neighlist_ws.d_cell_offsets,
        neighlist_ws.d_cell_counts, neighlist_ws.d_cell_atoms, cutsq_table_d,
        ld, neighstride, numneigh_d, neighlist_ws.d_neighbors_flat,
        neighlist_ws.d_overflow_flag);
  } else {
    const double cutsq_uniform = neighbor_cut * neighbor_cut;
    kernel_fill_neighbors_uniform<<<grid_inum, block, 0, stream>>>(
        inum, ilist_d, x_d, type_d, xmin, ymin, zmin, inv_binsize, nbinx, nbiny,
        nbinz, sx, sy, sz, neighlist_ws.d_cell_offsets,
        neighlist_ws.d_cell_counts, neighlist_ws.d_cell_atoms, cutsq_uniform,
        neighstride, numneigh_d, neighlist_ws.d_neighbors_flat,
        neighlist_ws.d_overflow_flag);
  }
  CHECK(cudaGetLastError());

  // ----------------------------------------------------------
  // 9. Overflow check
  // ----------------------------------------------------------
  int overflow = 0;
  CHECK(cudaMemcpyAsync(&overflow, neighlist_ws.d_overflow_flag, sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));

  if (overflow) {
    fprintf(stderr, "ERROR: neighbor list overflow detected.\n");
    std::abort();
  }

  update_neighbor_list_memory_usage();

  Utils::GPU::timer_stop(Profiling::TimedKernel::neighbor_list, 0.0);
}

}  // namespace TENSORMD::Kernels::GPU
