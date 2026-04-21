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
#include "../kernels_internal_types.h"
#include "../kernels_internal_v1_ops.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::V1::GPU {

/**
 * @brief CUDA kernel to set up i2a (LAMMPS_atom_index to TensorMD_atom_index)
 * and a2i (vice versa).
 *
 * This kernel parallelizes the loop over 'a' elements. It uses an atomic
 * operation to safely handle the increment of the elt_a_offset counter,
 * which is a critical step to avoid race conditions.
 */
__global__ void setup_index_maps_thread(int inum, const int* __restrict__ type,
                                        const int* __restrict__ fmap,
                                        const int* __restrict__ ilist,
                                        int* __restrict__ elt_a_offset,
                                        int* __restrict__ i2a,
                                        int* __restrict__ a2i) {
  // Standard 1D thread indexing
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  // Grid-stride check: ensures we don't go out of bounds
  if (ii < inum) {
    int i = ilist[ii];
    int elti = fmap[type[i]];
    int a = atomicAdd(&elt_a_offset[elti], 1);
    i2a[i] = a;
    a2i[a] = i;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void setup_index_maps(HostArgs<Scalar>& host_args,
                      DeviceArgs<Scalar>& device_args) {
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(host_args.elt_a_offset);
  check_kernel_arg(device_args.a);
  check_kernel_arg(device_args.b);
  check_kernel_arg(device_args.ilist);
  check_kernel_arg(device_args.type);
  check_kernel_arg(device_args.fmap);
  check_kernel_arg(device_args.elt_a_offset);
  check_kernel_arg(device_args.i2a);
  check_kernel_arg(device_args.a2i);
#endif
  if (device_args.b == 1) {
    return;
  }

  const int a = device_args.a;
  const int* __restrict__ ilist = device_args.ilist;
  const int* __restrict__ type = device_args.type;
  const int* __restrict__ fmap = device_args.fmap;
  int* __restrict__ elt_a_offset = device_args.elt_a_offset;
  int* __restrict__ i2a = device_args.i2a;
  int* __restrict__ a2i = device_args.a2i;

  CHECK(cudaMemcpy(elt_a_offset, host_args.elt_a_offset,
                   sizeof(int) * device_args.b, cudaMemcpyHostToDevice));

  int numBlocks = 128;
  int blockSize = (a + numBlocks - 1) / numBlocks;

  setup_index_maps_thread<<<blockSize, numBlocks>>>(a, type, fmap, ilist,
                                                    elt_a_offset, i2a, a2i);

  CHECK(cudaGetLastError());
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void setup_index_maps<float>(HostArgs<float>& host_args,
                                      DeviceArgs<float>& device_args);
template void setup_index_maps<double>(HostArgs<double>& host_args,
                                       DeviceArgs<double>& device_args);

}
