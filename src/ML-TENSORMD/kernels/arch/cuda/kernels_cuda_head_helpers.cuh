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

#ifndef TENSORMD_KERNELS_ARCH_CUDA_HEAD_HELPERS_CUH
#define TENSORMD_KERNELS_ARCH_CUDA_HEAD_HELPERS_CUH

#include <cub/cub.cuh>

namespace TENSORMD::Kernels::Head::GPU {

static void* d_temp_storage = nullptr;
static size_t temp_storage_bytes = 0;

template <typename Scalar>
void compute_total_energy(const Scalar* d_eatoms, Scalar* d_toten,
                          int total_size, cudaStream_t stream = 0) {
  // First call: d_temp_storage is nullptr.
  // This just computes the required allocation size.
  size_t required_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, required_bytes, d_eatoms, d_toten,
                         total_size, stream);

  // Allocate temporary storage
  if (required_bytes > temp_storage_bytes) {
    if (d_temp_storage) {
      cudaFree(d_temp_storage);
    }
    d_temp_storage = nullptr;
    temp_storage_bytes = required_bytes;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  }
  
  // Run the actual reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_eatoms, d_toten,
                         total_size, stream);
}

}  // namespace TENSORMD::Kernels::Head::GPU

#endif  // TENSORMD_KERNELS_ARCH_CUDA_HEAD_HELPERS_CUH
