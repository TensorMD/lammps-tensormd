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

#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::Utils::GPU {

/* ---------------------------------------------------------------------- */

template <typename T>
T* access_data_on_host(const T* device_ptr, int n,
                       std::vector<T>& backing_buffer) {
  // Resize the buffer to hold the data
  if (backing_buffer.size() < n) {
    backing_buffer.resize(n);
  }
  // Copy data from Device to Host
  CHECK(cudaMemcpy(backing_buffer.data(), device_ptr, n * sizeof(T),
                   cudaMemcpyDeviceToHost));

  // Return pointer to the vector's data
  return backing_buffer.data();
}

/* ---------------------------------------------------------------------- */

template <typename T>
void copy_data_to_device(T* device_ptr, const T* host_ptr, int n) {
  CHECK(
      cudaMemcpy(device_ptr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice));
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void __global__ cast_scalar_to_float_kernel(const Scalar* __restrict__ d_in,
                                            float* __restrict__ d_out,
                                            size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_out[idx] = static_cast<float>(d_in[idx]);
  }
}

template <typename Scalar>
void cast_scalar_to_float(const Scalar* __restrict__ d_in,
                          float* __restrict__ d_out, size_t n) {
  // Simple element-wise cast on GPU
  int threads_per_block = 256;
  int blocks = (n + threads_per_block - 1) / threads_per_block;
  cast_scalar_to_float_kernel<<<blocks, threads_per_block>>>(d_in, d_out, n);
  CHECK(cudaGetLastError());
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template int* access_data_on_host<int>(const int* device_ptr, int n,
                                       std::vector<int>& backing_buffer);
template float* access_data_on_host<float>(const float* device_ptr, int n,
                                           std::vector<float>& backing_buffer);
template double* access_data_on_host<double>(
    const double* device_ptr, int n, std::vector<double>& backing_buffer);

template void copy_data_to_device<int>(int* device_ptr, const int* host_ptr,
                                       int n);
template void copy_data_to_device<float>(float* device_ptr,
                                         const float* host_ptr, int n);
template void copy_data_to_device<double>(double* device_ptr,
                                          const double* host_ptr, int n);

template void cast_scalar_to_float<double>(const double* __restrict__ d_in,
                                           float* __restrict__ d_out, size_t n);
template void cast_scalar_to_float<float>(const float* __restrict__ d_in,
                                          float* __restrict__ d_out, size_t n);

}  // namespace TENSORMD::Kernels::Utils::GPU
