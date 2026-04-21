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

#ifndef TENSORMD_KERNELS_INTERNAL_UTILS_H
#define TENSORMD_KERNELS_INTERNAL_UTILS_H

#include <vector>

namespace TENSORMD::Kernels::Utils {

namespace CPU {
void timer_init();
void timer_start(int kernel_id);
void timer_stop(int kernel_id, double FLOPs);
void timer_accumulate();
void timer_collect(double* times, double* FLOPs);
void timer_destroy();
}  // namespace CPU

namespace GPU {
#ifdef TENSORMD_GPU

template <typename T>
T* access_data_on_host(const T* device_ptr, int n,
                       std::vector<T>& backing_buffer);

template <typename T>
void copy_data_to_device(T* device_ptr, const T* host_ptr, int n);

template <typename Scalar>
void cast_scalar_to_float(const Scalar* d_in, float* d_out, size_t n);

void sync_device();
void timer_init();
void timer_start(int kernel_id);
void timer_stop(int kernel_id, double FLOPs);
void timer_accumulate();
void timer_collect(double* times, double* FLOPs);
void timer_destroy();

#endif
}  // namespace GPU
}  // namespace TENSORMD::Kernels::Utils

#endif
