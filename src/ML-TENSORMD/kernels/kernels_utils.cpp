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

#include "kernels_utils.h"

#include "arch/kernels_internal_utils.h"

namespace TENSORMD::Kernels::Utils {

// ----------------------------------------------------------------------
// Memory Management
// ----------------------------------------------------------------------

template <typename T>
T* access_data_on_host(const T* device_ptr, int n, Device device,
                       std::vector<T>& backing_buffer) {
  if (device == Device::CPU) {
    // ---------------------------------------------------------
    // CPU CASE: ZERO COPY
    // ---------------------------------------------------------
    // We simply return the original pointer.
    // The vector 'backing_buffer' is untouched and unused.
    return const_cast<T*>(device_ptr);
  } else if (device == Device::GPU) {
    // ---------------------------------------------------------
    // GPU CASE: COPY REQUIRED
    // ---------------------------------------------------------
#if defined(TENSORMD_GPU)
    return GPU::access_data_on_host(device_ptr, n, backing_buffer);
#endif
  }
  return nullptr;  // Invalid device
}

template <typename T>
void copy_data_to_device(T* device_ptr, const T* host_ptr, int n,
                         Device device) {
  if (device == Device::CPU) {
    // Direct copy
    std::memcpy(device_ptr, host_ptr, n * sizeof(T));
  } else if (device == Device::GPU) {
#if defined(TENSORMD_GPU)
    // Call GPU implementation
    GPU::copy_data_to_device(device_ptr, host_ptr, n);
#endif
  }
}

template <typename Scalar>
void cast_scalar_to_float(Device device, const Scalar* d_in, float* d_out,
                          size_t n) {
  if (device == Device::CPU) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; ++i) {
      d_out[i] = static_cast<float>(d_in[i]);
    }
  } else if (device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::cast_scalar_to_float(d_in, d_out, n);
#endif
  }
}

// ----------------------------------------------------------------------
// Common Kernels
// ----------------------------------------------------------------------

void sync_device(Device dev) {
  if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::sync_device();
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template int* access_data_on_host<int>(const int* device_ptr, int n,
                                       Device device,
                                       std::vector<int>& backing_buffer);
template float* access_data_on_host<float>(const float* device_ptr, int n,
                                           Device device,
                                           std::vector<float>& backing_buffer);
template double* access_data_on_host<double>(
    const double* device_ptr, int n, Device device,
    std::vector<double>& backing_buffer);

template void copy_data_to_device<int>(int* device_ptr, const int* host_ptr,
                                       int n, Device device);
template void copy_data_to_device<float>(float* device_ptr,
                                         const float* host_ptr, int n,
                                         Device device);
template void copy_data_to_device<double>(double* device_ptr,
                                          const double* host_ptr, int n,
                                          Device device);

template void cast_scalar_to_float<float>(Device device, const float* d_in,
                                          float* d_out, size_t n);
template void cast_scalar_to_float<double>(Device device, const double* d_in,
                                           float* d_out, size_t n);

// ----------------------------------------------------------------------
// Kernels for timing and profiling
// ----------------------------------------------------------------------

namespace Profiling {

static bool timer_enabled = false;

void timer_init(Device dev) {
  if (dev == Device::CPU) {
    CPU::timer_init();
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_init();
#endif
  }
}

void timer_start(Device dev, int kernel_id) {
  if (dev == Device::CPU) {
    CPU::timer_start(kernel_id);
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_start(kernel_id);
#endif
  }
}

void timer_stop(Device dev, int kernel_id) {
  if (dev == Device::CPU) {
    CPU::timer_stop(kernel_id, 0.0);
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_stop(kernel_id, 0.0);
#endif
  }
}

void timer_accumulate(Device dev) {
  if (dev == Device::CPU) {
    CPU::timer_accumulate();
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_accumulate();
#endif
  }
}

void timer_collect(Device dev, double* times, double* FLOPs) {
  if (dev == Device::CPU) {
    CPU::timer_collect(times, FLOPs);
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_collect(times, FLOPs);
#endif
  }
}

void timer_destroy(Device dev) {
  if (dev == Device::CPU) {
    CPU::timer_destroy();
  } else if (dev == Device::GPU) {
#ifdef TENSORMD_GPU
    GPU::timer_destroy();
#endif
  }
}

}  // namespace Profiling

}  // namespace TENSORMD::Kernels::Utils
