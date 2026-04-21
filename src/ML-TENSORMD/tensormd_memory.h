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

#ifndef TENSORMD_MEMORY_H
#define TENSORMD_MEMORY_H

#include <cstdio>
#include <cstring>  // for memset
#include <stdexcept>
#include <utility>

#include "tensormd_device.h"
#include "tensormd_strategy.h"

#if defined(TENSORMD_GPU)
#if defined(TENSORMD_HIP)
#include <hip/hip_runtime.h>
#define GPU_MALLOC(ptr, size) hipMalloc((void**)(ptr), (size))
#define GPU_FREE(ptr) hipFree(ptr)
#define GPU_MEMSET(ptr, val, size) hipMemset((ptr), (val), (size))
#else
#include <cuda_runtime.h>
#define GPU_MALLOC(ptr, size) cudaMalloc((void**)(ptr), (size))
#define GPU_FREE(ptr) cudaFree(ptr)
#define GPU_MEMSET(ptr, val, size) cudaMemset((ptr), (val), (size))
#endif
#endif

namespace TENSORMD {

template <typename T>
class MemoryPool {
 public:
 public:
  MemoryPool(Device dev, const char* name_tag, bool dynamic = true)
      : device(dev),
        pool(nullptr),
        capacity(0),
        offset(0),
        name(name_tag),
        dynamic(dynamic) {}

  ~MemoryPool() { destroy(); }

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  MemoryPool(MemoryPool&& other) noexcept
      : device(other.device),
        pool(other.pool),
        capacity(other.capacity),
        offset(other.offset),
        name(other.name),
        dynamic(other.dynamic) {
    other.pool = nullptr;
    other.capacity = 0;
    other.offset = 0;
  }

  MemoryPool& operator=(MemoryPool&& other) noexcept {
    if (this != &other) {
      destroy();
      device = other.device;
      pool = other.pool;
      capacity = other.capacity;
      offset = other.offset;
      name = other.name;
      dynamic = other.dynamic;
      other.pool = nullptr;
      other.capacity = 0;
      other.offset = 0;
    }
    return *this;
  }

  // Return the memory usage (bytes)
  [[nodiscard]] double get_memory_usage() const {
    return static_cast<double>(capacity * sizeof(T));
  }

  // Set execution device
  void set_device(Device dev) {
    if (dev != device && capacity > 0) {
      // Warn: Changing device with allocated memory implies a reset
      destroy();
    }
    device = dev;
  }

  // Get a pointer from the pool and advance the offset
  T* request(size_t n) {
    if (n == 0) {
      return pool ? (pool + offset) : nullptr;
    }
    if (n > capacity || offset > capacity - n) {
      const size_t available = (offset <= capacity) ? (capacity - offset) : 0;
      char msg[512];
      std::snprintf(msg, sizeof(msg),
                    "TensorMD MemoryPool '%s' overflow: "
                    "requested %zu elements, but only %zu available. "
                    "Current offset: %zu, capacity: %zu. "
                    "Call grow() first to allocate more memory.",
                    name, n, available, offset, capacity);
      throw std::runtime_error(msg);
    }
    T* ptr = pool + offset;
    offset += n;
    return ptr;
  }

  // Reset the offset for the next timestep (reuse memory)
  void reset() { offset = 0; }

  // Resize the pool if necessary
  void grow(size_t n) {
    if (n <= capacity) return;

    size_t new_capacity = n;
    if (dynamic) {
      // Allocate N% more than requested to prevent frequent reallocs
      // during slight system expansions.
      new_capacity = static_cast<size_t>(static_cast<double>(n) *
                                         TENSORMD_MEMORY_GROWTH_RATE);
      if (new_capacity < n) {
        new_capacity = n;
      }
    }

    destroy();
    pool = allocate_raw(new_capacity);
    capacity = new_capacity;
    offset = 0;
  }

  // Zero out the entire pool
  void zero_all() {
    if (capacity == 0 || pool == nullptr) return;
    if (device == Device::CPU) {
      std::memset(pool, 0, capacity * sizeof(T));
    }
#if defined(TENSORMD_GPU)
    else {
      GPU_MEMSET(pool, 0, capacity * sizeof(T));
    }
#endif
  }

  void free() { destroy(); }

  [[nodiscard]] size_t get_capacity() const { return capacity; }
  [[nodiscard]] size_t get_bytes() const { return capacity * sizeof(T); }

 private:
  Device device;
  T* pool;
  size_t capacity;
  size_t offset;
  const char* name;
  bool dynamic;

  T* allocate_raw(size_t n) {
    if (n == 0) return nullptr;

    if (device == Device::CPU) {
      return new T[n];
    }

#if defined(TENSORMD_GPU)
    T* ptr = nullptr;
    if (GPU_MALLOC(&ptr, n * sizeof(T)) != 0) {
      throw std::runtime_error("TensorMD: GPU memory allocation failed");
    }
    return ptr;
#else
    throw std::runtime_error("TensorMD: GPU memory support is not enabled");
#endif
  }

  void destroy() {
    if (pool) {
      if (device == Device::CPU) {
        delete[] pool;
      }
#if defined(TENSORMD_GPU)
      else {
        GPU_FREE(pool);
      }
#endif
      pool = nullptr;
    }
    capacity = 0;
    offset = 0;
  }
};

}  // namespace TENSORMD

#endif