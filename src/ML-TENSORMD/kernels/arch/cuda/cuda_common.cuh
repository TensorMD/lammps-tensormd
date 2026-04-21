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

#ifndef TENSORMD_KERNELS_GPU_CUDA_COMMON
#define TENSORMD_KERNELS_GPU_CUDA_COMMON

#include <driver_types.h>
#include <cublas_v2.h>
#include <cstdio>

namespace TENSORMD::Kernels::GPU {

extern int GPU_Max_Thread;
extern cublasHandle_t handle;
extern cudaStream_t stream;

#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

#define check_kernel_arg(arg)                                           \
  do {                                                                  \
    if ((arg) == 0) {                                                   \
      char msg[512];                                                    \
      std::snprintf(msg, sizeof(msg),                                   \
                    "[Check Failed] %s:%d: Argument '%s' is 0/null.\n", \
                    __FILE__, __LINE__, #arg);                          \
      throw std::runtime_error(msg);                                    \
    }                                                                   \
  } while (0)

void error_handle(cudaError_t err, const char* file, int line);

void get_memory_usage(size_t& total_mem, size_t& free_mem);

#define CHECK(err) \
  (TENSORMD::Kernels::GPU::error_handle(err, __FILE__, __LINE__))

}  // namespace TENSORMD::Kernels::GPU

#endif