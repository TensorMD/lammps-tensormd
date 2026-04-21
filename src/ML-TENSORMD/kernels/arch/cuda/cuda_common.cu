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
#include <stdio.h>
#include <cublas_v2.h>

#include "cuda_common.cuh"
#include "../kernels_internal.h"

namespace TENSORMD::Kernels::GPU {

int GPU_Max_Thread = 128;
cublasHandle_t handle;
cudaStream_t stream;

void error_handle(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err),
            cudaGetErrorString(err), file, line);
    exit(int(err));
  }
}

void setup_device(int mpiid, bool verbose_device_map) {
  cudaDeviceProp prop;
  int deviceId, deviceCount;

  CHECK(cudaGetDeviceCount(&deviceCount));
  deviceId = (mpiid + 1) % deviceCount;
  CHECK(cudaSetDevice(deviceId));
  CHECK(cudaGetDeviceProperties(&prop, deviceId));
  GPU_Max_Thread = prop.maxThreadsDim[0];
  if (verbose_device_map) {
    printf("[TensorMD] Rank %d mapped to GPU device %d: %s\n", mpiid, deviceId,
           prop.name);
  }
  cublasCreate(&handle);
  cudaStreamCreate(&stream);
}

void finalize()
{
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  clear_neighbor_list_workspace();
}

void get_memory_usage(size_t& total_mem, size_t& free_mem) {
  CHECK(cudaMemGetInfo(&free_mem, &total_mem));
}

}  // namespace TENSORMD::Kernels::GPU
