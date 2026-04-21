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

#include "../../../tensormd_constants.h"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::GPU {

void sync_lammps_data(HostArgs<double>& host_args,
                      DeviceArgs<double>& device_args) {
  Utils::GPU::timer_start(Profiling::TimedKernel::sync_lammps_data);
#if defined(TENSORMD_DEBUG)
  check_kernel_arg(host_args.inum);
  check_kernel_arg(host_args.nlocal);
  check_kernel_arg(host_args.ilist);
  check_kernel_arg(host_args.x);
  check_kernel_arg(host_args.type);
  check_kernel_arg(device_args.ilist);
  check_kernel_arg(device_args.x);
  check_kernel_arg(device_args.type);
#endif
  const int nall = host_args.nlocal + host_args.nghost;
  CHECK(cudaMemcpyAsync(device_args.type, host_args.type, sizeof(int) * nall,
                        cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(device_args.ilist, host_args.ilist,
                        sizeof(int) * host_args.inum, cudaMemcpyHostToDevice,
                        stream));
  CHECK(cudaMemcpyAsync(device_args.x, host_args.x[0],
                        sizeof(double) * nall * 3, cudaMemcpyHostToDevice,
                        stream));

  // Should we copy f and vatoms to device side?
  cudaStreamSynchronize(stream);
  CHECK(cudaGetLastError());
  Utils::GPU::timer_stop(Profiling::TimedKernel::sync_lammps_data, 0.0);
}

}  // namespace TENSORMD::Kernels::GPU
