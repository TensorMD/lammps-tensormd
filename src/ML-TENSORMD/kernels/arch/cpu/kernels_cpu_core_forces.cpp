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

namespace TENSORMD::Kernels::CPU {

using Profiling::TimedKernel;

template <typename Scalar>
void sum_forces(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args) {
  Utils::CPU::timer_start(TimedKernel::SumForces);
  int stride2, stride3;
  int size = device_args.a * device_args.b * device_args.c;
  Scalar df[3], virial[6];
  Scalar scale = device_args.scale;
  const Scalar* __restrict__ Fr = device_args.Fr;
  const Scalar* __restrict__ Fa = device_args.Fa;

  for (int stride1 = 0; stride1 < size; stride1++) {
    stride2 = stride1 * 2;
    stride3 = stride1 * 3;
    if (device_args.tijlist[stride1] < 0) {
      continue;
    }
    const int i = device_args.ijlist[stride2 + 0];
    const int j = device_args.ijlist[stride2 + 1];
    if (i >= 0 && j >= 0) {
      if (Fa != nullptr) {
        for (int x = 0; x < 3; x++) {
          df[x] = (Fa[stride3 + x] + Fr[stride3 + x]) * scale;
          host_args.f[i][x] += df[x];
          host_args.f[j][x] -= df[x];
        }
      } else {
        for (int x = 0; x < 3; x++) {
          host_args.f[i][x] += Fr[stride3 + x] * scale;
          host_args.f[j][x] -= Fr[stride3 + x] * scale;
        }
      }

      if (host_args.vatom) {
        Scalar rij = device_args.R[stride1];
        virial[0] = -device_args.drdrx[stride3 + 0] * df[0] * rij;
        virial[1] = -device_args.drdrx[stride3 + 1] * df[1] * rij;
        virial[2] = -device_args.drdrx[stride3 + 2] * df[2] * rij;
        virial[3] = -device_args.drdrx[stride3 + 0] * df[1] * rij;
        virial[4] = -device_args.drdrx[stride3 + 0] * df[2] * rij;
        virial[5] = -device_args.drdrx[stride3 + 1] * df[2] * rij;
        host_args.vatom[i][0] += 0.5 * virial[0];
        host_args.vatom[i][1] += 0.5 * virial[1];
        host_args.vatom[i][2] += 0.5 * virial[2];
        host_args.vatom[i][3] += 0.5 * virial[3];
        host_args.vatom[i][4] += 0.5 * virial[4];
        host_args.vatom[i][5] += 0.5 * virial[5];
        host_args.vatom[j][0] += 0.5 * virial[0];
        host_args.vatom[j][1] += 0.5 * virial[1];
        host_args.vatom[j][2] += 0.5 * virial[2];
        host_args.vatom[j][3] += 0.5 * virial[3];
        host_args.vatom[j][4] += 0.5 * virial[4];
        host_args.vatom[j][5] += 0.5 * virial[5];
      }
    }
  }
  double FLOPs = size * 12.0;
  if (host_args.vatom) {
    FLOPs += 42.0 * size;
  }
  Utils::CPU::timer_stop(TimedKernel::SumForces, FLOPs);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void sum_forces<float>(HostArgs<float>& host_args,
                                DeviceArgs<float>& device_arg);
template void sum_forces<double>(HostArgs<double>& host_args,
                                 DeviceArgs<double>& device_arg);

}  // namespace TENSORMD::Kernels::CPU