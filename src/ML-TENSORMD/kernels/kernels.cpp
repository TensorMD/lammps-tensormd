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

#include "kernels.h"

#include "arch/kernels_internal.h"

namespace TENSORMD::Kernels {

// ----------------------------------------------------------------------
// Initialization and Finalization
// ----------------------------------------------------------------------

template <typename Scalar>
void setup_device(TensorMDContext<Scalar>* ctx) {
  if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::setup_device(ctx->mpiid, ctx->verbose_device_map);
#endif
  }
}

template <typename Scalar>
void initialize(const int* eltij_pos, const int* fmap,
                std::vector<int>& numbers, TensorMDContext<Scalar>* context) {
  HostArgs<Scalar> host_args{};
  host_args.eltij_pos = const_cast<int*>(eltij_pos);
  host_args.numbers = numbers.data();
  host_args.n_numbers = static_cast<int>(numbers.size());
  host_args.fmap = const_cast<int*>(fmap);
  host_args.lmp_ntypes = context->lmpdata.ntypes;

  DeviceArgs<Scalar> device_args{};
  device_args.eltij_pos = context->data->eltij_pos;
  device_args.numbers = context->data->numbers;
  device_args.b = context->data->dims.b;
  device_args.m = context->data->dims.m;
  device_args.d = context->data->dims.d;
  device_args.fmap = context->lmpdata.fmap;

  if (context->data->device == Device::CPU) {
    CPU::initialize(host_args, device_args);
  } else if (context->data->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::initialize(host_args, device_args);
#endif
  }
}

template <typename Scalar>
void finalize(TensorMDContext<Scalar>* ctx) {
  if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::finalize();
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void setup_device<float>(TensorMDContext<float>* context);
template void setup_device<double>(TensorMDContext<double>* context);

template void initialize<float>(const int* eltij_pos,
                                const int* fmap, std::vector<int>& numbers,
                                TensorMDContext<float>* context);
template void initialize<double>(const int* eltij_pos,
                                 const int* fmap, std::vector<int>& numbers,
                                 TensorMDContext<double>* context);

template void finalize<float>(TensorMDContext<float>* context);
template void finalize<double>(TensorMDContext<double>* context);

}  // namespace TENSORMD::Kernels
