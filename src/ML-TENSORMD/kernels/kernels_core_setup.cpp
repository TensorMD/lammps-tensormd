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

#include "arch/kernels_internal.h"
#include "kernels.h"

namespace TENSORMD::Kernels {

// ----------------------------------------------------------------------
// Mirror LAMMPS data and build neighbor list
// ----------------------------------------------------------------------

void sync_lammps_data(int* ilist, int* type, int* fmap, double** x,
                      int* numneigh, int** firstneigh, double** f,
                      double** vatom, LAMMPSData& lmp) {
  if (lmp.device == Device::CPU) {
    // For CPU, we use the pointers from LAMMPSData without copying.
    lmp.ilist = ilist;
    lmp.type = type;
    lmp.fmap = fmap;
    lmp.x = x[0];
    lmp.numneigh = numneigh;
    lmp.firstneigh = firstneigh;
    lmp.f = f ? f[0] : nullptr;
    lmp.vatom = vatom ? vatom[0] : nullptr;
  } else if (lmp.device == Device::GPU) {
#if defined(TENSORMD_GPU)
    HostArgs<double> host_args{};
    host_args.ilist = ilist;
    host_args.type = type;
    host_args.x = x;
    host_args.inum = lmp.inum;
    host_args.nlocal = lmp.nlocal;
    host_args.nghost = lmp.nghost;

    DeviceArgs<double> device_args{};
    device_args.ilist = lmp.ilist;
    device_args.type = lmp.type;
    device_args.x = lmp.x;

    // Launch the kernel
    GPU::sync_lammps_data(host_args, device_args);
#endif
  }
}

void build_neighbor_list(LAMMPSData& lmp) {
  if (lmp.device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::build_neighbor_list(lmp.x, lmp.type, lmp.ilist, lmp.inum, lmp.nlocal,
                             lmp.nghost, lmp.cutneighmax, lmp.subdomain,
                             nullptr, 0, lmp.numneigh, lmp.firstneigh_flat,
                             lmp.numneigh_max);
#endif
  }
}

void clear_neighbor_list_workspace(Device dev) {
  if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::clear_neighbor_list_workspace();
#endif
  }
}

double get_neighbor_list_memory_usage(Device dev) {
  if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    return GPU::get_neighbor_list_memory_usage();
#else
    return 0.0;
#endif
  } else {
    return 0.0;
  }
}

// ----------------------------------------------------------------------
// Get exact cmax and setup tensors
// ----------------------------------------------------------------------

template <typename Scalar>
int get_exact_cmax(TensorMDContext<Scalar>* ctx) {
  DeviceArgs<Scalar> device_args{};
  device_args.ilist = ctx->lmpdata.ilist;
  device_args.inum = ctx->lmpdata.inum;
  device_args.type = ctx->lmpdata.type;
  device_args.x = ctx->lmpdata.x;
  device_args.numneigh = ctx->lmpdata.numneigh;
  device_args.cutforcesq = ctx->cutforcesq;
  device_args.eltij_pos = ctx->data->eltij_pos;
  device_args.b = ctx->data->dims.b;
  device_args.fmap = ctx->lmpdata.fmap;

  if (ctx->device == Device::CPU) {
    // For CPU, the neighbor list is already built by LAMMPS and we can directly
    // use it to calculate cmax.
    device_args.firstneigh = ctx->lmpdata.firstneigh;
    return CPU::get_exact_cmax(device_args);
  } else if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    device_args.firstneigh_flat = ctx->lmpdata.firstneigh_flat;
    device_args.numneigh_max = ctx->lmpdata.numneigh_max;
    return GPU::get_exact_cmax(device_args);
#endif
  }
  return 0;
}

template <typename Scalar>
void setup_tensors(TensorMDContext<Scalar>* ctx) {
  DeviceArgs<Scalar> device_args{};
  device_args.a = ctx->data->dims.a;
  device_args.b = ctx->data->dims.b;
  device_args.c = ctx->data->dims.c;
  device_args.d = ctx->data->dims.d;
  device_args.m = ctx->data->dims.m;
  device_args.M = ctx->data->M;
  device_args.R = ctx->data->R;
  device_args.eltij_pos = ctx->data->eltij_pos;
  device_args.drdrx = ctx->data->drdrx;
  device_args.ijlist = ctx->data->ijlist;
  device_args.i2a = ctx->data->i2a;
  device_args.a2i = ctx->data->a2i;
  device_args.zijlist = ctx->data->zijlist;
  device_args.tlist = ctx->data->tlist;
  device_args.tijlist = ctx->data->tijlist;
  device_args.numbers = ctx->data->numbers;
  device_args.cutforcesq = ctx->cutforcesq;
  device_args.ilist = ctx->lmpdata.ilist;
  device_args.inum = ctx->lmpdata.inum;
  device_args.type = ctx->lmpdata.type;
  device_args.fmap = ctx->lmpdata.fmap;
  device_args.numneigh = ctx->lmpdata.numneigh;
  device_args.x = ctx->lmpdata.x;
  device_args.pair_map = const_cast<int*>(ctx->encoder->d_pair_map);
  device_args.nlocal = ctx->lmpdata.nlocal;
  device_args.nghost = ctx->lmpdata.nghost;
  device_args.T_proj_algo = ctx->data->T_proj_algo;

  if (ctx->device == Device::CPU) {
    device_args.firstneigh = ctx->lmpdata.firstneigh;
    CPU::setup_tensors(device_args);
  } else if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    device_args.firstneigh_flat = ctx->lmpdata.firstneigh_flat;
    device_args.numneigh_max = ctx->lmpdata.numneigh_max;
    GPU::setup_tensors(device_args);
#endif
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template int get_exact_cmax<float>(TensorMDContext<float>* context);
template int get_exact_cmax<double>(TensorMDContext<double>* context);

template void setup_tensors<float>(TensorMDContext<float>* context);
template void setup_tensors<double>(TensorMDContext<double>* context);

}  // namespace TENSORMD::Kernels
