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

#include "kernels_v1_ops.h"

#include "arch/kernels_internal_v1_ops.h"

namespace TENSORMD::Kernels::V1 {

template <typename Scalar>
void setup_index_maps(const int* elt_a_offset, TensorMDContext<Scalar>* ctx) {
  // No need to setup index maps if the potential has only one atom type.
  if (ctx->data->dims.b == 1) {
    return;
  }
  HostArgs<Scalar> host_args{};
  host_args.elt_a_offset = elt_a_offset;

  DeviceArgs<Scalar> device_args{};
  device_args.a = ctx->data->dims.a;
  device_args.b = ctx->data->dims.b;
  device_args.ilist = ctx->lmpdata.ilist;
  device_args.elt_a_offset = ctx->data->elt_a_offset;
  device_args.i2a = ctx->data->i2a;
  device_args.a2i = ctx->data->a2i;
  device_args.type = ctx->lmpdata.type;
  device_args.fmap = ctx->lmpdata.fmap;

  if (ctx->device == Device::CPU) {
    CPU::setup_index_maps(host_args, device_args);
  } else if (ctx->device == Device::GPU) {
#if defined(TENSORMD_GPU)
    GPU::setup_index_maps(host_args, device_args);
#endif
  }
}

template <typename Scalar>
void tdnp_inject_temperature(Device dev, int n, int ydim, Scalar* y,
                             bool use_atom_etemp, const Scalar* etemp) {
  if (dev == Device::CPU) {
    CPU::tdnp_inject_temperature(n, ydim, y, use_atom_etemp, etemp);
  }
}

template <typename Scalar>
void tdnp_combine_free_energy(Device dev, int n, bool squared_temp,
                              bool use_atom_etemp, const Scalar* etemp,
                              const Scalar* E_atom, Scalar E_tot,
                              Scalar* S_atom, Scalar& S_tot, Scalar* F_atom,
                              Scalar& F_tot) {
  if (dev == Device::CPU) {
    CPU::tdnp_combine_free_energy(n, squared_temp, use_atom_etemp, etemp,
                                  E_atom, E_tot, S_atom, S_tot, F_atom, F_tot);
  }
}

template <typename Scalar>
void tdnp_combine_gradients(Device dev, int n, int dim, bool squared_temp,
                            bool use_atom_etemp, const Scalar* etemp,
                            const Scalar* S_grad, Scalar* U_grad) {
  if (dev == Device::CPU) {
    CPU::tdnp_combine_gradients(n, dim, squared_temp, use_atom_etemp, etemp,
                                S_grad, U_grad);
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void setup_index_maps<float>(const int* elt_a_offset,
                                      TensorMDContext<float>* ctx);
template void setup_index_maps<double>(const int* elt_a_offset,
                                       TensorMDContext<double>* ctx);

template void tdnp_inject_temperature<float>(Device dev, int n, int ydim,
                                             float* y, bool use_atom_etemp,
                                             const float* etemp);
template void tdnp_inject_temperature<double>(Device dev, int n, int ydim,
                                              double* y, bool use_atom_etemp,
                                              const double* etemp);

template void tdnp_combine_free_energy<float>(
    Device dev, int n, bool squared_temp, bool use_atom_etemp,
    const float* etemp, const float* E_atom, float E_tot, float* S_atom,
    float& S_tot, float* F_atom, float& F_tot);
template void tdnp_combine_free_energy<double>(
    Device dev, int n, bool squared_temp, bool use_atom_etemp,
    const double* etemp, const double* E_atom, double E_tot, double* S_atom,
    double& S_tot, double* F_atom, double& F_tot);

template void tdnp_combine_gradients<float>(Device dev, int n, int dim,
                                            bool squared_temp,
                                            bool use_atom_etemp,
                                            const float* etemp,
                                            const float* S_grad, float* U_grad);
template void tdnp_combine_gradients<double>(
    Device dev, int n, int dim, bool squared_temp, bool use_atom_etemp,
    const double* etemp, const double* S_grad, double* U_grad);

}  // namespace TENSORMD::Kernels::V1
