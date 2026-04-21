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

#include <stdexcept>

#include "../kernels_internal_v1_ops.h"
#include "cppblas.hpp"

namespace TENSORMD::Kernels::V1::CPU {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void setup_index_maps(HostArgs<Scalar>& host_args,
                      DeviceArgs<Scalar>& device_args) {
  if (device_args.b == 1) {
    return;
  }
  for (int elti = 0; elti < device_args.b; elti++) {
    device_args.elt_a_offset[elti] = host_args.elt_a_offset[elti];
  }
  int* fmap = device_args.fmap;
  int* type = device_args.type;
  int* i2a = device_args.i2a;
  int* a2i = device_args.a2i;
  int* ilist = device_args.ilist;
  int* elt_a_offset = device_args.elt_a_offset;
  int i, ii, a, elti;
  for (ii = 0; ii < device_args.a; ii++) {
    i = ilist[ii];
    elti = fmap[type[i]];
    a = elt_a_offset[elti];
    elt_a_offset[elti]++;
    i2a[i] = a;
    a2i[a] = i;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void tdnp_inject_temperature(int n, int h_dim, Scalar* Ht, bool use_atom_etemp,
                             const Scalar* etemp) {
  int i;
  if (use_atom_etemp) {
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; ++i) {
      Ht[i * h_dim + h_dim - 1] = etemp[i];
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < n; ++i) {
      Ht[i * h_dim + h_dim - 1] = etemp[0];
    }
  }
}

template <typename Scalar>
void tdnp_combine_free_energy(int n, bool squared_temp, bool use_atom_etemp,
                              const Scalar* etemp, const Scalar* E_atom,
                              Scalar E_tot, Scalar* S_atom, Scalar& S_tot,
                              Scalar* F_atom, Scalar& F_tot) {
  int i;
  if (use_atom_etemp) {
    if (E_atom == nullptr || F_atom == nullptr || S_atom == nullptr) {
      throw std::invalid_argument(
          "E_atom, S_atom and F_atom must all be provided when use_atom_etemp "
          "is true.");
    }
    Scalar toten = 0.0;
    if (squared_temp) {
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+ : toten)
#endif
      for (i = 0; i < n; i++) {
        S_atom[i] *= etemp[i];
        toten += S_atom[i];
      }
      S_tot = toten;
    }
    toten = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+ : toten)
#endif
    for (i = 0; i < n; i++) {
      F_atom[i] = E_atom[i] - etemp[i] * S_atom[i];
      toten += F_atom[i];
    }
    F_tot = toten;
  } else {
    if (squared_temp) {
      // For the Sommerfeld model, the electron entropy is quadratic in T.
      // So we should multiply S by T to make unit consistent.
      S_tot *= etemp[0];
      if (S_atom != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
        for (i = 0; i < n; i++) {
          S_atom[i] *= etemp[0];
        }
      }
    }
    if (F_atom != nullptr && S_atom != nullptr && E_atom != nullptr) {
      Linalg::CPU::cppblas_copy(n, const_cast<Scalar*>(E_atom), 1, F_atom, 1);
      Linalg::CPU::cppblas_axpy(n, -etemp[0], S_atom, 1, F_atom, 1);
    }
    F_tot = E_tot - etemp[0] * S_tot;
  }
}

template <typename Scalar>
void tdnp_combine_gradients(int n, int h_dim, bool squared_temp,
                            bool use_atom_etemp, const Scalar* etemp,
                            const Scalar* S_grad, Scalar* U_grad) {
  int i;
  if (use_atom_etemp) {
    // TODO: these codes should be further checked. I doubt the correctness.
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < h_dim * n; i++) {
      if (squared_temp) {
        U_grad[i] -= etemp[i / h_dim] * etemp[i / h_dim] * S_grad[i];
      } else {
        U_grad[i] -= etemp[i / h_dim] * S_grad[i];
      }
    }
  } else {
    Scalar scale = etemp[0];
    if (squared_temp) {
      scale *= etemp[0];
    }
    Linalg::CPU::cppblas_axpy(h_dim * n, -scale, const_cast<Scalar*>(S_grad), 1,
                              U_grad, 1);
  }
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = h_dim - 1; i < n * h_dim; i += h_dim) {
    U_grad[i] = 0.0;
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void setup_index_maps<float>(HostArgs<float>& host_args,
                                      DeviceArgs<float>& device_arg);
template void setup_index_maps<double>(HostArgs<double>& host_args,
                                       DeviceArgs<double>& device_arg);

template void tdnp_inject_temperature<float>(int n, int h_dim, float* Ht,
                                             bool use_atom_etemp,
                                             const float* etemp);

template void tdnp_inject_temperature<double>(int n, int h_dim, double* Ht,
                                              bool use_atom_etemp,
                                              const double* etemp);

template void tdnp_combine_free_energy<float>(int n, bool squared_temp,
                                              bool use_atom_etemp,
                                              const float* etemp,
                                              const float* E_atom, float E_tot,
                                              float* S_atom, float& S_tot,
                                              float* F_atom, float& F_tot);

template void tdnp_combine_free_energy<double>(
    int n, bool squared_temp, bool use_atom_etemp, const double* etemp,
    const double* E_atom, double E_tot, double* S_atom, double& S_tot,
    double* F_atom, double& F_tot);

template void tdnp_combine_gradients<float>(int n, int dim, bool squared_temp,
                                            bool use_atom_etemp,
                                            const float* etemp,
                                            const float* S_grad, float* U_grad);

template void tdnp_combine_gradients<double>(int n, int dim, bool squared_temp,
                                             bool use_atom_etemp,
                                             const double* etemp,
                                             const double* S_grad,
                                             double* U_grad);
}  // namespace TENSORMD::Kernels::V1::CPU