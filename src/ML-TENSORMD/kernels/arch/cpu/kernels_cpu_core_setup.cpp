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

#include <cmath>
#include <stdexcept>

#include "../../../tensormd_constants.h"
#include "../../geometric/irreducible_moments.hpp"
#include "../../geometric/reducible_moments.hpp"
#include "../kernels_internal.h"
#include "../kernels_internal_utils.h"

namespace TENSORMD::Kernels::CPU {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void initialize(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args) {
  const int b = device_args.b;
  for (int i = 0; i < b * b; i++) {
    device_args.eltij_pos[i] = host_args.eltij_pos[i];
  }
  if (device_args.numbers != nullptr && host_args.numbers != nullptr) {
    for (int i = 0; i < host_args.n_numbers; i++) {
      device_args.numbers[i] = host_args.numbers[i];
    }
  }
  for (int i = 0; i < host_args.lmp_ntypes + 1; i++) {
    device_args.fmap[i] = host_args.fmap[i];
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int get_exact_cmax(DeviceArgs<Scalar>& device_args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::get_exact_cmax);
  int ii, elti, eltj, i, j, b, jn, neigh[TENSORMD_MAX_NSPECIES];
  double xtmp, ytmp, ztmp, dij[3], rij2;
  int cmax = 0;
#if defined(_OPENMP)
#pragma omp parallel for private(ii, elti, eltj, i, j, b, jn, neigh, xtmp, \
                                     ytmp, ztmp, dij, rij2)                \
    reduction(max : cmax)
#endif
  for (ii = 0; ii < device_args.inum; ii++) {
    for (elti = 0; elti < TENSORMD_MAX_NSPECIES; elti++) {
      neigh[elti] = 0;
    }
    i = device_args.ilist[ii];
    elti = device_args.fmap[device_args.type[i]];
    xtmp = device_args.x[i * 3 + 0];
    ytmp = device_args.x[i * 3 + 1];
    ztmp = device_args.x[i * 3 + 2];
    for (jn = 0; jn < device_args.numneigh[i]; jn++) {
      j = device_args.firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = device_args.x[j * 3 + 0] - xtmp;
      dij[1] = device_args.x[j * 3 + 1] - ytmp;
      dij[2] = device_args.x[j * 3 + 2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < device_args.cutforcesq && rij2 > 1e-10) {
        if (device_args.b > 1) {
          eltj = device_args.fmap[device_args.type[j]];
          b = device_args.eltij_pos[elti * device_args.b + eltj];
        } else {
          b = 0;
        }
        neigh[b]++;
      }
    }
    for (b = 0; b < device_args.b; b++) {
      if (neigh[b] > cmax) cmax = neigh[b];
    }
  }
  Utils::CPU::timer_stop(Profiling::TimedKernel::get_exact_cmax, 0.0);
  return cmax;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar, int m_val, bool UseReducible>
void setup_tensors_kernel(DeviceArgs<Scalar>& device_args) {
  Utils::CPU::timer_start(Profiling::TimedKernel::setup_tensors);

  if (device_args.zijlist != nullptr) {
    if (device_args.numbers == nullptr) {
      throw std::runtime_error(
          "Error in setup_tensors: device_args->numbers is empty but zijlist "
          "is "
          "requested.");
    }
  }

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<int> neigh(TENSORMD_MAX_NSPECIES, 0);
    int i, j, a, b, c, d;
    int ii, jn, elti, eltj;
    size_t stride1, stride2, stride3, strided;
    double xtmp, ytmp, ztmp;
    double x_exact, y_exact, z_exact;
    double rijinv_exact, rij_exact, rij2, dij[3];
    double cutforcesq = device_args.cutforcesq;
    Scalar x, y, z, rij, rijinv;
    int Zi, Zj;
    const int* __restrict__ pair_map = device_args.pair_map;
    const int* __restrict__ numbers = device_args.numbers;
    Scalar* __restrict__ R = device_args.R;
    Scalar* __restrict__ drdrx = device_args.drdrx;
    Scalar* __restrict__ M = device_args.M;
    const int* __restrict__ type = device_args.type;
    const int* __restrict__ fmap = device_args.fmap;
    const int* __restrict__ numneigh = device_args.numneigh;
    const int* __restrict__ ilist = device_args.ilist;
    const int* __restrict__ i2a = device_args.i2a;
    const int* __restrict__ eltij_pos = device_args.eltij_pos;
    int** __restrict__ firstneigh = device_args.firstneigh;
    int* __restrict__ ijlist = device_args.ijlist;
    int* __restrict__ zijlist = device_args.zijlist;
    int* __restrict__ tlist = device_args.tlist;
    int* __restrict__ tijlist = device_args.tijlist;
    const int A = device_args.a;
    const int B = device_args.b;
    const int C = device_args.c;
    const int D = device_args.d;

#if defined(_OPENMP)
#pragma omp for
#endif
    for (ii = 0; ii < A; ii++) {
      std::fill(neigh.begin(), neigh.end(), 0);
      i = ilist[ii];
      elti = fmap[type[i]];
      xtmp = device_args.x[i * 3 + 0];
      ytmp = device_args.x[i * 3 + 1];
      ztmp = device_args.x[i * 3 + 2];
      if (B == 1) {
        a = i;
      } else {
        // Map Lammps atom index 'i' to TensorMD re-ordered index 'a' so that
        // atoms of the same specie are grouped together.
        a = i2a[i];
      }
      if (tlist) {
        tlist[a] = elti;
      }
      for (jn = 0; jn < numneigh[i]; jn++) {
        j = firstneigh[i][jn];
        j &= NEIGHMASK;
        dij[0] = device_args.x[j * 3 + 0] - xtmp;
        dij[1] = device_args.x[j * 3 + 1] - ytmp;
        dij[2] = device_args.x[j * 3 + 2] - ztmp;
        rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
        if (rij2 < cutforcesq && rij2 > 1e-10) {
          eltj = fmap[type[j]];
          rij_exact = std::sqrt(rij2);
          rijinv_exact = 1.0 / rij_exact;
          if (B == 1) {
            b = 0;
          } else {
            b = eltij_pos[elti * B + eltj];
          }
          c = neigh[b];
          stride1 = a * C * B + b * C + c;
          stride2 = stride1 * 2;
          stride3 = stride1 * 3;
          strided = stride1 * D;
          if (c < C) {
            x_exact = dij[0] * rijinv_exact;
            y_exact = dij[1] * rijinv_exact;
            z_exact = dij[2] * rijinv_exact;
            x = static_cast<Scalar>(x_exact);
            y = static_cast<Scalar>(y_exact);
            z = static_cast<Scalar>(z_exact);
            rij = static_cast<Scalar>(rij_exact);
            R[stride1] = rij;
            drdrx[stride3 + 0] = x;
            drdrx[stride3 + 1] = y;
            drdrx[stride3 + 2] = z;
            ijlist[stride2 + 0] = i;
            ijlist[stride2 + 1] = j;
            Zi = numbers[elti];
            Zj = numbers[eltj];
            if (pair_map) {
              const int pair_index = PAIR_KEY(Zi, Zj);
              tijlist[stride1] = pair_map[pair_index];
            } else {
              tijlist[stride1] = 0;
            }
            if (zijlist) {
              zijlist[stride2 + 0] = Zi;
              zijlist[stride2 + 1] = Zj;
            }
            // Maybe we do not need to pre-computed and store M_dcba
            if (M) {
              if constexpr (UseReducible) {
                Geometric::Reducible::calculate_M_d_otf<Scalar, m_val>(
                    x, y, z, M + strided);
              } else {
                Geometric::Irreducible::calculate_M_u_otf<Scalar, m_val>(
                    x, y, z, M + strided);
              }
            }
          }
          neigh[b]++;
        }
      }

      for (b = 0; b < B; b++) {
        for (c = neigh[b]; c < C; c++) {
          stride1 = a * C * B + b * C + c;
          stride2 = stride1 * 2;
          stride3 = stride1 * 3;
          strided = stride1 * D;
          ijlist[stride2 + 0] = -1;
          ijlist[stride2 + 1] = -1;
          if (zijlist) {
            zijlist[stride2 + 0] = 0;
            zijlist[stride2 + 1] = 0;
          }
          tijlist[stride1] = -1;
          R[stride1] = 0.0;
          drdrx[stride3 + 0] = 0.0;
          drdrx[stride3 + 1] = 0.0;
          drdrx[stride3 + 2] = 0.0;
          if (M) {
            for (d = 0; d < D; d++) {
              M[strided + d] = 0.0;
            }
          }
        }
      }
    }
  }
  double FLOPs = 0.0;
  if (device_args.M) {
    if (UseReducible) {
      FLOPs = Geometric::Reducible::M_d_FLOPs[m_val];
    } else {
      FLOPs = Geometric::Irreducible::M_u_FLOPs[m_val];
    }
    FLOPs *= device_args.a * device_args.b * device_args.c;
  }
  Utils::CPU::timer_stop(Profiling::TimedKernel::setup_tensors, FLOPs);
}

template <typename Scalar>
void setup_tensors(DeviceArgs<Scalar>& device_args) {
  if (device_args.T_proj_algo < 0 || device_args.T_proj_algo > 2) {
    throw std::runtime_error("Unknown P->G projection algorithm.");
  }
  if (device_args.m == 1) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 1, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 1, true>(device_args);
    }
  } else if (device_args.m == 2) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 2, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 2, true>(device_args);
    }
  } else if (device_args.m == 3) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 3, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 3, true>(device_args);
    }
  } else if (device_args.m == 4) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 4, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 4, true>(device_args);
    }
  } else if (device_args.m == 5) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 5, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 5, true>(device_args);
    }
  } else if (device_args.m == 6) {
    if (device_args.T_proj_algo == 2) {
      setup_tensors_kernel<Scalar, 6, false>(device_args);
    } else {
      setup_tensors_kernel<Scalar, 6, true>(device_args);
    }
  } else {
    throw std::runtime_error("Invalid lmax for setup_tensors");
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void initialize<float>(HostArgs<float>& host_args,
                                DeviceArgs<float>& device_arg);
template void initialize<double>(HostArgs<double>& host_args,
                                 DeviceArgs<double>& device_arg);

template int get_exact_cmax<float>(DeviceArgs<float>& device_arg);
template int get_exact_cmax<double>(DeviceArgs<double>& device_arg);

template void setup_tensors<float>(DeviceArgs<float>& device_arg);
template void setup_tensors<double>(DeviceArgs<double>& device_ar);

}  // namespace TENSORMD::Kernels::CPU
