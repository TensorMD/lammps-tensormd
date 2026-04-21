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

#ifndef TENSORMD_KERNELS_INTERNAL_H
#define TENSORMD_KERNELS_INTERNAL_H

#include "kernels_internal_types.h"

namespace TENSORMD::Kernels {

namespace CPU {

template <typename Scalar>
void initialize(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args);

template <typename Scalar>
int get_exact_cmax(DeviceArgs<Scalar>& device_args);

template <typename Scalar>
void setup_tensors(DeviceArgs<Scalar>& device_args);

template <typename Scalar>
void calc_P(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_G(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_W(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_angular_forces(DeviceArgs<Scalar>& args);

template <typename Scalar>
void sum_forces(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args);

}  // namespace CPU

namespace GPU {
#ifdef TENSORMD_GPU

void setup_device(int mpiid, bool verbose_device_map);

template <typename Scalar>
void initialize(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args);

void finalize();

void sync_lammps_data(HostArgs<double>& host_args,
                      DeviceArgs<double>& device_args);

template <typename Scalar>
int get_exact_cmax(DeviceArgs<Scalar>& device_args);

template <typename Scalar>
void setup_tensors(DeviceArgs<Scalar>& device_args);

template <typename Scalar>
void calc_P(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_G(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_W(DeviceArgs<Scalar>& args);

template <typename Scalar>
void calc_angular_forces(DeviceArgs<Scalar>& args);

template <typename Scalar>
void sum_forces(HostArgs<Scalar>& host_args, DeviceArgs<Scalar>& device_args);

// ----------------------------------------------------------------------------
// GPU Neighbor List Utils
// ----------------------------------------------------------------------------

void build_neighbor_list(const double* d_x, const int* d_type,
                         const int* d_ilist, int inum, int nlocal, int nghost,
                         double cutneighmax, double* subdomain,
                         const double* cutsq_table_d, int ntypes,
                         int* d_numneigh, int*& d_firstneigh_flat,
                         int& neighstride);

void clear_neighbor_list_workspace();

double get_neighbor_list_memory_usage();

#endif
}  // namespace GPU
}  // namespace TENSORMD::Kernels

#endif
