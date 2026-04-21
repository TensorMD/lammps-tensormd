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

#ifndef TENSORMD_KERNELS_H
#define TENSORMD_KERNELS_H

#include "../encoders/encoder.h"
#include "../tensormd_context.h"
#include "../tensormd_timer.h"

namespace TENSORMD::Kernels {

// ----------------------------------------------------------------------
// Initialization and Finalization
// ----------------------------------------------------------------------

// Setup device
template <typename Scalar>
void setup_device(TensorMDContext<Scalar>* context);

// Setup fixed-size tensors, allocate cuBlasHandles, etc.
template <typename Scalar>
void initialize(const int* eltij_pos, const int* fmap,
                std::vector<int>& numbers, TensorMDContext<Scalar>* context);

// Finalize and free any resources allocated in initialize().
template <typename Scalar>
void finalize(TensorMDContext<Scalar>* context);

// ----------------------------------------------------------------------
// Setup Tensors
// ----------------------------------------------------------------------

void sync_lammps_data(int* ilist, int* type, int* fmap, double** x,
                      int* numneigh, int** firstneigh, double** f,
                      double** vatom, LAMMPSData& lmp);

void build_neighbor_list(LAMMPSData& lmp);
double get_neighbor_list_memory_usage(Device dev);

template <typename Scalar>
int get_exact_cmax(TensorMDContext<Scalar>* context);

template <typename Scalar>
void setup_tensors(TensorMDContext<Scalar>* context);

// ----------------------------------------------------------------------
// Descriptors
// ----------------------------------------------------------------------

// Calculate P_dkba = M_dcba * H_kcba
template <typename Scalar>
void calc_P(TensorData<Scalar>* data);

// Calculate G_mkba, given P_dkba and T_md
template <typename Scalar>
void calc_G(TensorData<Scalar>* data);

// ----------------------------------------------------------------------
// Gradients
// ----------------------------------------------------------------------

// Calculate W_dkba, the gradients of E_a with respect to P_dkba
template <typename Scalar>
void calc_W(TensorData<Scalar>* data);

// Calculate the angular forces Fa with tensor contraction:
// Fa_xcba = W_dkba * H_kcba * dMdrx_xdcba * V_dcba (Angular)
template <typename Scalar>
void calc_angular_forces(TensorData<Scalar>* data);

// ----------------------------------------------------------------------
// Forces Accumulation
// ----------------------------------------------------------------------

// Accumulate forces and virials with Fr and Fa, then copy them to host.
// Fr is handled by Encoder while Fa is a general kernel.
template <typename Scalar>
void sum_forces(double** f, double** vatom, double scale,
                TensorMDContext<Scalar>* context);

}  // namespace TENSORMD::Kernels

#endif  // TENSORMD_KERNELS_H