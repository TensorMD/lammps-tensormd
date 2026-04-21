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

#ifndef TENSORMD_KERNELS_V1_H
#define TENSORMD_KERNELS_V1_H

#include "../tensormd_context.h"

namespace TENSORMD::Kernels::V1 {

template <typename Scalar>
void setup_index_maps(const int* elt_a_offset, TensorMDContext<Scalar>* ctx);

// Inject electron temperature into the last column of the latent vector y
// y shape: [n, ydim]
// The last column (index ydim-1) is overwritten.
template <typename Scalar>
void tdnp_inject_temperature(Device dev, int n, int h_dim, Scalar* Ht,
                             bool use_atom_etemp, const Scalar* etemp);

// F = E - T*S (and atomic versions)
template <typename Scalar>
void tdnp_combine_free_energy(Device dev, int n, bool squared_temp,
                              bool use_atom_etemp, const Scalar* etemp,
                              const Scalar* E_atom, Scalar E_tot,
                              Scalar* S_atom, Scalar& S_tot, Scalar* F_atom,
                              Scalar& F_tot);

// Calculate gradients dU = - (1 or T or T^2) * dS
template <typename Scalar>
void tdnp_combine_gradients(Device dev, int n, int dim, bool squared_temp,
                            bool use_atom_etemp, const Scalar* etemp,
                            const Scalar* S_grad, Scalar* U_grad);

}  // namespace TENSORMD::Kernels::V1

#endif  // TENSORMD_KERNELS_V1_H