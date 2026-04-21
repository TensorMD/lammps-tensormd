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

#ifndef TENSORMD_KERNELS_ENCODER_H
#define TENSORMD_KERNELS_ENCODER_H

#include "../tensormd_types.h"

namespace TENSORMD::Kernels::Encoder {

// ----------------------------------------------------------------------
// Calculate radial forces with tensor contraction:
// Fr_xcba = W_dkba * M_dcba * dHdr_kcba * drdrx_xcba
// ----------------------------------------------------------------------

template <typename Scalar>
void interp_backward(TensorData<Scalar>* data, const Scalar* params_atlas,
                     int size, Scalar x0, Scalar dx);

// ----------------------------------------------------------------------
// Encoder and Cutoff Kernels
// ----------------------------------------------------------------------

template <typename Scalar>
void cosine_cutoff(Device dev, Scalar rcut, int n, Scalar* x, Scalar* f,
                   Scalar* df, int* tijlist);

template <typename Scalar>
void polynomial_cutoff(Device dev, Scalar rcut, int n, Scalar* x, Scalar* f,
                       Scalar* df, int* tijlist, Scalar gamma = 5.0);

template <typename Scalar>
void apply_cutoff(TensorData<Scalar>* data);

template <typename Scalar>
void apply_cutoff_forward_only(Device dev, int n, int k, const Scalar* sij,
                               Scalar* H);

template <typename Scalar>
void calc_U_abck(TensorData<Scalar>* data);

template <typename Scalar>
void calc_Up(TensorData<Scalar>* data);

template <typename Scalar>
void calc_Fr(TensorData<Scalar>* data);

}  // namespace TENSORMD::Kernels::Encoder

#endif  // TENSORMD_KERNELS_ENCODER_H