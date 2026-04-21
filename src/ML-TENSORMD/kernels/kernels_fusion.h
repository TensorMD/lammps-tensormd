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

#ifndef TENSORMD_KERNELS_FUSION_H
#define TENSORMD_KERNELS_FUSION_H

#include "../tensormd_context.h"

namespace TENSORMD::Kernels::Fusion {

// Calculate P_kdba and G_mkba with both M and H calculated on-the-fly.
template <typename Scalar, const bool WriteP>
void calc_descriptors_fused(TensorMDContext<Scalar>* context);

// The fusion kernel to calculate radial and angular forces simultaneously.
// Requires W_kdba, R_cba, drdrx_xcba, tijlist_cba and spline param atlas.
template <typename Scalar, const bool ReadP>
void calc_forces_fused(TensorMDContext<Scalar>* context);

// The ultimate fusion kernel that handle all calculations.
template <typename Scalar>
void calc_all_fused(TensorMDContext<Scalar>* context, double& h_etotal,
                    Scalar* d_eatoms);

}  // namespace TENSORMD::Kernels::Fusion

#endif
