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

#include "encoder.h"

#include "../kernels/kernels_encoder.h"
#include "../kernels/kernels_spline.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void Encoder<Scalar>::setup_interpolation(std::vector<std::string>& species,
                                          Scalar rmin, Scalar delta) {
  ctx.grid_dr = delta;
  ctx.grid_rmin = rmin;
  ctx.grid_size = static_cast<int>((ctx.grid_rmax - rmin) / delta) + 1;

  // Setup interpolation
  setup_interpolation_default(species, rmin, delta);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void Encoder<Scalar>::forward(TensorData<Scalar>* data) {
  if (is_interpolatable()) {
    // Call the interpolation function to compute radial(R) * sij(R) directly
    Kernels::Spline::cubic_interpolate(data, ctx.d_params_atlas, ctx.grid_size,
                                       ctx.grid_rmin, ctx.grid_dr);

  } else {
    // Call the cutoff function. This will update sij and dsij in-place.
    cutoff->compute(data);

    // Call the forward function to compute radial interaction tensor radial(R)
    // and its gradient dradial(R) w.r.t. the input R.
    this->radial_forward(data);

    // The chain rule: apply the cutoff to H and dHdr
    Kernels::Encoder::apply_cutoff(data);
  }
};

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void Encoder<Scalar>::backward(TensorData<Scalar>* data) {
  if (is_interpolatable()) {
    // Use the tensor contraction to calculate radial forces Fr:
    // Fr_xcba = W_dkba * M_dcba * dHdr_kcba * drdrx_xcba
    // This code path may not require precomputed U, if dHdr is calculated
    // on-the-fly.
    Kernels::Encoder::interp_backward(data, ctx.d_params_atlas, ctx.grid_size,
                                      ctx.grid_rmin, ctx.grid_dr);

  } else {
    // This contraction path explicitly requires U, or U_kcba explicitly.
    Kernels::Encoder::calc_U_abck(data);

    // Use the chain-rule to compute radial forces Fr
    // We have:
    //         dE/drx = (dE/dH * dH/dR + dE/dM * dM/dR) * dR/drx
    //              H = radial * sij
    //              U = dE/dH
    // Then:
    //          dH/dR = dradial * sij + radial * dsij
    // Thus,
    //             Fr = U * (dradial * sij + radial * dsij) * drdrx
    // In the forward phase, we already calculated dHdr = radial * dsij
    // In the backward phase, we first calculate U * sij and U * radial * dsij:
    // Up = U * sij
    // dHdr = U * dHdr
    // Note than U uses the memory of H and the following kernels only need Up.
    // Thus, Up can reuse the memory of U.
    Kernels::Encoder::calc_Up(data);

    // Then we calculate dHdrp = Up * dradial
    // dHdrp reuses the memory of dsij.
    this->radial_backward(data);

    // Finally we calculate Fr = (dHdrp + dHdr) * drdrx
    Kernels::Encoder::calc_Fr(data);
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class Encoder<float>;
template class Encoder<double>;

}  // namespace TENSORMD
