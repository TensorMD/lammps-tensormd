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

#include "cutoff.h"

#include "../kernels/kernels_encoder.h"

namespace TENSORMD {

template <typename Scalar>
void Cutoff<Scalar>::compute(TensorData<Scalar> *data) {
  const int n = data->dims.a * data->dims.b * data->dims.c;
  Scalar *x = data->R;
  Scalar *f = data->sij;
  Scalar *df = data->dsij;
  int *tijlist = data->tijlist;
  Device dev = data->device;
  if (cosine) {
    Kernels::Encoder::cosine_cutoff(dev, rcut, n, x, f, df, tijlist);
  } else {
    Kernels::Encoder::polynomial_cutoff(dev, rcut, n, x, f, df, tijlist, gamma);
  }
}

template <typename Scalar>
void Cutoff<Scalar>::compute(Device dev, int n, Scalar *x, Scalar *y) {
  Scalar *df = nullptr;
  int *tijlist = nullptr;
  if (cosine) {
    Kernels::Encoder::cosine_cutoff(dev, rcut, n, x, y, df, tijlist);
  } else {
    Kernels::Encoder::polynomial_cutoff(dev, rcut, n, x, y, df, tijlist, gamma);
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class Cutoff<float>;
template class Cutoff<double>;

}  // namespace TENSORMD
