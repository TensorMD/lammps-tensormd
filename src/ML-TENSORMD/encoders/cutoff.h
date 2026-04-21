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

#ifndef TENSORMD_CUTOFF_H
#define TENSORMD_CUTOFF_H

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../tensormd_types.h"

using Eigen::Tensor;
using Eigen::TensorMap;

namespace TENSORMD {

template <typename Scalar>
class Cutoff {
 public:
  explicit Cutoff(Scalar rc) {
    rcut = rc;
    cosine = true;
  }

  Cutoff(Scalar rc, Scalar gam) {
    rcut = rc;
    gamma = gam;
    cosine = false;
  }

  void compute(TensorData<Scalar> *data);
  void compute(Device dev, int n, Scalar *x, Scalar *y);

  bool is_cosine() { return cosine; }

 protected:
  bool cosine;
  Scalar rcut;
  Scalar gamma;
};

}  // namespace TENSORMD

#endif  // TENSORMD_CUTOFF_H
