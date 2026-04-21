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

#ifndef TENSORMD_KERNELS_ACTIVATION_HPP
#define TENSORMD_KERNELS_ACTIVATION_HPP

#include <cmath>
#include "../kernels_helpers.h"

namespace TENSORMD::Kernels {

// Activation (act_type) mapping:
// 0: ReLU, 1: Softplus, 2: Tanh, 3: Squareplus, 4: SiLU, 5: Sigmoid

constexpr static double activation_FLOPs[6] = {1, 4, 2, 7, 7, 5};

template <typename Scalar, int Activation>
KERNELS_DEVICE_INLINE void compute_activation(Scalar x, Scalar& y, Scalar& dy) {
  if constexpr (Activation == 0) {  // ReLU
    y = x > Scalar(0) ? x : Scalar(0);
    dy = x > Scalar(0) ? Scalar(1) : Scalar(0);
  } else if constexpr (Activation == 1) {  // Softplus
    // numerically stable enough for FP32
    const Scalar threshold = Scalar(20);
    if (x > threshold) {
      y = x;
      dy = Scalar(1);
    } else if (x < -threshold) {
      y = 0.0;
      dy = 0.0;
    } else {
      Scalar e = exp(x);
      y = log1p(e);
      dy = e / (Scalar(1) + e);  // sigmoid(x)
    }
  } else if constexpr (Activation == 2) {  // Tanh
    y = tanh(x);
    dy = Scalar(1) - y * y;

  } else if constexpr (Activation == 3) {  // Squareplus
    Scalar s = sqrt(x * x + Scalar(4));
    y = Scalar(0.5) * (x + s);
    dy = Scalar(0.5) * (Scalar(1) + x / s);
  } else if constexpr (Activation == 4) {  // SiLU
    Scalar s = Scalar(1) / (Scalar(1) + exp(-x));
    y = x * s;
    dy = s * (Scalar(1) + x * (Scalar(1) - s));
  } else if constexpr (Activation == 5) {  // Sigmoid
    y = Scalar(1) / (Scalar(1) + exp(-x));
    dy = y * (Scalar(1) - y);
  }
}

}  // namespace TENSORMD::Kernels

#endif
