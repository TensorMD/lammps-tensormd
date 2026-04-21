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

#ifndef TENSORMD_UTILS_TENSOR_DEBUG_H
#define TENSORMD_UTILS_TENSOR_DEBUG_H

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../tensormd_device.h"

using Eigen::Tensor;
using Eigen::TensorMap;

namespace TENSORMD::Debug {

template <typename Scalar>
void dump_tensor(Device dev, TensorMap<Tensor<Scalar, 3>>& in, const char* name,
                 int mpiid);

template <typename Scalar>
void dump_tensor(Device dev, TensorMap<Tensor<Scalar, 4>>& in, const char* name,
                 int mpiid);

}  // namespace TENSORMD::Debug

#endif
