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

#ifndef TENSORMD_KERNELS_ARCH_CPU_BATCH_MATMUL_OPS_HPP
#define TENSORMD_KERNELS_ARCH_CPU_BATCH_MATMUL_OPS_HPP

#include "../../../eigen/Eigen/Dense"
#include "../../../eigen/unsupported/Eigen/CXX11/Tensor"

#define NEIGHMASK 0x1FFFFFFF

using Eigen::Tensor;
using Eigen::TensorMap;

namespace TENSORMD::Kernels::CPU {

/* ----------------------------------------------------------------------
     Sequential batch matmul kernel that calls the regular Eigen matmul.
     We prefer this over the tensor contraction because it performs
     better on vector-matrix and matrix-vector products.
     Return the total FLOPs of this call.
     See also: tensorflow/core/kernels/batch_matmul_op_impl.h
  ---------------------------------------------------------------------- */
template <typename Scalar>
void batch_matmul(TensorMap<Tensor<Scalar, 3>>& in_x,
                  TensorMap<Tensor<Scalar, 3>>& in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 3>>* out, int start, int limit) {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  int i;
#if defined(_OPENMP) && !defined(MULTI_THREAD_BLAS)
#pragma omp parallel for private(i)
#endif
  for (i = start; i < limit; ++i) {
    auto x =
        ConstMatrixMap(in_x.data() + i * in_x.dimension(0) * in_x.dimension(1),
                       in_x.dimension(0), in_x.dimension(1));
    auto y =
        ConstMatrixMap(in_y.data() + i * in_y.dimension(0) * in_y.dimension(1),
                       in_y.dimension(0), in_y.dimension(1));
    auto z = MatrixMap(out->data() + i * out->dimension(0) * out->dimension(1),
                       out->dimension(0), out->dimension(1));
#if defined(_OPENMP) && !defined(MULTI_THREAD_BLAS)
#pragma omp parallel num_threads(1)
#endif
    {
      if (!adj_x) {
        if (!adj_y) {
          z.noalias() = x * y;
        } else {
          z.noalias() = x * y.adjoint();
        }
      } else {
        if (!adj_y) {
          z.noalias() = x.adjoint() * y;
        } else {
          z.noalias() = x.adjoint() * y.adjoint();
        }
      }
    }
  }
}

template <typename Scalar>
void batch_matmul(Tensor<Scalar, 4>& in_x, Tensor<Scalar, 4>& in_y, bool adj_x,
                  bool adj_y, Tensor<Scalar, 4>* out) {
  int a1 = static_cast<int>(in_x.dimension(0));
  int b1 = static_cast<int>(in_x.dimension(1));
  int c = static_cast<int>(in_x.dimension(2));
  int d = static_cast<int>(in_x.dimension(3));
  int a2 = static_cast<int>(in_y.dimension(0));
  int b2 = static_cast<int>(in_y.dimension(1));
  int a3 = static_cast<int>(out->dimension(0));
  int b3 = static_cast<int>(out->dimension(1));
  int limit = c * d;
  TensorMap<Tensor<Scalar, 3>> Tx{in_x.data(), {a1, b1, limit}};
  TensorMap<Tensor<Scalar, 3>> Ty{in_y.data(), {a2, b2, limit}};
  TensorMap<Tensor<Scalar, 3>> Tz{out->data(), {a3, b3, limit}};
  batch_matmul<Scalar>(Tx, Ty, adj_x, adj_y, &Tz, 0, limit);
}

template <typename Scalar>
void batch_matmul(Tensor<Scalar, 4>& in_x, TensorMap<Tensor<Scalar, 4>>& in_y,
                  bool adj_x, bool adj_y, Tensor<Scalar, 4>* out) {
  int a1 = static_cast<int>(in_x.dimension(0));
  int b1 = static_cast<int>(in_x.dimension(1));
  int c = static_cast<int>(in_x.dimension(2));
  int d = static_cast<int>(in_x.dimension(3));
  int a2 = static_cast<int>(in_y.dimension(0));
  int b2 = static_cast<int>(in_y.dimension(1));
  int a3 = static_cast<int>(out->dimension(0));
  int b3 = static_cast<int>(out->dimension(1));
  int limit = c * d;
  TensorMap<Tensor<Scalar, 3>> Tx{in_x.data(), {a1, b1, limit}};
  TensorMap<Tensor<Scalar, 3>> Ty{in_y.data(), {a2, b2, limit}};
  TensorMap<Tensor<Scalar, 3>> Tz{out->data(), {a3, b3, limit}};
  batch_matmul<Scalar>(Tx, Ty, adj_x, adj_y, &Tz, 0, limit);
}

template <typename Scalar>
void batch_matmul(TensorMap<Tensor<Scalar, 4>>& in_x,
                  TensorMap<Tensor<Scalar, 4>>& in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 4>>* out) {
  int a1 = static_cast<int>(in_x.dimension(0));
  int b1 = static_cast<int>(in_x.dimension(1));
  int c = static_cast<int>(in_x.dimension(2));
  int d = static_cast<int>(in_x.dimension(3));
  int a2 = static_cast<int>(in_y.dimension(0));
  int b2 = static_cast<int>(in_y.dimension(1));
  int a3 = static_cast<int>(out->dimension(0));
  int b3 = static_cast<int>(out->dimension(1));
  int limit = c * d;
  TensorMap<Tensor<Scalar, 3>> Tx{in_x.data(), {a1, b1, limit}};
  TensorMap<Tensor<Scalar, 3>> Ty{in_y.data(), {a2, b2, limit}};
  TensorMap<Tensor<Scalar, 3>> Tz{out->data(), {a3, b3, limit}};
  batch_matmul<Scalar>(Tx, Ty, adj_x, adj_y, &Tz, 0, limit);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void batch_matmul<float>(TensorMap<Tensor<float, 4>>& in_x,
                                  TensorMap<Tensor<float, 4>>& in_y, bool adj_x,
                                  bool adj_y, TensorMap<Tensor<float, 4>>* out);

template void batch_matmul<double>(TensorMap<Tensor<double, 4>>& in_x,
                                   TensorMap<Tensor<double, 4>>& in_y,
                                   bool adj_x, bool adj_y,
                                   TensorMap<Tensor<double, 4>>* out);

template void batch_matmul<float>(Tensor<float, 4>& in_x,
                                  TensorMap<Tensor<float, 4>>& in_y, bool adj_x,
                                  bool adj_y, Tensor<float, 4>* out);

template void batch_matmul<double>(Tensor<double, 4>& in_x,
                                   TensorMap<Tensor<double, 4>>& in_y,
                                   bool adj_x, bool adj_y,
                                   Tensor<double, 4>* out);

template void batch_matmul<float>(Tensor<float, 4>& in_x,
                                  Tensor<float, 4>& in_y, bool adj_x,
                                  bool adj_y, Tensor<float, 4>* out);

template void batch_matmul<double>(Tensor<double, 4>& in_x,
                                   Tensor<double, 4>& in_y, bool adj_x,
                                   bool adj_y, Tensor<double, 4>* out);

}  // namespace TENSORMD::Kernels::CPU

#endif
