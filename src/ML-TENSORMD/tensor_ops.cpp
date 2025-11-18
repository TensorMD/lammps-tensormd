/* ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "tensor_ops.h"
#include "cppblas.hpp"

/* ----------------------------------------------------------------------
Sequential batch matmul kernel that calls the regular Eigen matmul.
We prefer this over the tensor contraction because it performs
better on vector-matrix and matrix-vector products.

Return the total FLOPs of this call.

See also: tensorflow/core/kernels/batch_matmul_op_impl.h
------------------------------------------------------------------------- */

template <typename Scalar>
void batch_matmul(TensorMap<Tensor<Scalar, 3>> &in_x,
                  TensorMap<Tensor<Scalar, 3>> &in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 3>> *out, int start, int limit)
{
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
void batch_matmul(Tensor<Scalar, 4> &in_x, Tensor<Scalar, 4> &in_y, bool adj_x,
                  bool adj_y, Tensor<Scalar, 4> *out)
{
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
void batch_matmul(Tensor<Scalar, 4> &in_x, TensorMap<Tensor<Scalar, 4>> &in_y,
                  bool adj_x, bool adj_y, Tensor<Scalar, 4> *out)
{
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
void batch_matmul(TensorMap<Tensor<Scalar, 4>> &in_x,
                  TensorMap<Tensor<Scalar, 4>> &in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 4>> *out)
{
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
Kernel F1
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_F1(TensorMap<Tensor<Scalar, 4>> &U,
               TensorMap<Tensor<Scalar, 4>> &dHdr,
               TensorMap<Tensor<Scalar, 4>> &drdrx,
               TensorMap<Tensor<int, 3>> &mask,
               TensorMap<Tensor<Scalar, 4>> *F1)
{
  size_t amax = U.dimension(3);
  size_t bmax = U.dimension(2);
  size_t cmax = U.dimension(1);
  size_t kmax = U.dimension(0);
  size_t a, b, c, offset, pos;
  Scalar y;
#if defined(_OPENMP) && !defined(MULTI_THREAD_BLAS)
#pragma omp parallel for private(a, b, c, offset, pos, y)
#endif
  for (a = 0; a < amax; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        pos = (a * bmax * cmax + b * cmax + c) * 3;
        if (mask((long) c, (long) b, (long) a) == 0) {
          F1->data()[pos + 0] = 0;
          F1->data()[pos + 1] = 0;
          F1->data()[pos + 2] = 0;
        } else {
          offset = (a * bmax * cmax + b * cmax + c) * kmax;
          y = cppblas_dot((int) kmax, U.data() + offset, 1,
                          dHdr.data() + offset, 1);
          F1->data()[pos + 0] = drdrx.data()[pos + 0] * y;
          F1->data()[pos + 1] = drdrx.data()[pos + 1] * y;
          F1->data()[pos + 2] = drdrx.data()[pos + 2] * y;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
Kernel F2: F2_xcba = dMdrx_dxcba * V_dcba
------------------------------------------------------------------------- */

template <typename Scalar> void calculate_dM_s(Scalar *dM, int offset = 1)
{
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  dM[ix0 + 0] = 0.0;
  dM[iy0 + 0] = 0.0;
  dM[iz0 + 0] = 0.0;
}

template <typename Scalar>
void calculate_dM_p(Scalar rijinv, Scalar *drdrx, Scalar *dM, int offset = 4)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_s(dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;

  dM[ix0 + 1] = -rijinv * (xx - 1);
  dM[ix0 + 2] = -rijinv * xy;
  dM[ix0 + 3] = -rijinv * xz;

  dM[iy0 + 1] = -rijinv * xy;
  dM[iy0 + 2] = -rijinv * (yy - 1);
  dM[iy0 + 3] = -rijinv * yz;

  dM[iz0 + 1] = -rijinv * xz;
  dM[iz0 + 2] = -rijinv * yz;
  dM[iz0 + 3] = -rijinv * (zz - 1);
}

template <typename Scalar>
void calculate_dM_d(Scalar rijinv, Scalar *drdrx, Scalar *dM, int offset = 10)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_p(rijinv, drdrx, dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;

  dM[ix0 + 4] = -rijinv * (2 * xxx - 2 * x);
  dM[ix0 + 5] = -rijinv * (2 * xxy - y);
  dM[ix0 + 6] = -rijinv * (2 * xxz - z);
  dM[ix0 + 7] = -rijinv * 2 * xyy;
  dM[ix0 + 8] = -rijinv * 2 * xyz;
  dM[ix0 + 9] = -rijinv * 2 * xzz;

  dM[iy0 + 4] = -rijinv * 2 * xxy;
  dM[iy0 + 5] = -rijinv * (2 * xyy - x);
  dM[iy0 + 6] = -rijinv * 2 * xyz;
  dM[iy0 + 7] = -rijinv * (2 * yyy - 2 * y);
  dM[iy0 + 8] = -rijinv * (2 * yyz - z);
  dM[iy0 + 9] = -rijinv * 2 * yzz;

  dM[iz0 + 4] = -rijinv * 2 * xxz;
  dM[iz0 + 5] = -rijinv * 2 * xyz;
  dM[iz0 + 6] = -rijinv * (2 * xzz - x);
  dM[iz0 + 7] = -rijinv * 2 * yyz;
  dM[iz0 + 8] = -rijinv * (2 * yzz - y);
  dM[iz0 + 9] = -rijinv * (2 * zzz - 2 * z);
}

template <typename Scalar>
void calculate_dM_f(Scalar rijinv, Scalar *drdrx, Scalar *dM, int offset = 20)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_d(rijinv, drdrx, dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;
  xxxx = xxx * x;
  xxxy = xxx * y;
  xxxz = xxx * z;
  xxyy = xxy * y;
  xxyz = xxy * z;
  xxzz = xxz * z;
  xyyy = xyy * y;
  xyyz = xyy * z;
  xyzz = xyz * z;
  xzzz = xzz * z;
  yyyy = yyy * y;
  yyyz = yyy * z;
  yyzz = yyz * z;
  yzzz = yzz * z;
  zzzz = zzz * z;

  dM[ix0 + 10] = -rijinv * (3 * xxxx - 3 * xx);
  dM[ix0 + 11] = -rijinv * (3 * xxxy - 2 * xy);
  dM[ix0 + 12] = -rijinv * (3 * xxxz - 2 * xz);
  dM[ix0 + 13] = -rijinv * (3 * xxyy - yy);
  dM[ix0 + 14] = -rijinv * (3 * xxyz - yz);
  dM[ix0 + 15] = -rijinv * (3 * xxzz - zz);
  dM[ix0 + 16] = -rijinv * 3 * xyyy;
  dM[ix0 + 17] = -rijinv * 3 * xyyz;
  dM[ix0 + 18] = -rijinv * 3 * xyzz;
  dM[ix0 + 19] = -rijinv * 3 * xzzz;

  dM[iy0 + 10] = -rijinv * 3 * xxxy;
  dM[iy0 + 11] = -rijinv * (3 * xxyy - xx);
  dM[iy0 + 12] = -rijinv * 3 * xxyz;
  dM[iy0 + 13] = -rijinv * (3 * xyyy - 2 * xy);
  dM[iy0 + 14] = -rijinv * (3 * xyyz - xz);
  dM[iy0 + 15] = -rijinv * 3 * xyzz;
  dM[iy0 + 16] = -rijinv * (3 * yyyy - 3 * yy);
  dM[iy0 + 17] = -rijinv * (3 * yyyz - 2 * yz);
  dM[iy0 + 18] = -rijinv * (3 * yyzz - zz);
  dM[iy0 + 19] = -rijinv * 3 * yzzz;

  dM[iz0 + 10] = -rijinv * 3 * xxxz;
  dM[iz0 + 11] = -rijinv * 3 * xxyz;
  dM[iz0 + 12] = -rijinv * (3 * xxzz - xx);
  dM[iz0 + 13] = -rijinv * 3 * xyyz;
  dM[iz0 + 14] = -rijinv * (3 * xyzz - xy);
  dM[iz0 + 15] = -rijinv * (3 * xzzz - 2 * xz);
  dM[iz0 + 16] = -rijinv * 3 * yyyz;
  dM[iz0 + 17] = -rijinv * (3 * yyzz - yy);
  dM[iz0 + 18] = -rijinv * (3 * yzzz - 2 * yz);
  dM[iz0 + 19] = -rijinv * (3 * zzzz - 3 * zz);
}

template <typename Scalar>
void calculate_dM_g(Scalar rijinv, Scalar *drdrx, Scalar *dM, int offset = 35)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  Scalar xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz;
  Scalar xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz;
  Scalar yyzzz, yzzzz, zzzzz;

  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_f(rijinv, drdrx, dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;
  xxxx = xxx * x;
  xxxy = xxx * y;
  xxxz = xxx * z;
  xxyy = xxy * y;
  xxyz = xxy * z;
  xxzz = xxz * z;
  xyyy = xyy * y;
  xyyz = xyy * z;
  xyzz = xyz * z;
  xzzz = xzz * z;
  yyyy = yyy * y;
  yyyz = yyy * z;
  yyzz = yyz * z;
  yzzz = yzz * z;
  zzzz = zzz * z;
  xxxxx = xxxx * x;
  xxxxy = xxxx * y;
  xxxxz = xxxx * z;
  xxxyy = xxxy * y;
  xxxyz = xxxy * z;
  xxxzz = xxxz * z;
  xxyyy = xxyy * y;
  xxyyz = xxyy * z;
  xxyzz = xxyz * z;
  xxzzz = xxzz * z;
  xyyyy = xyyy * y;
  xyyyz = xyyy * z;
  xyyzz = xyyz * z;
  xyzzz = xyzz * z;
  xzzzz = xzzz * z;
  yyyyy = yyyy * y;
  yyyyz = yyyy * z;
  yyyzz = yyyz * z;
  yyzzz = yyzz * z;
  yzzzz = yzzz * z;
  zzzzz = zzzz * z;

  dM[ix0 + 20] = -rijinv * (4 * xxxxx - 4 * xxx);
  dM[ix0 + 21] = -rijinv * (4 * xxxxy - 3 * xxy);
  dM[ix0 + 22] = -rijinv * (4 * xxxxz - 3 * xxz);
  dM[ix0 + 23] = -rijinv * (4 * xxxyy - 2 * xyy);
  dM[ix0 + 24] = -rijinv * (4 * xxxyz - 2 * xyz);
  dM[ix0 + 25] = -rijinv * (4 * xxxzz - 2 * xzz);
  dM[ix0 + 26] = -rijinv * (4 * xxyyy - yyy);
  dM[ix0 + 27] = -rijinv * (4 * xxyyz - yyz);
  dM[ix0 + 28] = -rijinv * (4 * xxyzz - yzz);
  dM[ix0 + 29] = -rijinv * (4 * xxzzz - zzz);
  dM[ix0 + 30] = -rijinv * 4 * xyyyy;
  dM[ix0 + 31] = -rijinv * 4 * xyyyz;
  dM[ix0 + 32] = -rijinv * 4 * xyyzz;
  dM[ix0 + 33] = -rijinv * 4 * xyzzz;
  dM[ix0 + 34] = -rijinv * 4 * xzzzz;

  dM[iy0 + 20] = -rijinv * 4 * xxxxy;
  dM[iy0 + 21] = -rijinv * (4 * xxxyy - xxx);
  dM[iy0 + 22] = -rijinv * 4 * xxxyz;
  dM[iy0 + 23] = -rijinv * (4 * xxyyy - 2 * xxy);
  dM[iy0 + 24] = -rijinv * (4 * xxyyz - xxz);
  dM[iy0 + 25] = -rijinv * 4 * xxyzz;
  dM[iy0 + 26] = -rijinv * (4 * xyyyy - 3 * xyy);
  dM[iy0 + 27] = -rijinv * (4 * xyyyz - 2 * xyz);
  dM[iy0 + 28] = -rijinv * (4 * xyyzz - xzz);
  dM[iy0 + 29] = -rijinv * 4 * xyzzz;
  dM[iy0 + 30] = -rijinv * (4 * yyyyy - 4 * yyy);
  dM[iy0 + 31] = -rijinv * (4 * yyyyz - 3 * yyz);
  dM[iy0 + 32] = -rijinv * (4 * yyyzz - 2 * yzz);
  dM[iy0 + 33] = -rijinv * (4 * yyzzz - zzz);
  dM[iy0 + 34] = -rijinv * 4 * yzzzz;

  dM[iz0 + 20] = -rijinv * 4 * xxxxz;
  dM[iz0 + 21] = -rijinv * 4 * xxxyz;
  dM[iz0 + 22] = -rijinv * (4 * xxxzz - xxx);
  dM[iz0 + 23] = -rijinv * 4 * xxyyz;
  dM[iz0 + 24] = -rijinv * (4 * xxyzz - xxy);
  dM[iz0 + 25] = -rijinv * (4 * xxzzz - 2 * xxz);
  dM[iz0 + 26] = -rijinv * 4 * xyyyz;
  dM[iz0 + 27] = -rijinv * (4 * xyyzz - xyy);
  dM[iz0 + 28] = -rijinv * (4 * xyzzz - 2 * xyz);
  dM[iz0 + 29] = -rijinv * (4 * xzzzz - 3 * xzz);
  dM[iz0 + 30] = -rijinv * 4 * yyyyz;
  dM[iz0 + 31] = -rijinv * (4 * yyyzz - yyy);
  dM[iz0 + 32] = -rijinv * (4 * yyzzz - 2 * yyz);
  dM[iz0 + 33] = -rijinv * (4 * yzzzz - 3 * yzz);
  dM[iz0 + 34] = -rijinv * (4 * zzzzz - 4 * zzz);
}

template <typename Scalar>
void calculate_dM_h(Scalar rijinv, Scalar *drdrx, Scalar *dM, int offset = 56)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  Scalar xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz;
  Scalar xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz;
  Scalar yyzzz, yzzzz, zzzzz;
  Scalar xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz;
  Scalar xxxyzz, xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy;
  Scalar xyyyyz, xyyyzz, xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz;
  Scalar yyyzzz, yyzzzz, yzzzzz, zzzzzz;

  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_g(rijinv, drdrx, dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;
  xxxx = xxx * x;
  xxxy = xxx * y;
  xxxz = xxx * z;
  xxyy = xxy * y;
  xxyz = xxy * z;
  xxzz = xxz * z;
  xyyy = xyy * y;
  xyyz = xyy * z;
  xyzz = xyz * z;
  xzzz = xzz * z;
  yyyy = yyy * y;
  yyyz = yyy * z;
  yyzz = yyz * z;
  yzzz = yzz * z;
  zzzz = zzz * z;
  xxxxx = xxxx * x;
  xxxxy = xxxx * y;
  xxxxz = xxxx * z;
  xxxyy = xxxy * y;
  xxxyz = xxxy * z;
  xxxzz = xxxz * z;
  xxyyy = xxyy * y;
  xxyyz = xxyy * z;
  xxyzz = xxyz * z;
  xxzzz = xxzz * z;
  xyyyy = xyyy * y;
  xyyyz = xyyy * z;
  xyyzz = xyyz * z;
  xyzzz = xyzz * z;
  xzzzz = xzzz * z;
  yyyyy = yyyy * y;
  yyyyz = yyyy * z;
  yyyzz = yyyz * z;
  yyzzz = yyzz * z;
  yzzzz = yzzz * z;
  zzzzz = zzzz * z;
  xxxxxx = xxxxx * x;
  xxxxxy = xxxxx * y;
  xxxxxz = xxxxx * z;
  xxxxyy = xxxxy * y;
  xxxxyz = xxxxy * z;
  xxxxzz = xxxxz * z;
  xxxyyy = xxxyy * y;
  xxxyyz = xxxyy * z;
  xxxyzz = xxxyz * z;
  xxxzzz = xxxzz * z;
  xxyyyy = xxyyy * y;
  xxyyyz = xxyyy * z;
  xxyyzz = xxyyz * z;
  xxyzzz = xxyzz * z;
  xxzzzz = xxzzz * z;
  xyyyyy = xyyyy * y;
  xyyyyz = xyyyy * z;
  xyyyzz = xyyyz * z;
  xyyzzz = xyyzz * z;
  xyzzzz = xyzzz * z;
  xzzzzz = xzzzz * z;
  yyyyyy = yyyyy * y;
  yyyyyz = yyyyy * z;
  yyyyzz = yyyyz * z;
  yyyzzz = yyyzz * z;
  yyzzzz = yyzzz * z;
  yzzzzz = yzzzz * z;
  zzzzzz = zzzzz * z;

  dM[ix0 + 35] = -rijinv * (5 * xxxxxx - 5 * xxxx);
  dM[ix0 + 36] = -rijinv * (5 * xxxxxy - 4 * xxxy);
  dM[ix0 + 37] = -rijinv * (5 * xxxxxz - 4 * xxxz);
  dM[ix0 + 38] = -rijinv * (5 * xxxxyy - 3 * xxyy);
  dM[ix0 + 39] = -rijinv * (5 * xxxxyz - 3 * xxyz);
  dM[ix0 + 40] = -rijinv * (5 * xxxxzz - 3 * xxzz);
  dM[ix0 + 41] = -rijinv * (5 * xxxyyy - 2 * xyyy);
  dM[ix0 + 42] = -rijinv * (5 * xxxyyz - 2 * xyyz);
  dM[ix0 + 43] = -rijinv * (5 * xxxyzz - 2 * xyzz);
  dM[ix0 + 44] = -rijinv * (5 * xxxzzz - 2 * xzzz);
  dM[ix0 + 45] = -rijinv * (5 * xxyyyy - yyyy);
  dM[ix0 + 46] = -rijinv * (5 * xxyyyz - yyyz);
  dM[ix0 + 47] = -rijinv * (5 * xxyyzz - yyzz);
  dM[ix0 + 48] = -rijinv * (5 * xxyzzz - yzzz);
  dM[ix0 + 49] = -rijinv * (5 * xxzzzz - zzzz);
  dM[ix0 + 50] = -rijinv * 5 * xyyyyy;
  dM[ix0 + 51] = -rijinv * 5 * xyyyyz;
  dM[ix0 + 52] = -rijinv * 5 * xyyyzz;
  dM[ix0 + 53] = -rijinv * 5 * xyyzzz;
  dM[ix0 + 54] = -rijinv * 5 * xyzzzz;
  dM[ix0 + 55] = -rijinv * 5 * xzzzzz;

  dM[iy0 + 35] = -rijinv * 5 * xxxxxy;
  dM[iy0 + 36] = -rijinv * (5 * xxxxyy - xxxx);
  dM[iy0 + 37] = -rijinv * 5 * xxxxyz;
  dM[iy0 + 38] = -rijinv * (5 * xxxyyy - 2 * xxxy);
  dM[iy0 + 39] = -rijinv * (5 * xxxyyz - xxxz);
  dM[iy0 + 40] = -rijinv * 5 * xxxyzz;
  dM[iy0 + 41] = -rijinv * (5 * xxyyyy - 3 * xxyy);
  dM[iy0 + 42] = -rijinv * (5 * xxyyyz - 2 * xxyz);
  dM[iy0 + 43] = -rijinv * (5 * xxyyzz - xxzz);
  dM[iy0 + 44] = -rijinv * 5 * xxyzzz;
  dM[iy0 + 45] = -rijinv * (5 * xyyyyy - 4 * xyyy);
  dM[iy0 + 46] = -rijinv * (5 * xyyyyz - 3 * xyyz);
  dM[iy0 + 47] = -rijinv * (5 * xyyyzz - 2 * xyzz);
  dM[iy0 + 48] = -rijinv * (5 * xyyzzz - xzzz);
  dM[iy0 + 49] = -rijinv * 5 * xyzzzz;
  dM[iy0 + 50] = -rijinv * (5 * yyyyyy - 5 * yyyy);
  dM[iy0 + 51] = -rijinv * (5 * yyyyyz - 4 * yyyz);
  dM[iy0 + 52] = -rijinv * (5 * yyyyzz - 3 * yyzz);
  dM[iy0 + 53] = -rijinv * (5 * yyyzzz - 2 * yzzz);
  dM[iy0 + 54] = -rijinv * (5 * yyzzzz - zzzz);
  dM[iy0 + 55] = -rijinv * 5 * yzzzzz;

  dM[iz0 + 35] = -rijinv * 5 * xxxxxz;
  dM[iz0 + 36] = -rijinv * 5 * xxxxyz;
  dM[iz0 + 37] = -rijinv * (5 * xxxxzz - xxxx);
  dM[iz0 + 38] = -rijinv * 5 * xxxyyz;
  dM[iz0 + 39] = -rijinv * (5 * xxxyzz - xxxy);
  dM[iz0 + 40] = -rijinv * (5 * xxxzzz - 2 * xxxz);
  dM[iz0 + 41] = -rijinv * 5 * xxyyyz;
  dM[iz0 + 42] = -rijinv * (5 * xxyyzz - xxyy);
  dM[iz0 + 43] = -rijinv * (5 * xxyzzz - 2 * xxyz);
  dM[iz0 + 44] = -rijinv * (5 * xxzzzz - 3 * xxzz);
  dM[iz0 + 45] = -rijinv * 5 * xyyyyz;
  dM[iz0 + 46] = -rijinv * (5 * xyyyzz - xyyy);
  dM[iz0 + 47] = -rijinv * (5 * xyyzzz - 2 * xyyz);
  dM[iz0 + 48] = -rijinv * (5 * xyzzzz - 3 * xyzz);
  dM[iz0 + 49] = -rijinv * (5 * xzzzzz - 4 * xzzz);
  dM[iz0 + 50] = -rijinv * 5 * yyyyyz;
  dM[iz0 + 51] = -rijinv * (5 * yyyyzz - yyyy);
  dM[iz0 + 52] = -rijinv * (5 * yyyzzz - 2 * yyyz);
  dM[iz0 + 53] = -rijinv * (5 * yyzzzz - 3 * yyzz);
  dM[iz0 + 54] = -rijinv * (5 * yzzzzz - 4 * yzzz);
  dM[iz0 + 55] = -rijinv * (5 * zzzzzz - 5 * zzzz);
}

template <typename Scalar>
void calculate_dM(int max_moment, Scalar rijinv, Scalar *drdrx, Scalar *dM)
{
  if (max_moment == 0)
    calculate_dM_s<Scalar>(dM, 1);
  else if (max_moment == 1)
    calculate_dM_p<Scalar>(rijinv, drdrx, dM, 4);
  else if (max_moment == 2)
    calculate_dM_d<Scalar>(rijinv, drdrx, dM, 10);
  else if (max_moment == 3)
    calculate_dM_f<Scalar>(rijinv, drdrx, dM, 20);
  else if (max_moment == 4)
    calculate_dM_g<Scalar>(rijinv, drdrx, dM, 35);
  else
    calculate_dM_h<Scalar>(rijinv, drdrx, dM, 56);
}

template <typename Scalar>
void kernel_fused_F2(int max_moment, TensorMap<Tensor<Scalar, 4>> &V,
                     TensorMap<Tensor<Scalar, 3>> &R,
                     TensorMap<Tensor<Scalar, 4>> &drdrx,
                     TensorMap<Tensor<int, 3>> &mask,
                     TensorMap<Tensor<Scalar, 4>> *F2)
{
  int adim = static_cast<int>(V.dimension(3));
  int bdim = static_cast<int>(V.dimension(2));
  int cdim = static_cast<int>(V.dimension(1));
  int ddim = static_cast<int>(V.dimension(0));
  int loc;
  int a, b, c;
  const int dmax = 56;
  Scalar dM[3 * dmax];
  Scalar rij, rijinv;
#if defined(_OPENMP) && !defined(MULTI_THREAD_BLAS)
#pragma omp parallel for private(a, b, c, dM, rij, rijinv, loc)
#endif
  for (a = 0; a < adim; a++) {
    for (b = 0; b < bdim; b++) {
      for (c = 0; c < cdim; c++) {
        if (mask(c, b, a) == 0) continue;
        loc = a * bdim * cdim + b * cdim + c;
        rij = R(c, b, a);
        rijinv = 1.0 / rij;
        calculate_dM<Scalar>(max_moment, rijinv, drdrx.data() + loc * 3, dM);
        Scalar *V_ic = V.data() + loc * ddim;
        for (int x = 0; x < 3; x++) {
          (*F2)(x, c, b, a) = cppblas_dot(ddim, V_ic, 1, &dM[x * ddim], 1);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
Kernel U: U_kcba = dEdS_dkba * M_dcba
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_U(TensorMap<Tensor<Scalar, 4>> &dEdS,
              TensorMap<Tensor<Scalar, 4>> &M, TensorMap<Tensor<Scalar, 4>> *U)
{
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
  size_t limit = dEdS.dimension(2) * dEdS.dimension(3);
  const int d = static_cast<int>(dEdS.dimension(0));    // gemm M
  const int c = static_cast<int>(M.dimension(1));       // gemm N
  const int k = static_cast<int>(dEdS.dimension(1));    // gemm K
  const int lda = d;
  const int ldb = d;
  const int ldc = k;
  const int stridea = d * k;
  const int strideb = c * d;
  const int stridec = k * c;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  cppblas_gemm_batch_strided(CblasColMajor, CblasTrans, CblasNoTrans, k, c, d,
                             alpha, dEdS.data(), lda, stridea, M.data(), ldb,
                             strideb, beta, U->data(), ldc, stridec, limit);
#else
  batch_matmul<Scalar>(dEdS, M, true, false, U);
#endif
}

/* ----------------------------------------------------------------------
Kernel V: V_dcba = dEdS_dkba * H_kcba
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_V(TensorMap<Tensor<Scalar, 4>> &dEdS,
              TensorMap<Tensor<Scalar, 4>> &H, TensorMap<Tensor<Scalar, 4>> *V)
{
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
  size_t limit = dEdS.dimension(2) * dEdS.dimension(3);
  const int d = static_cast<int>(dEdS.dimension(0));    // gemm M
  const int c = static_cast<int>(H.dimension(1));       // gemm N
  const int k = static_cast<int>(dEdS.dimension(1));    // gemm K
  const int lda = d;
  const int ldb = k;
  const int ldc = d;
  const int stridea = d * k;
  const int strideb = c * k;
  const int stridec = d * c;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  cppblas_gemm_batch_strided(CblasColMajor, CblasNoTrans, CblasNoTrans, d, c, k,
                             alpha, dEdS.data(), lda, stridea, H.data(), ldb,
                             strideb, beta, V->data(), ldc, stridec, limit);
#else
  batch_matmul<Scalar>(dEdS, H, false, false, V);
#endif
}

/* ----------------------------------------------------------------------
Kernel P: P_dkba = M_dcba * H_kcba
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_P(TensorMap<Tensor<Scalar, 4>> &M, TensorMap<Tensor<Scalar, 4>> &H,
              TensorMap<Tensor<Scalar, 4>> *P)
{
#if defined(EIGEN_USE_MKL_ALL) && (INTEL_MKL_VERSION >= 20210000)
  size_t limit = M.dimension(2) * M.dimension(3);
  const int d = static_cast<int>(M.dimension(0));    // gemm M
  const int k = static_cast<int>(H.dimension(0));    // gemm N
  const int c = static_cast<int>(M.dimension(1));    // gemm K
  const int lda = d;
  const int ldb = k;
  const int ldc = d;
  const int stridea = d * c;
  const int strideb = c * k;
  const int stridec = d * k;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  const int incx = 1;
  const int incy = 1;
  cppblas_gemm_batch_strided(CblasColMajor, CblasNoTrans, CblasTrans, d, k, c,
                             alpha, M.data(), lda, stridea, H.data(), ldb,
                             strideb, beta, P->data(), ldc, stridec, limit);
#else
  batch_matmul<Scalar>(M, H, false, true, P);
#endif
}

/* ----------------------------------------------------------------------
Kernel dEdS_dkba: 2 * T_md * dEdG_mkba * P_dkba
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_dEdS(Tensor<Scalar, 2> &T, TensorMap<Tensor<Scalar, 4>> &dEdG,
                 TensorMap<Tensor<Scalar, 4>> &P,
                 TensorMap<Tensor<Scalar, 4>> *dEdS)
{
  const int m = static_cast<int>(T.dimension(0));
  const int d = static_cast<int>(T.dimension(1));
  const int k = static_cast<int>(dEdG.dimension(1));
  const int b = static_cast<int>(dEdG.dimension(2));
  const int a = static_cast<int>(dEdG.dimension(3));
  cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, k * b * a, d, m, 2.0,
               dEdG.data(), m, T.data(), m, 0.0, dEdS->data(), d);
#if EIGEN_USE_MKL_ALL
  cppvml_mul(a * b * k * d, dEdS->data(), P.data(), dEdS->data());
#else
  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < a * b * k * d; i++) (*dEdS)(i) *= P(i);
#endif
}

/* ----------------------------------------------------------------------
Kernel dEdG
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_dEdG(TensorMap<Tensor<Scalar, 4>> &G,
                 TensorMap<Tensor<Scalar, 4>> *dEdG)
{
  const int m = static_cast<int>(G.dimension(0));
  const int size = static_cast<int>(G.size());
  Scalar *src = G.data();
  Scalar *dst = dEdG->data();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < size; i += m) {
    if (fabs(src[i]) < 1e-14)
      dst[i] = 0.0;
    else
      dst[i] = dst[i] / src[i] * 0.5;
  }
}

/* ----------------------------------------------------------------------
Kernel Q_mkba = T_md * P_dkba
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_Q(TensorMap<Tensor<Scalar, 4>> &P, Tensor<Scalar, 2> &T,
              TensorMap<Tensor<Scalar, 4>> *Q)
{
  //   const int m = static_cast<int>(T.dimension(0));
  //   const int d = static_cast<int>(T.dimension(1));
  //   const int k = static_cast<int>(P.dimension(1));
  //   const int b = static_cast<int>(P.dimension(2));
  //   const int a = static_cast<int>(P.dimension(3));
  // #if defined(EIGEN_USE_MKL_ALL)
  //   Scalar *S = (Scalar *)malloc(a * b * k * d * sizeof(Scalar));
  //   cppvml_square(a * b * k * d, P.data(), S);
  //   cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a * b * k, m, d, 1.0,
  //                S, d, T.data(), m, 0.0, Q->data(), m);
  //   free(S);
  // #else
  Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
      Eigen::IndexPair<int>(1, 0)};
  *Q = T.contract(P.square(), contract_dims);
  // #endif
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void batch_matmul<double>(Tensor<double, 4> &in_x,
                                   Tensor<double, 4> &in_y, bool adj_x,
                                   bool adj_y, Tensor<double, 4> *out);
template void batch_matmul<float>(Tensor<float, 4> &in_x,
                                  Tensor<float, 4> &in_y, bool adj_x,
                                  bool adj_y, Tensor<float, 4> *out);
template void batch_matmul<double>(Tensor<double, 4> &in_x,
                                   TensorMap<Tensor<double, 4>> &in_y,
                                   bool adj_x, bool adj_y,
                                   Tensor<double, 4> *out);
template void batch_matmul<float>(Tensor<float, 4> &in_x,
                                  TensorMap<Tensor<float, 4>> &in_y, bool adj_x,
                                  bool adj_y, Tensor<float, 4> *out);
template void batch_matmul<double>(TensorMap<Tensor<double, 3>> &in_x,
                                   TensorMap<Tensor<double, 3>> &in_y,
                                   bool adj_x, bool adj_y,
                                   TensorMap<Tensor<double, 3>> *out, int start,
                                   int limit);
template void batch_matmul<float>(TensorMap<Tensor<float, 3>> &in_x,
                                  TensorMap<Tensor<float, 3>> &in_y, bool adj_x,
                                  bool adj_y, TensorMap<Tensor<float, 3>> *out,
                                  int start, int limit);
template void batch_matmul<float>(TensorMap<Tensor<float, 4>> &in_x,
                                  TensorMap<Tensor<float, 4>> &in_y, bool adj_x,
                                  bool adj_y, TensorMap<Tensor<float, 4>> *out);
template void batch_matmul<double>(TensorMap<Tensor<double, 4>> &in_x,
                                   TensorMap<Tensor<double, 4>> &in_y,
                                   bool adj_x, bool adj_y,
                                   TensorMap<Tensor<double, 4>> *out);
template void kernel_F1<double>(TensorMap<Tensor<double, 4>> &U,
                                TensorMap<Tensor<double, 4>> &dHdr,
                                TensorMap<Tensor<double, 4>> &drdrx,
                                TensorMap<Tensor<int, 3>> &mask,
                                TensorMap<Tensor<double, 4>> *F1);
template void kernel_F1<float>(TensorMap<Tensor<float, 4>> &U,
                               TensorMap<Tensor<float, 4>> &dHdr,
                               TensorMap<Tensor<float, 4>> &drdrx,
                               TensorMap<Tensor<int, 3>> &mask,
                               TensorMap<Tensor<float, 4>> *F1);
template void kernel_fused_F2<float>(int max_moment,
                                     TensorMap<Tensor<float, 4>> &V,
                                     TensorMap<Tensor<float, 3>> &R,
                                     TensorMap<Tensor<float, 4>> &drdrx,
                                     TensorMap<Tensor<int, 3>> &mask,
                                     TensorMap<Tensor<float, 4>> *F2);
template void kernel_fused_F2<double>(int max_moment,
                                      TensorMap<Tensor<double, 4>> &V,
                                      TensorMap<Tensor<double, 3>> &R,
                                      TensorMap<Tensor<double, 4>> &drdrx,
                                      TensorMap<Tensor<int, 3>> &mask,
                                      TensorMap<Tensor<double, 4>> *F2);

template void kernel_U<double>(TensorMap<Tensor<double, 4>> &dEdS,
                               TensorMap<Tensor<double, 4>> &M,
                               TensorMap<Tensor<double, 4>> *U);
template void kernel_U<float>(TensorMap<Tensor<float, 4>> &dEdS,
                              TensorMap<Tensor<float, 4>> &M,
                              TensorMap<Tensor<float, 4>> *U);
template void kernel_V<double>(TensorMap<Tensor<double, 4>> &dEdS,
                               TensorMap<Tensor<double, 4>> &H,
                               TensorMap<Tensor<double, 4>> *V);
template void kernel_V<float>(TensorMap<Tensor<float, 4>> &dEdS,
                              TensorMap<Tensor<float, 4>> &H,
                              TensorMap<Tensor<float, 4>> *V);
template void kernel_P<double>(TensorMap<Tensor<double, 4>> &M,
                               TensorMap<Tensor<double, 4>> &H,
                               TensorMap<Tensor<double, 4>> *P);
template void kernel_P<float>(TensorMap<Tensor<float, 4>> &M,
                              TensorMap<Tensor<float, 4>> &H,
                              TensorMap<Tensor<float, 4>> *P);
template void kernel_dEdS<double>(Tensor<double, 2> &T,
                                  TensorMap<Tensor<double, 4>> &dEdG,
                                  TensorMap<Tensor<double, 4>> &P,
                                  TensorMap<Tensor<double, 4>> *dEdS);
template void kernel_dEdS<float>(Tensor<float, 2> &T,
                                 TensorMap<Tensor<float, 4>> &dEdG,
                                 TensorMap<Tensor<float, 4>> &P,
                                 TensorMap<Tensor<float, 4>> *dEdS);
template void kernel_dEdG<double>(TensorMap<Tensor<double, 4>> &G,
                                  TensorMap<Tensor<double, 4>> *dEdG);
template void kernel_dEdG<float>(TensorMap<Tensor<float, 4>> &G,
                                 TensorMap<Tensor<float, 4>> *dEdG);
template void kernel_Q<double>(TensorMap<Tensor<double, 4>> &P,
                               Tensor<double, 2> &T,
                               TensorMap<Tensor<double, 4>> *Q);
template void kernel_Q<float>(TensorMap<Tensor<float, 4>> &P,
                              Tensor<float, 2> &T,
                              TensorMap<Tensor<float, 4>> *Q);
