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
  for (i = start; i < limit; ++i) {
    auto x =
        ConstMatrixMap(in_x.data() + i * in_x.dimension(0) * in_x.dimension(1),
                       in_x.dimension(0), in_x.dimension(1));
    auto y =
        ConstMatrixMap(in_y.data() + i * in_y.dimension(0) * in_y.dimension(1),
                       in_y.dimension(0), in_y.dimension(1));
    auto z = MatrixMap(out->data() + i * out->dimension(0) * out->dimension(1),
                       out->dimension(0), out->dimension(1));
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
Kernel F1: BLAS
------------------------------------------------------------------------- */

template <typename Scalar>
void kernel_F1(Tensor<Scalar, 4> &U, TensorMap<Tensor<Scalar, 4>> &dHdr,
               TensorMap<Tensor<Scalar, 4>> &drdrx, 
               TensorMap<Tensor<Scalar, 4>> *F1)
{
  size_t a = U.dimension(3);
  size_t b = U.dimension(2);
  size_t c = U.dimension(1);
  size_t k = U.dimension(0);

  size_t i;
  int n = static_cast<int>(k);
  size_t pos = 0;
  Scalar y;
  for (i = 0; i < a * b * c; ++i) {
    y = cppblas_dot(n, U.data() + i * k, 1, dHdr.data() + i * k, 1);
    F1->data()[pos + 0] = drdrx.data()[pos + 0] * y;
    F1->data()[pos + 1] = drdrx.data()[pos + 1] * y;
    F1->data()[pos + 2] = drdrx.data()[pos + 2] * y;
    pos += 3;
  }
}

template <typename Scalar>
void kernel_F1(Tensor<Scalar, 4> &U, TensorMap<Tensor<Scalar, 4>> &dHdr,
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
#if defined(USE_OPENMP)
  #pragma omp parallel for private(a, b, c, offset, pos, y)
#endif
  for (a = 0; a < amax; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        pos = (a * bmax * cmax + b * cmax + c) * 3;
        if (mask((long)c, (long)b, (long)a) == 0) {
          F1->data()[pos + 0] = 0;
          F1->data()[pos + 1] = 0;
          F1->data()[pos + 2] = 0;
        } else {
          offset = (a * bmax * cmax + b * cmax + c) * kmax;
#if defined(USE_OPENMP)
#pragma omp parallel num_threads(1)
#endif
          {
            y = cppblas_dot((int)kmax, U.data() + offset, 1, dHdr.data() + offset, 1);
          }
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

template <typename Scalar>
void kernel_F2(TensorMap<Tensor<Scalar, 3>> &dMdrx,
               TensorMap<Tensor<Scalar, 3>> &V,
               TensorMap<Tensor<Scalar, 3>> *F2)
{
  size_t limit = dMdrx.dimension(2);

#if defined(USE_MKL_BATCH) && (INTEL_MKL_VERSION >= 20210000)
  const int d = static_cast<int>(dMdrx.dimension(1));
  const int x = 3;
  const int lda = 1;
  const int ldb = d;
  const int ldc = 1;
  const int stridea = d;
  const int strideb = x * d;
  const int stridec = x;
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  const int incx = 1;
  const int incy = 1;
  cppblas_gemm_batch_strided(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, x, d,
                             alpha, V.data(), lda, stridea, dMdrx.data(), ldb,
                             strideb, beta, F2->data(), ldc, stridec, limit);
#else
  batch_matmul<Scalar>(V, dMdrx, false, false, F2, 0, limit);
#endif
}

template <typename Scalar>
void kernel_F2(TensorMap<Tensor<Scalar, 5>> &dMdrx,
               TensorMap<Tensor<Scalar, 4>> &V,
               TensorMap<Tensor<int, 3>> &mask,
               Tensor<Scalar, 4> *F2)
{
  int amax = static_cast<int>(V.dimension(3));
  int bmax = static_cast<int>(V.dimension(2));
  int cmax = static_cast<int>(V.dimension(1));
  int dmax = static_cast<int>(V.dimension(0));
  int loc;
  int a, b, c;
#if defined(USE_OPENMP)
  #pragma omp parallel for
#endif
  for (a = 0; a < amax; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        if (mask(c, b, a) == 0) continue;
        loc = a * bmax * cmax + b * cmax + c;
        Scalar *V_ic = V.data() + loc * dmax;
#if defined(USE_OPENMP)
#pragma omp parallel num_threads(1)
#endif
        {
          Scalar *dM = dMdrx.data() + loc * 3 * dmax;
          for (int x = 0; x < 3; x++) {
            (*F2)(x, c, b, a) = cppblas_dot(dmax, V_ic, 1, dM + x * dmax, 1);
          }
        }
      }
    }
  }
}

template <typename Scalar> void calculate_dM_s(Scalar *dM)
{
  dM[0] = 0.0;
  dM[1] = 0.0;
  dM[2] = 0.0;
}

template <typename Scalar>
void calculate_dM_p(Scalar rijinv, Scalar *drdrx, Scalar *dM)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = 0.0;
  dM[5] = -rijinv * xy;
  dM[6] = -rijinv * (yy - 1);
  dM[7] = -rijinv * yz;
  dM[8] = 0.0;
  dM[9] = -rijinv * xz;
  dM[10] = -rijinv * yz;
  dM[11] = -rijinv * (zz - 1);
}

template <typename Scalar>
void calculate_dM_d(Scalar rijinv, Scalar *drdrx, Scalar *dM)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;

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

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = -rijinv * (2 * xxx - 2 * x);
  dM[5] = -rijinv * (2 * xxy - y);
  dM[6] = -rijinv * (2 * xxz - z);
  dM[7] = -rijinv * 2 * xyy;
  dM[8] = -rijinv * 2 * xyz;
  dM[9] = -rijinv * 2 * xzz;
  dM[10] = 0.0;
  dM[11] = -rijinv * xy;
  dM[12] = -rijinv * (yy - 1);
  dM[13] = -rijinv * yz;
  dM[14] = -rijinv * 2 * xxy;
  dM[15] = -rijinv * (2 * xyy - x);
  dM[16] = -rijinv * 2 * xyz;
  dM[17] = -rijinv * (2 * yyy - 2 * y);
  dM[18] = -rijinv * (2 * yyz - z);
  dM[19] = -rijinv * 2 * yzz;
  dM[20] = 0.0;
  dM[21] = -rijinv * xz;
  dM[22] = -rijinv * yz;
  dM[23] = -rijinv * (zz - 1);
  dM[24] = -rijinv * 2 * xxz;
  dM[25] = -rijinv * 2 * xyz;
  dM[26] = -rijinv * (2 * xzz - x);
  dM[27] = -rijinv * 2 * yyz;
  dM[28] = -rijinv * (2 * yzz - y);
  dM[29] = -rijinv * (2 * zzz - 2 * z);
}

template <typename Scalar>
void calculate_dM_f(Scalar rijinv, Scalar *drdrx, Scalar *dM)
{
  Scalar x = drdrx[0];
  Scalar y = drdrx[1];
  Scalar z = drdrx[2];
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;

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

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = -rijinv * (2 * xxx - 2 * x);
  dM[5] = -rijinv * (2 * xxy - y);
  dM[6] = -rijinv * (2 * xxz - z);
  dM[7] = -rijinv * 2 * xyy;
  dM[8] = -rijinv * 2 * xyz;
  dM[9] = -rijinv * 2 * xzz;
  dM[10] = -rijinv * (3 * xxxx - 3 * xx);
  dM[11] = -rijinv * (3 * xxxy - 2 * xy);
  dM[12] = -rijinv * (3 * xxxz - 2 * xz);
  dM[13] = -rijinv * (3 * xxyy - yy);
  dM[14] = -rijinv * (3 * xxyz - yz);
  dM[15] = -rijinv * (3 * xxzz - zz);
  dM[16] = -rijinv * 3 * xyyy;
  dM[17] = -rijinv * 3 * xyyz;
  dM[18] = -rijinv * 3 * xyzz;
  dM[19] = -rijinv * 3 * xzzz;

  dM[20] = 0.0;
  dM[21] = -rijinv * xy;
  dM[22] = -rijinv * (yy - 1);
  dM[23] = -rijinv * yz;
  dM[24] = -rijinv * 2 * xxy;
  dM[25] = -rijinv * (2 * xyy - x);
  dM[26] = -rijinv * 2 * xyz;
  dM[27] = -rijinv * (2 * yyy - 2 * y);
  dM[28] = -rijinv * (2 * yyz - z);
  dM[29] = -rijinv * 2 * yzz;
  dM[30] = -rijinv * 3 * xxxy;
  dM[31] = -rijinv * (3 * xxyy - xx);
  dM[32] = -rijinv * 3 * xxyz;
  dM[33] = -rijinv * (3 * xyyy - 2 * xy);
  dM[34] = -rijinv * (3 * xyyz - xz);
  dM[35] = -rijinv * 3 * xyzz;
  dM[36] = -rijinv * (3 * yyyy - 3 * yy);
  dM[37] = -rijinv * (3 * yyyz - 2 * yz);
  dM[38] = -rijinv * (3 * yyzz - zz);
  dM[39] = -rijinv * 3 * yzzz;

  dM[40] = 0.0;
  dM[41] = -rijinv * xz;
  dM[42] = -rijinv * yz;
  dM[43] = -rijinv * (zz - 1);
  dM[44] = -rijinv * 2 * xxz;
  dM[45] = -rijinv * 2 * xyz;
  dM[46] = -rijinv * (2 * xzz - x);
  dM[47] = -rijinv * 2 * yyz;
  dM[48] = -rijinv * (2 * yzz - y);
  dM[49] = -rijinv * (2 * zzz - 2 * z);
  dM[50] = -rijinv * 3 * xxxz;
  dM[51] = -rijinv * 3 * xxyz;
  dM[52] = -rijinv * (3 * xxzz - xx);
  dM[53] = -rijinv * 3 * xyyz;
  dM[54] = -rijinv * (3 * xyzz - xy);
  dM[55] = -rijinv * (3 * xzzz - 2 * xz);
  dM[56] = -rijinv * 3 * yyyz;
  dM[57] = -rijinv * (3 * yyzz - yy);
  dM[58] = -rijinv * (3 * yzzz - 2 * yz);
  dM[59] = -rijinv * (3 * zzzz - 3 * zz);
}

template <typename Scalar>
void calculate_dM(int max_moment, Scalar rijinv, Scalar *drdrx, Scalar *dM)
{
  if (max_moment == 0)
    calculate_dM_s<Scalar>(dM);
  else if (max_moment == 1)
    calculate_dM_p<Scalar>(rijinv, drdrx, dM);
  else if (max_moment == 2)
    calculate_dM_d<Scalar>(rijinv, drdrx, dM);
  else
    calculate_dM_f<Scalar>(rijinv, drdrx, dM);
}

template <typename Scalar>
void kernel_fused_F2(int max_moment, TensorMap<Tensor<Scalar, 3>> &V,
                     TensorMap<Tensor<Scalar, 2>> &R,
                     TensorMap<Tensor<Scalar, 3>> &drdrx,
                     TensorMap<Tensor<Scalar, 3>> *F2)
{
  int limit = static_cast<int>(V.dimension(2));
  int cmax = static_cast<int>(V.dimension(1));
  int dmax = static_cast<int>(V.dimension(0));
  int loc;
  Scalar dM[3 * dmax];
  Scalar rij, rijinv;
  for (int i = 0; i < limit; i++) {
    for (int c = 0; c < cmax; c++) {
      loc = i * cmax * 3 + c * 3;
      rij = R(c, i);
      if (rij < 1e-4) break;
      rijinv = 1.0 / rij;
      calculate_dM<Scalar>(max_moment, rijinv, drdrx.data() + loc, dM);
      Scalar *V_ic = V.data() + i * cmax * dmax + c * dmax;
      for (int x = 0; x < 3; x++) {
        F2->data()[loc + x] = cppblas_dot(dmax, V_ic, 1, &dM[x * dmax], 1);
      }
    }
  }
}

template <typename Scalar>
void kernel_fused_F2(int max_moment, TensorMap<Tensor<Scalar, 4>> &V,
                     TensorMap<Tensor<Scalar, 3>> &R,
                     TensorMap<Tensor<Scalar, 4>> &drdrx,
                     TensorMap<Tensor<int, 3>> &mask,
                     Tensor<Scalar, 4> *F2)
{
  int amax = static_cast<int>(V.dimension(3));
  int bmax = static_cast<int>(V.dimension(2));
  int cmax = static_cast<int>(V.dimension(1));
  int dmax = static_cast<int>(V.dimension(0));
  int loc;
  int a, b, c;
  Scalar dM[3 * dmax];
  Scalar rij, rijinv;
#if defined(USE_OPENMP)
  #pragma omp parallel for private(a, b, c, dM, rij, rijinv, loc)
#endif
  for (a = 0; a < amax; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        if (mask(c, b, a) == 0) continue;
        loc = a * bmax * cmax + b * cmax + c;
        rij = R(c, b, a);
        rijinv = 1.0 / rij;
        calculate_dM<Scalar>(max_moment, rijinv, drdrx.data() + loc * 3, dM);
        Scalar *V_ic = V.data() + loc * dmax;
#if defined(USE_OPENMP)
#pragma omp parallel num_threads(1)
#endif
        {
          for (int x = 0; x < 3; x++) {
            (*F2)(x, c, b, a) = cppblas_dot(dmax, V_ic, 1, &dM[x * dmax], 1);
          }
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
              TensorMap<Tensor<Scalar, 4>> &M,
              TensorMap<Tensor<Scalar, 4>> *U)
{
#if defined(USE_MKL_BATCH) && (INTEL_MKL_VERSION >= 20210000)
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
              TensorMap<Tensor<Scalar, 4>> &H,
              TensorMap<Tensor<Scalar, 4>> *V)
{
#if defined(USE_MKL_BATCH) && (INTEL_MKL_VERSION >= 20210000)
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

#if defined(USE_MKL_BATCH) && (INTEL_MKL_VERSION >= 20210000)
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
  for (int i = 0; i < a * b * k * d; i++)
    (*dEdS)(i) *= P(i);
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

#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < size; i += m) {
    if (fabs(src[i]) < 1e-8) dst[i] = 0.0;
    else dst[i] = dst[i] / src[i] * 0.5;
  }
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
template void kernel_F1<double>(Tensor<double, 4> &U,
                                TensorMap<Tensor<double, 4>> &dHdr,
                                TensorMap<Tensor<double, 4>> &drdrx,
                                TensorMap<Tensor<double, 4>> *F1);
template void kernel_F1<float>(Tensor<float, 4> &U,
                               TensorMap<Tensor<float, 4>> &dHdr,
                               TensorMap<Tensor<float, 4>> &drdrx,
                               TensorMap<Tensor<float, 4>> *F1);

template void kernel_F1<double>(Tensor<double, 4> &U,
                                TensorMap<Tensor<double, 4>> &dHdr,
                                TensorMap<Tensor<double, 4>> &drdrx,
                                TensorMap<Tensor<int, 3>> &mask,
                                TensorMap<Tensor<double, 4>> *F1);
template void kernel_F1<float>(Tensor<float, 4> &U,
                               TensorMap<Tensor<float, 4>> &dHdr,
                               TensorMap<Tensor<float, 4>> &drdrx,
                               TensorMap<Tensor<int, 3>> &mask,
                               TensorMap<Tensor<float, 4>> *F1);

template void kernel_F2<double>(TensorMap<Tensor<double, 3>> &dMdrx,
                                TensorMap<Tensor<double, 3>> &V,
                                TensorMap<Tensor<double, 3>> *F2);
template void kernel_F2<float>(TensorMap<Tensor<float, 3>> &dMdrx,
                               TensorMap<Tensor<float, 3>> &V,
                               TensorMap<Tensor<float, 3>> *F2);

template void kernel_F2(TensorMap<Tensor<double, 5>> &dMdrx,
                        TensorMap<Tensor<double, 4>> &V,
                        TensorMap<Tensor<int, 3>> &mask,
                        Tensor<double, 4> *F2);
template void kernel_F2(TensorMap<Tensor<float, 5>> &dMdrx,
                        TensorMap<Tensor<float, 4>> &V,
                        TensorMap<Tensor<int, 3>> &mask,
                        Tensor<float, 4> *F2);

template void calculate_dM_s<float>(float *dM);
template void calculate_dM_p<float>(float rijinv, float *drdrx, float *dM);
template void calculate_dM_d<float>(float rijinv, float *drdrx, float *dM);
template void calculate_dM_f<float>(float rijinv, float *drdrx, float *dM);
template void calculate_dM<float>(int m, float rijinv, float *drdrx, float *dM);

template void calculate_dM_s<double>(double *dM);
template void calculate_dM_p<double>(double rijinv, double *drdrx, double *dM);
template void calculate_dM_d<double>(double rijinv, double *drdrx, double *dM);
template void calculate_dM_f<double>(double rijinv, double *drdrx, double *dM);
template void calculate_dM<double>(int m, double rijinv, double *drdrx,
                                   double *dM);

template void kernel_fused_F2<float>(int max_moment,
                                     TensorMap<Tensor<float, 3>> &V,
                                     TensorMap<Tensor<float, 2>> &R,
                                     TensorMap<Tensor<float, 3>> &drdrx,
                                     TensorMap<Tensor<float, 3>> *F2);
template void kernel_fused_F2<double>(int max_moment,
                                      TensorMap<Tensor<double, 3>> &V,
                                      TensorMap<Tensor<double, 2>> &R,
                                      TensorMap<Tensor<double, 3>> &drdrx,
                                      TensorMap<Tensor<double, 3>> *F2);
template void kernel_fused_F2<float>(int max_moment,
                                     TensorMap<Tensor<float, 4>> &V,
                                     TensorMap<Tensor<float, 3>> &R,
                                     TensorMap<Tensor<float, 4>> &drdrx,
                                     TensorMap<Tensor<int, 3>> &mask,
                                     Tensor<float, 4> *F2);
template void kernel_fused_F2<double>(int max_moment,
                                      TensorMap<Tensor<double, 4>> &V,
                                      TensorMap<Tensor<double, 3>> &R,
                                      TensorMap<Tensor<double, 4>> &drdrx,
                                      TensorMap<Tensor<int, 3>> &mask,
                                      Tensor<double, 4> *F2);

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
