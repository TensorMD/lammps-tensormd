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

#ifndef TENSORMD_KERNELS_GEOMETRIC_REDUCIBLE_MOMENTS_HPP
#define TENSORMD_KERNELS_GEOMETRIC_REDUCIBLE_MOMENTS_HPP

#include "../kernels_helpers.h"

namespace TENSORMD::Kernels::Geometric::Reducible {

// clang-format off
//                                              s   p    d    f    g     h
constexpr static double M_d_FLOPs[7]      = {0, 0,  0,   6,  16,  31,   52};
constexpr static double P_kd_FLOPs[7]     = {0, 0,  3,  15,  35,  65,  107};
constexpr static double dM_d_FLOPs[7]     = {0, 0, 33,  99, 215, 395,  653};
constexpr static double M_dM_d_FLOPs[7]   = {0, 0, 36, 108, 234, 429,  708};
constexpr static double W_M_d_FLOPs[7]    = {0, 1,  7,  25,  55, 100,  162};
constexpr static double W_dM_d_FLOPs[7]   = {0, 0, 51, 153, 329, 599,  983};
constexpr static double W_M_dM_d_FLOPs[7] = {0, 1, 61, 181, 387, 702, 1149};
// clang-format on

template <typename Scalar, int m_val>
KERNELS_DEVICE_INLINE void calculate_M_d_otf(Scalar x, Scalar y, Scalar z,
                                             Scalar* __restrict__ M_d) {
  M_d[0] = 1.0;

  if constexpr (m_val == 1) return;

  M_d[1] = x;
  M_d[2] = y;
  M_d[3] = z;

  if constexpr (m_val == 2) return;

  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;
  M_d[4] = xx;
  M_d[5] = xy;
  M_d[6] = xz;
  M_d[7] = yy;
  M_d[8] = yz;
  M_d[9] = zz;

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;
  M_d[10] = xxx;
  M_d[11] = xxy;
  M_d[12] = xxz;
  M_d[13] = xyy;
  M_d[14] = xyz;
  M_d[15] = xzz;
  M_d[16] = yyy;
  M_d[17] = yyz;
  M_d[18] = yzz;
  M_d[19] = zzz;

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;
  M_d[20] = xxxx;
  M_d[21] = xxxy;
  M_d[22] = xxxz;
  M_d[23] = xxyy;
  M_d[24] = xxyz;
  M_d[25] = xxzz;
  M_d[26] = xyyy;
  M_d[27] = xyyz;
  M_d[28] = xyzz;
  M_d[29] = xzzz;
  M_d[30] = yyyy;
  M_d[31] = yyyz;
  M_d[32] = yyzz;
  M_d[33] = yzzz;
  M_d[34] = zzzz;

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;
  M_d[35] = xxxxx;
  M_d[36] = xxxxy;
  M_d[37] = xxxxz;
  M_d[38] = xxxyy;
  M_d[39] = xxxyz;
  M_d[40] = xxxzz;
  M_d[41] = xxyyy;
  M_d[42] = xxyyz;
  M_d[43] = xxyzz;
  M_d[44] = xxzzz;
  M_d[45] = xyyyy;
  M_d[46] = xyyyz;
  M_d[47] = xyyzz;
  M_d[48] = xyzzz;
  M_d[49] = xzzzz;
  M_d[50] = yyyyy;
  M_d[51] = yyyyz;
  M_d[52] = yyyzz;
  M_d[53] = yyzzz;
  M_d[54] = yzzzz;
  M_d[55] = zzzzz;
}

template <typename Scalar, int m_val>
KERNELS_DEVICE_INLINE void calculate_P_kd_otf(Scalar x, Scalar y, Scalar z,
                                              Scalar hk,
                                              Scalar* __restrict__ P_kd) {
  P_kd[0] += hk;

  if constexpr (m_val == 1) return;

  P_kd[1] += x * hk;
  P_kd[2] += y * hk;
  P_kd[3] += z * hk;

  if constexpr (m_val == 2) return;

  const Scalar xx = x * x;
  const Scalar yy = y * y;
  const Scalar zz = z * z;
  const Scalar xy = x * y;
  const Scalar xz = x * z;
  const Scalar yz = y * z;

  P_kd[4] += hk * xx;
  P_kd[5] += hk * xy;
  P_kd[6] += hk * xz;
  P_kd[7] += hk * yy;
  P_kd[8] += hk * yz;
  P_kd[9] += hk * zz;

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;
  P_kd[10] += hk * xxx;
  P_kd[11] += hk * xxy;
  P_kd[12] += hk * xxz;
  P_kd[13] += hk * xyy;
  P_kd[14] += hk * xyz;
  P_kd[15] += hk * xzz;
  P_kd[16] += hk * yyy;
  P_kd[17] += hk * yyz;
  P_kd[18] += hk * yzz;
  P_kd[19] += hk * zzz;

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;
  P_kd[20] += hk * xxxx;
  P_kd[21] += hk * xxxy;
  P_kd[22] += hk * xxxz;
  P_kd[23] += hk * xxyy;
  P_kd[24] += hk * xxyz;
  P_kd[25] += hk * xxzz;
  P_kd[26] += hk * xyyy;
  P_kd[27] += hk * xyyz;
  P_kd[28] += hk * xyzz;
  P_kd[29] += hk * xzzz;
  P_kd[30] += hk * yyyy;
  P_kd[31] += hk * yyyz;
  P_kd[32] += hk * yyzz;
  P_kd[33] += hk * yzzz;
  P_kd[34] += hk * zzzz;

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;
  P_kd[35] += hk * xxxxx;
  P_kd[36] += hk * xxxxy;
  P_kd[37] += hk * xxxxz;
  P_kd[38] += hk * xxxyy;
  P_kd[39] += hk * xxxyz;
  P_kd[40] += hk * xxxzz;
  P_kd[41] += hk * xxyyy;
  P_kd[42] += hk * xxyyz;
  P_kd[43] += hk * xxyzz;
  P_kd[44] += hk * xxzzz;
  P_kd[45] += hk * xyyyy;
  P_kd[46] += hk * xyyyz;
  P_kd[47] += hk * xyyzz;
  P_kd[48] += hk * xyzzz;
  P_kd[49] += hk * xzzzz;
  P_kd[50] += hk * yyyyy;
  P_kd[51] += hk * yyyyz;
  P_kd[52] += hk * yyyzz;
  P_kd[53] += hk * yyzzz;
  P_kd[54] += hk * yzzzz;
  P_kd[55] += hk * zzzzz;
}

template <typename Scalar, const int m_val, const int D>
KERNELS_DEVICE_INLINE void calculate_dM_otf(Scalar x, Scalar y, Scalar z,
                                            Scalar rijinv,
                                            Scalar* __restrict__ dM) {
  constexpr int Dx = D * 0;
  constexpr int Dy = D * 1;
  constexpr int Dz = D * 2;

  dM[Dx] = 0.0;
  dM[Dy] = 0.0;
  dM[Dz] = 0.0;

  if constexpr (m_val == 1) return;

  Scalar r1 = rijinv;
  Scalar r2 = r1 * static_cast<Scalar>(2.0);
  Scalar r3 = r1 * static_cast<Scalar>(3.0);
  Scalar r4 = r1 * static_cast<Scalar>(4.0);
  Scalar r5 = r1 * static_cast<Scalar>(5.0);

  Scalar m1 = -r1;
  Scalar m2 = -r2;
  Scalar m3 = -r3;
  Scalar m4 = -r4;
  Scalar m5 = -r5;

  Scalar v_x = m1 * x;
  Scalar v_y = m1 * y;
  Scalar v_z = m1 * z;

  dM[Dx + 1] = v_x * x + r1;
  dM[Dy + 1] = v_x * y;
  dM[Dz + 1] = v_x * z;

  dM[Dx + 2] = v_y * x;
  dM[Dy + 2] = v_y * y + r1;
  dM[Dz + 2] = v_y * z;

  dM[Dx + 3] = v_z * x;
  dM[Dy + 3] = v_z * y;
  dM[Dz + 3] = v_z * z + r1;

  if constexpr (m_val == 2) return;

  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;

  Scalar v_xx = m2 * xx;
  Scalar v_xy = m2 * xy;
  Scalar v_xz = m2 * xz;
  Scalar v_yy = m2 * yy;
  Scalar v_yz = m2 * yz;
  Scalar v_zz = m2 * zz;

  dM[Dx + 4] = v_xx * x + r2 * x;
  dM[Dy + 4] = v_xx * y;
  dM[Dz + 4] = v_xx * z;

  dM[Dx + 5] = v_xy * x + r1 * y;
  dM[Dy + 5] = v_xy * y + r1 * x;
  dM[Dz + 5] = v_xy * z;

  dM[Dx + 6] = v_xz * x + r1 * z;
  dM[Dy + 6] = v_xz * y;
  dM[Dz + 6] = v_xz * z + r1 * x;

  dM[Dx + 7] = v_yy * x;
  dM[Dy + 7] = v_yy * y + r2 * y;
  dM[Dz + 7] = v_yy * z;

  dM[Dx + 8] = v_yz * x;
  dM[Dy + 8] = v_yz * y + r1 * z;
  dM[Dz + 8] = v_yz * z + r1 * y;

  dM[Dx + 9] = v_zz * x;
  dM[Dy + 9] = v_zz * y;
  dM[Dz + 9] = v_zz * z + r2 * z;

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;

  Scalar v_xxx = m3 * xxx;
  Scalar v_xxy = m3 * xxy;
  Scalar v_xxz = m3 * xxz;
  Scalar v_xyy = m3 * xyy;
  Scalar v_xyz = m3 * xyz;
  Scalar v_xzz = m3 * xzz;
  Scalar v_yyy = m3 * yyy;
  Scalar v_yyz = m3 * yyz;
  Scalar v_yzz = m3 * yzz;
  Scalar v_zzz = m3 * zzz;

  dM[Dx + 10] = v_xxx * x + r3 * xx;
  dM[Dy + 10] = v_xxx * y;
  dM[Dz + 10] = v_xxx * z;

  dM[Dx + 11] = v_xxy * x + r2 * xy;
  dM[Dy + 11] = v_xxy * y + r1 * xx;
  dM[Dz + 11] = v_xxy * z;

  dM[Dx + 12] = v_xxz * x + r2 * xz;
  dM[Dy + 12] = v_xxz * y;
  dM[Dz + 12] = v_xxz * z + r1 * xx;

  dM[Dx + 13] = v_xyy * x + r1 * yy;
  dM[Dy + 13] = v_xyy * y + r2 * xy;
  dM[Dz + 13] = v_xyy * z;

  dM[Dx + 14] = v_xyz * x + r1 * yz;
  dM[Dy + 14] = v_xyz * y + r1 * xz;
  dM[Dz + 14] = v_xyz * z + r1 * xy;

  dM[Dx + 15] = v_xzz * x + r1 * zz;
  dM[Dy + 15] = v_xzz * y;
  dM[Dz + 15] = v_xzz * z + r2 * xz;

  dM[Dx + 16] = v_yyy * x;
  dM[Dy + 16] = v_yyy * y + r3 * yy;
  dM[Dz + 16] = v_yyy * z;

  dM[Dx + 17] = v_yyz * x;
  dM[Dy + 17] = v_yyz * y + r2 * yz;
  dM[Dz + 17] = v_yyz * z + r1 * yy;

  dM[Dx + 18] = v_yzz * x;
  dM[Dy + 18] = v_yzz * y + r1 * zz;
  dM[Dz + 18] = v_yzz * z + r2 * yz;

  dM[Dx + 19] = v_zzz * x;
  dM[Dy + 19] = v_zzz * y;
  dM[Dz + 19] = v_zzz * z + r3 * zz;

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;

  Scalar v_xxxx = m4 * xxxx;
  Scalar v_xxxy = m4 * xxxy;
  Scalar v_xxxz = m4 * xxxz;
  Scalar v_xxyy = m4 * xxyy;
  Scalar v_xxyz = m4 * xxyz;
  Scalar v_xxzz = m4 * xxzz;
  Scalar v_xyyy = m4 * xyyy;
  Scalar v_xyyz = m4 * xyyz;
  Scalar v_xyzz = m4 * xyzz;
  Scalar v_xzzz = m4 * xzzz;
  Scalar v_yyyy = m4 * yyyy;
  Scalar v_yyyz = m4 * yyyz;
  Scalar v_yyzz = m4 * yyzz;
  Scalar v_yzzz = m4 * yzzz;
  Scalar v_zzzz = m4 * zzzz;

  dM[Dx + 20] = v_xxxx * x + r4 * xxx;
  dM[Dy + 20] = v_xxxx * y;
  dM[Dz + 20] = v_xxxx * z;

  dM[Dx + 21] = v_xxxy * x + r3 * xxy;
  dM[Dy + 21] = v_xxxy * y + r1 * xxx;
  dM[Dz + 21] = v_xxxy * z;

  dM[Dx + 22] = v_xxxz * x + r3 * xxz;
  dM[Dy + 22] = v_xxxz * y;
  dM[Dz + 22] = v_xxxz * z + r1 * xxx;

  dM[Dx + 23] = v_xxyy * x + r2 * xyy;
  dM[Dy + 23] = v_xxyy * y + r2 * xxy;
  dM[Dz + 23] = v_xxyy * z;

  dM[Dx + 24] = v_xxyz * x + r2 * xyz;
  dM[Dy + 24] = v_xxyz * y + r1 * xxz;
  dM[Dz + 24] = v_xxyz * z + r1 * xxy;

  dM[Dx + 25] = v_xxzz * x + r2 * xzz;
  dM[Dy + 25] = v_xxzz * y;
  dM[Dz + 25] = v_xxzz * z + r2 * xxz;

  dM[Dx + 26] = v_xyyy * x + r1 * yyy;
  dM[Dy + 26] = v_xyyy * y + r3 * xyy;
  dM[Dz + 26] = v_xyyy * z;

  dM[Dx + 27] = v_xyyz * x + r1 * yyz;
  dM[Dy + 27] = v_xyyz * y + r2 * xyz;
  dM[Dz + 27] = v_xyyz * z + r1 * xyy;

  dM[Dx + 28] = v_xyzz * x + r1 * yzz;
  dM[Dy + 28] = v_xyzz * y + r1 * xzz;
  dM[Dz + 28] = v_xyzz * z + r2 * xyz;

  dM[Dx + 29] = v_xzzz * x + r1 * zzz;
  dM[Dy + 29] = v_xzzz * y;
  dM[Dz + 29] = v_xzzz * z + r3 * xzz;

  dM[Dx + 30] = v_yyyy * x;
  dM[Dy + 30] = v_yyyy * y + r4 * yyy;
  dM[Dz + 30] = v_yyyy * z;

  dM[Dx + 31] = v_yyyz * x;
  dM[Dy + 31] = v_yyyz * y + r3 * yyz;
  dM[Dz + 31] = v_yyyz * z + r1 * yyy;

  dM[Dx + 32] = v_yyzz * x;
  dM[Dy + 32] = v_yyzz * y + r2 * yzz;
  dM[Dz + 32] = v_yyzz * z + r2 * yyz;

  dM[Dx + 33] = v_yzzz * x;
  dM[Dy + 33] = v_yzzz * y + r1 * zzz;
  dM[Dz + 33] = v_yzzz * z + r3 * yzz;

  dM[Dx + 34] = v_zzzz * x;
  dM[Dy + 34] = v_zzzz * y;
  dM[Dz + 34] = v_zzzz * z + r4 * zzz;

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;

  Scalar v_xxxxx = m5 * xxxxx;
  Scalar v_xxxxy = m5 * xxxxy;
  Scalar v_xxxxz = m5 * xxxxz;
  Scalar v_xxxyy = m5 * xxxyy;
  Scalar v_xxxyz = m5 * xxxyz;
  Scalar v_xxxzz = m5 * xxxzz;
  Scalar v_xxyyy = m5 * xxyyy;
  Scalar v_xxyyz = m5 * xxyyz;
  Scalar v_xxyzz = m5 * xxyzz;
  Scalar v_xxzzz = m5 * xxzzz;
  Scalar v_xyyyy = m5 * xyyyy;
  Scalar v_xyyyz = m5 * xyyyz;
  Scalar v_xyyzz = m5 * xyyzz;
  Scalar v_xyzzz = m5 * xyzzz;
  Scalar v_xzzzz = m5 * xzzzz;
  Scalar v_yyyyy = m5 * yyyyy;
  Scalar v_yyyyz = m5 * yyyyz;
  Scalar v_yyyzz = m5 * yyyzz;
  Scalar v_yyzzz = m5 * yyzzz;
  Scalar v_yzzzz = m5 * yzzzz;
  Scalar v_zzzzz = m5 * zzzzz;

  dM[Dx + 35] = v_xxxxx * x + r5 * xxxx;
  dM[Dy + 35] = v_xxxxx * y;
  dM[Dz + 35] = v_xxxxx * z;

  dM[Dx + 36] = v_xxxxy * x + r4 * xxxy;
  dM[Dy + 36] = v_xxxxy * y + r1 * xxxx;
  dM[Dz + 36] = v_xxxxy * z;

  dM[Dx + 37] = v_xxxxz * x + r4 * xxxz;
  dM[Dy + 37] = v_xxxxz * y;
  dM[Dz + 37] = v_xxxxz * z + r1 * xxxx;

  dM[Dx + 38] = v_xxxyy * x + r3 * xxyy;
  dM[Dy + 38] = v_xxxyy * y + r2 * xxxy;
  dM[Dz + 38] = v_xxxyy * z;

  dM[Dx + 39] = v_xxxyz * x + r3 * xxyz;
  dM[Dy + 39] = v_xxxyz * y + r1 * xxxz;
  dM[Dz + 39] = v_xxxyz * z + r1 * xxxy;

  dM[Dx + 40] = v_xxxzz * x + r3 * xxzz;
  dM[Dy + 40] = v_xxxzz * y;
  dM[Dz + 40] = v_xxxzz * z + r2 * xxxz;

  dM[Dx + 41] = v_xxyyy * x + r2 * xyyy;
  dM[Dy + 41] = v_xxyyy * y + r3 * xxyy;
  dM[Dz + 41] = v_xxyyy * z;

  dM[Dx + 42] = v_xxyyz * x + r2 * xyyz;
  dM[Dy + 42] = v_xxyyz * y + r2 * xxyz;
  dM[Dz + 42] = v_xxyyz * z + r1 * xxyy;

  dM[Dx + 43] = v_xxyzz * x + r2 * xyzz;
  dM[Dy + 43] = v_xxyzz * y + r1 * xxzz;
  dM[Dz + 43] = v_xxyzz * z + r2 * xxyz;

  dM[Dx + 44] = v_xxzzz * x + r2 * xzzz;
  dM[Dy + 44] = v_xxzzz * y;
  dM[Dz + 44] = v_xxzzz * z + r3 * xxzz;

  dM[Dx + 45] = v_xyyyy * x + r1 * yyyy;
  dM[Dy + 45] = v_xyyyy * y + r4 * xyyy;
  dM[Dz + 45] = v_xyyyy * z;

  dM[Dx + 46] = v_xyyyz * x + r1 * yyyz;
  dM[Dy + 46] = v_xyyyz * y + r3 * xyyz;
  dM[Dz + 46] = v_xyyyz * z + r1 * xyyy;

  dM[Dx + 47] = v_xyyzz * x + r1 * yyzz;
  dM[Dy + 47] = v_xyyzz * y + r2 * xyzz;
  dM[Dz + 47] = v_xyyzz * z + r2 * xyyz;

  dM[Dx + 48] = v_xyzzz * x + r1 * yzzz;
  dM[Dy + 48] = v_xyzzz * y + r1 * xzzz;
  dM[Dz + 48] = v_xyzzz * z + r3 * xyzz;

  dM[Dx + 49] = v_xzzzz * x + r1 * zzzz;
  dM[Dy + 49] = v_xzzzz * y;
  dM[Dz + 49] = v_xzzzz * z + r4 * xzzz;

  dM[Dx + 50] = v_yyyyy * x;
  dM[Dy + 50] = v_yyyyy * y + r5 * yyyy;
  dM[Dz + 50] = v_yyyyy * z;

  dM[Dx + 51] = v_yyyyz * x;
  dM[Dy + 51] = v_yyyyz * y + r4 * yyyz;
  dM[Dz + 51] = v_yyyyz * z + r1 * yyyy;

  dM[Dx + 52] = v_yyyzz * x;
  dM[Dy + 52] = v_yyyzz * y + r3 * yyzz;
  dM[Dz + 52] = v_yyyzz * z + r2 * yyyz;

  dM[Dx + 53] = v_yyzzz * x;
  dM[Dy + 53] = v_yyzzz * y + r2 * yzzz;
  dM[Dz + 53] = v_yyzzz * z + r3 * yyzz;

  dM[Dx + 54] = v_yzzzz * x;
  dM[Dy + 54] = v_yzzzz * y + r1 * zzzz;
  dM[Dz + 54] = v_yzzzz * z + r4 * yzzz;

  dM[Dx + 55] = v_zzzzz * x;
  dM[Dy + 55] = v_zzzzz * y;
  dM[Dz + 55] = v_zzzzz * z + r5 * zzzz;
}

template <typename Scalar, const int m_val, const int D>
KERNELS_DEVICE_INLINE void calculate_M_dM_otf(Scalar x, Scalar y, Scalar z,
                                              Scalar rijinv,
                                              Scalar* __restrict__ M_dM) {
  constexpr int Dr = D * 0;
  constexpr int Dx = D * 1;
  constexpr int Dy = D * 2;
  constexpr int Dz = D * 3;

  M_dM[Dr] = 1.0;
  M_dM[Dx] = 0.0;
  M_dM[Dy] = 0.0;
  M_dM[Dz] = 0.0;

  if constexpr (m_val == 1) return;

  Scalar r1 = rijinv;
  Scalar r2 = r1 * static_cast<Scalar>(2.0);
  Scalar r3 = r1 * static_cast<Scalar>(3.0);
  Scalar r4 = r1 * static_cast<Scalar>(4.0);
  Scalar r5 = r1 * static_cast<Scalar>(5.0);

  Scalar m1 = -r1;
  Scalar m2 = -r2;
  Scalar m3 = -r3;
  Scalar m4 = -r4;
  Scalar m5 = -r5;

  Scalar v_x = m1 * x;
  Scalar v_y = m1 * y;
  Scalar v_z = m1 * z;

  M_dM[Dr + 1] = x;
  M_dM[Dx + 1] = v_x * x + r1;
  M_dM[Dy + 1] = v_x * y;
  M_dM[Dz + 1] = v_x * z;

  M_dM[Dr + 2] = y;
  M_dM[Dx + 2] = v_y * x;
  M_dM[Dy + 2] = v_y * y + r1;
  M_dM[Dz + 2] = v_y * z;

  M_dM[Dr + 3] = z;
  M_dM[Dx + 3] = v_z * x;
  M_dM[Dy + 3] = v_z * y;
  M_dM[Dz + 3] = v_z * z + r1;

  if constexpr (m_val == 2) return;

  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;

  Scalar v_xx = m2 * xx;
  Scalar v_xy = m2 * xy;
  Scalar v_xz = m2 * xz;
  Scalar v_yy = m2 * yy;
  Scalar v_yz = m2 * yz;
  Scalar v_zz = m2 * zz;

  M_dM[Dr + 4] = xx;
  M_dM[Dx + 4] = v_xx * x + r2 * x;
  M_dM[Dy + 4] = v_xx * y;
  M_dM[Dz + 4] = v_xx * z;

  M_dM[Dr + 5] = xy;
  M_dM[Dx + 5] = v_xy * x + r1 * y;
  M_dM[Dy + 5] = v_xy * y + r1 * x;
  M_dM[Dz + 5] = v_xy * z;

  M_dM[Dr + 6] = xz;
  M_dM[Dx + 6] = v_xz * x + r1 * z;
  M_dM[Dy + 6] = v_xz * y;
  M_dM[Dz + 6] = v_xz * z + r1 * x;

  M_dM[Dr + 7] = yy;
  M_dM[Dx + 7] = v_yy * x;
  M_dM[Dy + 7] = v_yy * y + r2 * y;
  M_dM[Dz + 7] = v_yy * z;

  M_dM[Dr + 8] = yz;
  M_dM[Dx + 8] = v_yz * x;
  M_dM[Dy + 8] = v_yz * y + r1 * z;
  M_dM[Dz + 8] = v_yz * z + r1 * y;

  M_dM[Dr + 9] = zz;
  M_dM[Dx + 9] = v_zz * x;
  M_dM[Dy + 9] = v_zz * y;
  M_dM[Dz + 9] = v_zz * z + r2 * z;

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;

  Scalar v_xxx = m3 * xxx;
  Scalar v_xxy = m3 * xxy;
  Scalar v_xxz = m3 * xxz;
  Scalar v_xyy = m3 * xyy;
  Scalar v_xyz = m3 * xyz;
  Scalar v_xzz = m3 * xzz;
  Scalar v_yyy = m3 * yyy;
  Scalar v_yyz = m3 * yyz;
  Scalar v_yzz = m3 * yzz;
  Scalar v_zzz = m3 * zzz;

  M_dM[Dr + 10] = xxx;
  M_dM[Dx + 10] = v_xxx * x + r3 * xx;
  M_dM[Dy + 10] = v_xxx * y;
  M_dM[Dz + 10] = v_xxx * z;

  M_dM[Dr + 11] = xxy;
  M_dM[Dx + 11] = v_xxy * x + r2 * xy;
  M_dM[Dy + 11] = v_xxy * y + r1 * xx;
  M_dM[Dz + 11] = v_xxy * z;

  M_dM[Dr + 12] = xxz;
  M_dM[Dx + 12] = v_xxz * x + r2 * xz;
  M_dM[Dy + 12] = v_xxz * y;
  M_dM[Dz + 12] = v_xxz * z + r1 * xx;

  M_dM[Dr + 13] = xyy;
  M_dM[Dx + 13] = v_xyy * x + r1 * yy;
  M_dM[Dy + 13] = v_xyy * y + r2 * xy;
  M_dM[Dz + 13] = v_xyy * z;

  M_dM[Dr + 14] = xyz;
  M_dM[Dx + 14] = v_xyz * x + r1 * yz;
  M_dM[Dy + 14] = v_xyz * y + r1 * xz;
  M_dM[Dz + 14] = v_xyz * z + r1 * xy;

  M_dM[Dr + 15] = xzz;
  M_dM[Dx + 15] = v_xzz * x + r1 * zz;
  M_dM[Dy + 15] = v_xzz * y;
  M_dM[Dz + 15] = v_xzz * z + r2 * xz;

  M_dM[Dr + 16] = yyy;
  M_dM[Dx + 16] = v_yyy * x;
  M_dM[Dy + 16] = v_yyy * y + r3 * yy;
  M_dM[Dz + 16] = v_yyy * z;

  M_dM[Dr + 17] = yyz;
  M_dM[Dx + 17] = v_yyz * x;
  M_dM[Dy + 17] = v_yyz * y + r2 * yz;
  M_dM[Dz + 17] = v_yyz * z + r1 * yy;

  M_dM[Dr + 18] = yzz;
  M_dM[Dx + 18] = v_yzz * x;
  M_dM[Dy + 18] = v_yzz * y + r1 * zz;
  M_dM[Dz + 18] = v_yzz * z + r2 * yz;

  M_dM[Dr + 19] = zzz;
  M_dM[Dx + 19] = v_zzz * x;
  M_dM[Dy + 19] = v_zzz * y;
  M_dM[Dz + 19] = v_zzz * z + r3 * zz;

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;

  Scalar v_xxxx = m4 * xxxx;
  Scalar v_xxxy = m4 * xxxy;
  Scalar v_xxxz = m4 * xxxz;
  Scalar v_xxyy = m4 * xxyy;
  Scalar v_xxyz = m4 * xxyz;
  Scalar v_xxzz = m4 * xxzz;
  Scalar v_xyyy = m4 * xyyy;
  Scalar v_xyyz = m4 * xyyz;
  Scalar v_xyzz = m4 * xyzz;
  Scalar v_xzzz = m4 * xzzz;
  Scalar v_yyyy = m4 * yyyy;
  Scalar v_yyyz = m4 * yyyz;
  Scalar v_yyzz = m4 * yyzz;
  Scalar v_yzzz = m4 * yzzz;
  Scalar v_zzzz = m4 * zzzz;

  M_dM[Dr + 20] = xxxx;
  M_dM[Dx + 20] = v_xxxx * x + r4 * xxx;
  M_dM[Dy + 20] = v_xxxx * y;
  M_dM[Dz + 20] = v_xxxx * z;

  M_dM[Dr + 21] = xxxy;
  M_dM[Dx + 21] = v_xxxy * x + r3 * xxy;
  M_dM[Dy + 21] = v_xxxy * y + r1 * xxx;
  M_dM[Dz + 21] = v_xxxy * z;

  M_dM[Dr + 22] = xxxz;
  M_dM[Dx + 22] = v_xxxz * x + r3 * xxz;
  M_dM[Dy + 22] = v_xxxz * y;
  M_dM[Dz + 22] = v_xxxz * z + r1 * xxx;

  M_dM[Dr + 23] = xxyy;
  M_dM[Dx + 23] = v_xxyy * x + r2 * xyy;
  M_dM[Dy + 23] = v_xxyy * y + r2 * xxy;
  M_dM[Dz + 23] = v_xxyy * z;

  M_dM[Dr + 24] = xxyz;
  M_dM[Dx + 24] = v_xxyz * x + r2 * xyz;
  M_dM[Dy + 24] = v_xxyz * y + r1 * xxz;
  M_dM[Dz + 24] = v_xxyz * z + r1 * xxy;

  M_dM[Dr + 25] = xxzz;
  M_dM[Dx + 25] = v_xxzz * x + r2 * xzz;
  M_dM[Dy + 25] = v_xxzz * y;
  M_dM[Dz + 25] = v_xxzz * z + r2 * xxz;

  M_dM[Dr + 26] = xyyy;
  M_dM[Dx + 26] = v_xyyy * x + r1 * yyy;
  M_dM[Dy + 26] = v_xyyy * y + r3 * xyy;
  M_dM[Dz + 26] = v_xyyy * z;

  M_dM[Dr + 27] = xyyz;
  M_dM[Dx + 27] = v_xyyz * x + r1 * yyz;
  M_dM[Dy + 27] = v_xyyz * y + r2 * xyz;
  M_dM[Dz + 27] = v_xyyz * z + r1 * xyy;

  M_dM[Dr + 28] = xyzz;
  M_dM[Dx + 28] = v_xyzz * x + r1 * yzz;
  M_dM[Dy + 28] = v_xyzz * y + r1 * xzz;
  M_dM[Dz + 28] = v_xyzz * z + r2 * xyz;

  M_dM[Dr + 29] = xzzz;
  M_dM[Dx + 29] = v_xzzz * x + r1 * zzz;
  M_dM[Dy + 29] = v_xzzz * y;
  M_dM[Dz + 29] = v_xzzz * z + r3 * xzz;

  M_dM[Dr + 30] = yyyy;
  M_dM[Dx + 30] = v_yyyy * x;
  M_dM[Dy + 30] = v_yyyy * y + r4 * yyy;
  M_dM[Dz + 30] = v_yyyy * z;

  M_dM[Dr + 31] = yyyz;
  M_dM[Dx + 31] = v_yyyz * x;
  M_dM[Dy + 31] = v_yyyz * y + r3 * yyz;
  M_dM[Dz + 31] = v_yyyz * z + r1 * yyy;

  M_dM[Dr + 32] = yyzz;
  M_dM[Dx + 32] = v_yyzz * x;
  M_dM[Dy + 32] = v_yyzz * y + r2 * yzz;
  M_dM[Dz + 32] = v_yyzz * z + r2 * yyz;

  M_dM[Dr + 33] = yzzz;
  M_dM[Dx + 33] = v_yzzz * x;
  M_dM[Dy + 33] = v_yzzz * y + r1 * zzz;
  M_dM[Dz + 33] = v_yzzz * z + r3 * yzz;

  M_dM[Dr + 34] = zzzz;
  M_dM[Dx + 34] = v_zzzz * x;
  M_dM[Dy + 34] = v_zzzz * y;
  M_dM[Dz + 34] = v_zzzz * z + r4 * zzz;

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;

  Scalar v_xxxxx = m5 * xxxxx;
  Scalar v_xxxxy = m5 * xxxxy;
  Scalar v_xxxxz = m5 * xxxxz;
  Scalar v_xxxyy = m5 * xxxyy;
  Scalar v_xxxyz = m5 * xxxyz;
  Scalar v_xxxzz = m5 * xxxzz;
  Scalar v_xxyyy = m5 * xxyyy;
  Scalar v_xxyyz = m5 * xxyyz;
  Scalar v_xxyzz = m5 * xxyzz;
  Scalar v_xxzzz = m5 * xxzzz;
  Scalar v_xyyyy = m5 * xyyyy;
  Scalar v_xyyyz = m5 * xyyyz;
  Scalar v_xyyzz = m5 * xyyzz;
  Scalar v_xyzzz = m5 * xyzzz;
  Scalar v_xzzzz = m5 * xzzzz;
  Scalar v_yyyyy = m5 * yyyyy;
  Scalar v_yyyyz = m5 * yyyyz;
  Scalar v_yyyzz = m5 * yyyzz;
  Scalar v_yyzzz = m5 * yyzzz;
  Scalar v_yzzzz = m5 * yzzzz;
  Scalar v_zzzzz = m5 * zzzzz;

  M_dM[Dr + 35] = xxxxx;
  M_dM[Dx + 35] = v_xxxxx * x + r5 * xxxx;
  M_dM[Dy + 35] = v_xxxxx * y;
  M_dM[Dz + 35] = v_xxxxx * z;

  M_dM[Dr + 36] = xxxxy;
  M_dM[Dx + 36] = v_xxxxy * x + r4 * xxxy;
  M_dM[Dy + 36] = v_xxxxy * y + r1 * xxxx;
  M_dM[Dz + 36] = v_xxxxy * z;

  M_dM[Dr + 37] = xxxxz;
  M_dM[Dx + 37] = v_xxxxz * x + r4 * xxxz;
  M_dM[Dy + 37] = v_xxxxz * y;
  M_dM[Dz + 37] = v_xxxxz * z + r1 * xxxx;

  M_dM[Dr + 38] = xxxyy;
  M_dM[Dx + 38] = v_xxxyy * x + r3 * xxyy;
  M_dM[Dy + 38] = v_xxxyy * y + r2 * xxxy;
  M_dM[Dz + 38] = v_xxxyy * z;

  M_dM[Dr + 39] = xxxyz;
  M_dM[Dx + 39] = v_xxxyz * x + r3 * xxyz;
  M_dM[Dy + 39] = v_xxxyz * y + r1 * xxxz;
  M_dM[Dz + 39] = v_xxxyz * z + r1 * xxxy;

  M_dM[Dr + 40] = xxxzz;
  M_dM[Dx + 40] = v_xxxzz * x + r3 * xxzz;
  M_dM[Dy + 40] = v_xxxzz * y;
  M_dM[Dz + 40] = v_xxxzz * z + r2 * xxxz;

  M_dM[Dr + 41] = xxyyy;
  M_dM[Dx + 41] = v_xxyyy * x + r2 * xyyy;
  M_dM[Dy + 41] = v_xxyyy * y + r3 * xxyy;
  M_dM[Dz + 41] = v_xxyyy * z;

  M_dM[Dr + 42] = xxyyz;
  M_dM[Dx + 42] = v_xxyyz * x + r2 * xyyz;
  M_dM[Dy + 42] = v_xxyyz * y + r2 * xxyz;
  M_dM[Dz + 42] = v_xxyyz * z + r1 * xxyy;

  M_dM[Dr + 43] = xxyzz;
  M_dM[Dx + 43] = v_xxyzz * x + r2 * xyzz;
  M_dM[Dy + 43] = v_xxyzz * y + r1 * xxzz;
  M_dM[Dz + 43] = v_xxyzz * z + r2 * xxyz;

  M_dM[Dr + 44] = xxzzz;
  M_dM[Dx + 44] = v_xxzzz * x + r2 * xzzz;
  M_dM[Dy + 44] = v_xxzzz * y;
  M_dM[Dz + 44] = v_xxzzz * z + r3 * xxzz;

  M_dM[Dr + 45] = xyyyy;
  M_dM[Dx + 45] = v_xyyyy * x + r1 * yyyy;
  M_dM[Dy + 45] = v_xyyyy * y + r4 * xyyy;
  M_dM[Dz + 45] = v_xyyyy * z;

  M_dM[Dr + 46] = xyyyz;
  M_dM[Dx + 46] = v_xyyyz * x + r1 * yyyz;
  M_dM[Dy + 46] = v_xyyyz * y + r3 * xyyz;
  M_dM[Dz + 46] = v_xyyyz * z + r1 * xyyy;

  M_dM[Dr + 47] = xyyzz;
  M_dM[Dx + 47] = v_xyyzz * x + r1 * yyzz;
  M_dM[Dy + 47] = v_xyyzz * y + r2 * xyzz;
  M_dM[Dz + 47] = v_xyyzz * z + r2 * xyyz;

  M_dM[Dr + 48] = xyzzz;
  M_dM[Dx + 48] = v_xyzzz * x + r1 * yzzz;
  M_dM[Dy + 48] = v_xyzzz * y + r1 * xzzz;
  M_dM[Dz + 48] = v_xyzzz * z + r3 * xyzz;

  M_dM[Dr + 49] = xzzzz;
  M_dM[Dx + 49] = v_xzzzz * x + r1 * zzzz;
  M_dM[Dy + 49] = v_xzzzz * y;
  M_dM[Dz + 49] = v_xzzzz * z + r4 * xzzz;

  M_dM[Dr + 50] = yyyyy;
  M_dM[Dx + 50] = v_yyyyy * x;
  M_dM[Dy + 50] = v_yyyyy * y + r5 * yyyy;
  M_dM[Dz + 50] = v_yyyyy * z;

  M_dM[Dr + 51] = yyyyz;
  M_dM[Dx + 51] = v_yyyyz * x;
  M_dM[Dy + 51] = v_yyyyz * y + r4 * yyyz;
  M_dM[Dz + 51] = v_yyyyz * z + r1 * yyyy;

  M_dM[Dr + 52] = yyyzz;
  M_dM[Dx + 52] = v_yyyzz * x;
  M_dM[Dy + 52] = v_yyyzz * y + r3 * yyzz;
  M_dM[Dz + 52] = v_yyyzz * z + r2 * yyyz;

  M_dM[Dr + 53] = yyzzz;
  M_dM[Dx + 53] = v_yyzzz * x;
  M_dM[Dy + 53] = v_yyzzz * y + r2 * yzzz;
  M_dM[Dz + 53] = v_yyzzz * z + r3 * yyzz;

  M_dM[Dr + 54] = yzzzz;
  M_dM[Dx + 54] = v_yzzzz * x;
  M_dM[Dy + 54] = v_yzzzz * y + r1 * zzzz;
  M_dM[Dz + 54] = v_yzzzz * z + r4 * yzzz;

  M_dM[Dr + 55] = zzzzz;
  M_dM[Dx + 55] = v_zzzzz * x;
  M_dM[Dy + 55] = v_zzzzz * y;
  M_dM[Dz + 55] = v_zzzzz * z + r5 * zzzz;
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_dM_otf(Scalar x, Scalar y, Scalar z,
                                             Scalar rijinv,
                                             const Scalar* __restrict__ W_kd,
                                             Scalar& fa_x, Scalar& fa_y,
                                             Scalar& fa_z) {
  if constexpr (m_val == 1) return;

  Scalar w;
  Scalar r1 = rijinv;
  Scalar m1 = -r1;

  Scalar v_x = m1 * x;
  Scalar v_y = m1 * y;
  Scalar v_z = m1 * z;

  w = W_kd[1];
  fa_x += w * (v_x * x + r1);
  fa_y += w * (v_x * y);
  fa_z += w * (v_x * z);

  w = W_kd[2];
  fa_x += w * (v_y * x);
  fa_y += w * (v_y * y + r1);
  fa_z += w * (v_y * z);

  w = W_kd[3];
  fa_x += w * (v_z * x);
  fa_y += w * (v_z * y);
  fa_z += w * (v_z * z + r1);

  if constexpr (m_val == 2) return;

  Scalar r2 = r1 * static_cast<Scalar>(2.0);
  Scalar m2 = -r2;
  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;

  Scalar v_xx = m2 * xx;
  Scalar v_xy = m2 * xy;
  Scalar v_xz = m2 * xz;
  Scalar v_yy = m2 * yy;
  Scalar v_yz = m2 * yz;
  Scalar v_zz = m2 * zz;

  w = W_kd[4];
  fa_x += w * (v_xx * x + r2 * x);
  fa_y += w * (v_xx * y);
  fa_z += w * (v_xx * z);

  w = W_kd[5];
  fa_x += w * (v_xy * x + r1 * y);
  fa_y += w * (v_xy * y + r1 * x);
  fa_z += w * (v_xy * z);

  w = W_kd[6];
  fa_x += w * (v_xz * x + r1 * z);
  fa_y += w * (v_xz * y);
  fa_z += w * (v_xz * z + r1 * x);

  w = W_kd[7];
  fa_x += w * (v_yy * x);
  fa_y += w * (v_yy * y + r2 * y);
  fa_z += w * (v_yy * z);

  w = W_kd[8];
  fa_x += w * (v_yz * x);
  fa_y += w * (v_yz * y + r1 * z);
  fa_z += w * (v_yz * z + r1 * y);

  w = W_kd[9];
  fa_x += w * (v_zz * x);
  fa_y += w * (v_zz * y);
  fa_z += w * (v_zz * z + r2 * z);

  if constexpr (m_val == 3) return;

  Scalar r3 = r1 * static_cast<Scalar>(3.0);
  Scalar m3 = -r3;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;

  Scalar v_xxx = m3 * xxx;
  Scalar v_xxy = m3 * xxy;
  Scalar v_xxz = m3 * xxz;
  Scalar v_xyy = m3 * xyy;
  Scalar v_xyz = m3 * xyz;
  Scalar v_xzz = m3 * xzz;
  Scalar v_yyy = m3 * yyy;
  Scalar v_yyz = m3 * yyz;
  Scalar v_yzz = m3 * yzz;
  Scalar v_zzz = m3 * zzz;

  w = W_kd[10];
  fa_x += w * (v_xxx * x + r3 * xx);
  fa_y += w * (v_xxx * y);
  fa_z += w * (v_xxx * z);

  w = W_kd[11];
  fa_x += w * (v_xxy * x + r2 * xy);
  fa_y += w * (v_xxy * y + r1 * xx);
  fa_z += w * (v_xxy * z);

  w = W_kd[12];
  fa_x += w * (v_xxz * x + r2 * xz);
  fa_y += w * (v_xxz * y);
  fa_z += w * (v_xxz * z + r1 * xx);

  w = W_kd[13];
  fa_x += w * (v_xyy * x + r1 * yy);
  fa_y += w * (v_xyy * y + r2 * xy);
  fa_z += w * (v_xyy * z);

  w = W_kd[14];
  fa_x += w * (v_xyz * x + r1 * yz);
  fa_y += w * (v_xyz * y + r1 * xz);
  fa_z += w * (v_xyz * z + r1 * xy);

  w = W_kd[15];
  fa_x += w * (v_xzz * x + r1 * zz);
  fa_y += w * (v_xzz * y);
  fa_z += w * (v_xzz * z + r2 * xz);

  w = W_kd[16];
  fa_x += w * (v_yyy * x);
  fa_y += w * (v_yyy * y + r3 * yy);
  fa_z += w * (v_yyy * z);

  w = W_kd[17];
  fa_x += w * (v_yyz * x);
  fa_y += w * (v_yyz * y + r2 * yz);
  fa_z += w * (v_yyz * z + r1 * yy);

  w = W_kd[18];
  fa_x += w * (v_yzz * x);
  fa_y += w * (v_yzz * y + r1 * zz);
  fa_z += w * (v_yzz * z + r2 * yz);

  w = W_kd[19];
  fa_x += w * (v_zzz * x);
  fa_y += w * (v_zzz * y);
  fa_z += w * (v_zzz * z + r3 * zz);

  if constexpr (m_val == 4) return;

  Scalar r4 = r1 * static_cast<Scalar>(4.0);
  Scalar m4 = -r4;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;

  Scalar v_xxxx = m4 * xxxx;
  Scalar v_xxxy = m4 * xxxy;
  Scalar v_xxxz = m4 * xxxz;
  Scalar v_xxyy = m4 * xxyy;
  Scalar v_xxyz = m4 * xxyz;
  Scalar v_xxzz = m4 * xxzz;
  Scalar v_xyyy = m4 * xyyy;
  Scalar v_xyyz = m4 * xyyz;
  Scalar v_xyzz = m4 * xyzz;
  Scalar v_xzzz = m4 * xzzz;
  Scalar v_yyyy = m4 * yyyy;
  Scalar v_yyyz = m4 * yyyz;
  Scalar v_yyzz = m4 * yyzz;
  Scalar v_yzzz = m4 * yzzz;
  Scalar v_zzzz = m4 * zzzz;

  w = W_kd[20];
  fa_x += w * (v_xxxx * x + r4 * xxx);
  fa_y += w * (v_xxxx * y);
  fa_z += w * (v_xxxx * z);

  w = W_kd[21];
  fa_x += w * (v_xxxy * x + r3 * xxy);
  fa_y += w * (v_xxxy * y + r1 * xxx);
  fa_z += w * (v_xxxy * z);

  w = W_kd[22];
  fa_x += w * (v_xxxz * x + r3 * xxz);
  fa_y += w * (v_xxxz * y);
  fa_z += w * (v_xxxz * z + r1 * xxx);

  w = W_kd[23];
  fa_x += w * (v_xxyy * x + r2 * xyy);
  fa_y += w * (v_xxyy * y + r2 * xxy);
  fa_z += w * (v_xxyy * z);

  w = W_kd[24];
  fa_x += w * (v_xxyz * x + r2 * xyz);
  fa_y += w * (v_xxyz * y + r1 * xxz);
  fa_z += w * (v_xxyz * z + r1 * xxy);

  w = W_kd[25];
  fa_x += w * (v_xxzz * x + r2 * xzz);
  fa_y += w * (v_xxzz * y);
  fa_z += w * (v_xxzz * z + r2 * xxz);

  w = W_kd[26];
  fa_x += w * (v_xyyy * x + r1 * yyy);
  fa_y += w * (v_xyyy * y + r3 * xyy);
  fa_z += w * (v_xyyy * z);

  w = W_kd[27];
  fa_x += w * (v_xyyz * x + r1 * yyz);
  fa_y += w * (v_xyyz * y + r2 * xyz);
  fa_z += w * (v_xyyz * z + r1 * xyy);

  w = W_kd[28];
  fa_x += w * (v_xyzz * x + r1 * yzz);
  fa_y += w * (v_xyzz * y + r1 * xzz);
  fa_z += w * (v_xyzz * z + r2 * xyz);

  w = W_kd[29];
  fa_x += w * (v_xzzz * x + r1 * zzz);
  fa_y += w * (v_xzzz * y);
  fa_z += w * (v_xzzz * z + r3 * xzz);

  w = W_kd[30];
  fa_x += w * (v_yyyy * x);
  fa_y += w * (v_yyyy * y + r4 * yyy);
  fa_z += w * (v_yyyy * z);

  w = W_kd[31];
  fa_x += w * (v_yyyz * x);
  fa_y += w * (v_yyyz * y + r3 * yyz);
  fa_z += w * (v_yyyz * z + r1 * yyy);

  w = W_kd[32];
  fa_x += w * (v_yyzz * x);
  fa_y += w * (v_yyzz * y + r2 * yzz);
  fa_z += w * (v_yyzz * z + r2 * yyz);

  w = W_kd[33];
  fa_x += w * (v_yzzz * x);
  fa_y += w * (v_yzzz * y + r1 * zzz);
  fa_z += w * (v_yzzz * z + r3 * yzz);

  w = W_kd[34];
  fa_x += w * (v_zzzz * x);
  fa_y += w * (v_zzzz * y);
  fa_z += w * (v_zzzz * z + r4 * zzz);

  if constexpr (m_val == 5) return;

  Scalar r5 = r1 * static_cast<Scalar>(5.0);
  Scalar m5 = -r5;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;

  Scalar v_xxxxx = m5 * xxxxx;
  Scalar v_xxxxy = m5 * xxxxy;
  Scalar v_xxxxz = m5 * xxxxz;
  Scalar v_xxxyy = m5 * xxxyy;
  Scalar v_xxxyz = m5 * xxxyz;
  Scalar v_xxxzz = m5 * xxxzz;
  Scalar v_xxyyy = m5 * xxyyy;
  Scalar v_xxyyz = m5 * xxyyz;
  Scalar v_xxyzz = m5 * xxyzz;
  Scalar v_xxzzz = m5 * xxzzz;
  Scalar v_xyyyy = m5 * xyyyy;
  Scalar v_xyyyz = m5 * xyyyz;
  Scalar v_xyyzz = m5 * xyyzz;
  Scalar v_xyzzz = m5 * xyzzz;
  Scalar v_xzzzz = m5 * xzzzz;
  Scalar v_yyyyy = m5 * yyyyy;
  Scalar v_yyyyz = m5 * yyyyz;
  Scalar v_yyyzz = m5 * yyyzz;
  Scalar v_yyzzz = m5 * yyzzz;
  Scalar v_yzzzz = m5 * yzzzz;
  Scalar v_zzzzz = m5 * zzzzz;

  w = W_kd[35];
  fa_x += w * (v_xxxxx * x + r5 * xxxx);
  fa_y += w * (v_xxxxx * y);
  fa_z += w * (v_xxxxx * z);

  w = W_kd[36];
  fa_x += w * (v_xxxxy * x + r4 * xxxy);
  fa_y += w * (v_xxxxy * y + r1 * xxxx);
  fa_z += w * (v_xxxxy * z);

  w = W_kd[37];
  fa_x += w * (v_xxxxz * x + r4 * xxxz);
  fa_y += w * (v_xxxxz * y);
  fa_z += w * (v_xxxxz * z + r1 * xxxx);

  w = W_kd[38];
  fa_x += w * (v_xxxyy * x + r3 * xxyy);
  fa_y += w * (v_xxxyy * y + r2 * xxxy);
  fa_z += w * (v_xxxyy * z);

  w = W_kd[39];
  fa_x += w * (v_xxxyz * x + r3 * xxyz);
  fa_y += w * (v_xxxyz * y + r1 * xxxz);
  fa_z += w * (v_xxxyz * z + r1 * xxxy);

  w = W_kd[40];
  fa_x += w * (v_xxxzz * x + r3 * xxzz);
  fa_y += w * (v_xxxzz * y);
  fa_z += w * (v_xxxzz * z + r2 * xxxz);

  w = W_kd[41];
  fa_x += w * (v_xxyyy * x + r2 * xyyy);
  fa_y += w * (v_xxyyy * y + r3 * xxyy);
  fa_z += w * (v_xxyyy * z);

  w = W_kd[42];
  fa_x += w * (v_xxyyz * x + r2 * xyyz);
  fa_y += w * (v_xxyyz * y + r2 * xxyz);
  fa_z += w * (v_xxyyz * z + r1 * xxyy);

  w = W_kd[43];
  fa_x += w * (v_xxyzz * x + r2 * xyzz);
  fa_y += w * (v_xxyzz * y + r1 * xxzz);
  fa_z += w * (v_xxyzz * z + r2 * xxyz);

  w = W_kd[44];
  fa_x += w * (v_xxzzz * x + r2 * xzzz);
  fa_y += w * (v_xxzzz * y);
  fa_z += w * (v_xxzzz * z + r3 * xxzz);

  w = W_kd[45];
  fa_x += w * (v_xyyyy * x + r1 * yyyy);
  fa_y += w * (v_xyyyy * y + r4 * xyyy);
  fa_z += w * (v_xyyyy * z);

  w = W_kd[46];
  fa_x += w * (v_xyyyz * x + r1 * yyyz);
  fa_y += w * (v_xyyyz * y + r3 * xyyz);
  fa_z += w * (v_xyyyz * z + r1 * xyyy);

  w = W_kd[47];
  fa_x += w * (v_xyyzz * x + r1 * yyzz);
  fa_y += w * (v_xyyzz * y + r2 * xyzz);
  fa_z += w * (v_xyyzz * z + r2 * xyyz);

  w = W_kd[48];
  fa_x += w * (v_xyzzz * x + r1 * yzzz);
  fa_y += w * (v_xyzzz * y + r1 * xzzz);
  fa_z += w * (v_xyzzz * z + r3 * xyzz);

  w = W_kd[49];
  fa_x += w * (v_xzzzz * x + r1 * zzzz);
  fa_y += w * (v_xzzzz * y);
  fa_z += w * (v_xzzzz * z + r4 * xzzz);

  w = W_kd[50];
  fa_x += w * (v_yyyyy * x);
  fa_y += w * (v_yyyyy * y + r5 * yyyy);
  fa_z += w * (v_yyyyy * z);

  w = W_kd[51];
  fa_x += w * (v_yyyyz * x);
  fa_y += w * (v_yyyyz * y + r4 * yyyz);
  fa_z += w * (v_yyyyz * z + r1 * yyyy);

  w = W_kd[52];
  fa_x += w * (v_yyyzz * x);
  fa_y += w * (v_yyyzz * y + r3 * yyzz);
  fa_z += w * (v_yyyzz * z + r2 * yyyz);

  w = W_kd[53];
  fa_x += w * (v_yyzzz * x);
  fa_y += w * (v_yyzzz * y + r2 * yzzz);
  fa_z += w * (v_yyzzz * z + r3 * yyzz);

  w = W_kd[54];
  fa_x += w * (v_yzzzz * x);
  fa_y += w * (v_yzzzz * y + r1 * zzzz);
  fa_z += w * (v_yzzzz * z + r4 * yzzz);

  w = W_kd[55];
  fa_x += w * (v_zzzzz * x);
  fa_y += w * (v_zzzzz * y);
  fa_z += w * (v_zzzzz * z + r5 * zzzz);
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_M_otf(Scalar x, Scalar y, Scalar z,

                                            const Scalar* __restrict__ W_kd,
                                            Scalar& fr) {
  Scalar w;

  fr += W_kd[0];

  if constexpr (m_val == 1) return;

  w = W_kd[1];
  fr += w * x;

  w = W_kd[2];
  fr += w * y;

  w = W_kd[3];
  fr += w * z;

  if constexpr (m_val == 2) return;

  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;

  w = W_kd[4];
  fr += w * xx;

  w = W_kd[5];
  fr += w * xy;

  w = W_kd[6];
  fr += w * xz;

  w = W_kd[7];
  fr += w * yy;

  w = W_kd[8];
  fr += w * yz;

  w = W_kd[9];
  fr += w * zz;

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;

  w = W_kd[10];
  fr += w * xxx;

  w = W_kd[11];
  fr += w * xxy;

  w = W_kd[12];
  fr += w * xxz;

  w = W_kd[13];
  fr += w * xyy;

  w = W_kd[14];
  fr += w * xyz;

  w = W_kd[15];
  fr += w * xzz;

  w = W_kd[16];
  fr += w * yyy;

  w = W_kd[17];
  fr += w * yyz;

  w = W_kd[18];
  fr += w * yzz;

  w = W_kd[19];
  fr += w * zzz;

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;

  w = W_kd[20];
  fr += w * xxxx;

  w = W_kd[21];
  fr += w * xxxy;

  w = W_kd[22];
  fr += w * xxxz;

  w = W_kd[23];
  fr += w * xxyy;

  w = W_kd[24];
  fr += w * xxyz;

  w = W_kd[25];
  fr += w * xxzz;

  w = W_kd[26];
  fr += w * xyyy;

  w = W_kd[27];
  fr += w * xyyz;

  w = W_kd[28];
  fr += w * xyzz;

  w = W_kd[29];
  fr += w * xzzz;

  w = W_kd[30];
  fr += w * yyyy;

  w = W_kd[31];
  fr += w * yyyz;

  w = W_kd[32];
  fr += w * yyzz;

  w = W_kd[33];
  fr += w * yzzz;

  w = W_kd[34];
  fr += w * zzzz;

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;

  w = W_kd[35];
  fr += w * xxxxx;

  w = W_kd[36];
  fr += w * xxxxy;

  w = W_kd[37];
  fr += w * xxxxz;

  w = W_kd[38];
  fr += w * xxxyy;

  w = W_kd[39];
  fr += w * xxxyz;

  w = W_kd[40];
  fr += w * xxxzz;

  w = W_kd[41];
  fr += w * xxyyy;

  w = W_kd[42];
  fr += w * xxyyz;

  w = W_kd[43];
  fr += w * xxyzz;

  w = W_kd[44];
  fr += w * xxzzz;

  w = W_kd[45];
  fr += w * xyyyy;

  w = W_kd[46];
  fr += w * xyyyz;

  w = W_kd[47];
  fr += w * xyyzz;

  w = W_kd[48];
  fr += w * xyzzz;

  w = W_kd[49];
  fr += w * xzzzz;

  w = W_kd[50];
  fr += w * yyyyy;

  w = W_kd[51];
  fr += w * yyyyz;

  w = W_kd[52];
  fr += w * yyyzz;

  w = W_kd[53];
  fr += w * yyzzz;

  w = W_kd[54];
  fr += w * yzzzz;

  w = W_kd[55];
  fr += w * zzzzz;
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_M_dM_otf(Scalar x, Scalar y, Scalar z,
                                               Scalar rijinv,
                                               const Scalar* __restrict__ W_kd,
                                               Scalar& fr, Scalar& fa_x,
                                               Scalar& fa_y, Scalar& fa_z) {
  Scalar w;

  fr += W_kd[0];

  if constexpr (m_val == 1) return;

  Scalar r1 = rijinv;
  Scalar r2 = r1 * static_cast<Scalar>(2.0);
  Scalar r3 = r1 * static_cast<Scalar>(3.0);
  Scalar r4 = r1 * static_cast<Scalar>(4.0);
  Scalar r5 = r1 * static_cast<Scalar>(5.0);

  Scalar m1 = -r1;
  Scalar m2 = -r2;
  Scalar m3 = -r3;
  Scalar m4 = -r4;
  Scalar m5 = -r5;

  Scalar v_x = m1 * x;
  Scalar v_y = m1 * y;
  Scalar v_z = m1 * z;

  w = W_kd[1];
  fr += w * x;
  fa_x += w * (v_x * x + r1);
  fa_y += w * (v_x * y);
  fa_z += w * (v_x * z);

  w = W_kd[2];
  fr += w * y;
  fa_x += w * (v_y * x);
  fa_y += w * (v_y * y + r1);
  fa_z += w * (v_y * z);

  w = W_kd[3];
  fr += w * z;
  fa_x += w * (v_z * x);
  fa_y += w * (v_z * y);
  fa_z += w * (v_z * z + r1);

  if constexpr (m_val == 2) return;

  Scalar xx = x * x;
  Scalar xy = x * y;
  Scalar xz = x * z;
  Scalar yy = y * y;
  Scalar yz = y * z;
  Scalar zz = z * z;

  Scalar v_xx = m2 * xx;
  Scalar v_xy = m2 * xy;
  Scalar v_xz = m2 * xz;
  Scalar v_yy = m2 * yy;
  Scalar v_yz = m2 * yz;
  Scalar v_zz = m2 * zz;

  w = W_kd[4];
  fr += w * xx;
  fa_x += w * (v_xx * x + r2 * x);
  fa_y += w * (v_xx * y);
  fa_z += w * (v_xx * z);

  w = W_kd[5];
  fr += w * xy;
  fa_x += w * (v_xy * x + r1 * y);
  fa_y += w * (v_xy * y + r1 * x);
  fa_z += w * (v_xy * z);

  w = W_kd[6];
  fr += w * xz;
  fa_x += w * (v_xz * x + r1 * z);
  fa_y += w * (v_xz * y);
  fa_z += w * (v_xz * z + r1 * x);

  w = W_kd[7];
  fr += w * yy;
  fa_x += w * (v_yy * x);
  fa_y += w * (v_yy * y + r2 * y);
  fa_z += w * (v_yy * z);

  w = W_kd[8];
  fr += w * yz;
  fa_x += w * (v_yz * x);
  fa_y += w * (v_yz * y + r1 * z);
  fa_z += w * (v_yz * z + r1 * y);

  w = W_kd[9];
  fr += w * zz;
  fa_x += w * (v_zz * x);
  fa_y += w * (v_zz * y);
  fa_z += w * (v_zz * z + r2 * z);

  if constexpr (m_val == 3) return;

  Scalar xxx = xx * x;
  Scalar xxy = xx * y;
  Scalar xxz = xx * z;
  Scalar xyy = xy * y;
  Scalar xyz = xy * z;
  Scalar xzz = xz * z;
  Scalar yyy = yy * y;
  Scalar yyz = yy * z;
  Scalar yzz = yz * z;
  Scalar zzz = zz * z;

  Scalar v_xxx = m3 * xxx;
  Scalar v_xxy = m3 * xxy;
  Scalar v_xxz = m3 * xxz;
  Scalar v_xyy = m3 * xyy;
  Scalar v_xyz = m3 * xyz;
  Scalar v_xzz = m3 * xzz;
  Scalar v_yyy = m3 * yyy;
  Scalar v_yyz = m3 * yyz;
  Scalar v_yzz = m3 * yzz;
  Scalar v_zzz = m3 * zzz;

  w = W_kd[10];
  fr += w * xxx;
  fa_x += w * (v_xxx * x + r3 * xx);
  fa_y += w * (v_xxx * y);
  fa_z += w * (v_xxx * z);

  w = W_kd[11];
  fr += w * xxy;
  fa_x += w * (v_xxy * x + r2 * xy);
  fa_y += w * (v_xxy * y + r1 * xx);
  fa_z += w * (v_xxy * z);

  w = W_kd[12];
  fr += w * xxz;
  fa_x += w * (v_xxz * x + r2 * xz);
  fa_y += w * (v_xxz * y);
  fa_z += w * (v_xxz * z + r1 * xx);

  w = W_kd[13];
  fr += w * xyy;
  fa_x += w * (v_xyy * x + r1 * yy);
  fa_y += w * (v_xyy * y + r2 * xy);
  fa_z += w * (v_xyy * z);

  w = W_kd[14];
  fr += w * xyz;
  fa_x += w * (v_xyz * x + r1 * yz);
  fa_y += w * (v_xyz * y + r1 * xz);
  fa_z += w * (v_xyz * z + r1 * xy);

  w = W_kd[15];
  fr += w * xzz;
  fa_x += w * (v_xzz * x + r1 * zz);
  fa_y += w * (v_xzz * y);
  fa_z += w * (v_xzz * z + r2 * xz);

  w = W_kd[16];
  fr += w * yyy;
  fa_x += w * (v_yyy * x);
  fa_y += w * (v_yyy * y + r3 * yy);
  fa_z += w * (v_yyy * z);

  w = W_kd[17];
  fr += w * yyz;
  fa_x += w * (v_yyz * x);
  fa_y += w * (v_yyz * y + r2 * yz);
  fa_z += w * (v_yyz * z + r1 * yy);

  w = W_kd[18];
  fr += w * yzz;
  fa_x += w * (v_yzz * x);
  fa_y += w * (v_yzz * y + r1 * zz);
  fa_z += w * (v_yzz * z + r2 * yz);

  w = W_kd[19];
  fr += w * zzz;
  fa_x += w * (v_zzz * x);
  fa_y += w * (v_zzz * y);
  fa_z += w * (v_zzz * z + r3 * zz);

  if constexpr (m_val == 4) return;

  Scalar xxxx = xxx * x;
  Scalar xxxy = xxx * y;
  Scalar xxxz = xxx * z;
  Scalar xxyy = xxy * y;
  Scalar xxyz = xxy * z;
  Scalar xxzz = xxz * z;
  Scalar xyyy = xyy * y;
  Scalar xyyz = xyy * z;
  Scalar xyzz = xyz * z;
  Scalar xzzz = xzz * z;
  Scalar yyyy = yyy * y;
  Scalar yyyz = yyy * z;
  Scalar yyzz = yyz * z;
  Scalar yzzz = yzz * z;
  Scalar zzzz = zzz * z;

  Scalar v_xxxx = m4 * xxxx;
  Scalar v_xxxy = m4 * xxxy;
  Scalar v_xxxz = m4 * xxxz;
  Scalar v_xxyy = m4 * xxyy;
  Scalar v_xxyz = m4 * xxyz;
  Scalar v_xxzz = m4 * xxzz;
  Scalar v_xyyy = m4 * xyyy;
  Scalar v_xyyz = m4 * xyyz;
  Scalar v_xyzz = m4 * xyzz;
  Scalar v_xzzz = m4 * xzzz;
  Scalar v_yyyy = m4 * yyyy;
  Scalar v_yyyz = m4 * yyyz;
  Scalar v_yyzz = m4 * yyzz;
  Scalar v_yzzz = m4 * yzzz;
  Scalar v_zzzz = m4 * zzzz;

  w = W_kd[20];
  fr += w * xxxx;
  fa_x += w * (v_xxxx * x + r4 * xxx);
  fa_y += w * (v_xxxx * y);
  fa_z += w * (v_xxxx * z);

  w = W_kd[21];
  fr += w * xxxy;
  fa_x += w * (v_xxxy * x + r3 * xxy);
  fa_y += w * (v_xxxy * y + r1 * xxx);
  fa_z += w * (v_xxxy * z);

  w = W_kd[22];
  fr += w * xxxz;
  fa_x += w * (v_xxxz * x + r3 * xxz);
  fa_y += w * (v_xxxz * y);
  fa_z += w * (v_xxxz * z + r1 * xxx);

  w = W_kd[23];
  fr += w * xxyy;
  fa_x += w * (v_xxyy * x + r2 * xyy);
  fa_y += w * (v_xxyy * y + r2 * xxy);
  fa_z += w * (v_xxyy * z);

  w = W_kd[24];
  fr += w * xxyz;
  fa_x += w * (v_xxyz * x + r2 * xyz);
  fa_y += w * (v_xxyz * y + r1 * xxz);
  fa_z += w * (v_xxyz * z + r1 * xxy);

  w = W_kd[25];
  fr += w * xxzz;
  fa_x += w * (v_xxzz * x + r2 * xzz);
  fa_y += w * (v_xxzz * y);
  fa_z += w * (v_xxzz * z + r2 * xxz);

  w = W_kd[26];
  fr += w * xyyy;
  fa_x += w * (v_xyyy * x + r1 * yyy);
  fa_y += w * (v_xyyy * y + r3 * xyy);
  fa_z += w * (v_xyyy * z);

  w = W_kd[27];
  fr += w * xyyz;
  fa_x += w * (v_xyyz * x + r1 * yyz);
  fa_y += w * (v_xyyz * y + r2 * xyz);
  fa_z += w * (v_xyyz * z + r1 * xyy);

  w = W_kd[28];
  fr += w * xyzz;
  fa_x += w * (v_xyzz * x + r1 * yzz);
  fa_y += w * (v_xyzz * y + r1 * xzz);
  fa_z += w * (v_xyzz * z + r2 * xyz);

  w = W_kd[29];
  fr += w * xzzz;
  fa_x += w * (v_xzzz * x + r1 * zzz);
  fa_y += w * (v_xzzz * y);
  fa_z += w * (v_xzzz * z + r3 * xzz);

  w = W_kd[30];
  fr += w * yyyy;
  fa_x += w * (v_yyyy * x);
  fa_y += w * (v_yyyy * y + r4 * yyy);
  fa_z += w * (v_yyyy * z);

  w = W_kd[31];
  fr += w * yyyz;
  fa_x += w * (v_yyyz * x);
  fa_y += w * (v_yyyz * y + r3 * yyz);
  fa_z += w * (v_yyyz * z + r1 * yyy);

  w = W_kd[32];
  fr += w * yyzz;
  fa_x += w * (v_yyzz * x);
  fa_y += w * (v_yyzz * y + r2 * yzz);
  fa_z += w * (v_yyzz * z + r2 * yyz);

  w = W_kd[33];
  fr += w * yzzz;
  fa_x += w * (v_yzzz * x);
  fa_y += w * (v_yzzz * y + r1 * zzz);
  fa_z += w * (v_yzzz * z + r3 * yzz);

  w = W_kd[34];
  fr += w * zzzz;
  fa_x += w * (v_zzzz * x);
  fa_y += w * (v_zzzz * y);
  fa_z += w * (v_zzzz * z + r4 * zzz);

  if constexpr (m_val == 5) return;

  Scalar xxxxx = xxxx * x;
  Scalar xxxxy = xxxx * y;
  Scalar xxxxz = xxxx * z;
  Scalar xxxyy = xxxy * y;
  Scalar xxxyz = xxxy * z;
  Scalar xxxzz = xxxz * z;
  Scalar xxyyy = xxyy * y;
  Scalar xxyyz = xxyy * z;
  Scalar xxyzz = xxyz * z;
  Scalar xxzzz = xxzz * z;
  Scalar xyyyy = xyyy * y;
  Scalar xyyyz = xyyy * z;
  Scalar xyyzz = xyyz * z;
  Scalar xyzzz = xyzz * z;
  Scalar xzzzz = xzzz * z;
  Scalar yyyyy = yyyy * y;
  Scalar yyyyz = yyyy * z;
  Scalar yyyzz = yyyz * z;
  Scalar yyzzz = yyzz * z;
  Scalar yzzzz = yzzz * z;
  Scalar zzzzz = zzzz * z;

  Scalar v_xxxxx = m5 * xxxxx;
  Scalar v_xxxxy = m5 * xxxxy;
  Scalar v_xxxxz = m5 * xxxxz;
  Scalar v_xxxyy = m5 * xxxyy;
  Scalar v_xxxyz = m5 * xxxyz;
  Scalar v_xxxzz = m5 * xxxzz;
  Scalar v_xxyyy = m5 * xxyyy;
  Scalar v_xxyyz = m5 * xxyyz;
  Scalar v_xxyzz = m5 * xxyzz;
  Scalar v_xxzzz = m5 * xxzzz;
  Scalar v_xyyyy = m5 * xyyyy;
  Scalar v_xyyyz = m5 * xyyyz;
  Scalar v_xyyzz = m5 * xyyzz;
  Scalar v_xyzzz = m5 * xyzzz;
  Scalar v_xzzzz = m5 * xzzzz;
  Scalar v_yyyyy = m5 * yyyyy;
  Scalar v_yyyyz = m5 * yyyyz;
  Scalar v_yyyzz = m5 * yyyzz;
  Scalar v_yyzzz = m5 * yyzzz;
  Scalar v_yzzzz = m5 * yzzzz;
  Scalar v_zzzzz = m5 * zzzzz;

  w = W_kd[35];
  fr += w * xxxxx;
  fa_x += w * (v_xxxxx * x + r5 * xxxx);
  fa_y += w * (v_xxxxx * y);
  fa_z += w * (v_xxxxx * z);

  w = W_kd[36];
  fr += w * xxxxy;
  fa_x += w * (v_xxxxy * x + r4 * xxxy);
  fa_y += w * (v_xxxxy * y + r1 * xxxx);
  fa_z += w * (v_xxxxy * z);

  w = W_kd[37];
  fr += w * xxxxz;
  fa_x += w * (v_xxxxz * x + r4 * xxxz);
  fa_y += w * (v_xxxxz * y);
  fa_z += w * (v_xxxxz * z + r1 * xxxx);

  w = W_kd[38];
  fr += w * xxxyy;
  fa_x += w * (v_xxxyy * x + r3 * xxyy);
  fa_y += w * (v_xxxyy * y + r2 * xxxy);
  fa_z += w * (v_xxxyy * z);

  w = W_kd[39];
  fr += w * xxxyz;
  fa_x += w * (v_xxxyz * x + r3 * xxyz);
  fa_y += w * (v_xxxyz * y + r1 * xxxz);
  fa_z += w * (v_xxxyz * z + r1 * xxxy);

  w = W_kd[40];
  fr += w * xxxzz;
  fa_x += w * (v_xxxzz * x + r3 * xxzz);
  fa_y += w * (v_xxxzz * y);
  fa_z += w * (v_xxxzz * z + r2 * xxxz);

  w = W_kd[41];
  fr += w * xxyyy;
  fa_x += w * (v_xxyyy * x + r2 * xyyy);
  fa_y += w * (v_xxyyy * y + r3 * xxyy);
  fa_z += w * (v_xxyyy * z);

  w = W_kd[42];
  fr += w * xxyyz;
  fa_x += w * (v_xxyyz * x + r2 * xyyz);
  fa_y += w * (v_xxyyz * y + r2 * xxyz);
  fa_z += w * (v_xxyyz * z + r1 * xxyy);

  w = W_kd[43];
  fr += w * xxyzz;
  fa_x += w * (v_xxyzz * x + r2 * xyzz);
  fa_y += w * (v_xxyzz * y + r1 * xxzz);
  fa_z += w * (v_xxyzz * z + r2 * xxyz);

  w = W_kd[44];
  fr += w * xxzzz;
  fa_x += w * (v_xxzzz * x + r2 * xzzz);
  fa_y += w * (v_xxzzz * y);
  fa_z += w * (v_xxzzz * z + r3 * xxzz);

  w = W_kd[45];
  fr += w * xyyyy;
  fa_x += w * (v_xyyyy * x + r1 * yyyy);
  fa_y += w * (v_xyyyy * y + r4 * xyyy);
  fa_z += w * (v_xyyyy * z);

  w = W_kd[46];
  fr += w * xyyyz;
  fa_x += w * (v_xyyyz * x + r1 * yyyz);
  fa_y += w * (v_xyyyz * y + r3 * xyyz);
  fa_z += w * (v_xyyyz * z + r1 * xyyy);

  w = W_kd[47];
  fr += w * xyyzz;
  fa_x += w * (v_xyyzz * x + r1 * yyzz);
  fa_y += w * (v_xyyzz * y + r2 * xyzz);
  fa_z += w * (v_xyyzz * z + r2 * xyyz);

  w = W_kd[48];
  fr += w * xyzzz;
  fa_x += w * (v_xyzzz * x + r1 * yzzz);
  fa_y += w * (v_xyzzz * y + r1 * xzzz);
  fa_z += w * (v_xyzzz * z + r3 * xyzz);

  w = W_kd[49];
  fr += w * xzzzz;
  fa_x += w * (v_xzzzz * x + r1 * zzzz);
  fa_y += w * (v_xzzzz * y);
  fa_z += w * (v_xzzzz * z + r4 * xzzz);

  w = W_kd[50];
  fr += w * yyyyy;
  fa_x += w * (v_yyyyy * x);
  fa_y += w * (v_yyyyy * y + r5 * yyyy);
  fa_z += w * (v_yyyyy * z);

  w = W_kd[51];
  fr += w * yyyyz;
  fa_x += w * (v_yyyyz * x);
  fa_y += w * (v_yyyyz * y + r4 * yyyz);
  fa_z += w * (v_yyyyz * z + r1 * yyyy);

  w = W_kd[52];
  fr += w * yyyzz;
  fa_x += w * (v_yyyzz * x);
  fa_y += w * (v_yyyzz * y + r3 * yyzz);
  fa_z += w * (v_yyyzz * z + r2 * yyyz);

  w = W_kd[53];
  fr += w * yyzzz;
  fa_x += w * (v_yyzzz * x);
  fa_y += w * (v_yyzzz * y + r2 * yzzz);
  fa_z += w * (v_yyzzz * z + r3 * yyzz);

  w = W_kd[54];
  fr += w * yzzzz;
  fa_x += w * (v_yzzzz * x);
  fa_y += w * (v_yzzzz * y + r1 * zzzz);
  fa_z += w * (v_yzzzz * z + r4 * yzzz);

  w = W_kd[55];
  fr += w * zzzzz;
  fa_x += w * (v_zzzzz * x);
  fa_y += w * (v_zzzzz * y);
  fa_z += w * (v_zzzzz * z + r5 * zzzz);
}

}  // namespace TENSORMD::Kernels::Geometric::Reducible

#endif