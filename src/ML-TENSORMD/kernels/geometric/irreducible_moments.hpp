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

#ifndef TENSORMD_KERNELS_GEOMETRIC_IRREDUCIBLE_MOMENTS_HPP
#define TENSORMD_KERNELS_GEOMETRIC_IRREDUCIBLE_MOMENTS_HPP

#include "../kernels_helpers.h"

namespace TENSORMD::Kernels::Geometric::Irreducible {

// clang-format off
//                                              s   p    d    f    g     h
constexpr static double M_u_FLOPs[7]      = {0, 0,  0,  15,  59, 121,  225};
constexpr static double P_ku_FLOPs[7]     = {0, 0,  3,  23,  65, 136,  251};
constexpr static double dM_u_FLOPs[7]     = {0, 0, 31, 134, 297, 582, 1051};
constexpr static double M_dM_u_FLOPs[7]   = {0, 0, 30, 140, 310, 605, 1086};
constexpr static double W_M_u_FLOPs[7]    = {0, 1,  7,  32,  84, 167,  313};
constexpr static double W_dM_u_FLOPs[7]   = {0, 0, 45, 180, 385, 725, 1261};
constexpr static double W_M_dM_u_FLOPs[7] = {0, 1, 55, 205, 431, 798, 1367};
// clang-format on

template <typename Scalar>
static KERNELS_DEVICE_INLINE constexpr Scalar irc(double v) {
  return static_cast<Scalar>(v);
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void calculate_M_u_otf(Scalar x, Scalar y, Scalar z,
                                             Scalar* __restrict__ M_u) {
  M_u[0] = irc<Scalar>(1.0);

  if constexpr (m_val == 1) return;

  M_u[1] = x;
  M_u[2] = y;
  M_u[3] = z;

  if constexpr (m_val == 2) return;

  const Scalar xx = x * x;
  const Scalar yy = y * y;
  const Scalar zz = z * z;
  const Scalar xy = x * y;
  const Scalar xz = x * z;
  const Scalar yz = y * z;

  M_u[4] = irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
  M_u[5] = irc<Scalar>(1.41421356) * xy;
  M_u[6] = irc<Scalar>(1.41421356) * xz;
  M_u[7] = irc<Scalar>(0.70710678) * (yy - zz);
  M_u[8] = irc<Scalar>(1.41421356) * yz;

  if constexpr (m_val == 3) return;

  const Scalar xxx = xx * x;
  const Scalar yyy = yy * y;
  const Scalar zzz = zz * z;
  const Scalar xyy = x * yy;
  const Scalar xzz = x * zz;
  const Scalar xxy = xx * y;
  const Scalar xxz = xx * z;
  const Scalar yzz = y * zz;
  const Scalar yyz = yy * z;
  const Scalar xyz = xy * z;

  M_u[9] =
      x * (irc<Scalar>(0.26261287) * xx - irc<Scalar>(1.18175790) * (yy + zz));
  M_u[10] = y * (irc<Scalar>(1.66410059) * xx - irc<Scalar>(0.13867505) * yy -
                 irc<Scalar>(0.41602515) * zz);
  M_u[11] = z * (irc<Scalar>(1.66410059) * xx - irc<Scalar>(0.41602515) * yy -
                 irc<Scalar>(0.13867505) * zz);
  M_u[12] = x * irc<Scalar>(1.22474487) * (yy - zz);
  M_u[13] = irc<Scalar>(2.44948974) * xyz;
  M_u[14] = y * (-irc<Scalar>(0.35682062) * xx + irc<Scalar>(0.22301289) * yy -
                 irc<Scalar>(1.65029537) * zz);
  M_u[15] = z * (irc<Scalar>(0.35682062) * xx + irc<Scalar>(1.65029537) * yy -
                 irc<Scalar>(0.22301289) * zz);

  if constexpr (m_val == 4) return;

  const Scalar xxxx = xx * xx;
  const Scalar yyyy = yy * yy;
  const Scalar zzzz = zz * zz;
  const Scalar xxyy = xx * yy;
  const Scalar xxzz = xx * zz;
  const Scalar yyzz = yy * zz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  M_u[16] = irc<Scalar>(0.09421550) * xxxx -
            irc<Scalar>(1.69587899) * xx * yy_plus_zz +
            irc<Scalar>(0.03533081) * (yyyy + zzzz) +
            irc<Scalar>(0.42396975) * yyzz;
  M_u[17] = xy * (irc<Scalar>(1.10940039) * xx - irc<Scalar>(0.83205029) * yy -
                  irc<Scalar>(2.49615088) * zz);
  M_u[18] = xz * (irc<Scalar>(1.10940039) * xx - irc<Scalar>(2.49615088) * yy -
                  irc<Scalar>(0.83205029) * zz);
  M_u[19] = xx * irc<Scalar>(1.72805530) * yy_minus_zz -
            irc<Scalar>(0.04800154) * (yyyy - zzzz);
  M_u[20] = yz * (irc<Scalar>(3.43246532) * xx -
                  irc<Scalar>(0.19069252) * yy_plus_zz);
  M_u[21] = xy * (-irc<Scalar>(0.89754911) * xx + irc<Scalar>(1.15933427) * yy -
                  irc<Scalar>(2.35606641) * zz);
  M_u[22] = xz * (irc<Scalar>(0.89754911) * xx + irc<Scalar>(2.35606641) * yy -
                  irc<Scalar>(1.15933427) * zz);
  M_u[23] = irc<Scalar>(0.01600757) * xxxx -
            irc<Scalar>(0.28813629) * xx * yy_plus_zz +
            irc<Scalar>(0.07470200) * (yyyy + zzzz) -
            irc<Scalar>(2.40113574) * yyzz;
  M_u[24] = yz * irc<Scalar>(1.41421356) * yy_minus_zz;

  if constexpr (m_val == 5) return;

  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;

  M_u[25] = x * (irc<Scalar>(0.03230801) * xxxx -
                 irc<Scalar>(1.61540033) * xx * yy_plus_zz +
                 irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
                 irc<Scalar>(3.63465074) * yyzz);
  M_u[26] =
      y * (irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(1.53317860) * xxyy -
           irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.01277649) * yyyy +
           irc<Scalar>(0.25552977) * yyzz + irc<Scalar>(0.06388244) * zzzz);
  M_u[27] =
      z * (irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
           irc<Scalar>(1.53317860) * xxzz + irc<Scalar>(0.06388244) * yyyy +
           irc<Scalar>(0.25552977) * yyzz + irc<Scalar>(0.01277649) * zzzz);
  M_u[28] = x * (irc<Scalar>(2.10818511) * xx * yy_minus_zz -
                 irc<Scalar>(0.52704628) * yyyy_minus_zzzz);
  M_u[29] = xyz * (irc<Scalar>(3.65148372) * xx -
                   irc<Scalar>(1.82574186) * yy_plus_zz);
  M_u[30] =
      y * (-irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.54996401) * xxyy -
           irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.02492138) * yyyy -
           irc<Scalar>(0.05782623) * yyzz + irc<Scalar>(0.09569378) * zzzz);
  M_u[31] =
      z * (irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
           irc<Scalar>(2.54996401) * xxzz - irc<Scalar>(0.09569378) * yyyy +
           irc<Scalar>(0.05782623) * yyzz + irc<Scalar>(0.02492138) * zzzz);
  M_u[32] = x * (irc<Scalar>(0.02706701) * xxxx -
                 irc<Scalar>(1.35335068) * xx * yy_plus_zz +
                 irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
                 irc<Scalar>(4.06747614) * yyzz);
  M_u[33] = xyz * (irc<Scalar>(3.16227766) * yy_minus_zz);
  M_u[34] =
      y * (irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.18046516) * xxyy -
           irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.03149789) * yyyy -
           irc<Scalar>(2.96932339) * yyzz + irc<Scalar>(0.74831340) * zzzz);
  M_u[35] =
      z * (irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
           irc<Scalar>(0.18046516) * xxzz + irc<Scalar>(0.74831340) * yyyy -
           irc<Scalar>(2.96932339) * yyzz + irc<Scalar>(0.03149789) * zzzz);
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void calculate_P_ku_otf(Scalar x, Scalar y, Scalar z,
                                              Scalar hk,
                                              Scalar* __restrict__ P_ku) {
  P_ku[0] += irc<Scalar>(hk);

  if constexpr (m_val == 1) return;

  P_ku[1] += x * hk;
  P_ku[2] += y * hk;
  P_ku[3] += z * hk;

  if constexpr (m_val == 2) return;

  const Scalar xx = x * x;
  const Scalar yy = y * y;
  const Scalar zz = z * z;
  const Scalar xy = x * y;
  const Scalar xz = x * z;
  const Scalar yz = y * z;

  P_ku[4] +=
      hk * (irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz));
  P_ku[5] += hk * (irc<Scalar>(1.41421356) * xy);
  P_ku[6] += hk * (irc<Scalar>(1.41421356) * xz);
  P_ku[7] += hk * (irc<Scalar>(0.70710678) * (yy - zz));
  P_ku[8] += hk * (irc<Scalar>(1.41421356) * yz);

  if constexpr (m_val == 3) return;

  const Scalar xyz = xy * z;

  P_ku[9] +=
      hk * x *
      (irc<Scalar>(0.26261287) * xx - irc<Scalar>(1.18175790) * (yy + zz));
  P_ku[10] += hk * y *
              (irc<Scalar>(1.66410059) * xx - irc<Scalar>(0.13867505) * yy -
               irc<Scalar>(0.41602515) * zz);
  P_ku[11] += hk * z *
              (irc<Scalar>(1.66410059) * xx - irc<Scalar>(0.41602515) * yy -
               irc<Scalar>(0.13867505) * zz);
  P_ku[12] += hk * x * irc<Scalar>(1.22474487) * (yy - zz);
  P_ku[13] += hk * irc<Scalar>(2.44948974) * xyz;
  P_ku[14] += hk * y *
              (-irc<Scalar>(0.35682062) * xx + irc<Scalar>(0.22301289) * yy -
               irc<Scalar>(1.65029537) * zz);
  P_ku[15] += hk * z *
              (irc<Scalar>(0.35682062) * xx + irc<Scalar>(1.65029537) * yy -
               irc<Scalar>(0.22301289) * zz);

  if constexpr (m_val == 4) return;

  const Scalar xxxx = xx * xx;
  const Scalar yyyy = yy * yy;
  const Scalar zzzz = zz * zz;
  const Scalar xxyy = xx * yy;
  const Scalar xxzz = xx * zz;
  const Scalar yyzz = yy * zz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  P_ku[16] += hk * (irc<Scalar>(0.09421550) * xxxx -
                    irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                    irc<Scalar>(0.03533081) * (yyyy + zzzz) +
                    irc<Scalar>(0.42396975) * yyzz);
  P_ku[17] += hk * xy *
              (irc<Scalar>(1.10940039) * xx - irc<Scalar>(0.83205029) * yy -
               irc<Scalar>(2.49615088) * zz);
  P_ku[18] += hk * xz *
              (irc<Scalar>(1.10940039) * xx - irc<Scalar>(2.49615088) * yy -
               irc<Scalar>(0.83205029) * zz);
  P_ku[19] += hk * (xx * irc<Scalar>(1.72805530) * yy_minus_zz -
                    irc<Scalar>(0.04800154) * (yyyy - zzzz));
  P_ku[20] +=
      hk * yz *
      (irc<Scalar>(3.43246532) * xx - irc<Scalar>(0.19069252) * yy_plus_zz);
  P_ku[21] += hk * xy *
              (-irc<Scalar>(0.89754911) * xx + irc<Scalar>(1.15933427) * yy -
               irc<Scalar>(2.35606641) * zz);
  P_ku[22] += hk * xz *
              (irc<Scalar>(0.89754911) * xx + irc<Scalar>(2.35606641) * yy -
               irc<Scalar>(1.15933427) * zz);
  P_ku[23] += hk * (irc<Scalar>(0.01600757) * xxxx -
                    irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                    irc<Scalar>(0.07470200) * (yyyy + zzzz) -
                    irc<Scalar>(2.40113574) * yyzz);
  P_ku[24] += hk * yz * irc<Scalar>(1.41421356) * yy_minus_zz;

  if constexpr (m_val == 5) return;

  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;

  P_ku[25] += hk * x *
              (irc<Scalar>(0.03230801) * xxxx -
               irc<Scalar>(1.61540033) * xx * yy_plus_zz +
               irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
               irc<Scalar>(3.63465074) * yyzz);
  P_ku[26] += hk * y *
              (irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(1.53317860) * xxyy -
               irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.01277649) * yyyy +
               irc<Scalar>(0.25552977) * yyzz + irc<Scalar>(0.06388244) * zzzz);
  P_ku[27] += hk * z *
              (irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
               irc<Scalar>(1.53317860) * xxzz + irc<Scalar>(0.06388244) * yyyy +
               irc<Scalar>(0.25552977) * yyzz + irc<Scalar>(0.01277649) * zzzz);
  P_ku[28] += hk * x *
              (irc<Scalar>(2.10818511) * xx * yy_minus_zz -
               irc<Scalar>(0.52704628) * yyyy_minus_zzzz);
  P_ku[29] +=
      hk * xyz *
      (irc<Scalar>(3.65148372) * xx - irc<Scalar>(1.82574186) * yy_plus_zz);
  P_ku[30] +=
      hk * y *
      (-irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.54996401) * xxyy -
       irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.02492138) * yyyy -
       irc<Scalar>(0.05782623) * yyzz + irc<Scalar>(0.09569378) * zzzz);
  P_ku[31] += hk * z *
              (irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
               irc<Scalar>(2.54996401) * xxzz - irc<Scalar>(0.09569378) * yyyy +
               irc<Scalar>(0.05782623) * yyzz + irc<Scalar>(0.02492138) * zzzz);
  P_ku[32] += hk * x *
              (irc<Scalar>(0.02706701) * xxxx -
               irc<Scalar>(1.35335068) * xx * yy_plus_zz +
               irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
               irc<Scalar>(4.06747614) * yyzz);
  P_ku[33] += hk * xyz * (irc<Scalar>(3.16227766) * yy_minus_zz);
  P_ku[34] += hk * y *
              (irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.18046516) * xxyy -
               irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.03149789) * yyyy -
               irc<Scalar>(2.96932339) * yyzz + irc<Scalar>(0.74831340) * zzzz);
  P_ku[35] += hk * z *
              (irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
               irc<Scalar>(0.18046516) * xxzz + irc<Scalar>(0.74831340) * yyyy -
               irc<Scalar>(2.96932339) * yyzz + irc<Scalar>(0.03149789) * zzzz);
}

template <typename Scalar, const int m_val, const int U>
KERNELS_DEVICE_INLINE void calculate_dM_otf(Scalar x, Scalar y, Scalar z,
                                            Scalar rijinv,
                                            Scalar* __restrict__ dM) {
  constexpr int Ux = U * 0;
  constexpr int Uy = U * 1;
  constexpr int Uz = U * 2;

  dM[Ux] = irc<Scalar>(0.0);
  dM[Uy] = irc<Scalar>(0.0);
  dM[Uz] = irc<Scalar>(0.0);

  if constexpr (m_val == 1) return;

  const Scalar r1 = rijinv;
  const Scalar r2 = irc<Scalar>(2.0) * r1;
  const Scalar r3 = irc<Scalar>(3.0) * r1;
  const Scalar r4 = irc<Scalar>(4.0) * r1;
  const Scalar r5 = irc<Scalar>(5.0) * r1;

  const Scalar v_x = -r1 * x, v_y = -r1 * y, v_z = -r1 * z;
  dM[Ux + 1] = v_x * x + r1;
  dM[Uy + 1] = v_x * y;
  dM[Uz + 1] = v_x * z;
  dM[Ux + 2] = v_y * x;
  dM[Uy + 2] = v_y * y + r1;
  dM[Uz + 2] = v_y * z;
  dM[Ux + 3] = v_z * x;
  dM[Uy + 3] = v_z * y;
  dM[Uz + 3] = v_z * z + r1;

  if constexpr (m_val == 2) return;

  const Scalar v_x2 = -r2 * x, v_y2 = -r2 * y, v_z2 = -r2 * z;
  const Scalar v_x3 = -r3 * x, v_y3 = -r3 * y, v_z3 = -r3 * z;

  const Scalar xx = x * x, yy = y * y, zz = z * z;
  const Scalar xy = x * y, xz = x * z, yz = y * z;
  const Scalar xxx = xx * x, yyy = yy * y, zzz = zz * z;
  const Scalar xxy = xx * y, xxz = xx * z;
  const Scalar xyy = x * yy, xzz = x * zz;
  const Scalar yyz = yy * z, yzz = y * zz;
  const Scalar xyz = xy * z;

  {
    const Scalar Y =
        irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
    const Scalar dx = irc<Scalar>(1.63299316) * x;
    const Scalar dy = -irc<Scalar>(0.81649658) * y;
    const Scalar dz = -irc<Scalar>(0.81649658) * z;
    dM[Ux + 4] = v_x2 * Y + r1 * dx;
    dM[Uy + 4] = v_y2 * Y + r1 * dy;
    dM[Uz + 4] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xy;
    const Scalar dx = irc<Scalar>(1.41421356) * y;
    const Scalar dy = irc<Scalar>(1.41421356) * x;
    dM[Ux + 5] = v_x2 * Y + r1 * dx;
    dM[Uy + 5] = v_y2 * Y + r1 * dy;
    dM[Uz + 5] = v_z2 * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xz;
    const Scalar dx = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * x;
    dM[Ux + 6] = v_x2 * Y + r1 * dx;
    dM[Uy + 6] = v_y2 * Y;
    dM[Uz + 6] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.70710678) * (yy - zz);
    const Scalar dy = irc<Scalar>(1.41421356) * y;
    const Scalar dz = -irc<Scalar>(1.41421356) * z;
    dM[Ux + 7] = v_x2 * Y;
    dM[Uy + 7] = v_y2 * Y + r1 * dy;
    dM[Uz + 7] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * yz;
    const Scalar dy = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * y;
    dM[Ux + 8] = v_x2 * Y;
    dM[Uy + 8] = v_y2 * Y + r1 * dy;
    dM[Uz + 8] = v_z2 * Y + r1 * dz;
  }
  if constexpr (m_val == 3) return;

  {
    const Scalar Y = irc<Scalar>(0.26261287) * xxx -
                     irc<Scalar>(1.18175790) * xyy -
                     irc<Scalar>(1.18175790) * xzz;
    const Scalar dx = irc<Scalar>(0.78783861) * xx -
                      irc<Scalar>(1.18175790) * yy -
                      irc<Scalar>(1.18175790) * zz;
    const Scalar dy = -irc<Scalar>(2.36351580) * xy;
    const Scalar dz = -irc<Scalar>(2.36351580) * xz;
    dM[Ux + 9] = v_x3 * Y + r1 * dx;
    dM[Uy + 9] = v_y3 * Y + r1 * dy;
    dM[Uz + 9] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxy -
                     irc<Scalar>(0.13867505) * yyy -
                     irc<Scalar>(0.41602515) * yzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xy;
    const Scalar dy = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    const Scalar dz = -irc<Scalar>(0.83205030) * yz;
    dM[Ux + 10] = v_x3 * Y + r1 * dx;
    dM[Uy + 10] = v_y3 * Y + r1 * dy;
    dM[Uz + 10] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxz -
                     irc<Scalar>(0.41602515) * yyz -
                     irc<Scalar>(0.13867505) * zzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xz;
    const Scalar dy = -irc<Scalar>(0.83205030) * yz;
    const Scalar dz = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    dM[Ux + 11] = v_x3 * Y + r1 * dx;
    dM[Uy + 11] = v_y3 * Y + r1 * dy;
    dM[Uz + 11] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.22474487) * (xyy - xzz);
    const Scalar dx = irc<Scalar>(1.22474487) * (yy - zz);
    const Scalar dy = irc<Scalar>(2.44948974) * xy;
    const Scalar dz = -irc<Scalar>(2.44948974) * xz;
    dM[Ux + 12] = v_x3 * Y + r1 * dx;
    dM[Uy + 12] = v_y3 * Y + r1 * dy;
    dM[Uz + 12] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(2.44948974) * xyz;
    const Scalar dx = irc<Scalar>(2.44948974) * yz;
    const Scalar dy = irc<Scalar>(2.44948974) * xz;
    const Scalar dz = irc<Scalar>(2.44948974) * xy;
    dM[Ux + 13] = v_x3 * Y + r1 * dx;
    dM[Uy + 13] = v_y3 * Y + r1 * dy;
    dM[Uz + 13] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = -irc<Scalar>(0.35682062) * xxy +
                     irc<Scalar>(0.22301289) * yyy -
                     irc<Scalar>(1.65029537) * yzz;
    const Scalar dx = -irc<Scalar>(0.71364124) * xy;
    const Scalar dy = -irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(0.66903867) * yy -
                      irc<Scalar>(1.65029537) * zz;
    const Scalar dz = -irc<Scalar>(3.30059074) * yz;
    dM[Ux + 14] = v_x3 * Y + r1 * dx;
    dM[Uy + 14] = v_y3 * Y + r1 * dy;
    dM[Uz + 14] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.35682062) * xxz +
                     irc<Scalar>(1.65029537) * yyz -
                     irc<Scalar>(0.22301289) * zzz;
    const Scalar dx = irc<Scalar>(0.71364124) * xz;
    const Scalar dy = irc<Scalar>(3.30059074) * yz;
    const Scalar dz = irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(1.65029537) * yy -
                      irc<Scalar>(0.66903867) * zz;
    dM[Ux + 15] = v_x3 * Y + r1 * dx;
    dM[Uy + 15] = v_y3 * Y + r1 * dy;
    dM[Uz + 15] = v_z3 * Y + r1 * dz;
  }

  if constexpr (m_val == 4) return;

  const Scalar v_x4 = -r4 * x, v_y4 = -r4 * y, v_z4 = -r4 * z;
  const Scalar xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
  const Scalar xxyy = xx * yy, xxzz = xx * zz, yyzz = yy * zz;
  const Scalar xxxy = xxx * y, xxxz = xxx * z;
  const Scalar xyyy = x * yyy, xzzz = x * zzz;
  const Scalar xxyz = xx * yz, xyyz = x * yyz, xyzz = x * yzz;
  const Scalar yyyz = yyy * z, yzzz = y * zzz;
  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  {
    const Scalar Y = irc<Scalar>(0.09421550) * xxxx -
                     irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                     irc<Scalar>(0.03533081) * yyyy_plus_zzzz +
                     irc<Scalar>(0.42396975) * yyzz;
    const Scalar dx = irc<Scalar>(0.37686200) * xxx -
                      irc<Scalar>(3.39175798) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(3.39175798) * xxy +
                      irc<Scalar>(0.14132324) * yyy +
                      irc<Scalar>(0.84793950) * yzz;
    const Scalar dz = -irc<Scalar>(3.39175798) * xxz +
                      irc<Scalar>(0.14132324) * zzz +
                      irc<Scalar>(0.84793950) * yyz;
    dM[Ux + 16] = v_x4 * Y + r1 * dx;
    dM[Uy + 16] = v_y4 * Y + r1 * dy;
    dM[Uz + 16] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxy -
                     irc<Scalar>(0.83205029) * xyyy -
                     irc<Scalar>(2.49615088) * xyzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxy -
                      irc<Scalar>(0.83205029) * yyy -
                      irc<Scalar>(2.49615088) * yzz;
    const Scalar dy = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615087) * xyy -
                      irc<Scalar>(2.49615088) * xzz;
    const Scalar dz = -irc<Scalar>(4.99230176) * xyz;
    dM[Ux + 17] = v_x4 * Y + r1 * dx;
    dM[Uy + 17] = v_y4 * Y + r1 * dy;
    dM[Uz + 17] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxz -
                     irc<Scalar>(2.49615088) * xyyz -
                     irc<Scalar>(0.83205029) * xzzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxz -
                      irc<Scalar>(2.49615088) * yyz -
                      irc<Scalar>(0.83205029) * zzz;
    const Scalar dy = -irc<Scalar>(4.99230176) * xyz;
    const Scalar dz = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615088) * xyy -
                      irc<Scalar>(2.49615087) * xzz;
    dM[Ux + 18] = v_x4 * Y + r1 * dx;
    dM[Uy + 18] = v_y4 * Y + r1 * dy;
    dM[Uz + 18] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.72805530) * (xxyy - xxzz) -
                     irc<Scalar>(0.04800154) * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(3.45611060) * x * yy_minus_zz;
    const Scalar dy =
        irc<Scalar>(3.45611060) * xxy - irc<Scalar>(0.19200616) * yyy;
    const Scalar dz =
        -irc<Scalar>(3.45611060) * xxz + irc<Scalar>(0.19200616) * zzz;
    dM[Ux + 19] = v_x4 * Y + r1 * dx;
    dM[Uy + 19] = v_y4 * Y + r1 * dy;
    dM[Uz + 19] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.43246532) * xxyz -
                     irc<Scalar>(0.19069252) * (yyyz + yzzz);
    const Scalar dx = irc<Scalar>(6.86493064) * xyz;
    const Scalar dy = irc<Scalar>(3.43246532) * xxz -
                      irc<Scalar>(0.57207756) * yyz -
                      irc<Scalar>(0.19069252) * zzz;
    const Scalar dz = irc<Scalar>(3.43246532) * xxy -
                      irc<Scalar>(0.19069252) * yyy -
                      irc<Scalar>(0.57207756) * yzz;
    dM[Ux + 20] = v_x4 * Y + r1 * dx;
    dM[Uy + 20] = v_y4 * Y + r1 * dy;
    dM[Uz + 20] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = -irc<Scalar>(0.89754911) * xxxy +
                     irc<Scalar>(1.15933427) * xyyy -
                     irc<Scalar>(2.35606641) * xyzz;
    const Scalar dx = -irc<Scalar>(2.69264733) * xxy +
                      irc<Scalar>(1.15933427) * yyy -
                      irc<Scalar>(2.35606641) * yzz;
    const Scalar dy = -irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(3.47800281) * xyy -
                      irc<Scalar>(2.35606641) * xzz;
    const Scalar dz = -irc<Scalar>(4.71213282) * xyz;
    dM[Ux + 21] = v_x4 * Y + r1 * dx;
    dM[Uy + 21] = v_y4 * Y + r1 * dy;
    dM[Uz + 21] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.89754911) * xxxz +
                     irc<Scalar>(2.35606641) * xyyz -
                     irc<Scalar>(1.15933427) * xzzz;
    const Scalar dx = irc<Scalar>(2.69264733) * xxz +
                      irc<Scalar>(2.35606641) * yyz -
                      irc<Scalar>(1.15933427) * zzz;
    const Scalar dy = irc<Scalar>(4.71213282) * xyz;
    const Scalar dz = irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(2.35606641) * xyy -
                      irc<Scalar>(3.47800281) * xzz;
    dM[Ux + 22] = v_x4 * Y + r1 * dx;
    dM[Uy + 22] = v_y4 * Y + r1 * dy;
    dM[Uz + 22] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.01600757) * xxxx -
                     irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                     irc<Scalar>(0.07470200) * yyyy_plus_zzzz -
                     irc<Scalar>(2.40113574) * yyzz;
    const Scalar dx = irc<Scalar>(0.06403028) * xxx -
                      irc<Scalar>(0.57627258) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(0.57627258) * xxy +
                      irc<Scalar>(0.29880800) * yyy -
                      irc<Scalar>(4.80227148) * yzz;
    const Scalar dz = -irc<Scalar>(0.57627258) * xxz +
                      irc<Scalar>(0.29880800) * zzz -
                      irc<Scalar>(4.80227148) * yyz;
    dM[Ux + 23] = v_x4 * Y + r1 * dx;
    dM[Uy + 23] = v_y4 * Y + r1 * dy;
    dM[Uz + 23] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(4.24264068) * yyz - irc<Scalar>(1.41421356) * zzz;
    const Scalar dz =
        irc<Scalar>(1.41421356) * yyy - irc<Scalar>(4.24264068) * yzz;
    dM[Ux + 24] = v_x4 * Y;
    dM[Uy + 24] = v_y4 * Y + r1 * dy;
    dM[Uz + 24] = v_z4 * Y + r1 * dz;
  }

  if constexpr (m_val == 5) return;

  const Scalar v_x5 = -r5 * x, v_y5 = -r5 * y, v_z5 = -r5 * z;
  const Scalar xxxxy = xxxx * y;
  const Scalar xxxxz = xxxx * z;
  const Scalar xxyyy = xx * yyy;
  const Scalar xxyzz = xx * yzz;
  const Scalar xxyyz = xx * yyz;
  const Scalar xxzzz = xx * zzz;
  const Scalar yyyzz = yyy * zz;
  const Scalar yyzzz = yy * zzz;

  {
    const Scalar Y = irc<Scalar>(0.03230801) * xxxx * x -
                     irc<Scalar>(1.61540033) * xxx * yy_plus_zz +
                     irc<Scalar>(0.30288756) * x * yyyy_plus_zzzz +
                     irc<Scalar>(3.63465074) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.16154005) * xxxx -
                      irc<Scalar>(4.84620099) * xx * yy_plus_zz +
                      irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
                      irc<Scalar>(3.63465074) * yyzz;
    const Scalar dy = -irc<Scalar>(3.23080066) * xxxy +
                      irc<Scalar>(1.21155024) * xyyy +
                      irc<Scalar>(7.26930148) * xyzz;
    const Scalar dz = -irc<Scalar>(3.23080066) * xxxz +
                      irc<Scalar>(1.21155024) * xzzz +
                      irc<Scalar>(7.26930148) * xyyz;
    dM[Ux + 25] = v_x5 * Y + r1 * dx;
    dM[Uy + 25] = v_y5 * Y + r1 * dy;
    dM[Uz + 25] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxy - irc<Scalar>(1.53317860) * xxyyy -
        irc<Scalar>(4.59953581) * xxyzz + irc<Scalar>(0.01277649) * yyyy * y +
        irc<Scalar>(0.25552977) * yyyzz + irc<Scalar>(0.06388244) * yzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxy -
                      irc<Scalar>(3.06635720) * xyyy -
                      irc<Scalar>(9.19907162) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953580) * xxyy -
        irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.06388245) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388244) * zzzz;
    const Scalar dz = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.51105954) * yyyz +
                      irc<Scalar>(0.25552976) * yzzz;
    dM[Ux + 26] = v_x5 * Y + r1 * dx;
    dM[Uy + 26] = v_y5 * Y + r1 * dy;
    dM[Uz + 26] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxz - irc<Scalar>(4.59953581) * xxyyz -
        irc<Scalar>(1.53317860) * xxzzz + irc<Scalar>(0.06388244) * yyyy * z +
        irc<Scalar>(0.25552977) * yyzzz + irc<Scalar>(0.01277649) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxz -
                      irc<Scalar>(9.19907162) * xyyz -
                      irc<Scalar>(3.06635720) * xzzz;
    const Scalar dy = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.25552976) * yyyz +
                      irc<Scalar>(0.51105954) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
        irc<Scalar>(4.59953580) * xxzz + irc<Scalar>(0.06388244) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388245) * zzzz;
    dM[Ux + 27] = v_x5 * Y + r1 * dx;
    dM[Uy + 27] = v_y5 * Y + r1 * dy;
    dM[Uz + 27] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(2.10818511) * xxx * yy_minus_zz -
                     irc<Scalar>(0.52704628) * x * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(6.32455533) * xx * yy_minus_zz -
                      irc<Scalar>(0.52704628) * yyyy_minus_zzzz;
    const Scalar dy =
        irc<Scalar>(4.21637022) * xxxy - irc<Scalar>(2.10818512) * xyyy;
    const Scalar dz =
        -irc<Scalar>(4.21637022) * xxxz + irc<Scalar>(2.10818512) * xzzz;
    dM[Ux + 28] = v_x5 * Y + r1 * dx;
    dM[Uy + 28] = v_y5 * Y + r1 * dy;
    dM[Uz + 28] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.65148372) * xxxy * z -
                     irc<Scalar>(1.82574186) * (xyyy * z + x * yzzz);
    const Scalar dx = irc<Scalar>(10.95445116) * xxyz -
                      irc<Scalar>(1.82574186) * (yyyz + yzzz);
    const Scalar dy = irc<Scalar>(3.65148372) * xxxz -
                      irc<Scalar>(5.47722558) * xyyz -
                      irc<Scalar>(1.82574186) * xzzz;
    const Scalar dz = irc<Scalar>(3.65148372) * xxxy -
                      irc<Scalar>(1.82574186) * xyyy -
                      irc<Scalar>(5.47722558) * xyzz;
    dM[Ux + 29] = v_x5 * Y + r1 * dx;
    dM[Uy + 29] = v_y5 * Y + r1 * dy;
    dM[Uz + 29] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        -irc<Scalar>(0.55625378) * xxxxy + irc<Scalar>(2.54996401) * xxyyy -
        irc<Scalar>(2.92453987) * xxyzz - irc<Scalar>(0.02492138) * yyyy * y -
        irc<Scalar>(0.05782623) * yyyzz + irc<Scalar>(0.09569378) * yzzz * z;
    const Scalar dx = -irc<Scalar>(2.22501512) * xxxy +
                      irc<Scalar>(5.09992802) * xyyy -
                      irc<Scalar>(5.84907974) * xyzz;
    const Scalar dy =
        -irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(7.64989203) * xxyy -
        irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.12460690) * yyyy -
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.09569378) * zzzz;
    const Scalar dz = -irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.11565246) * yyyz +
                      irc<Scalar>(0.38277512) * yzzz;
    dM[Ux + 30] = v_x5 * Y + r1 * dx;
    dM[Uy + 30] = v_y5 * Y + r1 * dy;
    dM[Uz + 30] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.55625378) * xxxxz + irc<Scalar>(2.92453987) * xxyyz -
        irc<Scalar>(2.54996401) * xxzzz - irc<Scalar>(0.09569378) * yyyy * z +
        irc<Scalar>(0.05782623) * yyzzz + irc<Scalar>(0.02492138) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.22501512) * xxxz +
                      irc<Scalar>(5.84907974) * xyyz -
                      irc<Scalar>(5.09992802) * xzzz;
    const Scalar dy = irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.38277512) * yyyz +
                      irc<Scalar>(0.11565246) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
        irc<Scalar>(7.64989203) * xxzz - irc<Scalar>(0.09569378) * yyyy +
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.12460690) * zzzz;
    dM[Ux + 31] = v_x5 * Y + r1 * dx;
    dM[Uy + 31] = v_y5 * Y + r1 * dy;
    dM[Uz + 31] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.02706701) * xxxx * x -
                     irc<Scalar>(1.35335068) * xxx * yy_plus_zz +
                     irc<Scalar>(0.45132312) * x * yyyy_plus_zzzz -
                     irc<Scalar>(4.06747614) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.13533505) * xxxx -
                      irc<Scalar>(4.06005204) * xx * yy_plus_zz +
                      irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
                      irc<Scalar>(4.06747614) * yyzz;
    const Scalar dy = -irc<Scalar>(2.70670136) * xxxy +
                      irc<Scalar>(1.80529248) * xyyy -
                      irc<Scalar>(8.13495228) * xyzz;
    const Scalar dz = -irc<Scalar>(2.70670136) * xxxz +
                      irc<Scalar>(1.80529248) * xzzz -
                      irc<Scalar>(8.13495228) * xyyz;
    dM[Ux + 32] = v_x5 * Y + r1 * dx;
    dM[Uy + 32] = v_y5 * Y + r1 * dy;
    dM[Uz + 32] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.16227766) * (xyyy * z - x * yzzz);
    const Scalar dx = irc<Scalar>(3.16227766) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(9.48683298) * xyyz - irc<Scalar>(3.16227766) * xzzz;
    const Scalar dz =
        irc<Scalar>(3.16227766) * xyyy - irc<Scalar>(9.48683298) * xyzz;
    dM[Ux + 33] = v_x5 * Y + r1 * dx;
    dM[Uy + 33] = v_y5 * Y + r1 * dy;
    dM[Uz + 33] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxy - irc<Scalar>(0.18046516) * xxyyy -
        irc<Scalar>(0.21537191) * xxyzz + irc<Scalar>(0.03149789) * yyyy * y -
        irc<Scalar>(2.96932339) * yyyzz + irc<Scalar>(0.74831340) * yzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxy -
                      irc<Scalar>(0.36093032) * xyyy -
                      irc<Scalar>(0.43074382) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.54139548) * xxyy -
        irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.15748945) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.74831340) * zzzz;
    const Scalar dz = -irc<Scalar>(0.43074382) * xxyz -
                      irc<Scalar>(5.93864678) * yyyz +
                      irc<Scalar>(2.99325360) * yzzz;
    dM[Ux + 34] = v_x5 * Y + r1 * dx;
    dM[Uy + 34] = v_y5 * Y + r1 * dy;
    dM[Uz + 34] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxz - irc<Scalar>(0.21537191) * xxyyz -
        irc<Scalar>(0.18046516) * xxzzz + irc<Scalar>(0.74831340) * yyyy * z -
        irc<Scalar>(2.96932339) * yyzzz + irc<Scalar>(0.03149789) * zzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxz -
                      irc<Scalar>(0.43074382) * xyyz -
                      irc<Scalar>(0.36093032) * xzzz;
    const Scalar dy = -irc<Scalar>(0.43074382) * xxyz +
                      irc<Scalar>(2.99325360) * yyyz -
                      irc<Scalar>(5.93864678) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
        irc<Scalar>(0.54139548) * xxzz + irc<Scalar>(0.74831340) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.15748945) * zzzz;
    dM[Ux + 35] = v_x5 * Y + r1 * dx;
    dM[Uy + 35] = v_y5 * Y + r1 * dy;
    dM[Uz + 35] = v_z5 * Y + r1 * dz;
  }
}

template <typename Scalar, const int m_val, const int U>
KERNELS_DEVICE_INLINE void calculate_M_dM_otf(Scalar x, Scalar y, Scalar z,
                                              Scalar rijinv,
                                              Scalar* __restrict__ M_dM) {
  constexpr int Ur = U * 0;
  constexpr int Ux = U * 1;
  constexpr int Uy = U * 2;
  constexpr int Uz = U * 3;

  M_dM[Ur] = irc<Scalar>(1.0);
  M_dM[Ux] = irc<Scalar>(0.0);
  M_dM[Uy] = irc<Scalar>(0.0);
  M_dM[Uz] = irc<Scalar>(0.0);

  if constexpr (m_val == 1) return;

  const Scalar r1 = rijinv;

  const Scalar v_x = -r1 * x, v_y = -r1 * y, v_z = -r1 * z;
  M_dM[Ur + 1] = x;
  M_dM[Ux + 1] = v_x * x + r1;
  M_dM[Uy + 1] = v_x * y;
  M_dM[Uz + 1] = v_x * z;
  M_dM[Ur + 2] = y;
  M_dM[Ux + 2] = v_y * x;
  M_dM[Uy + 2] = v_y * y + r1;
  M_dM[Uz + 2] = v_y * z;
  M_dM[Ur + 3] = z;
  M_dM[Ux + 3] = v_z * x;
  M_dM[Uy + 3] = v_z * y;
  M_dM[Uz + 3] = v_z * z + r1;

  if constexpr (m_val == 2) return;

  const Scalar r2 = irc<Scalar>(2.0) * r1;
  const Scalar r3 = irc<Scalar>(3.0) * r1;

  const Scalar v_x2 = -r2 * x, v_y2 = -r2 * y, v_z2 = -r2 * z;
  const Scalar v_x3 = -r3 * x, v_y3 = -r3 * y, v_z3 = -r3 * z;

  const Scalar xx = x * x, yy = y * y, zz = z * z;
  const Scalar xy = x * y, xz = x * z, yz = y * z;
  const Scalar xxx = xx * x, yyy = yy * y, zzz = zz * z;
  const Scalar xxy = xx * y, xxz = xx * z;
  const Scalar xyy = x * yy, xzz = x * zz;
  const Scalar yyz = yy * z, yzz = y * zz;
  const Scalar xyz = xy * z;

  {
    const Scalar Y =
        irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
    const Scalar dx = irc<Scalar>(1.63299316) * x;
    const Scalar dy = -irc<Scalar>(0.81649658) * y;
    const Scalar dz = -irc<Scalar>(0.81649658) * z;
    M_dM[Ur + 4] = Y;
    M_dM[Ux + 4] = v_x2 * Y + r1 * dx;
    M_dM[Uy + 4] = v_y2 * Y + r1 * dy;
    M_dM[Uz + 4] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xy;
    const Scalar dx = irc<Scalar>(1.41421356) * y;
    const Scalar dy = irc<Scalar>(1.41421356) * x;
    M_dM[Ur + 5] = Y;
    M_dM[Ux + 5] = v_x2 * Y + r1 * dx;
    M_dM[Uy + 5] = v_y2 * Y + r1 * dy;
    M_dM[Uz + 5] = v_z2 * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xz;
    const Scalar dx = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * x;
    M_dM[Ur + 6] = Y;
    M_dM[Ux + 6] = v_x2 * Y + r1 * dx;
    M_dM[Uy + 6] = v_y2 * Y;
    M_dM[Uz + 6] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.70710678) * (yy - zz);
    const Scalar dy = irc<Scalar>(1.41421356) * y;
    const Scalar dz = -irc<Scalar>(1.41421356) * z;
    M_dM[Ur + 7] = Y;
    M_dM[Ux + 7] = v_x2 * Y;
    M_dM[Uy + 7] = v_y2 * Y + r1 * dy;
    M_dM[Uz + 7] = v_z2 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * yz;
    const Scalar dy = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * y;
    M_dM[Ur + 8] = Y;
    M_dM[Ux + 8] = v_x2 * Y;
    M_dM[Uy + 8] = v_y2 * Y + r1 * dy;
    M_dM[Uz + 8] = v_z2 * Y + r1 * dz;
  }
  if constexpr (m_val == 3) return;

  {
    const Scalar Y = irc<Scalar>(0.26261287) * xxx -
                     irc<Scalar>(1.18175790) * xyy -
                     irc<Scalar>(1.18175790) * xzz;
    const Scalar dx = irc<Scalar>(0.78783861) * xx -
                      irc<Scalar>(1.18175790) * yy -
                      irc<Scalar>(1.18175790) * zz;
    const Scalar dy = -irc<Scalar>(2.36351580) * xy;
    const Scalar dz = -irc<Scalar>(2.36351580) * xz;
    M_dM[Ur + 9] = Y;
    M_dM[Ux + 9] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 9] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 9] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxy -
                     irc<Scalar>(0.13867505) * yyy -
                     irc<Scalar>(0.41602515) * yzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xy;
    const Scalar dy = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    const Scalar dz = -irc<Scalar>(0.83205030) * yz;
    M_dM[Ur + 10] = Y;
    M_dM[Ux + 10] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 10] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 10] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxz -
                     irc<Scalar>(0.41602515) * yyz -
                     irc<Scalar>(0.13867505) * zzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xz;
    const Scalar dy = -irc<Scalar>(0.83205030) * yz;
    const Scalar dz = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    M_dM[Ur + 11] = Y;
    M_dM[Ux + 11] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 11] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 11] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.22474487) * (xyy - xzz);
    const Scalar dx = irc<Scalar>(1.22474487) * (yy - zz);
    const Scalar dy = irc<Scalar>(2.44948974) * xy;
    const Scalar dz = -irc<Scalar>(2.44948974) * xz;
    M_dM[Ur + 12] = Y;
    M_dM[Ux + 12] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 12] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 12] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(2.44948974) * xyz;
    const Scalar dx = irc<Scalar>(2.44948974) * yz;
    const Scalar dy = irc<Scalar>(2.44948974) * xz;
    const Scalar dz = irc<Scalar>(2.44948974) * xy;
    M_dM[Ur + 13] = Y;
    M_dM[Ux + 13] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 13] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 13] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = -irc<Scalar>(0.35682062) * xxy +
                     irc<Scalar>(0.22301289) * yyy -
                     irc<Scalar>(1.65029537) * yzz;
    const Scalar dx = -irc<Scalar>(0.71364124) * xy;
    const Scalar dy = -irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(0.66903867) * yy -
                      irc<Scalar>(1.65029537) * zz;
    const Scalar dz = -irc<Scalar>(3.30059074) * yz;
    M_dM[Ur + 14] = Y;
    M_dM[Ux + 14] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 14] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 14] = v_z3 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.35682062) * xxz +
                     irc<Scalar>(1.65029537) * yyz -
                     irc<Scalar>(0.22301289) * zzz;
    const Scalar dx = irc<Scalar>(0.71364124) * xz;
    const Scalar dy = irc<Scalar>(3.30059074) * yz;
    const Scalar dz = irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(1.65029537) * yy -
                      irc<Scalar>(0.66903867) * zz;
    M_dM[Ur + 15] = Y;
    M_dM[Ux + 15] = v_x3 * Y + r1 * dx;
    M_dM[Uy + 15] = v_y3 * Y + r1 * dy;
    M_dM[Uz + 15] = v_z3 * Y + r1 * dz;
  }

  if constexpr (m_val == 4) return;

  const Scalar r4 = irc<Scalar>(4.0) * r1;

  const Scalar v_x4 = -r4 * x, v_y4 = -r4 * y, v_z4 = -r4 * z;
  const Scalar xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
  const Scalar xxyy = xx * yy, xxzz = xx * zz, yyzz = yy * zz;
  const Scalar xxxy = xxx * y, xxxz = xxx * z;
  const Scalar xyyy = x * yyy, xzzz = x * zzz;
  const Scalar xxyz = xx * yz, xyyz = x * yyz, xyzz = x * yzz;
  const Scalar yyyz = yyy * z, yzzz = y * zzz;
  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  {
    const Scalar Y = irc<Scalar>(0.09421550) * xxxx -
                     irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                     irc<Scalar>(0.03533081) * yyyy_plus_zzzz +
                     irc<Scalar>(0.42396975) * yyzz;
    const Scalar dx = irc<Scalar>(0.37686200) * xxx -
                      irc<Scalar>(3.39175798) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(3.39175798) * xxy +
                      irc<Scalar>(0.14132324) * yyy +
                      irc<Scalar>(0.84793950) * yzz;
    const Scalar dz = -irc<Scalar>(3.39175798) * xxz +
                      irc<Scalar>(0.14132324) * zzz +
                      irc<Scalar>(0.84793950) * yyz;
    M_dM[Ur + 16] = Y;
    M_dM[Ux + 16] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 16] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 16] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxy -
                     irc<Scalar>(0.83205029) * xyyy -
                     irc<Scalar>(2.49615088) * xyzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxy -
                      irc<Scalar>(0.83205029) * yyy -
                      irc<Scalar>(2.49615088) * yzz;
    const Scalar dy = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615087) * xyy -
                      irc<Scalar>(2.49615088) * xzz;
    const Scalar dz = -irc<Scalar>(4.99230176) * xyz;
    M_dM[Ur + 17] = Y;
    M_dM[Ux + 17] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 17] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 17] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxz -
                     irc<Scalar>(2.49615088) * xyyz -
                     irc<Scalar>(0.83205029) * xzzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxz -
                      irc<Scalar>(2.49615088) * yyz -
                      irc<Scalar>(0.83205029) * zzz;
    const Scalar dy = -irc<Scalar>(4.99230176) * xyz;
    const Scalar dz = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615088) * xyy -
                      irc<Scalar>(2.49615087) * xzz;
    M_dM[Ur + 18] = Y;
    M_dM[Ux + 18] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 18] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 18] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.72805530) * (xxyy - xxzz) -
                     irc<Scalar>(0.04800154) * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(3.45611060) * x * yy_minus_zz;
    const Scalar dy =
        irc<Scalar>(3.45611060) * xxy - irc<Scalar>(0.19200616) * yyy;
    const Scalar dz =
        -irc<Scalar>(3.45611060) * xxz + irc<Scalar>(0.19200616) * zzz;
    M_dM[Ur + 19] = Y;
    M_dM[Ux + 19] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 19] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 19] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.43246532) * xxyz -
                     irc<Scalar>(0.19069252) * (yyyz + yzzz);
    const Scalar dx = irc<Scalar>(6.86493064) * xyz;
    const Scalar dy = irc<Scalar>(3.43246532) * xxz -
                      irc<Scalar>(0.57207756) * yyz -
                      irc<Scalar>(0.19069252) * zzz;
    const Scalar dz = irc<Scalar>(3.43246532) * xxy -
                      irc<Scalar>(0.19069252) * yyy -
                      irc<Scalar>(0.57207756) * yzz;
    M_dM[Ur + 20] = Y;
    M_dM[Ux + 20] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 20] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 20] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = -irc<Scalar>(0.89754911) * xxxy +
                     irc<Scalar>(1.15933427) * xyyy -
                     irc<Scalar>(2.35606641) * xyzz;
    const Scalar dx = -irc<Scalar>(2.69264733) * xxy +
                      irc<Scalar>(1.15933427) * yyy -
                      irc<Scalar>(2.35606641) * yzz;
    const Scalar dy = -irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(3.47800281) * xyy -
                      irc<Scalar>(2.35606641) * xzz;
    const Scalar dz = -irc<Scalar>(4.71213282) * xyz;
    M_dM[Ur + 21] = Y;
    M_dM[Ux + 21] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 21] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 21] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.89754911) * xxxz +
                     irc<Scalar>(2.35606641) * xyyz -
                     irc<Scalar>(1.15933427) * xzzz;
    const Scalar dx = irc<Scalar>(2.69264733) * xxz +
                      irc<Scalar>(2.35606641) * yyz -
                      irc<Scalar>(1.15933427) * zzz;
    const Scalar dy = irc<Scalar>(4.71213282) * xyz;
    const Scalar dz = irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(2.35606641) * xyy -
                      irc<Scalar>(3.47800281) * xzz;
    M_dM[Ur + 22] = Y;
    M_dM[Ux + 22] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 22] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 22] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.01600757) * xxxx -
                     irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                     irc<Scalar>(0.07470200) * yyyy_plus_zzzz -
                     irc<Scalar>(2.40113574) * yyzz;
    const Scalar dx = irc<Scalar>(0.06403028) * xxx -
                      irc<Scalar>(0.57627258) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(0.57627258) * xxy +
                      irc<Scalar>(0.29880800) * yyy -
                      irc<Scalar>(4.80227148) * yzz;
    const Scalar dz = -irc<Scalar>(0.57627258) * xxz +
                      irc<Scalar>(0.29880800) * zzz -
                      irc<Scalar>(4.80227148) * yyz;
    M_dM[Ur + 23] = Y;
    M_dM[Ux + 23] = v_x4 * Y + r1 * dx;
    M_dM[Uy + 23] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 23] = v_z4 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(4.24264068) * yyz - irc<Scalar>(1.41421356) * zzz;
    const Scalar dz =
        irc<Scalar>(1.41421356) * yyy - irc<Scalar>(4.24264068) * yzz;
    M_dM[Ur + 24] = Y;
    M_dM[Ux + 24] = v_x4 * Y;
    M_dM[Uy + 24] = v_y4 * Y + r1 * dy;
    M_dM[Uz + 24] = v_z4 * Y + r1 * dz;
  }

  if constexpr (m_val == 5) return;

  const Scalar r5 = irc<Scalar>(5.0) * r1;

  const Scalar v_x5 = -r5 * x, v_y5 = -r5 * y, v_z5 = -r5 * z;
  const Scalar xxxxy = xxxx * y;
  const Scalar xxxxz = xxxx * z;
  const Scalar xxyyy = xx * yyy;
  const Scalar xxyzz = xx * yzz;
  const Scalar xxyyz = xx * yyz;
  const Scalar xxzzz = xx * zzz;
  const Scalar yyyzz = yyy * zz;
  const Scalar yyzzz = yy * zzz;

  {
    const Scalar Y = irc<Scalar>(0.03230801) * xxxx * x -
                     irc<Scalar>(1.61540033) * xxx * yy_plus_zz +
                     irc<Scalar>(0.30288756) * x * yyyy_plus_zzzz +
                     irc<Scalar>(3.63465074) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.16154005) * xxxx -
                      irc<Scalar>(4.84620099) * xx * yy_plus_zz +
                      irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
                      irc<Scalar>(3.63465074) * yyzz;
    const Scalar dy = -irc<Scalar>(3.23080066) * xxxy +
                      irc<Scalar>(1.21155024) * xyyy +
                      irc<Scalar>(7.26930148) * xyzz;
    const Scalar dz = -irc<Scalar>(3.23080066) * xxxz +
                      irc<Scalar>(1.21155024) * xzzz +
                      irc<Scalar>(7.26930148) * xyyz;
    M_dM[Ur + 25] = Y;
    M_dM[Ux + 25] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 25] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 25] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxy - irc<Scalar>(1.53317860) * xxyyy -
        irc<Scalar>(4.59953581) * xxyzz + irc<Scalar>(0.01277649) * yyyy * y +
        irc<Scalar>(0.25552977) * yyyzz + irc<Scalar>(0.06388244) * yzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxy -
                      irc<Scalar>(3.06635720) * xyyy -
                      irc<Scalar>(9.19907162) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953580) * xxyy -
        irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.06388245) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388244) * zzzz;
    const Scalar dz = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.51105954) * yyyz +
                      irc<Scalar>(0.25552976) * yzzz;
    M_dM[Ur + 26] = Y;
    M_dM[Ux + 26] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 26] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 26] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxz - irc<Scalar>(4.59953581) * xxyyz -
        irc<Scalar>(1.53317860) * xxzzz + irc<Scalar>(0.06388244) * yyyy * z +
        irc<Scalar>(0.25552977) * yyzzz + irc<Scalar>(0.01277649) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxz -
                      irc<Scalar>(9.19907162) * xyyz -
                      irc<Scalar>(3.06635720) * xzzz;
    const Scalar dy = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.25552976) * yyyz +
                      irc<Scalar>(0.51105954) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
        irc<Scalar>(4.59953580) * xxzz + irc<Scalar>(0.06388244) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388245) * zzzz;
    M_dM[Ur + 27] = Y;
    M_dM[Ux + 27] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 27] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 27] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(2.10818511) * xxx * yy_minus_zz -
                     irc<Scalar>(0.52704628) * x * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(6.32455533) * xx * yy_minus_zz -
                      irc<Scalar>(0.52704628) * yyyy_minus_zzzz;
    const Scalar dy =
        irc<Scalar>(4.21637022) * xxxy - irc<Scalar>(2.10818512) * xyyy;
    const Scalar dz =
        -irc<Scalar>(4.21637022) * xxxz + irc<Scalar>(2.10818512) * xzzz;
    M_dM[Ur + 28] = Y;
    M_dM[Ux + 28] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 28] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 28] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.65148372) * xxxy * z -
                     irc<Scalar>(1.82574186) * (xyyy * z + x * yzzz);
    const Scalar dx = irc<Scalar>(10.95445116) * xxyz -
                      irc<Scalar>(1.82574186) * (yyyz + yzzz);
    const Scalar dy = irc<Scalar>(3.65148372) * xxxz -
                      irc<Scalar>(5.47722558) * xyyz -
                      irc<Scalar>(1.82574186) * xzzz;
    const Scalar dz = irc<Scalar>(3.65148372) * xxxy -
                      irc<Scalar>(1.82574186) * xyyy -
                      irc<Scalar>(5.47722558) * xyzz;
    M_dM[Ur + 29] = Y;
    M_dM[Ux + 29] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 29] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 29] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        -irc<Scalar>(0.55625378) * xxxxy + irc<Scalar>(2.54996401) * xxyyy -
        irc<Scalar>(2.92453987) * xxyzz - irc<Scalar>(0.02492138) * yyyy * y -
        irc<Scalar>(0.05782623) * yyyzz + irc<Scalar>(0.09569378) * yzzz * z;
    const Scalar dx = -irc<Scalar>(2.22501512) * xxxy +
                      irc<Scalar>(5.09992802) * xyyy -
                      irc<Scalar>(5.84907974) * xyzz;
    const Scalar dy =
        -irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(7.64989203) * xxyy -
        irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.12460690) * yyyy -
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.09569378) * zzzz;
    const Scalar dz = -irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.11565246) * yyyz +
                      irc<Scalar>(0.38277512) * yzzz;
    M_dM[Ur + 30] = Y;
    M_dM[Ux + 30] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 30] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 30] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.55625378) * xxxxz + irc<Scalar>(2.92453987) * xxyyz -
        irc<Scalar>(2.54996401) * xxzzz - irc<Scalar>(0.09569378) * yyyy * z +
        irc<Scalar>(0.05782623) * yyzzz + irc<Scalar>(0.02492138) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.22501512) * xxxz +
                      irc<Scalar>(5.84907974) * xyyz -
                      irc<Scalar>(5.09992802) * xzzz;
    const Scalar dy = irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.38277512) * yyyz +
                      irc<Scalar>(0.11565246) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
        irc<Scalar>(7.64989203) * xxzz - irc<Scalar>(0.09569378) * yyyy +
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.12460690) * zzzz;
    M_dM[Ur + 31] = Y;
    M_dM[Ux + 31] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 31] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 31] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(0.02706701) * xxxx * x -
                     irc<Scalar>(1.35335068) * xxx * yy_plus_zz +
                     irc<Scalar>(0.45132312) * x * yyyy_plus_zzzz -
                     irc<Scalar>(4.06747614) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.13533505) * xxxx -
                      irc<Scalar>(4.06005204) * xx * yy_plus_zz +
                      irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
                      irc<Scalar>(4.06747614) * yyzz;
    const Scalar dy = -irc<Scalar>(2.70670136) * xxxy +
                      irc<Scalar>(1.80529248) * xyyy -
                      irc<Scalar>(8.13495228) * xyzz;
    const Scalar dz = -irc<Scalar>(2.70670136) * xxxz +
                      irc<Scalar>(1.80529248) * xzzz -
                      irc<Scalar>(8.13495228) * xyyz;
    M_dM[Ur + 32] = Y;
    M_dM[Ux + 32] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 32] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 32] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y = irc<Scalar>(3.16227766) * (xyyy * z - x * yzzz);
    const Scalar dx = irc<Scalar>(3.16227766) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(9.48683298) * xyyz - irc<Scalar>(3.16227766) * xzzz;
    const Scalar dz =
        irc<Scalar>(3.16227766) * xyyy - irc<Scalar>(9.48683298) * xyzz;
    M_dM[Ur + 33] = Y;
    M_dM[Ux + 33] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 33] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 33] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxy - irc<Scalar>(0.18046516) * xxyyy -
        irc<Scalar>(0.21537191) * xxyzz + irc<Scalar>(0.03149789) * yyyy * y -
        irc<Scalar>(2.96932339) * yyyzz + irc<Scalar>(0.74831340) * yzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxy -
                      irc<Scalar>(0.36093032) * xyyy -
                      irc<Scalar>(0.43074382) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.54139548) * xxyy -
        irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.15748945) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.74831340) * zzzz;
    const Scalar dz = -irc<Scalar>(0.43074382) * xxyz -
                      irc<Scalar>(5.93864678) * yyyz +
                      irc<Scalar>(2.99325360) * yzzz;
    M_dM[Ur + 34] = Y;
    M_dM[Ux + 34] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 34] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 34] = v_z5 * Y + r1 * dz;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxz - irc<Scalar>(0.21537191) * xxyyz -
        irc<Scalar>(0.18046516) * xxzzz + irc<Scalar>(0.74831340) * yyyy * z -
        irc<Scalar>(2.96932339) * yyzzz + irc<Scalar>(0.03149789) * zzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxz -
                      irc<Scalar>(0.43074382) * xyyz -
                      irc<Scalar>(0.36093032) * xzzz;
    const Scalar dy = -irc<Scalar>(0.43074382) * xxyz +
                      irc<Scalar>(2.99325360) * yyyz -
                      irc<Scalar>(5.93864678) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
        irc<Scalar>(0.54139548) * xxzz + irc<Scalar>(0.74831340) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.15748945) * zzzz;
    M_dM[Ur + 35] = Y;
    M_dM[Ux + 35] = v_x5 * Y + r1 * dx;
    M_dM[Uy + 35] = v_y5 * Y + r1 * dy;
    M_dM[Uz + 35] = v_z5 * Y + r1 * dz;
  }
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_dM_otf(Scalar x, Scalar y, Scalar z,
                                             Scalar rijinv,
                                             const Scalar* __restrict__ W_ku,
                                             Scalar& fa_x, Scalar& fa_y,
                                             Scalar& fa_z) {
  Scalar w;

  if constexpr (m_val == 1) return;

  const Scalar r1 = rijinv;
  const Scalar v_x = -r1 * x, v_y = -r1 * y, v_z = -r1 * z;

  w = W_ku[1];
  fa_x += w * (v_x * x + r1);
  fa_y += w * (v_x * y);
  fa_z += w * (v_x * z);

  w = W_ku[2];
  fa_x += w * (v_y * x);
  fa_y += w * (v_y * y + r1);
  fa_z += w * (v_y * z);

  w = W_ku[3];
  fa_x += w * (v_z * x);
  fa_y += w * (v_z * y);
  fa_z += w * (v_z * z + r1);

  if constexpr (m_val == 2) return;

  const Scalar r2 = irc<Scalar>(2.0) * r1;
  const Scalar r3 = irc<Scalar>(3.0) * r1;
  const Scalar v_x2 = -r2 * x, v_y2 = -r2 * y, v_z2 = -r2 * z;
  const Scalar v_x3 = -r3 * x, v_y3 = -r3 * y, v_z3 = -r3 * z;

  const Scalar xx = x * x, yy = y * y, zz = z * z;
  const Scalar xy = x * y, xz = x * z, yz = y * z;
  const Scalar xxx = xx * x, yyy = yy * y, zzz = zz * z;
  const Scalar xxy = xx * y, xxz = xx * z;
  const Scalar xyy = x * yy, xzz = x * zz;
  const Scalar yyz = yy * z, yzz = y * zz;
  const Scalar xyz = xy * z;

  {
    const Scalar Y =
        irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
    const Scalar dx = irc<Scalar>(1.63299316) * x;
    const Scalar dy = -irc<Scalar>(0.81649658) * y;
    const Scalar dz = -irc<Scalar>(0.81649658) * z;
    w = W_ku[4];
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xy;
    const Scalar dx = irc<Scalar>(1.41421356) * y;
    const Scalar dy = irc<Scalar>(1.41421356) * x;
    w = W_ku[5];
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xz;
    const Scalar dx = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * x;
    w = W_ku[6];
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.70710678) * (yy - zz);
    const Scalar dy = irc<Scalar>(1.41421356) * y;
    const Scalar dz = -irc<Scalar>(1.41421356) * z;
    w = W_ku[7];
    fa_x += w * (v_x2 * Y);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * yz;
    const Scalar dy = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * y;
    w = W_ku[8];
    fa_x += w * (v_x2 * Y);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }

  if constexpr (m_val == 3) return;

  {
    const Scalar Y = irc<Scalar>(0.26261287) * xxx -
                     irc<Scalar>(1.18175790) * xyy -
                     irc<Scalar>(1.18175790) * xzz;
    const Scalar dx = irc<Scalar>(0.78783861) * xx -
                      irc<Scalar>(1.18175790) * yy -
                      irc<Scalar>(1.18175790) * zz;
    const Scalar dy = -irc<Scalar>(2.36351580) * xy;
    const Scalar dz = -irc<Scalar>(2.36351580) * xz;
    w = W_ku[9];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxy -
                     irc<Scalar>(0.13867505) * yyy -
                     irc<Scalar>(0.41602515) * yzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xy;
    const Scalar dy = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    const Scalar dz = -irc<Scalar>(0.83205030) * yz;
    w = W_ku[10];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxz -
                     irc<Scalar>(0.41602515) * yyz -
                     irc<Scalar>(0.13867505) * zzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xz;
    const Scalar dy = -irc<Scalar>(0.83205030) * yz;
    const Scalar dz = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    w = W_ku[11];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.22474487) * (xyy - xzz);
    const Scalar dx = irc<Scalar>(1.22474487) * (yy - zz);
    const Scalar dy = irc<Scalar>(2.44948974) * xy;
    const Scalar dz = -irc<Scalar>(2.44948974) * xz;
    w = W_ku[12];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(2.44948974) * xyz;
    const Scalar dx = irc<Scalar>(2.44948974) * yz;
    const Scalar dy = irc<Scalar>(2.44948974) * xz;
    const Scalar dz = irc<Scalar>(2.44948974) * xy;
    w = W_ku[13];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = -irc<Scalar>(0.35682062) * xxy +
                     irc<Scalar>(0.22301289) * yyy -
                     irc<Scalar>(1.65029537) * yzz;
    const Scalar dx = -irc<Scalar>(0.71364124) * xy;
    const Scalar dy = -irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(0.66903867) * yy -
                      irc<Scalar>(1.65029537) * zz;
    const Scalar dz = -irc<Scalar>(3.30059074) * yz;
    w = W_ku[14];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.35682062) * xxz +
                     irc<Scalar>(1.65029537) * yyz -
                     irc<Scalar>(0.22301289) * zzz;
    const Scalar dx = irc<Scalar>(0.71364124) * xz;
    const Scalar dy = irc<Scalar>(3.30059074) * yz;
    const Scalar dz = irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(1.65029537) * yy -
                      irc<Scalar>(0.66903867) * zz;
    w = W_ku[15];
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }

  if constexpr (m_val == 4) return;

  const Scalar r4 = irc<Scalar>(4.0) * r1;
  const Scalar v_x4 = -r4 * x, v_y4 = -r4 * y, v_z4 = -r4 * z;
  const Scalar xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
  const Scalar xxyy = xx * yy, xxzz = xx * zz, yyzz = yy * zz;
  const Scalar xxxy = xxx * y, xxxz = xxx * z;
  const Scalar xyyy = x * yyy, xzzz = x * zzz;
  const Scalar xxyz = xx * yz, xyyz = x * yyz, xyzz = x * yzz;
  const Scalar yyyz = yyy * z, yzzz = y * zzz;
  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  {
    const Scalar Y = irc<Scalar>(0.09421550) * xxxx -
                     irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                     irc<Scalar>(0.03533081) * yyyy_plus_zzzz +
                     irc<Scalar>(0.42396975) * yyzz;
    const Scalar dx = irc<Scalar>(0.37686200) * xxx -
                      irc<Scalar>(3.39175798) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(3.39175798) * xxy +
                      irc<Scalar>(0.14132324) * yyy +
                      irc<Scalar>(0.84793950) * yzz;
    const Scalar dz = -irc<Scalar>(3.39175798) * xxz +
                      irc<Scalar>(0.14132324) * zzz +
                      irc<Scalar>(0.84793950) * yyz;
    w = W_ku[16];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxy -
                     irc<Scalar>(0.83205029) * xyyy -
                     irc<Scalar>(2.49615088) * xyzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxy -
                      irc<Scalar>(0.83205029) * yyy -
                      irc<Scalar>(2.49615088) * yzz;
    const Scalar dy = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615087) * xyy -
                      irc<Scalar>(2.49615088) * xzz;
    const Scalar dz = -irc<Scalar>(4.99230176) * xyz;
    w = W_ku[17];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxz -
                     irc<Scalar>(2.49615088) * xyyz -
                     irc<Scalar>(0.83205029) * xzzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxz -
                      irc<Scalar>(2.49615088) * yyz -
                      irc<Scalar>(0.83205029) * zzz;
    const Scalar dy = -irc<Scalar>(4.99230176) * xyz;
    const Scalar dz = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615088) * xyy -
                      irc<Scalar>(2.49615087) * xzz;
    w = W_ku[18];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.72805530) * (xxyy - xxzz) -
                     irc<Scalar>(0.04800154) * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(3.45611060) * x * yy_minus_zz;
    const Scalar dy =
        irc<Scalar>(3.45611060) * xxy - irc<Scalar>(0.19200616) * yyy;
    const Scalar dz =
        -irc<Scalar>(3.45611060) * xxz + irc<Scalar>(0.19200616) * zzz;
    w = W_ku[19];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.43246532) * xxyz -
                     irc<Scalar>(0.19069252) * (yyyz + yzzz);
    const Scalar dx = irc<Scalar>(6.86493064) * xyz;
    const Scalar dy = irc<Scalar>(3.43246532) * xxz -
                      irc<Scalar>(0.57207756) * yyz -
                      irc<Scalar>(0.19069252) * zzz;
    const Scalar dz = irc<Scalar>(3.43246532) * xxy -
                      irc<Scalar>(0.19069252) * yyy -
                      irc<Scalar>(0.57207756) * yzz;
    w = W_ku[20];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = -irc<Scalar>(0.89754911) * xxxy +
                     irc<Scalar>(1.15933427) * xyyy -
                     irc<Scalar>(2.35606641) * xyzz;
    const Scalar dx = -irc<Scalar>(2.69264733) * xxy +
                      irc<Scalar>(1.15933427) * yyy -
                      irc<Scalar>(2.35606641) * yzz;
    const Scalar dy = -irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(3.47800281) * xyy -
                      irc<Scalar>(2.35606641) * xzz;
    const Scalar dz = -irc<Scalar>(4.71213282) * xyz;
    w = W_ku[21];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.89754911) * xxxz +
                     irc<Scalar>(2.35606641) * xyyz -
                     irc<Scalar>(1.15933427) * xzzz;
    const Scalar dx = irc<Scalar>(2.69264733) * xxz +
                      irc<Scalar>(2.35606641) * yyz -
                      irc<Scalar>(1.15933427) * zzz;
    const Scalar dy = irc<Scalar>(4.71213282) * xyz;
    const Scalar dz = irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(2.35606641) * xyy -
                      irc<Scalar>(3.47800281) * xzz;
    w = W_ku[22];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.01600757) * xxxx -
                     irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                     irc<Scalar>(0.07470200) * yyyy_plus_zzzz -
                     irc<Scalar>(2.40113574) * yyzz;
    const Scalar dx = irc<Scalar>(0.06403028) * xxx -
                      irc<Scalar>(0.57627258) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(0.57627258) * xxy +
                      irc<Scalar>(0.29880800) * yyy -
                      irc<Scalar>(4.80227148) * yzz;
    const Scalar dz = -irc<Scalar>(0.57627258) * xxz +
                      irc<Scalar>(0.29880800) * zzz -
                      irc<Scalar>(4.80227148) * yyz;
    w = W_ku[23];
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(4.24264068) * yyz - irc<Scalar>(1.41421356) * zzz;
    const Scalar dz =
        irc<Scalar>(1.41421356) * yyy - irc<Scalar>(4.24264068) * yzz;
    w = W_ku[24];
    fa_x += w * v_x4 * Y;
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }

  if constexpr (m_val == 5) return;

  const Scalar r5 = irc<Scalar>(5.0) * r1;
  const Scalar v_x5 = -r5 * x, v_y5 = -r5 * y, v_z5 = -r5 * z;
  const Scalar xxxxy = xxxx * y;
  const Scalar xxxxz = xxxx * z;
  const Scalar xxyyy = xx * yyy;
  const Scalar xxyzz = xx * yzz;
  const Scalar xxyyz = xx * yyz;
  const Scalar xxzzz = xx * zzz;
  const Scalar yyyzz = yyy * zz;
  const Scalar yyzzz = yy * zzz;

  {
    const Scalar Y = irc<Scalar>(0.03230801) * xxxx * x -
                     irc<Scalar>(1.61540033) * xxx * yy_plus_zz +
                     irc<Scalar>(0.30288756) * x * yyyy_plus_zzzz +
                     irc<Scalar>(3.63465074) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.16154005) * xxxx -
                      irc<Scalar>(4.84620099) * xx * yy_plus_zz +
                      irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
                      irc<Scalar>(3.63465074) * yyzz;
    const Scalar dy = -irc<Scalar>(3.23080066) * xxxy +
                      irc<Scalar>(1.21155024) * xyyy +
                      irc<Scalar>(7.26930148) * xyzz;
    const Scalar dz = -irc<Scalar>(3.23080066) * xxxz +
                      irc<Scalar>(1.21155024) * xzzz +
                      irc<Scalar>(7.26930148) * xyyz;
    w = W_ku[25];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxy - irc<Scalar>(1.53317860) * xxyyy -
        irc<Scalar>(4.59953581) * xxyzz + irc<Scalar>(0.01277649) * yyyy * y +
        irc<Scalar>(0.25552977) * yyyzz + irc<Scalar>(0.06388244) * yzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxy -
                      irc<Scalar>(3.06635720) * xyyy -
                      irc<Scalar>(9.19907162) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953580) * xxyy -
        irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.06388245) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388244) * zzzz;
    const Scalar dz = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.51105954) * yyyz +
                      irc<Scalar>(0.25552976) * yzzz;
    w = W_ku[26];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxz - irc<Scalar>(4.59953581) * xxyyz -
        irc<Scalar>(1.53317860) * xxzzz + irc<Scalar>(0.06388244) * yyyy * z +
        irc<Scalar>(0.25552977) * yyzzz + irc<Scalar>(0.01277649) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxz -
                      irc<Scalar>(9.19907162) * xyyz -
                      irc<Scalar>(3.06635720) * xzzz;
    const Scalar dy = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.25552976) * yyyz +
                      irc<Scalar>(0.51105954) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
        irc<Scalar>(4.59953580) * xxzz + irc<Scalar>(0.06388244) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388245) * zzzz;
    w = W_ku[27];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(2.10818511) * xxx * yy_minus_zz -
                     irc<Scalar>(0.52704628) * x * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(6.32455533) * xx * yy_minus_zz -
                      irc<Scalar>(0.52704628) * yyyy_minus_zzzz;
    const Scalar dy =
        irc<Scalar>(4.21637022) * xxxy - irc<Scalar>(2.10818512) * xyyy;
    const Scalar dz =
        -irc<Scalar>(4.21637022) * xxxz + irc<Scalar>(2.10818512) * xzzz;
    w = W_ku[28];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.65148372) * xxxy * z -
                     irc<Scalar>(1.82574186) * (xyyy * z + x * yzzz);
    const Scalar dx = irc<Scalar>(10.95445116) * xxyz -
                      irc<Scalar>(1.82574186) * (yyyz + yzzz);
    const Scalar dy = irc<Scalar>(3.65148372) * xxxz -
                      irc<Scalar>(5.47722558) * xyyz -
                      irc<Scalar>(1.82574186) * xzzz;
    const Scalar dz = irc<Scalar>(3.65148372) * xxxy -
                      irc<Scalar>(1.82574186) * xyyy -
                      irc<Scalar>(5.47722558) * xyzz;
    w = W_ku[29];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        -irc<Scalar>(0.55625378) * xxxxy + irc<Scalar>(2.54996401) * xxyyy -
        irc<Scalar>(2.92453987) * xxyzz - irc<Scalar>(0.02492138) * yyyy * y -
        irc<Scalar>(0.05782623) * yyyzz + irc<Scalar>(0.09569378) * yzzz * z;
    const Scalar dx = -irc<Scalar>(2.22501512) * xxxy +
                      irc<Scalar>(5.09992802) * xyyy -
                      irc<Scalar>(5.84907974) * xyzz;
    const Scalar dy =
        -irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(7.64989203) * xxyy -
        irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.12460690) * yyyy -
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.09569378) * zzzz;
    const Scalar dz = -irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.11565246) * yyyz +
                      irc<Scalar>(0.38277512) * yzzz;
    w = W_ku[30];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.55625378) * xxxxz + irc<Scalar>(2.92453987) * xxyyz -
        irc<Scalar>(2.54996401) * xxzzz - irc<Scalar>(0.09569378) * yyyy * z +
        irc<Scalar>(0.05782623) * yyzzz + irc<Scalar>(0.02492138) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.22501512) * xxxz +
                      irc<Scalar>(5.84907974) * xyyz -
                      irc<Scalar>(5.09992802) * xzzz;
    const Scalar dy = irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.38277512) * yyyz +
                      irc<Scalar>(0.11565246) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
        irc<Scalar>(7.64989203) * xxzz - irc<Scalar>(0.09569378) * yyyy +
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.12460690) * zzzz;
    w = W_ku[31];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.02706701) * xxxx * x -
                     irc<Scalar>(1.35335068) * xxx * yy_plus_zz +
                     irc<Scalar>(0.45132312) * x * yyyy_plus_zzzz -
                     irc<Scalar>(4.06747614) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.13533505) * xxxx -
                      irc<Scalar>(4.06005204) * xx * yy_plus_zz +
                      irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
                      irc<Scalar>(4.06747614) * yyzz;
    const Scalar dy = -irc<Scalar>(2.70670136) * xxxy +
                      irc<Scalar>(1.80529248) * xyyy -
                      irc<Scalar>(8.13495228) * xyzz;
    const Scalar dz = -irc<Scalar>(2.70670136) * xxxz +
                      irc<Scalar>(1.80529248) * xzzz -
                      irc<Scalar>(8.13495228) * xyyz;
    w = W_ku[32];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.16227766) * (xyyy * z - x * yzzz);
    const Scalar dx = irc<Scalar>(3.16227766) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(9.48683298) * xyyz - irc<Scalar>(3.16227766) * xzzz;
    const Scalar dz =
        irc<Scalar>(3.16227766) * xyyy - irc<Scalar>(9.48683298) * xyzz;
    w = W_ku[33];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxy - irc<Scalar>(0.18046516) * xxyyy -
        irc<Scalar>(0.21537191) * xxyzz + irc<Scalar>(0.03149789) * yyyy * y -
        irc<Scalar>(2.96932339) * yyyzz + irc<Scalar>(0.74831340) * yzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxy -
                      irc<Scalar>(0.36093032) * xyyy -
                      irc<Scalar>(0.43074382) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.54139548) * xxyy -
        irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.15748945) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.74831340) * zzzz;
    const Scalar dz = -irc<Scalar>(0.43074382) * xxyz -
                      irc<Scalar>(5.93864678) * yyyz +
                      irc<Scalar>(2.99325360) * yzzz;
    w = W_ku[34];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxz - irc<Scalar>(0.21537191) * xxyyz -
        irc<Scalar>(0.18046516) * xxzzz + irc<Scalar>(0.74831340) * yyyy * z -
        irc<Scalar>(2.96932339) * yyzzz + irc<Scalar>(0.03149789) * zzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxz -
                      irc<Scalar>(0.43074382) * xyyz -
                      irc<Scalar>(0.36093032) * xzzz;
    const Scalar dy = -irc<Scalar>(0.43074382) * xxyz +
                      irc<Scalar>(2.99325360) * yyyz -
                      irc<Scalar>(5.93864678) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
        irc<Scalar>(0.54139548) * xxzz + irc<Scalar>(0.74831340) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.15748945) * zzzz;
    w = W_ku[35];
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_M_otf(Scalar x, Scalar y, Scalar z,
                                            const Scalar* __restrict__ W_ku,
                                            Scalar& fr) {
  Scalar w;

  w = W_ku[0];
  fr += w;

  if constexpr (m_val == 1) return;

  w = W_ku[1];
  fr += w * x;

  w = W_ku[2];
  fr += w * y;

  w = W_ku[3];
  fr += w * z;

  if constexpr (m_val == 2) return;

  const Scalar xx = x * x, yy = y * y, zz = z * z;
  const Scalar xy = x * y, xz = x * z, yz = y * z;

  {
    const Scalar Y =
        irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
    w = W_ku[4];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xy;
    w = W_ku[5];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xz;
    w = W_ku[6];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(0.70710678) * (yy - zz);
    w = W_ku[7];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * yz;
    w = W_ku[8];
    fr += w * Y;
  }

  if constexpr (m_val == 3) return;

  const Scalar xxx = xx * x, yyy = yy * y, zzz = zz * z;
  const Scalar xxy = xx * y, xxz = xx * z;
  const Scalar xyy = x * yy, xzz = x * zz;
  const Scalar yyz = yy * z, yzz = y * zz;
  const Scalar xyz = xy * z;

  {
    const Scalar Y = irc<Scalar>(0.26261287) * xxx -
                     irc<Scalar>(1.18175790) * xyy -
                     irc<Scalar>(1.18175790) * xzz;
    w = W_ku[9];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxy -
                     irc<Scalar>(0.13867505) * yyy -
                     irc<Scalar>(0.41602515) * yzz;
    w = W_ku[10];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxz -
                     irc<Scalar>(0.41602515) * yyz -
                     irc<Scalar>(0.13867505) * zzz;
    w = W_ku[11];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.22474487) * (xyy - xzz);
    w = W_ku[12];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(2.44948974) * xyz;
    w = W_ku[13];
    fr += w * Y;
  }
  {
    const Scalar Y = -irc<Scalar>(0.35682062) * xxy +
                     irc<Scalar>(0.22301289) * yyy -
                     irc<Scalar>(1.65029537) * yzz;
    w = W_ku[14];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(0.35682062) * xxz +
                     irc<Scalar>(1.65029537) * yyz -
                     irc<Scalar>(0.22301289) * zzz;
    w = W_ku[15];
    fr += w * Y;
  }

  if constexpr (m_val == 4) return;

  const Scalar xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
  const Scalar xxyy = xx * yy, xxzz = xx * zz, yyzz = yy * zz;
  const Scalar xxxy = xxx * y, xxxz = xxx * z;
  const Scalar xyyy = x * yyy, xzzz = x * zzz;
  const Scalar xxyz = xx * yz, xyyz = x * yyz, xyzz = x * yzz;
  const Scalar yyyz = yyy * z, yzzz = y * zzz;
  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  {
    const Scalar Y = irc<Scalar>(0.09421550) * xxxx -
                     irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                     irc<Scalar>(0.03533081) * yyyy_plus_zzzz +
                     irc<Scalar>(0.42396975) * yyzz;
    w = W_ku[16];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxy -
                     irc<Scalar>(0.83205029) * xyyy -
                     irc<Scalar>(2.49615088) * xyzz;
    w = W_ku[17];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxz -
                     irc<Scalar>(2.49615088) * xyyz -
                     irc<Scalar>(0.83205029) * xzzz;
    w = W_ku[18];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.72805530) * (xxyy - xxzz) -
                     irc<Scalar>(0.04800154) * yyyy_minus_zzzz;
    w = W_ku[19];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(3.43246532) * xxyz -
                     irc<Scalar>(0.19069252) * (yyyz + yzzz);
    w = W_ku[20];
    fr += w * Y;
  }
  {
    const Scalar Y = -irc<Scalar>(0.89754911) * xxxy +
                     irc<Scalar>(1.15933427) * xyyy -
                     irc<Scalar>(2.35606641) * xyzz;
    w = W_ku[21];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(0.89754911) * xxxz +
                     irc<Scalar>(2.35606641) * xyyz -
                     irc<Scalar>(1.15933427) * xzzz;
    w = W_ku[22];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(0.01600757) * xxxx -
                     irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                     irc<Scalar>(0.07470200) * yyyy_plus_zzzz -
                     irc<Scalar>(2.40113574) * yyzz;
    w = W_ku[23];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * (yyyz - yzzz);
    w = W_ku[24];
    fr += w * Y;
  }

  if constexpr (m_val == 5) return;

  const Scalar xxxxy = xxxx * y;
  const Scalar xxxxz = xxxx * z;
  const Scalar xxyyy = xx * yyy;
  const Scalar xxyzz = xx * yzz;
  const Scalar xxyyz = xx * yyz;
  const Scalar xxzzz = xx * zzz;
  const Scalar yyyzz = yyy * zz;
  const Scalar yyzzz = yy * zzz;

  {
    const Scalar Y = irc<Scalar>(0.03230801) * xxxx * x -
                     irc<Scalar>(1.61540033) * xxx * yy_plus_zz +
                     irc<Scalar>(0.30288756) * x * yyyy_plus_zzzz +
                     irc<Scalar>(3.63465074) * x * yyzz;
    w = W_ku[25];
    fr += w * Y;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxy - irc<Scalar>(1.53317860) * xxyyy -
        irc<Scalar>(4.59953581) * xxyzz + irc<Scalar>(0.01277649) * yyyy * y +
        irc<Scalar>(0.25552977) * yyyzz + irc<Scalar>(0.06388244) * yzzz * z;
    w = W_ku[26];
    fr += w * Y;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxz - irc<Scalar>(4.59953581) * xxyyz -
        irc<Scalar>(1.53317860) * xxzzz + irc<Scalar>(0.06388244) * yyyy * z +
        irc<Scalar>(0.25552977) * yyzzz + irc<Scalar>(0.01277649) * zzzz * z;
    w = W_ku[27];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(2.10818511) * xxx * yy_minus_zz -
                     irc<Scalar>(0.52704628) * x * yyyy_minus_zzzz;
    w = W_ku[28];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(3.65148372) * xxxy * z -
                     irc<Scalar>(1.82574186) * (xyyy * z + x * yzzz);
    w = W_ku[29];
    fr += w * Y;
  }
  {
    const Scalar Y =
        -irc<Scalar>(0.55625378) * xxxxy + irc<Scalar>(2.54996401) * xxyyy -
        irc<Scalar>(2.92453987) * xxyzz - irc<Scalar>(0.02492138) * yyyy * y -
        irc<Scalar>(0.05782623) * yyyzz + irc<Scalar>(0.09569378) * yzzz * z;
    w = W_ku[30];
    fr += w * Y;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.55625378) * xxxxz + irc<Scalar>(2.92453987) * xxyyz -
        irc<Scalar>(2.54996401) * xxzzz - irc<Scalar>(0.09569378) * yyyy * z +
        irc<Scalar>(0.05782623) * yyzzz + irc<Scalar>(0.02492138) * zzzz * z;
    w = W_ku[31];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(0.02706701) * xxxx * x -
                     irc<Scalar>(1.35335068) * xxx * yy_plus_zz +
                     irc<Scalar>(0.45132312) * x * yyyy_plus_zzzz -
                     irc<Scalar>(4.06747614) * x * yyzz;
    w = W_ku[32];
    fr += w * Y;
  }
  {
    const Scalar Y = irc<Scalar>(3.16227766) * (xyyy * z - x * yzzz);
    w = W_ku[33];
    fr += w * Y;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxy - irc<Scalar>(0.18046516) * xxyyy -
        irc<Scalar>(0.21537191) * xxyzz + irc<Scalar>(0.03149789) * yyyy * y -
        irc<Scalar>(2.96932339) * yyyzz + irc<Scalar>(0.74831340) * yzzz * z;
    w = W_ku[34];
    fr += w * Y;
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxz - irc<Scalar>(0.21537191) * xxyyz -
        irc<Scalar>(0.18046516) * xxzzz + irc<Scalar>(0.74831340) * yyyy * z -
        irc<Scalar>(2.96932339) * yyzzz + irc<Scalar>(0.03149789) * zzzz * z;
    w = W_ku[35];
    fr += w * Y;
  }
}

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_W_M_dM_otf(Scalar x, Scalar y, Scalar z,
                                               Scalar rijinv,
                                               const Scalar* __restrict__ W_ku,
                                               Scalar& fr, Scalar& fa_x,
                                               Scalar& fa_y, Scalar& fa_z) {
  Scalar w;

  w = W_ku[0];
  fr += w;

  if constexpr (m_val == 1) return;

  const Scalar r1 = rijinv;
  const Scalar v_x = -r1 * x, v_y = -r1 * y, v_z = -r1 * z;

  w = W_ku[1];
  fr += w * x;
  fa_x += w * (v_x * x + r1);
  fa_y += w * (v_x * y);
  fa_z += w * (v_x * z);

  w = W_ku[2];
  fr += w * y;
  fa_x += w * (v_y * x);
  fa_y += w * (v_y * y + r1);
  fa_z += w * (v_y * z);

  w = W_ku[3];
  fr += w * z;
  fa_x += w * (v_z * x);
  fa_y += w * (v_z * y);
  fa_z += w * (v_z * z + r1);

  if constexpr (m_val == 2) return;

  const Scalar r2 = irc<Scalar>(2.0) * r1;
  const Scalar r3 = irc<Scalar>(3.0) * r1;
  const Scalar v_x2 = -r2 * x, v_y2 = -r2 * y, v_z2 = -r2 * z;
  const Scalar v_x3 = -r3 * x, v_y3 = -r3 * y, v_z3 = -r3 * z;

  const Scalar xx = x * x, yy = y * y, zz = z * z;
  const Scalar xy = x * y, xz = x * z, yz = y * z;
  const Scalar xxx = xx * x, yyy = yy * y, zzz = zz * z;
  const Scalar xxy = xx * y, xxz = xx * z;
  const Scalar xyy = x * yy, xzz = x * zz;
  const Scalar yyz = yy * z, yzz = y * zz;
  const Scalar xyz = xy * z;

  {
    const Scalar Y =
        irc<Scalar>(0.81649658) * xx - irc<Scalar>(0.40824829) * (yy + zz);
    const Scalar dx = irc<Scalar>(1.63299316) * x;
    const Scalar dy = -irc<Scalar>(0.81649658) * y;
    const Scalar dz = -irc<Scalar>(0.81649658) * z;
    w = W_ku[4];
    fr += w * Y;
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xy;
    const Scalar dx = irc<Scalar>(1.41421356) * y;
    const Scalar dy = irc<Scalar>(1.41421356) * x;
    w = W_ku[5];
    fr += w * Y;
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * xz;
    const Scalar dx = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * x;
    w = W_ku[6];
    fr += w * Y;
    fa_x += w * (v_x2 * Y + r1 * dx);
    fa_y += w * (v_y2 * Y);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.70710678) * (yy - zz);
    const Scalar dy = irc<Scalar>(1.41421356) * y;
    const Scalar dz = -irc<Scalar>(1.41421356) * z;
    w = W_ku[7];
    fr += w * Y;
    fa_x += w * (v_x2 * Y);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * yz;
    const Scalar dy = irc<Scalar>(1.41421356) * z;
    const Scalar dz = irc<Scalar>(1.41421356) * y;
    w = W_ku[8];
    fr += w * Y;
    fa_x += w * (v_x2 * Y);
    fa_y += w * (v_y2 * Y + r1 * dy);
    fa_z += w * (v_z2 * Y + r1 * dz);
  }

  if constexpr (m_val == 3) return;

  {
    const Scalar Y = irc<Scalar>(0.26261287) * xxx -
                     irc<Scalar>(1.18175790) * xyy -
                     irc<Scalar>(1.18175790) * xzz;
    const Scalar dx = irc<Scalar>(0.78783861) * xx -
                      irc<Scalar>(1.18175790) * yy -
                      irc<Scalar>(1.18175790) * zz;
    const Scalar dy = -irc<Scalar>(2.36351580) * xy;
    const Scalar dz = -irc<Scalar>(2.36351580) * xz;
    w = W_ku[9];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxy -
                     irc<Scalar>(0.13867505) * yyy -
                     irc<Scalar>(0.41602515) * yzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xy;
    const Scalar dy = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    const Scalar dz = -irc<Scalar>(0.83205030) * yz;
    w = W_ku[10];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.66410059) * xxz -
                     irc<Scalar>(0.41602515) * yyz -
                     irc<Scalar>(0.13867505) * zzz;
    const Scalar dx = irc<Scalar>(3.32820118) * xz;
    const Scalar dy = -irc<Scalar>(0.83205030) * yz;
    const Scalar dz = irc<Scalar>(1.66410059) * xx -
                      irc<Scalar>(0.41602515) * yy -
                      irc<Scalar>(0.41602515) * zz;
    w = W_ku[11];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.22474487) * (xyy - xzz);
    const Scalar dx = irc<Scalar>(1.22474487) * (yy - zz);
    const Scalar dy = irc<Scalar>(2.44948974) * xy;
    const Scalar dz = -irc<Scalar>(2.44948974) * xz;
    w = W_ku[12];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(2.44948974) * xyz;
    const Scalar dx = irc<Scalar>(2.44948974) * yz;
    const Scalar dy = irc<Scalar>(2.44948974) * xz;
    const Scalar dz = irc<Scalar>(2.44948974) * xy;
    w = W_ku[13];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = -irc<Scalar>(0.35682062) * xxy +
                     irc<Scalar>(0.22301289) * yyy -
                     irc<Scalar>(1.65029537) * yzz;
    const Scalar dx = -irc<Scalar>(0.71364124) * xy;
    const Scalar dy = -irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(0.66903867) * yy -
                      irc<Scalar>(1.65029537) * zz;
    const Scalar dz = -irc<Scalar>(3.30059074) * yz;
    w = W_ku[14];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.35682062) * xxz +
                     irc<Scalar>(1.65029537) * yyz -
                     irc<Scalar>(0.22301289) * zzz;
    const Scalar dx = irc<Scalar>(0.71364124) * xz;
    const Scalar dy = irc<Scalar>(3.30059074) * yz;
    const Scalar dz = irc<Scalar>(0.35682062) * xx +
                      irc<Scalar>(1.65029537) * yy -
                      irc<Scalar>(0.66903867) * zz;
    w = W_ku[15];
    fr += w * Y;
    fa_x += w * (v_x3 * Y + r1 * dx);
    fa_y += w * (v_y3 * Y + r1 * dy);
    fa_z += w * (v_z3 * Y + r1 * dz);
  }

  if constexpr (m_val == 4) return;

  const Scalar r4 = irc<Scalar>(4.0) * r1;
  const Scalar v_x4 = -r4 * x, v_y4 = -r4 * y, v_z4 = -r4 * z;
  const Scalar xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
  const Scalar xxyy = xx * yy, xxzz = xx * zz, yyzz = yy * zz;
  const Scalar xxxy = xxx * y, xxxz = xxx * z;
  const Scalar xyyy = x * yyy, xzzz = x * zzz;
  const Scalar xxyz = xx * yz, xyyz = x * yyz, xyzz = x * yzz;
  const Scalar yyyz = yyy * z, yzzz = y * zzz;
  const Scalar yyyy_plus_zzzz = yyyy + zzzz;
  const Scalar yyyy_minus_zzzz = yyyy - zzzz;
  const Scalar yy_plus_zz = yy + zz;
  const Scalar yy_minus_zz = yy - zz;

  {
    const Scalar Y = irc<Scalar>(0.09421550) * xxxx -
                     irc<Scalar>(1.69587899) * xx * yy_plus_zz +
                     irc<Scalar>(0.03533081) * yyyy_plus_zzzz +
                     irc<Scalar>(0.42396975) * yyzz;
    const Scalar dx = irc<Scalar>(0.37686200) * xxx -
                      irc<Scalar>(3.39175798) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(3.39175798) * xxy +
                      irc<Scalar>(0.14132324) * yyy +
                      irc<Scalar>(0.84793950) * yzz;
    const Scalar dz = -irc<Scalar>(3.39175798) * xxz +
                      irc<Scalar>(0.14132324) * zzz +
                      irc<Scalar>(0.84793950) * yyz;
    w = W_ku[16];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxy -
                     irc<Scalar>(0.83205029) * xyyy -
                     irc<Scalar>(2.49615088) * xyzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxy -
                      irc<Scalar>(0.83205029) * yyy -
                      irc<Scalar>(2.49615088) * yzz;
    const Scalar dy = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615087) * xyy -
                      irc<Scalar>(2.49615088) * xzz;
    const Scalar dz = -irc<Scalar>(4.99230176) * xyz;
    w = W_ku[17];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.10940039) * xxxz -
                     irc<Scalar>(2.49615088) * xyyz -
                     irc<Scalar>(0.83205029) * xzzz;
    const Scalar dx = irc<Scalar>(3.32820117) * xxz -
                      irc<Scalar>(2.49615088) * yyz -
                      irc<Scalar>(0.83205029) * zzz;
    const Scalar dy = -irc<Scalar>(4.99230176) * xyz;
    const Scalar dz = irc<Scalar>(1.10940039) * xxx -
                      irc<Scalar>(2.49615088) * xyy -
                      irc<Scalar>(2.49615087) * xzz;
    w = W_ku[18];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.72805530) * (xxyy - xxzz) -
                     irc<Scalar>(0.04800154) * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(3.45611060) * x * yy_minus_zz;
    const Scalar dy =
        irc<Scalar>(3.45611060) * xxy - irc<Scalar>(0.19200616) * yyy;
    const Scalar dz =
        -irc<Scalar>(3.45611060) * xxz + irc<Scalar>(0.19200616) * zzz;
    w = W_ku[19];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.43246532) * xxyz -
                     irc<Scalar>(0.19069252) * (yyyz + yzzz);
    const Scalar dx = irc<Scalar>(6.86493064) * xyz;
    const Scalar dy = irc<Scalar>(3.43246532) * xxz -
                      irc<Scalar>(0.57207756) * yyz -
                      irc<Scalar>(0.19069252) * zzz;
    const Scalar dz = irc<Scalar>(3.43246532) * xxy -
                      irc<Scalar>(0.19069252) * yyy -
                      irc<Scalar>(0.57207756) * yzz;
    w = W_ku[20];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = -irc<Scalar>(0.89754911) * xxxy +
                     irc<Scalar>(1.15933427) * xyyy -
                     irc<Scalar>(2.35606641) * xyzz;
    const Scalar dx = -irc<Scalar>(2.69264733) * xxy +
                      irc<Scalar>(1.15933427) * yyy -
                      irc<Scalar>(2.35606641) * yzz;
    const Scalar dy = -irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(3.47800281) * xyy -
                      irc<Scalar>(2.35606641) * xzz;
    const Scalar dz = -irc<Scalar>(4.71213282) * xyz;
    w = W_ku[21];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.89754911) * xxxz +
                     irc<Scalar>(2.35606641) * xyyz -
                     irc<Scalar>(1.15933427) * xzzz;
    const Scalar dx = irc<Scalar>(2.69264733) * xxz +
                      irc<Scalar>(2.35606641) * yyz -
                      irc<Scalar>(1.15933427) * zzz;
    const Scalar dy = irc<Scalar>(4.71213282) * xyz;
    const Scalar dz = irc<Scalar>(0.89754911) * xxx +
                      irc<Scalar>(2.35606641) * xyy -
                      irc<Scalar>(3.47800281) * xzz;
    w = W_ku[22];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.01600757) * xxxx -
                     irc<Scalar>(0.28813629) * xx * yy_plus_zz +
                     irc<Scalar>(0.07470200) * yyyy_plus_zzzz -
                     irc<Scalar>(2.40113574) * yyzz;
    const Scalar dx = irc<Scalar>(0.06403028) * xxx -
                      irc<Scalar>(0.57627258) * x * yy_plus_zz;
    const Scalar dy = -irc<Scalar>(0.57627258) * xxy +
                      irc<Scalar>(0.29880800) * yyy -
                      irc<Scalar>(4.80227148) * yzz;
    const Scalar dz = -irc<Scalar>(0.57627258) * xxz +
                      irc<Scalar>(0.29880800) * zzz -
                      irc<Scalar>(4.80227148) * yyz;
    w = W_ku[23];
    fr += w * Y;
    fa_x += w * (v_x4 * Y + r1 * dx);
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(1.41421356) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(4.24264068) * yyz - irc<Scalar>(1.41421356) * zzz;
    const Scalar dz =
        irc<Scalar>(1.41421356) * yyy - irc<Scalar>(4.24264068) * yzz;
    w = W_ku[24];
    fr += w * Y;
    fa_x += w * v_x4 * Y;
    fa_y += w * (v_y4 * Y + r1 * dy);
    fa_z += w * (v_z4 * Y + r1 * dz);
  }

  if constexpr (m_val == 5) return;

  const Scalar r5 = irc<Scalar>(5.0) * r1;
  const Scalar v_x5 = -r5 * x, v_y5 = -r5 * y, v_z5 = -r5 * z;
  const Scalar xxxxy = xxxx * y;
  const Scalar xxxxz = xxxx * z;
  const Scalar xxyyy = xx * yyy;
  const Scalar xxyzz = xx * yzz;
  const Scalar xxyyz = xx * yyz;
  const Scalar xxzzz = xx * zzz;
  const Scalar yyyzz = yyy * zz;
  const Scalar yyzzz = yy * zzz;

  {
    const Scalar Y = irc<Scalar>(0.03230801) * xxxx * x -
                     irc<Scalar>(1.61540033) * xxx * yy_plus_zz +
                     irc<Scalar>(0.30288756) * x * yyyy_plus_zzzz +
                     irc<Scalar>(3.63465074) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.16154005) * xxxx -
                      irc<Scalar>(4.84620099) * xx * yy_plus_zz +
                      irc<Scalar>(0.30288756) * yyyy_plus_zzzz +
                      irc<Scalar>(3.63465074) * yyzz;
    const Scalar dy = -irc<Scalar>(3.23080066) * xxxy +
                      irc<Scalar>(1.21155024) * xyyy +
                      irc<Scalar>(7.26930148) * xyzz;
    const Scalar dz = -irc<Scalar>(3.23080066) * xxxz +
                      irc<Scalar>(1.21155024) * xzzz +
                      irc<Scalar>(7.26930148) * xyyz;
    w = W_ku[25];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxy - irc<Scalar>(1.53317860) * xxyyy -
        irc<Scalar>(4.59953581) * xxyzz + irc<Scalar>(0.01277649) * yyyy * y +
        irc<Scalar>(0.25552977) * yyyzz + irc<Scalar>(0.06388244) * yzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxy -
                      irc<Scalar>(3.06635720) * xyyy -
                      irc<Scalar>(9.19907162) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953580) * xxyy -
        irc<Scalar>(4.59953581) * xxzz + irc<Scalar>(0.06388245) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388244) * zzzz;
    const Scalar dz = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.51105954) * yyyz +
                      irc<Scalar>(0.25552976) * yzzz;
    w = W_ku[26];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.51105953) * xxxxz - irc<Scalar>(4.59953581) * xxyyz -
        irc<Scalar>(1.53317860) * xxzzz + irc<Scalar>(0.06388244) * yyyy * z +
        irc<Scalar>(0.25552977) * yyzzz + irc<Scalar>(0.01277649) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.04423812) * xxxz -
                      irc<Scalar>(9.19907162) * xyyz -
                      irc<Scalar>(3.06635720) * xzzz;
    const Scalar dy = -irc<Scalar>(9.19907162) * xxyz +
                      irc<Scalar>(0.25552976) * yyyz +
                      irc<Scalar>(0.51105954) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.51105953) * xxxx - irc<Scalar>(4.59953581) * xxyy -
        irc<Scalar>(4.59953580) * xxzz + irc<Scalar>(0.06388244) * yyyy +
        irc<Scalar>(0.76658931) * yyzz + irc<Scalar>(0.06388245) * zzzz;
    w = W_ku[27];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(2.10818511) * xxx * yy_minus_zz -
                     irc<Scalar>(0.52704628) * x * yyyy_minus_zzzz;
    const Scalar dx = irc<Scalar>(6.32455533) * xx * yy_minus_zz -
                      irc<Scalar>(0.52704628) * yyyy_minus_zzzz;
    const Scalar dy =
        irc<Scalar>(4.21637022) * xxxy - irc<Scalar>(2.10818512) * xyyy;
    const Scalar dz =
        -irc<Scalar>(4.21637022) * xxxz + irc<Scalar>(2.10818512) * xzzz;
    w = W_ku[28];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.65148372) * xxxy * z -
                     irc<Scalar>(1.82574186) * (xyyy * z + x * yzzz);
    const Scalar dx = irc<Scalar>(10.95445116) * xxyz -
                      irc<Scalar>(1.82574186) * (yyyz + yzzz);
    const Scalar dy = irc<Scalar>(3.65148372) * xxxz -
                      irc<Scalar>(5.47722558) * xyyz -
                      irc<Scalar>(1.82574186) * xzzz;
    const Scalar dz = irc<Scalar>(3.65148372) * xxxy -
                      irc<Scalar>(1.82574186) * xyyy -
                      irc<Scalar>(5.47722558) * xyzz;
    w = W_ku[29];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        -irc<Scalar>(0.55625378) * xxxxy + irc<Scalar>(2.54996401) * xxyyy -
        irc<Scalar>(2.92453987) * xxyzz - irc<Scalar>(0.02492138) * yyyy * y -
        irc<Scalar>(0.05782623) * yyyzz + irc<Scalar>(0.09569378) * yzzz * z;
    const Scalar dx = -irc<Scalar>(2.22501512) * xxxy +
                      irc<Scalar>(5.09992802) * xyyy -
                      irc<Scalar>(5.84907974) * xyzz;
    const Scalar dy =
        -irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(7.64989203) * xxyy -
        irc<Scalar>(2.92453987) * xxzz - irc<Scalar>(0.12460690) * yyyy -
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.09569378) * zzzz;
    const Scalar dz = -irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.11565246) * yyyz +
                      irc<Scalar>(0.38277512) * yzzz;
    w = W_ku[30];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.55625378) * xxxxz + irc<Scalar>(2.92453987) * xxyyz -
        irc<Scalar>(2.54996401) * xxzzz - irc<Scalar>(0.09569378) * yyyy * z +
        irc<Scalar>(0.05782623) * yyzzz + irc<Scalar>(0.02492138) * zzzz * z;
    const Scalar dx = irc<Scalar>(2.22501512) * xxxz +
                      irc<Scalar>(5.84907974) * xyyz -
                      irc<Scalar>(5.09992802) * xzzz;
    const Scalar dy = irc<Scalar>(5.84907974) * xxyz -
                      irc<Scalar>(0.38277512) * yyyz +
                      irc<Scalar>(0.11565246) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.55625378) * xxxx + irc<Scalar>(2.92453987) * xxyy -
        irc<Scalar>(7.64989203) * xxzz - irc<Scalar>(0.09569378) * yyyy +
        irc<Scalar>(0.17347869) * yyzz + irc<Scalar>(0.12460690) * zzzz;
    w = W_ku[31];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(0.02706701) * xxxx * x -
                     irc<Scalar>(1.35335068) * xxx * yy_plus_zz +
                     irc<Scalar>(0.45132312) * x * yyyy_plus_zzzz -
                     irc<Scalar>(4.06747614) * x * yyzz;
    const Scalar dx = irc<Scalar>(0.13533505) * xxxx -
                      irc<Scalar>(4.06005204) * xx * yy_plus_zz +
                      irc<Scalar>(0.45132312) * yyyy_plus_zzzz -
                      irc<Scalar>(4.06747614) * yyzz;
    const Scalar dy = -irc<Scalar>(2.70670136) * xxxy +
                      irc<Scalar>(1.80529248) * xyyy -
                      irc<Scalar>(8.13495228) * xyzz;
    const Scalar dz = -irc<Scalar>(2.70670136) * xxxz +
                      irc<Scalar>(1.80529248) * xzzz -
                      irc<Scalar>(8.13495228) * xyyz;
    w = W_ku[32];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y = irc<Scalar>(3.16227766) * (xyyy * z - x * yzzz);
    const Scalar dx = irc<Scalar>(3.16227766) * (yyyz - yzzz);
    const Scalar dy =
        irc<Scalar>(9.48683298) * xyyz - irc<Scalar>(3.16227766) * xzzz;
    const Scalar dz =
        irc<Scalar>(3.16227766) * xyyy - irc<Scalar>(9.48683298) * xyzz;
    w = W_ku[33];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxy - irc<Scalar>(0.18046516) * xxyyy -
        irc<Scalar>(0.21537191) * xxyzz + irc<Scalar>(0.03149789) * yyyy * y -
        irc<Scalar>(2.96932339) * yyyzz + irc<Scalar>(0.74831340) * yzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxy -
                      irc<Scalar>(0.36093032) * xyyy -
                      irc<Scalar>(0.43074382) * xyzz;
    const Scalar dy =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.54139548) * xxyy -
        irc<Scalar>(0.21537191) * xxzz + irc<Scalar>(0.15748945) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.74831340) * zzzz;
    const Scalar dz = -irc<Scalar>(0.43074382) * xxyz -
                      irc<Scalar>(5.93864678) * yyyz +
                      irc<Scalar>(2.99325360) * yzzz;
    w = W_ku[34];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
  {
    const Scalar Y =
        irc<Scalar>(0.05109884) * xxxxz - irc<Scalar>(0.21537191) * xxyyz -
        irc<Scalar>(0.18046516) * xxzzz + irc<Scalar>(0.74831340) * yyyy * z -
        irc<Scalar>(2.96932339) * yyzzz + irc<Scalar>(0.03149789) * zzzz * z;
    const Scalar dx = irc<Scalar>(0.20439536) * xxxz -
                      irc<Scalar>(0.43074382) * xyyz -
                      irc<Scalar>(0.36093032) * xzzz;
    const Scalar dy = -irc<Scalar>(0.43074382) * xxyz +
                      irc<Scalar>(2.99325360) * yyyz -
                      irc<Scalar>(5.93864678) * yzzz;
    const Scalar dz =
        irc<Scalar>(0.05109884) * xxxx - irc<Scalar>(0.21537191) * xxyy -
        irc<Scalar>(0.54139548) * xxzz + irc<Scalar>(0.74831340) * yyyy -
        irc<Scalar>(8.90797017) * yyzz + irc<Scalar>(0.15748945) * zzzz;
    w = W_ku[35];
    fr += w * Y;
    fa_x += w * (v_x5 * Y + r1 * dx);
    fa_y += w * (v_y5 * Y + r1 * dy);
    fa_z += w * (v_z5 * Y + r1 * dz);
  }
}

}  // namespace TENSORMD::Kernels::Geometric::Irreducible

#endif
