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

#ifndef TENSORMD_KERNELS_GEOMETRIC_REDUCIBLE_PGW_HPP
#define TENSORMD_KERNELS_GEOMETRIC_REDUCIBLE_PGW_HPP

#include "../kernels_helpers.h"

namespace TENSORMD::Kernels::Geometric::Reducible {

// clang-format off
//                                          s  p   d   f   g    h
constexpr static double G_km_FLOPs[7] = {0, 0, 5, 19, 45, 86, 144};
constexpr static double W_km_FLOPs[7] = {0, 0, 4, 14, 32, 60, 100};
// clang-format on

// -----------------------------------------------------------------------------
// Calculate G(m,k) on the fly: AoS layout, P_dkba, G_mkba
// -----------------------------------------------------------------------------

template <typename Scalar, const int m_val, const bool use_legacy_T>
KERNELS_DEVICE_INLINE void contract_G_mkba_otf(const Scalar* __restrict__ P_d,
                                               Scalar* __restrict__ G_m) {
  G_m[0] = P_d[0];

  if constexpr (m_val == 1) return;

  Scalar p_x = P_d[1] * P_d[1];
  Scalar p_y = P_d[2] * P_d[2];
  Scalar p_z = P_d[3] * P_d[3];

  G_m[1] = p_x + p_y + p_z;

  if constexpr (m_val == 2) return;

  Scalar p_xx = P_d[4] * P_d[4];
  Scalar p_xy = P_d[5] * P_d[5] * 2;
  Scalar p_xz = P_d[6] * P_d[6] * 2;
  Scalar p_yy = P_d[7] * P_d[7];
  Scalar p_yz = P_d[8] * P_d[8] * 2;
  Scalar p_zz = P_d[9] * P_d[9];

  G_m[2] = p_xx + p_xy + p_xz + p_yy + p_yz + p_zz;

  if (use_legacy_T) {
    G_m[2] -= 1.0 / 3.0 * G_m[0] * G_m[0];
  }

  if constexpr (m_val == 3) return;

  Scalar p_xxx = P_d[10] * P_d[10];
  Scalar p_xxy = P_d[11] * P_d[11] * 3;
  Scalar p_xxz = P_d[12] * P_d[12] * 3;
  Scalar p_xyy = P_d[13] * P_d[13] * 3;
  Scalar p_xyz = P_d[14] * P_d[14] * 6;
  Scalar p_xzz = P_d[15] * P_d[15] * 3;
  Scalar p_yyy = P_d[16] * P_d[16];
  Scalar p_yyz = P_d[17] * P_d[17] * 3;
  Scalar p_yzz = P_d[18] * P_d[18] * 3;
  Scalar p_zzz = P_d[19] * P_d[19];

  G_m[3] = p_xxx + p_xxy + p_xxz + p_xyy + p_xyz + p_xzz + p_yyy + p_yyz +
           p_yzz + p_zzz;

  if (use_legacy_T) {
    G_m[3] -= 3.0 / 5.0 * G_m[1];
  }

  if constexpr (m_val == 4) return;

  Scalar p_xxxx = P_d[20] * P_d[20];
  Scalar p_xxxy = P_d[21] * P_d[21] * 4;
  Scalar p_xxxz = P_d[22] * P_d[22] * 4;
  Scalar p_xxyy = P_d[23] * P_d[23] * 6;
  Scalar p_xxyz = P_d[24] * P_d[24] * 12;
  Scalar p_xxzz = P_d[25] * P_d[25] * 6;
  Scalar p_xyyy = P_d[26] * P_d[26] * 4;
  Scalar p_xyyz = P_d[27] * P_d[27] * 12;
  Scalar p_xyzz = P_d[28] * P_d[28] * 12;
  Scalar p_xzzz = P_d[29] * P_d[29] * 4;
  Scalar p_yyyy = P_d[30] * P_d[30];
  Scalar p_yyyz = P_d[31] * P_d[31] * 4;
  Scalar p_yyzz = P_d[32] * P_d[32] * 6;
  Scalar p_yzzz = P_d[33] * P_d[33] * 4;
  Scalar p_zzzz = P_d[34] * P_d[34];

  G_m[4] = p_xxxx + p_xxxy + p_xxxz + p_xxyy + p_xxyz + p_xxzz + p_xyyy +
           p_xyyz + p_xyzz + p_xzzz + p_yyyy + p_yyyz + p_yyzz + p_yzzz +
           p_zzzz;

  if constexpr (m_val == 5) return;

  Scalar p_xxxxx = P_d[35] * P_d[35];
  Scalar p_xxxxy = P_d[36] * P_d[36] * 5;
  Scalar p_xxxxz = P_d[37] * P_d[37] * 5;
  Scalar p_xxxyy = P_d[38] * P_d[38] * 10;
  Scalar p_xxxyz = P_d[39] * P_d[39] * 20;
  Scalar p_xxxzz = P_d[40] * P_d[40] * 10;
  Scalar p_xxyyy = P_d[41] * P_d[41] * 10;
  Scalar p_xxyyz = P_d[42] * P_d[42] * 30;
  Scalar p_xxyzz = P_d[43] * P_d[43] * 30;
  Scalar p_xxzzz = P_d[44] * P_d[44] * 10;
  Scalar p_xyyyy = P_d[45] * P_d[45] * 5;
  Scalar p_xyyyz = P_d[46] * P_d[46] * 20;
  Scalar p_xyyzz = P_d[47] * P_d[47] * 30;
  Scalar p_xyzzz = P_d[48] * P_d[48] * 20;
  Scalar p_xzzzz = P_d[49] * P_d[49] * 5;
  Scalar p_yyyyy = P_d[50] * P_d[50];
  Scalar p_yyyyz = P_d[51] * P_d[51] * 5;
  Scalar p_yyyzz = P_d[52] * P_d[52] * 10;
  Scalar p_yyzzz = P_d[53] * P_d[53] * 10;
  Scalar p_yzzzz = P_d[54] * P_d[54] * 5;
  Scalar p_zzzzz = P_d[55] * P_d[55];

  G_m[5] = p_xxxxx + p_xxxxy + p_xxxxz + p_xxxyy + p_xxxyz + p_xxxzz + p_xxyyy +
           p_xxyyz + p_xxyzz + p_xxzzz + p_xyyyy + p_xyyyz + p_xyyzz + p_xyzzz +
           p_xzzzz + p_yyyyy + p_yyyyz + p_yyyzz + p_yyzzz + p_yzzzz + p_zzzzz;
}

// -----------------------------------------------------------------------------
// Calculate G(k,m) on the fly: SoA layout, P_kdba, G_kmba
// -----------------------------------------------------------------------------

template <typename Scalar, int m_val, bool use_legacy_T>
KERNELS_DEVICE_INLINE void contract_G_kmba_otf(
    const Scalar* __restrict__ P_base, Scalar* __restrict__ G_base, const int k,
    const int K_stride) {
  // Helper macro to access P in SoA layout: P[dim * K + k]
  // This allows the compiler to see that for adjacent 'k', memory is
  // contiguous.
#define P(idx) P_base[(idx) * K_stride + k]

  // Helper for G in SoA layout (assuming G is also [M, K] layout)
#define G(idx) G_base[(idx) * K_stride + k]

  G(0) = P(0);

  if constexpr (m_val == 1) return;

  G(1) = P(1) * P(1);
  G(1) += P(2) * P(2);
  G(1) += P(3) * P(3);

  if constexpr (m_val == 2) return;

  G(2) = P(4) * P(4);
  G(2) += P(5) * P(5) * 2;
  G(2) += P(6) * P(6) * 2;
  G(2) += P(7) * P(7);
  G(2) += P(8) * P(8) * 2;
  G(2) += P(9) * P(9);

  if (use_legacy_T) {
    G(2) -= 1.0 / 3.0 * G(0) * G(0);
  }

  if constexpr (m_val == 3) return;

  G(3) = P(10) * P(10);
  G(3) += P(11) * P(11) * 3;
  G(3) += P(12) * P(12) * 3;
  G(3) += P(13) * P(13) * 3;
  G(3) += P(14) * P(14) * 6;
  G(3) += P(15) * P(15) * 3;
  G(3) += P(16) * P(16);
  G(3) += P(17) * P(17) * 3;
  G(3) += P(18) * P(18) * 3;
  G(3) += P(19) * P(19);

  if (use_legacy_T) {
    G(3) -= 3.0 / 5.0 * G(1);
  }

  if constexpr (m_val == 4) return;

  G(4) = P(20) * P(20);
  G(4) += P(21) * P(21) * 4;
  G(4) += P(22) * P(22) * 4;
  G(4) += P(23) * P(23) * 6;
  G(4) += P(24) * P(24) * 12;
  G(4) += P(25) * P(25) * 6;
  G(4) += P(26) * P(26) * 4;
  G(4) += P(27) * P(27) * 12;
  G(4) += P(28) * P(28) * 12;
  G(4) += P(29) * P(29) * 4;
  G(4) += P(30) * P(30);
  G(4) += P(31) * P(31) * 4;
  G(4) += P(32) * P(32) * 6;
  G(4) += P(33) * P(33) * 4;
  G(4) += P(34) * P(34);

  if constexpr (m_val == 5) return;

  G(5) = P(35) * P(35);
  G(5) += P(36) * P(36) * 5;
  G(5) += P(37) * P(37) * 5;
  G(5) += P(38) * P(38) * 10;
  G(5) += P(39) * P(39) * 20;
  G(5) += P(40) * P(40) * 10;
  G(5) += P(41) * P(41) * 10;
  G(5) += P(42) * P(42) * 30;
  G(5) += P(43) * P(43) * 30;
  G(5) += P(44) * P(44) * 10;
  G(5) += P(45) * P(45) * 5;
  G(5) += P(46) * P(46) * 20;
  G(5) += P(47) * P(47) * 30;
  G(5) += P(48) * P(48) * 20;
  G(5) += P(49) * P(49) * 5;
  G(5) += P(50) * P(50);
  G(5) += P(51) * P(51) * 5;
  G(5) += P(52) * P(52) * 10;
  G(5) += P(53) * P(53) * 10;
  G(5) += P(54) * P(54) * 5;
  G(5) += P(55) * P(55);

#undef P
#undef G
}

template <typename Scalar, typename AtlasType, int m_val, bool use_legacy_T>
KERNELS_DEVICE_INLINE void P_d_to_G_kmba_otf(AtlasType* __restrict__ P,
                                             Scalar* __restrict__ G_base,
                                             const int k, const int K_stride) {
  // Helper for G in SoA layout (assuming G is also [M, K] layout)
#define G(idx) G_base[(idx) * K_stride + k]

  G(0) = static_cast<Scalar>(P[0]);

  if constexpr (m_val == 1) return;

  Scalar p1 = static_cast<Scalar>(P[1]);
  Scalar p2 = static_cast<Scalar>(P[2]);
  Scalar p3 = static_cast<Scalar>(P[3]);
  G(1) = p1 * p1 + p2 * p2 + p3 * p3;

  if constexpr (m_val == 2) return;

  Scalar p4 = static_cast<Scalar>(P[4]);
  Scalar p5 = static_cast<Scalar>(P[5]);
  Scalar p6 = static_cast<Scalar>(P[6]);
  Scalar p7 = static_cast<Scalar>(P[7]);
  Scalar p8 = static_cast<Scalar>(P[8]);
  Scalar p9 = static_cast<Scalar>(P[9]);
  G(2) = p4 * p4;
  G(2) += p5 * p5 * 2;
  G(2) += p6 * p6 * 2;
  G(2) += p7 * p7;
  G(2) += p8 * p8 * 2;
  G(2) += p9 * p9;

  if (use_legacy_T) {
    G(2) -= static_cast<Scalar>(1.0) / static_cast<Scalar>(3.0) * G(0) * G(0);
  }

  if constexpr (m_val == 3) return;

  Scalar p10 = static_cast<Scalar>(P[10]);
  Scalar p11 = static_cast<Scalar>(P[11]);
  Scalar p12 = static_cast<Scalar>(P[12]);
  Scalar p13 = static_cast<Scalar>(P[13]);
  Scalar p14 = static_cast<Scalar>(P[14]);
  Scalar p15 = static_cast<Scalar>(P[15]);
  Scalar p16 = static_cast<Scalar>(P[16]);
  Scalar p17 = static_cast<Scalar>(P[17]);
  Scalar p18 = static_cast<Scalar>(P[18]);
  Scalar p19 = static_cast<Scalar>(P[19]);

  G(3) = p10 * p10;
  G(3) += p11 * p11 * static_cast<Scalar>(3);
  G(3) += p12 * p12 * static_cast<Scalar>(3);
  G(3) += p13 * p13 * static_cast<Scalar>(3);
  G(3) += p14 * p14 * static_cast<Scalar>(6);
  G(3) += p15 * p15 * static_cast<Scalar>(3);
  G(3) += p16 * p16;
  G(3) += p17 * p17 * static_cast<Scalar>(3);
  G(3) += p18 * p18 * static_cast<Scalar>(3);
  G(3) += p19 * p19;

  if (use_legacy_T) {
    G(3) -= static_cast<Scalar>(0.6) * G(1);
  }

  if constexpr (m_val == 4) return;

  Scalar p20 = static_cast<Scalar>(P[20]);
  Scalar p21 = static_cast<Scalar>(P[21]);
  Scalar p22 = static_cast<Scalar>(P[22]);
  Scalar p23 = static_cast<Scalar>(P[23]);
  Scalar p24 = static_cast<Scalar>(P[24]);
  Scalar p25 = static_cast<Scalar>(P[25]);
  Scalar p26 = static_cast<Scalar>(P[26]);
  Scalar p27 = static_cast<Scalar>(P[27]);
  Scalar p28 = static_cast<Scalar>(P[28]);
  Scalar p29 = static_cast<Scalar>(P[29]);
  Scalar p30 = static_cast<Scalar>(P[30]);
  Scalar p31 = static_cast<Scalar>(P[31]);
  Scalar p32 = static_cast<Scalar>(P[32]);
  Scalar p33 = static_cast<Scalar>(P[33]);
  Scalar p34 = static_cast<Scalar>(P[34]);

  G(4) = p20 * p20;
  G(4) += p21 * p21 * static_cast<Scalar>(4);
  G(4) += p22 * p22 * static_cast<Scalar>(4);
  G(4) += p23 * p23 * static_cast<Scalar>(6);
  G(4) += p24 * p24 * static_cast<Scalar>(12);
  G(4) += p25 * p25 * static_cast<Scalar>(6);
  G(4) += p26 * p26 * static_cast<Scalar>(4);
  G(4) += p27 * p27 * static_cast<Scalar>(12);
  G(4) += p28 * p28 * static_cast<Scalar>(12);
  G(4) += p29 * p29 * static_cast<Scalar>(4);
  G(4) += p30 * p30;
  G(4) += p31 * p31 * static_cast<Scalar>(4);
  G(4) += p32 * p32 * static_cast<Scalar>(6);
  G(4) += p33 * p33 * static_cast<Scalar>(4);
  G(4) += p34 * p34;

  if constexpr (m_val == 5) return;

  Scalar p35 = static_cast<Scalar>(P[35]);
  Scalar p36 = static_cast<Scalar>(P[36]);
  Scalar p37 = static_cast<Scalar>(P[37]);
  Scalar p38 = static_cast<Scalar>(P[38]);
  Scalar p39 = static_cast<Scalar>(P[39]);
  Scalar p40 = static_cast<Scalar>(P[40]);
  Scalar p41 = static_cast<Scalar>(P[41]);
  Scalar p42 = static_cast<Scalar>(P[42]);
  Scalar p43 = static_cast<Scalar>(P[43]);
  Scalar p44 = static_cast<Scalar>(P[44]);
  Scalar p45 = static_cast<Scalar>(P[45]);
  Scalar p46 = static_cast<Scalar>(P[46]);
  Scalar p47 = static_cast<Scalar>(P[47]);
  Scalar p48 = static_cast<Scalar>(P[48]);
  Scalar p49 = static_cast<Scalar>(P[49]);
  Scalar p50 = static_cast<Scalar>(P[50]);
  Scalar p51 = static_cast<Scalar>(P[51]);
  Scalar p52 = static_cast<Scalar>(P[52]);
  Scalar p53 = static_cast<Scalar>(P[53]);
  Scalar p54 = static_cast<Scalar>(P[54]);
  Scalar p55 = static_cast<Scalar>(P[55]);

  G(5) = p35 * p35;
  G(5) += p36 * p36 * static_cast<Scalar>(5);
  G(5) += p37 * p37 * static_cast<Scalar>(5);
  G(5) += p38 * p38 * static_cast<Scalar>(10);
  G(5) += p39 * p39 * static_cast<Scalar>(20);
  G(5) += p40 * p40 * static_cast<Scalar>(10);
  G(5) += p41 * p41 * static_cast<Scalar>(10);
  G(5) += p42 * p42 * static_cast<Scalar>(30);
  G(5) += p43 * p43 * static_cast<Scalar>(30);
  G(5) += p44 * p44 * static_cast<Scalar>(10);
  G(5) += p45 * p45 * static_cast<Scalar>(5);
  G(5) += p46 * p46 * static_cast<Scalar>(20);
  G(5) += p47 * p47 * static_cast<Scalar>(30);
  G(5) += p48 * p48 * static_cast<Scalar>(20);
  G(5) += p49 * p49 * static_cast<Scalar>(5);
  G(5) += p50 * p50;
  G(5) += p51 * p51 * static_cast<Scalar>(5);
  G(5) += p52 * p52 * static_cast<Scalar>(10);
  G(5) += p53 * p53 * static_cast<Scalar>(10);
  G(5) += p54 * p54 * static_cast<Scalar>(5);
  G(5) += p55 * p55;

#undef P
#undef G
}

// -----------------------------------------------------------------------------
// Calculate W(d,k) on the fly: AoS layout, P_dkba, dEdG_mkba
// -----------------------------------------------------------------------------

template <typename Scalar, int m_val, const bool use_legacy_T>
KERNELS_DEVICE_INLINE void contract_W_dkba_otf(const Scalar* __restrict__ Gp_m,
                                               Scalar* P_d, Scalar* W_d) {
  if (use_legacy_T && m_val > 2) {
    W_d[0] = Gp_m[0] - P_d[0] * Gp_m[2] * 2.0 / 3.0;
  } else {
    W_d[0] = Gp_m[0];
  }

  if constexpr (m_val == 1) return;

  Scalar Gp_p = Gp_m[1] * 2.0;

  if (use_legacy_T && m_val > 3) {
    W_d[1] = P_d[1] * (Gp_p - Gp_m[3] * 6.0 / 5.0);
    W_d[2] = P_d[2] * (Gp_p - Gp_m[3] * 6.0 / 5.0);
    W_d[3] = P_d[3] * (Gp_p - Gp_m[3] * 6.0 / 5.0);
  } else {
    W_d[1] = Gp_p * P_d[1];
    W_d[2] = Gp_p * P_d[2];
    W_d[3] = Gp_p * P_d[3];
  }

  if constexpr (m_val == 2) return;

  Scalar Gp_d = Gp_m[2] * 2.0;

  W_d[4] = Gp_d * P_d[4];
  W_d[5] = Gp_d * P_d[5] * 2;
  W_d[6] = Gp_d * P_d[6] * 2;
  W_d[7] = Gp_d * P_d[7];
  W_d[8] = Gp_d * P_d[8] * 2;
  W_d[9] = Gp_d * P_d[9];

  if constexpr (m_val == 3) return;

  Scalar Gp_f = Gp_m[3] * 2.0;

  W_d[10] = Gp_f * P_d[10];
  W_d[11] = Gp_f * P_d[11] * 3;
  W_d[12] = Gp_f * P_d[12] * 3;
  W_d[13] = Gp_f * P_d[13] * 3;
  W_d[14] = Gp_f * P_d[14] * 6;
  W_d[15] = Gp_f * P_d[15] * 3;
  W_d[16] = Gp_f * P_d[16];
  W_d[17] = Gp_f * P_d[17] * 3;
  W_d[18] = Gp_f * P_d[18] * 3;
  W_d[19] = Gp_f * P_d[19];

  if constexpr (m_val == 4) return;

  Scalar Gp_g = Gp_m[4] * 2.0;

  W_d[20] = Gp_g * P_d[20];
  W_d[21] = Gp_g * P_d[21] * 4;
  W_d[22] = Gp_g * P_d[22] * 4;
  W_d[23] = Gp_g * P_d[23] * 6;
  W_d[24] = Gp_g * P_d[24] * 12;
  W_d[25] = Gp_g * P_d[25] * 6;
  W_d[26] = Gp_g * P_d[26] * 4;
  W_d[27] = Gp_g * P_d[27] * 12;
  W_d[28] = Gp_g * P_d[28] * 12;
  W_d[29] = Gp_g * P_d[29] * 4;
  W_d[30] = Gp_g * P_d[30];
  W_d[31] = Gp_g * P_d[31] * 4;
  W_d[32] = Gp_g * P_d[32] * 6;
  W_d[33] = Gp_g * P_d[33] * 4;
  W_d[34] = Gp_g * P_d[34];

  if constexpr (m_val == 5) return;

  Scalar Gp_h = Gp_m[5] * 2.0;

  W_d[35] = Gp_h * P_d[35];
  W_d[36] = Gp_h * P_d[36] * 5;
  W_d[37] = Gp_h * P_d[37] * 5;
  W_d[38] = Gp_h * P_d[38] * 10;
  W_d[39] = Gp_h * P_d[39] * 20;
  W_d[40] = Gp_h * P_d[40] * 10;
  W_d[41] = Gp_h * P_d[41] * 10;
  W_d[42] = Gp_h * P_d[42] * 30;
  W_d[43] = Gp_h * P_d[43] * 30;
  W_d[44] = Gp_h * P_d[44] * 10;
  W_d[45] = Gp_h * P_d[45] * 5;
  W_d[46] = Gp_h * P_d[46] * 20;
  W_d[47] = Gp_h * P_d[47] * 30;
  W_d[48] = Gp_h * P_d[48] * 20;
  W_d[49] = Gp_h * P_d[49] * 5;
  W_d[50] = Gp_h * P_d[50];
  W_d[51] = Gp_h * P_d[51] * 5;
  W_d[52] = Gp_h * P_d[52] * 10;
  W_d[53] = Gp_h * P_d[53] * 10;
  W_d[54] = Gp_h * P_d[54] * 5;
  W_d[55] = Gp_h * P_d[55];
}

// -----------------------------------------------------------------------------
// Calculate W(k,d) on the fly: SoA layout, P_kdba, dEdG_kmba
// -----------------------------------------------------------------------------

template <typename Scalar, int m_val, const bool use_legacy_T>
KERNELS_DEVICE_INLINE void contract_W_kdba_otf(
    const Scalar* __restrict__ G_base, Scalar* P_base, Scalar* W_base,
    const int k, const int K_stride) {
  // Define macros for K-continuous access: Base[dim * Stride + k]
#define G(idx) G_base[(idx) * K_stride + k]
#define P(idx) P_base[(idx) * K_stride + k]
#define W(idx) W_base[(idx) * K_stride + k]

  // --- d = 0 ---
  if (use_legacy_T && m_val > 2) {
    W(0) = G(0) - P(0) * G(2) * (2.0 / 3.0);
  } else {
    W(0) = G(0);
  }

  if constexpr (m_val == 1) return;

  // --- d = 1, 2, 3 (p-shell) ---
  Scalar Gp_p = G(1) * 2.0;

  if (use_legacy_T && m_val > 3) {
    Scalar factor = Gp_p - G(3) * (6.0 / 5.0);
    W(1) = P(1) * factor;
    W(2) = P(2) * factor;
    W(3) = P(3) * factor;
  } else {
    W(1) = Gp_p * P(1);
    W(2) = Gp_p * P(2);
    W(3) = Gp_p * P(3);
  }

  if constexpr (m_val == 2) return;

  // --- d = 4..9 (d-shell) ---
  Scalar Gp_d = G(2) * 2.0;

  W(4) = Gp_d * P(4);
  W(5) = Gp_d * P(5) * 2;
  W(6) = Gp_d * P(6) * 2;
  W(7) = Gp_d * P(7);
  W(8) = Gp_d * P(8) * 2;
  W(9) = Gp_d * P(9);

  if constexpr (m_val == 3) return;

  // --- d = 10..19 (f-shell) ---
  Scalar Gp_f = G(3) * 2.0;

  W(10) = Gp_f * P(10);
  W(11) = Gp_f * P(11) * 3;
  W(12) = Gp_f * P(12) * 3;
  W(13) = Gp_f * P(13) * 3;
  W(14) = Gp_f * P(14) * 6;
  W(15) = Gp_f * P(15) * 3;
  W(16) = Gp_f * P(16);
  W(17) = Gp_f * P(17) * 3;
  W(18) = Gp_f * P(18) * 3;
  W(19) = Gp_f * P(19);

  if constexpr (m_val == 4) return;

  // --- d = 20..34 (g-shell) ---
  Scalar Gp_g = G(4) * 2.0;

  W(20) = Gp_g * P(20);
  W(21) = Gp_g * P(21) * 4;
  W(22) = Gp_g * P(22) * 4;
  W(23) = Gp_g * P(23) * 6;
  W(24) = Gp_g * P(24) * 12;
  W(25) = Gp_g * P(25) * 6;
  W(26) = Gp_g * P(26) * 4;
  W(27) = Gp_g * P(27) * 12;
  W(28) = Gp_g * P(28) * 12;
  W(29) = Gp_g * P(29) * 4;
  W(30) = Gp_g * P(30);
  W(31) = Gp_g * P(31) * 4;
  W(32) = Gp_g * P(32) * 6;
  W(33) = Gp_g * P(33) * 4;
  W(34) = Gp_g * P(34);

  if constexpr (m_val == 5) return;

  // --- d = 35..55 (h-shell) ---
  Scalar Gp_h = G(5) * 2.0;

  W(35) = Gp_h * P(35);
  W(36) = Gp_h * P(36) * 5;
  W(37) = Gp_h * P(37) * 5;
  W(38) = Gp_h * P(38) * 10;
  W(39) = Gp_h * P(39) * 20;
  W(40) = Gp_h * P(40) * 10;
  W(41) = Gp_h * P(41) * 10;
  W(42) = Gp_h * P(42) * 30;
  W(43) = Gp_h * P(43) * 30;
  W(44) = Gp_h * P(44) * 10;
  W(45) = Gp_h * P(45) * 5;
  W(46) = Gp_h * P(46) * 20;
  W(47) = Gp_h * P(47) * 30;
  W(48) = Gp_h * P(48) * 20;
  W(49) = Gp_h * P(49) * 5;
  W(50) = Gp_h * P(50);
  W(51) = Gp_h * P(51) * 5;
  W(52) = Gp_h * P(52) * 10;
  W(53) = Gp_h * P(53) * 10;
  W(54) = Gp_h * P(54) * 5;
  W(55) = Gp_h * P(55);

#undef G
#undef P
#undef W
}

template <typename Scalar, typename AtlasType, int m_val>
KERNELS_DEVICE_INLINE void P_kd_Gp_km_to_W_d(const Scalar* __restrict__ G_km,
                                             const AtlasType* __restrict__ P_kd,
                                             AtlasType* __restrict__ W,
                                             const int k, const int K_stride) {
#define G(m) G_km[m * K_stride + k]
#define P(d) P_kd[d * K_stride + k]
#define W(d) W[d]

  W(0) = G(0);

  if constexpr (m_val == 1) return;

  AtlasType Gp_p = static_cast<AtlasType>(G(1) * AtlasType(2.0));
  W(1) = P(1) * Gp_p;
  W(2) = P(2) * Gp_p;
  W(3) = P(3) * Gp_p;

  if constexpr (m_val == 2) return;

  AtlasType Gp_d = static_cast<AtlasType>(G(2) * AtlasType(2.0));
  W(4) = P(4) * Gp_d;
  W(5) = P(5) * Gp_d * AtlasType(2.0);
  W(6) = P(6) * Gp_d * AtlasType(2.0);
  W(7) = P(7) * Gp_d;
  W(8) = P(8) * Gp_d * AtlasType(2.0);
  W(9) = P(9) * Gp_d;

  if constexpr (m_val == 3) return;

  AtlasType Gp_f = static_cast<AtlasType>(G(3) * AtlasType(2.0));
  W(10) = P(10) * Gp_f;
  W(11) = P(11) * Gp_f * AtlasType(3.0);
  W(12) = P(12) * Gp_f * AtlasType(3.0);
  W(13) = P(13) * Gp_f * AtlasType(3.0);
  W(14) = P(14) * Gp_f * AtlasType(6.0);
  W(15) = P(15) * Gp_f * AtlasType(3.0);
  W(16) = P(16) * Gp_f;
  W(17) = P(17) * Gp_f * AtlasType(3.0);
  W(18) = P(18) * Gp_f * AtlasType(3.0);
  W(19) = P(19) * Gp_f;

  if constexpr (m_val == 4) return;

  AtlasType Gp_g = static_cast<AtlasType>(G(4) * AtlasType(2.0));
  W(20) = P(20) * Gp_g;
  W(21) = P(21) * Gp_g * AtlasType(4.0);
  W(22) = P(22) * Gp_g * AtlasType(4.0);
  W(23) = P(23) * Gp_g * AtlasType(6.0);
  W(24) = P(24) * Gp_g * AtlasType(12.0);
  W(25) = P(25) * Gp_g * AtlasType(6.0);
  W(26) = P(26) * Gp_g * AtlasType(4.0);
  W(27) = P(27) * Gp_g * AtlasType(12.0);
  W(28) = P(28) * Gp_g * AtlasType(12.0);
  W(29) = P(29) * Gp_g * AtlasType(4.0);
  W(30) = P(30) * Gp_g;
  W(31) = P(31) * Gp_g * AtlasType(4.0);
  W(32) = P(32) * Gp_g * AtlasType(6.0);
  W(33) = P(33) * Gp_g * AtlasType(4.0);
  W(34) = P(34) * Gp_g;

  if constexpr (m_val == 5) return;

  AtlasType Gp_h = static_cast<AtlasType>(G(5) * AtlasType(2.0));
  W(35) = P(35) * Gp_h;
  W(36) = P(36) * Gp_h * AtlasType(5.0);
  W(37) = P(37) * Gp_h * AtlasType(5.0);
  W(38) = P(38) * Gp_h * AtlasType(10.0);
  W(39) = P(39) * Gp_h * AtlasType(20.0);
  W(40) = P(40) * Gp_h * AtlasType(10.0);
  W(41) = P(41) * Gp_h * AtlasType(10.0);
  W(42) = P(42) * Gp_h * AtlasType(30.0);
  W(43) = P(43) * Gp_h * AtlasType(30.0);
  W(44) = P(44) * Gp_h * AtlasType(10.0);
  W(45) = P(45) * Gp_h * AtlasType(5.0);
  W(46) = P(46) * Gp_h * AtlasType(20.0);
  W(47) = P(47) * Gp_h * AtlasType(30.0);
  W(48) = P(48) * Gp_h * AtlasType(20.0);
  W(49) = P(49) * Gp_h * AtlasType(5.0);
  W(50) = P(50) * Gp_h;
  W(51) = P(51) * Gp_h * AtlasType(5.0);
  W(52) = P(52) * Gp_h * AtlasType(10.0);
  W(53) = P(53) * Gp_h * AtlasType(10.0);
  W(54) = P(54) * Gp_h * AtlasType(5.0);
  W(55) = P(55) * Gp_h;

#undef P
#undef W
#undef G
}

template <typename Scalar, typename AtlasType, int m_val>
KERNELS_DEVICE_INLINE void P_d_Gp_km_to_W_d_inplace(
    const Scalar* __restrict__ G_km, AtlasType* P_d, const int k,
    const int K_stride) {
#define G(m) G_km[m * K_stride + k]

  P_d[0] = G(0);

  if constexpr (m_val == 1) return;

  AtlasType Gp_p = static_cast<AtlasType>(G(1) * AtlasType(2.0));
  P_d[1] *= Gp_p;
  P_d[2] *= Gp_p;
  P_d[3] *= Gp_p;

  if constexpr (m_val == 2) return;

  AtlasType Gp_d = static_cast<AtlasType>(G(2) * AtlasType(2.0));
  P_d[4] *= Gp_d;
  P_d[5] *= Gp_d * AtlasType(2.0);
  P_d[6] *= Gp_d * AtlasType(2.0);
  P_d[7] *= Gp_d;
  P_d[8] *= Gp_d * AtlasType(2.0);
  P_d[9] *= Gp_d;

  if constexpr (m_val == 3) return;

  AtlasType Gp_f = static_cast<AtlasType>(G(3) * AtlasType(2.0));
  P_d[10] *= Gp_f;
  P_d[11] *= Gp_f * AtlasType(3.0);
  P_d[12] *= Gp_f * AtlasType(3.0);
  P_d[13] *= Gp_f * AtlasType(3.0);
  P_d[14] *= Gp_f * AtlasType(6.0);
  P_d[15] *= Gp_f * AtlasType(3.0);
  P_d[16] *= Gp_f;
  P_d[17] *= Gp_f * AtlasType(3.0);
  P_d[18] *= Gp_f * AtlasType(3.0);
  P_d[19] *= Gp_f;

  if constexpr (m_val == 4) return;

  AtlasType Gp_g = static_cast<AtlasType>(G(4) * AtlasType(2.0));
  P_d[20] *= Gp_g;
  P_d[21] *= Gp_g * AtlasType(4.0);
  P_d[22] *= Gp_g * AtlasType(4.0);
  P_d[23] *= Gp_g * AtlasType(6.0);
  P_d[24] *= Gp_g * AtlasType(12.0);
  P_d[25] *= Gp_g * AtlasType(6.0);
  P_d[26] *= Gp_g * AtlasType(4.0);
  P_d[27] *= Gp_g * AtlasType(12.0);
  P_d[28] *= Gp_g * AtlasType(12.0);
  P_d[29] *= Gp_g * AtlasType(4.0);
  P_d[30] *= Gp_g;
  P_d[31] *= Gp_g * AtlasType(4.0);
  P_d[32] *= Gp_g * AtlasType(6.0);
  P_d[33] *= Gp_g * AtlasType(4.0);
  P_d[34] *= Gp_g;

  if constexpr (m_val == 5) return;

  AtlasType Gp_h = static_cast<AtlasType>(G(5) * AtlasType(2.0));
  P_d[35] *= Gp_h;
  P_d[36] *= Gp_h * AtlasType(5.0);
  P_d[37] *= Gp_h * AtlasType(5.0);
  P_d[38] *= Gp_h * AtlasType(10.0);
  P_d[39] *= Gp_h * AtlasType(20.0);
  P_d[40] *= Gp_h * AtlasType(10.0);
  P_d[41] *= Gp_h * AtlasType(10.0);
  P_d[42] *= Gp_h * AtlasType(30.0);
  P_d[43] *= Gp_h * AtlasType(30.0);
  P_d[44] *= Gp_h * AtlasType(10.0);
  P_d[45] *= Gp_h * AtlasType(5.0);
  P_d[46] *= Gp_h * AtlasType(20.0);
  P_d[47] *= Gp_h * AtlasType(30.0);
  P_d[48] *= Gp_h * AtlasType(20.0);
  P_d[49] *= Gp_h * AtlasType(5.0);
  P_d[50] *= Gp_h;
  P_d[51] *= Gp_h * AtlasType(5.0);
  P_d[52] *= Gp_h * AtlasType(10.0);
  P_d[53] *= Gp_h * AtlasType(10.0);
  P_d[54] *= Gp_h * AtlasType(5.0);
  P_d[55] *= Gp_h;

#undef G
}

}  // namespace TENSORMD::Kernels::Geometric::Reducible

#endif
