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

#ifndef TENSORMD_KERNELS_GEOMETRIC_IRREDUCIBLE_PGW_HPP
#define TENSORMD_KERNELS_GEOMETRIC_IRREDUCIBLE_PGW_HPP

#include "../kernels_helpers.h"

namespace TENSORMD::Kernels::Geometric::Irreducible {

// clang-format off
//                                          s  p   d   f   g   h
constexpr static double G_km_FLOPs[7] = {0, 0, 5, 15, 29, 47, 69};
constexpr static double W_km_FLOPs[7] = {0, 0, 4, 10, 18, 28, 40};
// clang-format on

template <typename Scalar, const int m_val>
KERNELS_DEVICE_INLINE void contract_G_mkba_otf(const Scalar* __restrict__ P_u,
                                               Scalar* __restrict__ G_m) {
  G_m[0] = P_u[0];

  if constexpr (m_val == 1) return;

  G_m[1] = P_u[1] * P_u[1] + P_u[2] * P_u[2] + P_u[3] * P_u[3];

  if constexpr (m_val == 2) return;

  Scalar G_d = 0.0;
  for (int u = 4; u < 9; u++) {
    G_d += P_u[u] * P_u[u];
  }
  G_m[2] = G_d;

  if constexpr (m_val == 3) return;

  Scalar G_f = 0.0;
  for (int u = 9; u < 16; u++) {
    G_f += P_u[u] * P_u[u];
  }
  G_m[3] = G_f;

  if constexpr (m_val == 4) return;

  Scalar G_g = 0.0;
  for (int u = 16; u < 25; u++) {
    G_g += P_u[u] * P_u[u];
  }
  G_m[4] = G_g;

  if constexpr (m_val == 5) return;

  Scalar G_h = 0.0;
  for (int u = 25; u < 36; u++) {
    G_h += P_u[u] * P_u[u];
  }
  G_m[5] = G_h;
}

template <typename Scalar, const int m_val>
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

  G(1) = P(1) * P(1) + P(2) * P(2) + P(3) * P(3);

  if constexpr (m_val == 2) return;

  Scalar G_d = 0.0;
  for (int u = 4; u < 9; u++) {
    G_d += P(u) * P(u);
  }
  G(2) = G_d;

  if constexpr (m_val == 3) return;

  Scalar G_f = 0.0;
  for (int u = 9; u < 16; u++) {
    G_f += P(u) * P(u);
  }
  G(3) = G_f;

  if constexpr (m_val == 4) return;

  Scalar G_g = 0.0;
  for (int u = 16; u < 25; u++) {
    G_g += P(u) * P(u);
  }
  G(4) = G_g;

  if constexpr (m_val == 5) return;

  Scalar G_h = 0.0;
  for (int u = 25; u < 36; u++) {
    G_h += P(u) * P(u);
  }
  G(5) = G_h;

#undef P
#undef G
}

template <typename Scalar, typename AtlasType, const int m_val>
KERNELS_DEVICE_INLINE void P_u_to_G_kmba_otf(AtlasType* __restrict__ P,
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

  Scalar G_d = 0.0;
  for (int u = 4; u < 9; u++) {
    Scalar p = static_cast<Scalar>(P[u]);
    G_d += p * p;
  }
  G(2) = G_d;

  if constexpr (m_val == 3) return;

  Scalar G_f = 0.0;
  for (int u = 9; u < 16; u++) {
    Scalar p = static_cast<Scalar>(P[u]);
    G_f += p * p;
  }
  G(3) = G_f;

  if constexpr (m_val == 4) return;

  Scalar G_g = 0.0;
  for (int u = 16; u < 25; u++) {
    Scalar p = static_cast<Scalar>(P[u]);
    G_g += p * p;
  }
  G(4) = G_g;

  if constexpr (m_val == 5) return;

  Scalar G_h = 0.0;
  for (int u = 25; u < 36; u++) {
    Scalar p = static_cast<Scalar>(P[u]);
    G_h += p * p;
  }
  G(5) = G_h;

#undef G
}

template <typename Scalar, int m_val>
KERNELS_DEVICE_INLINE void contract_W_dkba_otf(const Scalar* __restrict__ Gp_m,
                                               Scalar* P_u, Scalar* W_u) {
  W_u[0] = Gp_m[0];

  if constexpr (m_val == 1) return;

  Scalar Gp_p = Gp_m[1] * 2.0;
  W_u[1] = Gp_p * P_u[1];
  W_u[2] = Gp_p * P_u[2];
  W_u[3] = Gp_p * P_u[3];

  if constexpr (m_val == 2) return;

  Scalar Gp_d = Gp_m[2] * 2.0;
  W_u[4] = Gp_d * P_u[4];
  W_u[5] = Gp_d * P_u[5];
  W_u[6] = Gp_d * P_u[6];
  W_u[7] = Gp_d * P_u[7];
  W_u[8] = Gp_d * P_u[8];

  if constexpr (m_val == 3) return;

  Scalar Gp_f = Gp_m[3] * 2.0;
  W_u[9] = Gp_f * P_u[9];
  W_u[10] = Gp_f * P_u[10];
  W_u[11] = Gp_f * P_u[11];
  W_u[12] = Gp_f * P_u[12];
  W_u[13] = Gp_f * P_u[13];
  W_u[14] = Gp_f * P_u[14];
  W_u[15] = Gp_f * P_u[15];

  if constexpr (m_val == 4) return;

  Scalar Gp_g = Gp_m[4] * 2.0;
  W_u[16] = Gp_g * P_u[16];
  W_u[17] = Gp_g * P_u[17];
  W_u[18] = Gp_g * P_u[18];
  W_u[19] = Gp_g * P_u[19];
  W_u[20] = Gp_g * P_u[20];
  W_u[21] = Gp_g * P_u[21];
  W_u[22] = Gp_g * P_u[22];
  W_u[23] = Gp_g * P_u[23];
  W_u[24] = Gp_g * P_u[24];

  if constexpr (m_val == 5) return;

  Scalar Gp_h = Gp_m[5] * 2.0;
  W_u[25] = Gp_h * P_u[25];
  W_u[26] = Gp_h * P_u[26];
  W_u[27] = Gp_h * P_u[27];
  W_u[28] = Gp_h * P_u[28];
  W_u[29] = Gp_h * P_u[29];
  W_u[30] = Gp_h * P_u[30];
  W_u[31] = Gp_h * P_u[31];
  W_u[32] = Gp_h * P_u[32];
  W_u[33] = Gp_h * P_u[33];
  W_u[34] = Gp_h * P_u[34];
  W_u[35] = Gp_h * P_u[35];
}

template <typename Scalar, int m_val>
KERNELS_DEVICE_INLINE void contract_W_kdba_otf(
    const Scalar* __restrict__ G_base, Scalar* P_base, Scalar* W_base,
    const int k, const int K_stride) {
  // Define macros for K-continuous access: Base[dim * Stride + k]
#define G(idx) G_base[(idx) * K_stride + k]
#define P(idx) P_base[(idx) * K_stride + k]
#define W(idx) W_base[(idx) * K_stride + k]

  W(0) = G(0);

  if constexpr (m_val == 1) return;

  Scalar Gp_p = G(1) * 2.0;
  W(1) = Gp_p * P(1);
  W(2) = Gp_p * P(2);
  W(3) = Gp_p * P(3);

  if constexpr (m_val == 2) return;

  Scalar Gp_d = G(2) * 2.0;
  W(4) = Gp_d * P(4);
  W(5) = Gp_d * P(5);
  W(6) = Gp_d * P(6);
  W(7) = Gp_d * P(7);
  W(8) = Gp_d * P(8);

  if constexpr (m_val == 3) return;

  Scalar Gp_f = G(3) * 2.0;
  W(9) = Gp_f * P(9);
  W(10) = Gp_f * P(10);
  W(11) = Gp_f * P(11);
  W(12) = Gp_f * P(12);
  W(13) = Gp_f * P(13);
  W(14) = Gp_f * P(14);
  W(15) = Gp_f * P(15);

  if constexpr (m_val == 4) return;

  Scalar Gp_g = G(4) * 2.0;
  W(16) = Gp_g * P(16);
  W(17) = Gp_g * P(17);
  W(18) = Gp_g * P(18);
  W(19) = Gp_g * P(19);
  W(20) = Gp_g * P(20);
  W(21) = Gp_g * P(21);
  W(22) = Gp_g * P(22);
  W(23) = Gp_g * P(23);
  W(24) = Gp_g * P(24);

  if constexpr (m_val == 5) return;

  Scalar Gp_h = G(5) * 2.0;
  W(25) = Gp_h * P(25);
  W(26) = Gp_h * P(26);
  W(27) = Gp_h * P(27);
  W(28) = Gp_h * P(28);
  W(29) = Gp_h * P(29);
  W(30) = Gp_h * P(30);
  W(31) = Gp_h * P(31);
  W(32) = Gp_h * P(32);
  W(33) = Gp_h * P(33);
  W(34) = Gp_h * P(34);
  W(35) = Gp_h * P(35);
#undef W
#undef P
#undef G
}

template <typename Scalar, typename AtlasType, int m_val>
KERNELS_DEVICE_INLINE void P_ku_Gp_km_to_W_u(const Scalar* __restrict__ G_km,
                                             const AtlasType* __restrict__ P_ku,
                                             AtlasType* __restrict__ W,
                                             const int k, const int K_stride) {
#define G(idx) G_km[(idx) * K_stride + k]
#define P(idx) P_ku[(idx) * K_stride + k]
#define W(idx) W[idx]

  W(0) = G(0);

  if constexpr (m_val == 1) return;

  AtlasType two = AtlasType(2.0);
  AtlasType Gp_p = static_cast<AtlasType>(G(1)) * two;
  W(1) = Gp_p * P(1);
  W(2) = Gp_p * P(2);
  W(3) = Gp_p * P(3);

  if constexpr (m_val == 2) return;

  AtlasType Gp_d = static_cast<AtlasType>(G(2)) * two;
  W(4) = Gp_d * P(4);
  W(5) = Gp_d * P(5);
  W(6) = Gp_d * P(6);
  W(7) = Gp_d * P(7);
  W(8) = Gp_d * P(8);

  if constexpr (m_val == 3) return;

  AtlasType Gp_f = static_cast<AtlasType>(G(3)) * two;
  W(9) = Gp_f * P(9);
  W(10) = Gp_f * P(10);
  W(11) = Gp_f * P(11);
  W(12) = Gp_f * P(12);
  W(13) = Gp_f * P(13);
  W(14) = Gp_f * P(14);
  W(15) = Gp_f * P(15);

  if constexpr (m_val == 4) return;

  AtlasType Gp_g = static_cast<AtlasType>(G(4)) * two;
  W(16) = Gp_g * P(16);
  W(17) = Gp_g * P(17);
  W(18) = Gp_g * P(18);
  W(19) = Gp_g * P(19);
  W(20) = Gp_g * P(20);
  W(21) = Gp_g * P(21);
  W(22) = Gp_g * P(22);
  W(23) = Gp_g * P(23);
  W(24) = Gp_g * P(24);

  if constexpr (m_val == 5) return;

  AtlasType Gp_h = static_cast<AtlasType>(G(5)) * two;
  W(25) = Gp_h * P(25);
  W(26) = Gp_h * P(26);
  W(27) = Gp_h * P(27);
  W(28) = Gp_h * P(28);
  W(29) = Gp_h * P(29);
  W(30) = Gp_h * P(30);
  W(31) = Gp_h * P(31);
  W(32) = Gp_h * P(32);
  W(33) = Gp_h * P(33);
  W(34) = Gp_h * P(34);
  W(35) = Gp_h * P(35);
#undef G
#undef P
#undef W
}

template <typename Scalar, typename AtlasType, int m_val>
KERNELS_DEVICE_INLINE void P_u_Gp_km_to_W_u_inplace(
    const Scalar* __restrict__ G_km, AtlasType* P_u, const int k,
    const int K_stride) {
#define G(m) G_km[m * K_stride + k]

  P_u[0] = static_cast<Scalar>(G(0));
  
  if constexpr (m_val == 1) return;

  AtlasType Gp_p = static_cast<AtlasType>(G(1) * AtlasType(2.0));
  P_u[1] *= Gp_p;
  P_u[2] *= Gp_p;
  P_u[3] *= Gp_p;
  
  if constexpr (m_val == 2) return;

  AtlasType Gp_d = static_cast<AtlasType>(G(2) * AtlasType(2.0));
  P_u[4] *= Gp_d;
  P_u[5] *= Gp_d;
  P_u[6] *= Gp_d;
  P_u[7] *= Gp_d;
  P_u[8] *= Gp_d;
  
  if constexpr (m_val == 3) return;

  AtlasType Gp_f = static_cast<AtlasType>(G(3) * AtlasType(2.0));
  P_u[9] *= Gp_f;
  P_u[10] *= Gp_f;
  P_u[11] *= Gp_f;
  P_u[12] *= Gp_f;
  P_u[13] *= Gp_f;
  P_u[14] *= Gp_f;
  P_u[15] *= Gp_f;
  
  if constexpr (m_val == 4) return;

  AtlasType Gp_g = static_cast<AtlasType>(G(4) * AtlasType(2.0));
  P_u[16] *= Gp_g;
  P_u[17] *= Gp_g;
  P_u[18] *= Gp_g;
  P_u[19] *= Gp_g;
  P_u[20] *= Gp_g;
  P_u[21] *= Gp_g;
  P_u[22] *= Gp_g;
  P_u[23] *= Gp_g;
  P_u[24] *= Gp_g;

  if constexpr (m_val == 5) return;

  AtlasType Gp_h = static_cast<AtlasType>(G(5) * AtlasType(2.0));
  P_u[25] *= Gp_h;
  P_u[26] *= Gp_h;
  P_u[27] *= Gp_h;
  P_u[28] *= Gp_h;
  P_u[29] *= Gp_h;
  P_u[30] *= Gp_h;
  P_u[31] *= Gp_h;
  P_u[32] *= Gp_h;
  P_u[33] *= Gp_h;
  P_u[34] *= Gp_h;
  P_u[35] *= Gp_h;
  
#undef G
}

}  // namespace TENSORMD::Kernels::Geometric::Irreducible

#endif
