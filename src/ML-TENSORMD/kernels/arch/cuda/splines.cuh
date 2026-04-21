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

#ifndef TENSORMD_KERNELS_ARCH_CUDA_SPLINES_CUH
#define TENSORMD_KERNELS_ARCH_CUDA_SPLINES_CUH

namespace TENSORMD::Kernels::Spline::GPU {

// ----------------------------------------------------------------------------
// Cubic Spline Interpolation Kernels
// ----------------------------------------------------------------------------

// Mixed-precision cubic spline kernel (only evaluate value)
template <typename Scalar, typename AtlasType>
__device__ __forceinline__ void eval_cubic_spline_single(
    Scalar r_val, int t, int u, int U,
    const AtlasType* __restrict__ params_atlas, int grid_size, int atlas_stride,
    Scalar x0, Scalar dx, Scalar xdx, AtlasType& h_u, AtlasType& dh_u) {
  const AtlasType* params = params_atlas + t * atlas_stride;
  constexpr AtlasType two = static_cast<AtlasType>(2.0);
  constexpr AtlasType three = static_cast<AtlasType>(3.0);

  Scalar dr_x0 = r_val - x0;
  int i = static_cast<int>(dr_x0 * xdx);
  i = max(0, min(grid_size - 2, i));
  Scalar q_exact = fma(-static_cast<Scalar>(i), dx, dr_x0);
  AtlasType q = static_cast<AtlasType>(q_exact);

  AtlasType c0, c1, c2, c3;

  if constexpr (std::is_same_v<AtlasType, float>) {
    // Cast the start of this grid interval to float4
    const float4* interval_f4 =
        reinterpret_cast<const float4*>(params + i * U * 4);

    // Load 16 bytes (c0, c1, c2, c3) in ONE instruction
    float4 coeffs = interval_f4[u];
    c0 = coeffs.x;
    c1 = coeffs.y;
    c2 = coeffs.z;
    c3 = coeffs.w;
  } else if constexpr (std::is_same_v<AtlasType, double>) {
    // Cast the start of this grid interval to double2
    const double2* interval_d2 =
        reinterpret_cast<const double2*>(params + i * U * 4);

    // Load 32 bytes in TWO 128-bit instructions
    double2 c01 = interval_d2[u * 2 + 0];
    double2 c23 = interval_d2[u * 2 + 1];
    c0 = c01.x;
    c1 = c01.y;
    c2 = c23.x;
    c3 = c23.y;
  }
  h_u = c0 + q * (c1 + q * (c2 + q * c3));
  dh_u = c1 + q * (two * c2 + three * c3 * q);
}

template <typename Scalar>
__device__ __forceinline__ void eval_cubic_spline_single(
    int i, Scalar q, int t, int u, int U,
    const Scalar* __restrict__ params_atlas, int atlas_stride, Scalar& h_u,
    Scalar& dh_u) {
  constexpr Scalar two = static_cast<Scalar>(2.0);
  constexpr Scalar three = static_cast<Scalar>(3.0);
  const Scalar* params = params_atlas + t * atlas_stride;

  Scalar c0, c1, c2, c3;

  if constexpr (std::is_same_v<Scalar, float>) {
    // Cast the start of this grid interval to float4
    const float4* interval_f4 =
        reinterpret_cast<const float4*>(params + i * U * 4);

    // Load 16 bytes (c0, c1, c2, c3) in ONE instruction
    float4 coeffs = interval_f4[u];
    c0 = coeffs.x;
    c1 = coeffs.y;
    c2 = coeffs.z;
    c3 = coeffs.w;
  } else if constexpr (std::is_same_v<Scalar, double>) {
    // Cast the start of this grid interval to double2
    const double2* interval_d2 =
        reinterpret_cast<const double2*>(params + i * U * 4);

    // Load 32 bytes in TWO 128-bit instructions
    double2 c01 = interval_d2[u * 2 + 0];
    double2 c23 = interval_d2[u * 2 + 1];
    c0 = c01.x;
    c1 = c01.y;
    c2 = c23.x;
    c3 = c23.y;
  }
  h_u = c0 + q * (c1 + q * (c2 + q * c3));
  dh_u = c1 + q * (two * c2 + three * c3 * q);
}

// Mixed-precision cubic spline kernel (only evaluate value)
template <typename Scalar, typename AtlasType>
__device__ __forceinline__ void eval_cubic_spline_single_value(
    Scalar r_val, int t, int u, int U,
    const AtlasType* __restrict__ params_atlas, int grid_size, int atlas_stride,
    Scalar x0, Scalar dx, Scalar xdx, AtlasType& h_u) {
  const AtlasType* params = params_atlas + t * atlas_stride;

  // Better and faster method to calculate `q`
  Scalar dr_x0 = r_val - x0;
  int i = static_cast<int>(dr_x0 * xdx);
  i = max(0, min(grid_size - 2, i));
  Scalar q_exact = fma(-static_cast<Scalar>(i), dx, dr_x0);
  AtlasType q = static_cast<AtlasType>(q_exact);

  AtlasType c0, c1, c2, c3;

  if constexpr (std::is_same_v<AtlasType, float>) {
    // Cast the start of this grid interval to float4
    const float4* interval_f4 =
        reinterpret_cast<const float4*>(params + i * U * 4);

    // Load 16 bytes (c0, c1, c2, c3) in ONE instruction
    float4 coeffs = interval_f4[u];
    c0 = coeffs.x;
    c1 = coeffs.y;
    c2 = coeffs.z;
    c3 = coeffs.w;
  } else if constexpr (std::is_same_v<AtlasType, double>) {
    // Cast the start of this grid interval to double2
    const double2* interval_d2 =
        reinterpret_cast<const double2*>(params + i * U * 4);

    // Load 32 bytes in TWO 128-bit instructions
    double2 c01 = interval_d2[u * 2 + 0];
    double2 c23 = interval_d2[u * 2 + 1];
    c0 = c01.x;
    c1 = c01.y;
    c2 = c23.x;
    c3 = c23.y;
  }
  h_u = c0 + q * (c1 + q * (c2 + q * c3));
}

template <typename Scalar>
__device__ __forceinline__ void eval_cubic_spline_single_value(
    int i, Scalar q, int t, int u, int U,
    const Scalar* __restrict__ params_atlas, int atlas_stride, Scalar& h_u) {
  const Scalar* params = params_atlas + t * atlas_stride;

  // Better and faster method to calculate `q`
  Scalar c0, c1, c2, c3;

  if constexpr (std::is_same_v<Scalar, float>) {
    // Cast the start of this grid interval to float4
    const float4* interval_f4 =
        reinterpret_cast<const float4*>(params + i * U * 4);

    // Load 16 bytes (c0, c1, c2, c3) in ONE instruction
    float4 coeffs = interval_f4[u];
    c0 = coeffs.x;
    c1 = coeffs.y;
    c2 = coeffs.z;
    c3 = coeffs.w;
  } else if constexpr (std::is_same_v<Scalar, double>) {
    // Cast the start of this grid interval to double2
    const double2* interval_d2 =
        reinterpret_cast<const double2*>(params + i * U * 4);

    // Load 32 bytes in TWO 128-bit instructions
    double2 c01 = interval_d2[u * 2 + 0];
    double2 c23 = interval_d2[u * 2 + 1];
    c0 = c01.x;
    c1 = c01.y;
    c2 = c23.x;
    c3 = c23.y;
  }
  h_u = c0 + q * (c1 + q * (c2 + q * c3));
}

// Mixed-precision cubic spline kernel (only evaluate derivative)
template <typename Scalar, typename AtlasType>
__device__ __forceinline__ void eval_cubic_spline_single_deriv(
    Scalar r_val, int t, int u, int U,
    const AtlasType* __restrict__ params_atlas, int grid_size, int atlas_stride,
    Scalar x0, Scalar dx, Scalar xdx, AtlasType& h_u, AtlasType& dh_u) {
  constexpr AtlasType two = static_cast<AtlasType>(2.0);
  constexpr AtlasType three = static_cast<AtlasType>(3.0);
  const AtlasType* params = params_atlas + t * atlas_stride;

  Scalar dr_x0 = r_val - x0;
  int i = static_cast<int>(dr_x0 * xdx);
  i = max(0, min(grid_size - 2, i));
  Scalar q_exact = fma(-static_cast<Scalar>(i), dx, dr_x0);
  AtlasType q = static_cast<AtlasType>(q_exact);

  AtlasType c1, c2, c3;

  if constexpr (std::is_same_v<AtlasType, float>) {
    // Cast the start of this grid interval to float4
    const float4* interval_f4 =
        reinterpret_cast<const float4*>(params + i * U * 4);

    // Load 16 bytes (c0, c1, c2, c3) in ONE instruction
    float4 coeffs = interval_f4[u];
    c1 = coeffs.y;
    c2 = coeffs.z;
    c3 = coeffs.w;
  } else if constexpr (std::is_same_v<AtlasType, double>) {
    // Cast the start of this grid interval to double2
    const double2* interval_d2 =
        reinterpret_cast<const double2*>(params + i * U * 4);

    // Load 32 bytes in TWO 128-bit instructions
    double2 c01 = interval_d2[u * 2 + 0];
    double2 c23 = interval_d2[u * 2 + 1];
    c1 = c01.y;
    c2 = c23.x;
    c3 = c23.y;
  }

  dh_u = c1 + q * (two * c2 + three * c3 * q);
}

template <typename Scalar, bool CalcVal, bool CalcDeriv>
__global__ void cubic_spline_kernel(
    int total_elements, int n_out, const Scalar* __restrict__ R,
    const int* __restrict__ tijlist, Scalar* __restrict__ H,
    Scalar* __restrict__ dHdr, const Scalar* __restrict__ params_atlas,
    int grid_size, int atlas_stride, Scalar x0, Scalar dx, Scalar xdx) {
  // Strategy: Flatten the 2D loop.
  // Global Thread ID maps to (Interaction k, Feature j)
  // dHdr can be null pointer if pre-computed derivative is not needed.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < total_elements) {
    // Decode 1D index -> (Interaction k, Feature j)
    int k = idx / n_out;
    int j = idx % n_out;

    // Lookup Atlas Index
    int t = tijlist[k];

    // Safety check
    if (t < 0) {
      if (CalcVal) H[idx] = 0.0;
      if (CalcDeriv) dHdr[idx] = 0.0;
      return;
    }

    if (CalcVal && CalcDeriv) {
      eval_cubic_spline_single<Scalar, Scalar>(R[k], t, j, n_out, params_atlas,
                                               grid_size, atlas_stride, x0, dx,
                                               xdx, H[idx], dHdr[idx]);
    } else if (CalcVal) {
      eval_cubic_spline_single_value<Scalar, Scalar>(
          R[k], t, j, n_out, params_atlas, grid_size, atlas_stride, x0, dx, xdx,
          H[idx]);
    } else if (CalcDeriv) {
      eval_cubic_spline_single_deriv<Scalar, Scalar>(
          R[k], t, j, n_out, params_atlas, grid_size, atlas_stride, x0, dx, xdx,
          H[idx], dHdr[idx]);
    }
  }
}

}  // namespace TENSORMD::Kernels::Spline::GPU

#endif  // TENSORMD_KERNELS_ARCH_CUDA_SPLINES_CUH