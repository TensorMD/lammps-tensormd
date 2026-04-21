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

#ifndef TENSORMD_KERNELS_ARCH_CUDA_KERNELS_FUSION_TYPES_CUH
#define TENSORMD_KERNELS_ARCH_CUDA_KERNELS_FUSION_TYPES_CUH

namespace TENSORMD::Kernels::Fusion::GPU {

// ============================================================================
// Descriptor dimensions
// ============================================================================

// clang-format off
template <bool UseReducible, int M> struct DescriptorDims;

template <> struct DescriptorDims<true, 1> { static constexpr int D = 1; };
template <> struct DescriptorDims<true, 2> { static constexpr int D = 4; };
template <> struct DescriptorDims<true, 3> { static constexpr int D = 10; };
template <> struct DescriptorDims<true, 4> { static constexpr int D = 20; };
template <> struct DescriptorDims<true, 5> { static constexpr int D = 35; };
template <> struct DescriptorDims<true, 6> { static constexpr int D = 56; };

template <> struct DescriptorDims<false, 1> { static constexpr int D = 1; };
template <> struct DescriptorDims<false, 2> { static constexpr int D = 4; };
template <> struct DescriptorDims<false, 3> { static constexpr int D = 9; };
template <> struct DescriptorDims<false, 4> { static constexpr int D = 16; };
template <> struct DescriptorDims<false, 5> { static constexpr int D = 25; };
template <> struct DescriptorDims<false, 6> { static constexpr int D = 36; };
// clang-format on

// ============================================================================
// M/dM storage stride
// ============================================================================
//
// Shared-memory row stride used for cached force moments / derivatives.
// Rule:
//   - FP32 bank-conflict avoidance for row-stepped access wants gcd(stride,
//   32)=1
//   - therefore choose an odd stride
//   - minimal padding choice: smallest odd >= logical row length
//
// For backward dM storage, logical row length is factor * D where factor is 3
// for saving dM only and 4 for saving both dM and M.
// For s/p (M=1/2), this optimization is not important and can be ignored,
// but we still provide consistent values here.
// ============================================================================

template <bool UseReducible, int M>
struct ForwardMStride {
  static constexpr int value = DescriptorDims<UseReducible, M>::D;
};

template <bool UseReducible, int M, bool SaveM = false>
struct BackwardMStride {
  static constexpr int factor = SaveM ? 4 : 3;
  static constexpr int logical = factor * DescriptorDims<UseReducible, M>::D;
  static constexpr int value = (logical & 1) ? logical : (logical + 1);
};

// ============================================================================
// Config
// ============================================================================

template <typename Scalar_, typename AtlasType_, int M_, int K_,
          bool UseReducible_, int BatchSize_ = 32, bool ForwardSaveM_ = false,
          bool BackwardSaveM_ = false>
struct KernelConfig {
  using Scalar = Scalar_;
  using AtlasType = AtlasType_;

  static constexpr int m_val = M_;
  static constexpr int K = K_;
  static constexpr bool UseReducible = UseReducible_;
  static constexpr int D = DescriptorDims<UseReducible, M_>::D;
  static constexpr int M_Stride = ForwardMStride<UseReducible, M_>::value;
  static constexpr int dM_Stride =
      BackwardMStride<UseReducible, M_, BackwardSaveM_>::value;
  static constexpr int BatchSize = BatchSize_;
  static constexpr bool ForwardSaveM = ForwardSaveM_;
  static constexpr bool BackwardSaveM = BackwardSaveM_;
};

}  // namespace TENSORMD::Kernels::Fusion::GPU

#endif
