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

#ifndef TENSORMD_ENCODER_CONTEXT_H
#define TENSORMD_ENCODER_CONTEXT_H

#include "../tensormd_device.h"

namespace TENSORMD {

template <typename Scalar>
class EncoderContext {
 public:
  EncoderContext(Device dev) : device(dev) {}

  Device device;
  Scalar grid_rmax = 0.0;
  Scalar grid_rmin = 0.0;
  Scalar grid_dr = 0.0;
  Scalar grid_rdr = 0.0;
  int grid_size = 0;
  int atlas_stride = 0;
  Scalar* d_params_atlas = nullptr;  // [num_pairs * block_size]
  float* d_params_atlas_fp32 = nullptr;
  int* d_pair_map = nullptr;
  int num_pairs = 0;
  int num_radial_features = 0;
  bool ready_to_interpolate = false;
};

}  // namespace TENSORMD

#endif
