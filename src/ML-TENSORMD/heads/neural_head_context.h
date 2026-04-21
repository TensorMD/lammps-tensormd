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

#ifndef TENSORMD_HEADS_CONTEXT_H
#define TENSORMD_HEADS_CONTEXT_H

#include <vector>

#include "../tensormd_device.h"
#include "../tensormd_types.h"

namespace TENSORMD {

template <typename Scalar>
class NeuralHeadContext {
 public:
  NeuralHeadContext(Device dev) : device(dev) {}

  ~NeuralHeadContext() {
#define delete_arrays(d_arrays)                          \
  if (d_arrays) {                                        \
    for (int elti = 0; elti < this->n_species; elti++) { \
      if (d_arrays[elti]) {                              \
        delete[] d_arrays[elti];                         \
      }                                                  \
    }                                                    \
    delete[] d_arrays;                                   \
  }
    delete_arrays(d_biases);
    delete_arrays(d_weights);
    delete_arrays(d_weights_T);
#undef delete_arrays
  }

  Device device;

  // Device pointers
  Scalar*** d_weights = nullptr;
  Scalar*** d_weights_T = nullptr;
  Scalar*** d_biases = nullptr;
  Scalar* d_eatom = nullptr;
  Scalar* d_output_sum = nullptr;

  // This flag decides whether to sum the output values.
  //    * If true, result.toten = sum(result.eatom).
  //    * If false, result.toten = 0.0.
  // By default, it is true. For MLP based encoders it should be false.
  bool sum_output = true;

  // Architecture
  int n_species = 0;
  int n_layers = 0;             // n_layers = n_hidden_layers + 1
  std::vector<int> full_sizes;  // Including input and output layers
  int n_in = 0;
  int n_out = 0;
  int eff_ndim_in = 0;
  bool use_resnet_dt = true;
  bool apply_output_bias = true;
  ActivationType actype = ActivationType::Silu;
};

}  // namespace TENSORMD

#endif
