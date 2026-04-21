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

#ifndef TENSORMD_ENERGY_HEAD_H
#define TENSORMD_ENERGY_HEAD_H

#include "../tensormd_types.h"
#include "cnpy++.hpp"
#include "neural_head.h"

namespace TENSORMD {

template <typename Scalar>
class EnergyHead : public NeuralHead<Scalar> {
 public:
  EnergyHead(Device dev) : NeuralHead<Scalar>(dev) {
    this->ctx.sum_output = true;
    this->use_W_T = true;
  }

  virtual void initialize(cnpypp::npz_t& npz) = 0;

  // Return true if this is a finite-temperature energy head
  [[nodiscard]] virtual bool is_finite_temp_head() const { return false; }

 protected:
  // Override this function because the first weight matrix maybe permuted
  void copy_params_to_device(Scalar*** h_weights, Scalar*** h_biases) override;

  virtual void permute_W0(std::vector<Scalar>& W0_kmh,
                          std::vector<Scalar>& W0_kmh_T, const Scalar* W0_mkh,
                          int n_rows, int n_cols) = 0;
  virtual void transpose_W0(std::vector<Scalar>& W0_mkh_T, const Scalar* W0_mkh,
                            int n_rows, int n_cols) = 0;
};

}  // namespace TENSORMD
#endif