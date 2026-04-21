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

#ifndef TENSORMD_MLP_ENCODER_H
#define TENSORMD_MLP_ENCODER_H

#include "encoder.h"
#include "mlp_head.h"

namespace TENSORMD {

template <typename Scalar>
class MLPEncoder : public Encoder<Scalar> {
 public:
  MLPEncoder(Device dev, Scalar rcut);
  ~MLPEncoder();

  void initialize(cnpypp::npz_t& npz) override;
  void radial_forward(TensorData<Scalar>* data) override;
  void radial_backward(TensorData<Scalar>* data) override;

  [[nodiscard]] double get_memory_usage() const override {
    return mlp.get_memory_usage();
  }

 protected:
  // The internal MLP model
  MLPHead<Scalar> mlp;

  void setup_interpolation_default(std::vector<std::string> species,
                                   Scalar rmin, Scalar delta) override;
};

}  // namespace TENSORMD
#endif