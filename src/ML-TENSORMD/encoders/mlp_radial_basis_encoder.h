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

#ifndef TENSORMD_RADIAL_BASIS_ENCODER_H
#define TENSORMD_RADIAL_BASIS_ENCODER_H

#include <cnpy++.hpp>

#include "../tensormd_memory.h"
#include "../tensormd_v2_config.hpp"
#include "encoder.h"
#include "mlp_head.h"

namespace TENSORMD {

template <typename Scalar>
class MLPRadialBasisEncoder : public Encoder<Scalar> {
 public:
  MLPRadialBasisEncoder(Device dev, ModelConfig& cfg);
  ~MLPRadialBasisEncoder() override;

  void initialize(cnpypp::npz_t& npz) override;
  void radial_forward(TensorData<Scalar>* data) override;
  void radial_backward(TensorData<Scalar>* data) override;

  [[nodiscard]] double get_memory_usage() const override {
    double usage = Encoder<Scalar>::get_memory_usage();
    usage += mlp.get_memory_usage();
    usage += embed_pool.get_memory_usage();
    usage += static_pool.get_memory_usage();
    return usage;
  }

 protected:
  // The YAML config used for training the model
  ModelConfig config;

  // MemoryPool for dynamic inputs
  MemoryPool<Scalar> embed_pool;
  MemoryPool<Scalar> static_pool;

  // Constant buffers (allocated once)
  Scalar* d_rbf_centers;
  Scalar* d_embeddings;
  Scalar* d_expanded;
  Scalar rbf_gamma;

  // The internal MLP model
  MLPHead<Scalar> mlp;

  // The internal encoding function
  void radial_forward(const Scalar* R, const int* zijlist, int n, Scalar* H);

  // The internal interpolation setup functions
  void setup_interpolation_default(std::vector<std::string> species,
                                   Scalar rmin, Scalar delta) override;
};

}  // namespace TENSORMD

#endif
