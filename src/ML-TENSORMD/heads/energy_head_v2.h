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

#ifndef TENSORMD_ENERGY_HEAD_V2_H
#define TENSORMD_ENERGY_HEAD_V2_H

#include "../tensormd_v2_config.hpp"
#include "energy_head.h"
#include "energy_head_v1.h"

namespace TENSORMD {

template <typename Scalar>
class EnergyHeadV2 : public EnergyHeadV1<Scalar> {
 public:
  EnergyHeadV2(Device dev, ModelConfig& config)
      : EnergyHeadV1<Scalar>(dev),
        config(config),
        static_pool(dev, "EnergyHeadV2::Pool::static", false),
        dynamic_pool(dev, "EnergyHeadV2::Pool::dynamic", true) {
    d_embeddings = nullptr;
    d_E0_embed = nullptr;
    d_h0_embed = nullptr;
    d_dynamic = nullptr;
    this->ctx.eff_ndim_in = config.k_dim * (config.max_moment + 1);
  }

  void initialize(cnpypp::npz_t& npz) override;

  void compute(int n, Scalar* G_nkm, const int* d_tlist, Scalar& h_total,
               Scalar* h_outputs, Scalar* dydx);
  void forward(int n, Scalar* G_nkm, const int* d_tlist, Scalar& d_total,
               Scalar* h_outputs);
  void backward(const Scalar* grad_in, Scalar* grad_out);

  [[nodiscard]] double get_memory_usage() const override;

  void release_dynamic_memory() override {
    EnergyHeadV1<Scalar>::release_dynamic_memory();
  }

 protected:
  ModelConfig config;

  // Extra memory pool for the expanded input and embeddings
  MemoryPool<Scalar> static_pool;
  MemoryPool<Scalar> dynamic_pool;
  Scalar* d_dynamic;
  Scalar* d_embeddings;
  Scalar* d_E0_embed;
  Scalar* d_h0_embed;

  // Setup the embeddings for the first layer and the final output layer.
  // Both layers use atom-type based bias vectors.
  void setup_embedded_biases(cnpypp::npz_t& npz);

  // The permute and transpose functions are overriden because we only need to
  // work on the feature part of the W0.

  void permute_W0(std::vector<Scalar>& W_kmh, std::vector<Scalar>& W_kmh_T,
                  const Scalar* W_mkh, int n_rows, int n_cols) override;
  void transpose_W0(std::vector<Scalar>& W_mkh_T, const Scalar* W_mkh,
                    int n_rows, int n_cols) override;

  bool use_hybrid_kernel = false;
};

}  // namespace TENSORMD

#endif
