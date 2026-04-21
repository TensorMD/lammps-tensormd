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

#include "energy_head_v1.h"

#include "../kernels/kernels_head.h"
#include "fmt/format.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV1<Scalar>::initialize(cnpypp::npz_t& npz) {
  int n_species_ = *npz.find("nelt")->second.data<int>();
  int actfn_ = *npz.find("actfn")->second.data<int>();
  bool use_resnet_dt_ = *npz.find("use_resnet_dt")->second.data<int>();
  bool apply_output_bias_ = *npz.find("apply_output_bias")->second.data<int>();

  // layer_sizes_ includes the hidden layers and the output layer
  // layer_sizes_.size() is n_layers_
  int n_layers_ = *npz.find("nlayers")->second.data<int>();
  std::vector<int> layer_sizes_(n_layers_);
  for (int i = 0; i < n_layers_; i++) {
    layer_sizes_[i] = npz.find("layer_sizes")->second.data<int>()[i];
  }
  auto*** weights_ = new Scalar**[n_species_];
  auto*** biases_ = new Scalar**[n_species_];
  for (int elti = 0; elti < n_species_; elti++) {
    weights_[elti] = new Scalar*[n_layers_];
    biases_[elti] = new Scalar*[n_layers_];
    for (int i = 0; i < n_layers_; i++) {
      weights_[elti][i] = npz.find(fmt::format("weights_{}_{}", elti, i))
                              ->second.data<Scalar>();
      if (i < n_layers_ - 1 || apply_output_bias_) {
        biases_[elti][i] = npz.find(fmt::format("biases_{}_{}", elti, i))
                               ->second.data<Scalar>();
      }
    }
  }
  int num_in = static_cast<int>(npz.find("weights_0_0")->second.shape[0]);
  ActivationType type = get_activation_type(actfn_);

  // The dimension is required if K-continuous memory layout is used for the
  // first layer weights.
  this->k_dim = *npz.find("fnn::num_filters")->second.data<int>();
  this->m_dim = *npz.find("max_moment")->second.data<int>() + 1;

  NeuralHead<Scalar>::setup(n_species_, num_in, n_layers_, layer_sizes_.data(),
                            type, weights_, biases_, use_resnet_dt_,
                            apply_output_bias_);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV1<Scalar>::permute_W0(std::vector<Scalar>& W_kmh,
                                      std::vector<Scalar>& W_kmh_T,
                                      const Scalar* W_mkh, int n_rows,
                                      int n_cols) {
  size_t bytes_per_row = sizeof(Scalar) * n_cols;
  size_t stride_b = n_rows / this->ctx.n_species * n_cols;
  for (int i_b = 0; i_b < this->ctx.n_species; ++i_b) {
    const Scalar* src = W_mkh + i_b * stride_b;
    Scalar* dst = W_kmh.data() + i_b * stride_b;
    for (int i_k = 0; i_k < this->k_dim; ++i_k) {
      for (int i_m = 0; i_m < this->m_dim; ++i_m) {
        // Calculate index in the OLD weight matrix
        // Row = i_k * m + i_m
        int old_row = (i_k * this->m_dim + i_m) * n_cols;

        // Calculate index in the NEW weight matrix
        // Row = i_m * k + i_k
        int new_row = (i_m * this->k_dim + i_k) * n_cols;

        // Copy the entire row (size n_cols)
        std::memcpy(dst + new_row, src + old_row, bytes_per_row);
      }
    }
  }
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      W_kmh_T[col * n_rows + row] = W_kmh[row * n_cols + col];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV1<Scalar>::transpose_W0(std::vector<Scalar>& WT,
                                        const Scalar* W, int n_rows,
                                        int n_cols) {
  for (int row = 0; row < n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      WT[col * n_rows + row] = W[row * n_cols + col];
    }
  }
}

/* ---------------------------------------------------------------------- */

template class EnergyHeadV1<float>;
template class EnergyHeadV1<double>;

}  // namespace TENSORMD