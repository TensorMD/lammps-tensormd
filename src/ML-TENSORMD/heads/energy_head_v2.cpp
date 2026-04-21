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

#include "energy_head_v2.h"

#include "../kernels/kernels_head.h"
#include "../kernels/kernels_utils.h"
#include "../kernels/kernels_v2_ops.h"
#include "../tensormd_strategy.h"
#include "../tensormd_types.h"
#include "../tensormd_v2_embeddings.hpp"
#include "../utils/tensor_debug.h"
#include "fmt/format.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::initialize(cnpypp::npz_t& npz) {
  const int n_species = 1;
  bool use_resnet_dt = config.energy_head.use_resnet_dt;

  // Initialize `layer_sizes`. The output layer size is always 1.
  int nlayers = static_cast<int>(config.energy_head.hidden_dims.size() + 1);
  std::vector<int> layer_sizes(nlayers);
  for (int i = 0; i < nlayers - 1; i++) {
    layer_sizes[i] = config.energy_head.hidden_dims[i];
  }
  layer_sizes[nlayers - 1] = 1;

  // Setup the activation function
  ActivationType actfn = get_activation_type(config.energy_head.actfn);

  // Setup the weights and biases
  // The naming rule is a bit complicated.
  auto** weights = new Scalar*[nlayers];
  auto** biases = new Scalar*[nlayers];

  int k = 0, i = 0;
  int num_in = 0;
  while (i < nlayers && k < nlayers * 2) {
    std::string weight_key, bias_key;
    if (i > 0 && layer_sizes[i] == layer_sizes[i - 1] && use_resnet_dt) {
      weight_key = fmt::format("energy_heads.{}.mlp.{}.linear.weight",
                               config.active_task_id, k);
      bias_key = fmt::format("energy_heads.{}.mlp.{}.linear.bias",
                             config.active_task_id, k);
    } else {
      weight_key = fmt::format("energy_heads.{}.mlp.{}.weight",
                               config.active_task_id, k);
      bias_key =
          fmt::format("energy_heads.{}.mlp.{}.bias", config.active_task_id, k);
    }
    auto it = npz.find(weight_key);
    if (it == npz.end()) {
      k++;
      continue;
    }
    weights[i] = it->second.data<Scalar>();
    if (i == 0) {
      num_in = static_cast<int>(it->second.shape[0]);
    }

    it = npz.find(bias_key);
    if (it == npz.end()) {
      if (i < nlayers - 1) {
        throw std::runtime_error(
            fmt::format("[TensorMDV2]: Missing bias for layer {}.", i));
      }
    } else {
      biases[i] = it->second.data<Scalar>();
    }
    k++;
    i++;
  }
  if (i != nlayers) {
    throw std::runtime_error(fmt::format(
        "[TensorMDV2]: only found {} of {} NNP layers.", i, nlayers));
  }

  EnergyHeadV1<Scalar>::setup(n_species, num_in, nlayers, layer_sizes.data(),
                              actfn, &weights, &biases, use_resnet_dt, false);

  setup_embedded_biases(npz);

  const char* var = std::getenv("TENSORMD_USE_HYBRID_KERNEL");
  if (var && is_var_enabled(var)) {
    use_hybrid_kernel = true;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::permute_W0(std::vector<Scalar>& W_kmh,
                                      std::vector<Scalar>& W_kmh_T,
                                      const Scalar* W_mkh, int n_rows,
                                      int n_cols) {
  int m_val = this->config.max_moment + 1;
  int K_dim = this->config.k_dim;
  size_t bytes = sizeof(Scalar) * n_cols;
  for (int i_k = 0; i_k < K_dim; ++i_k) {
    for (int i_m = 0; i_m < m_val; ++i_m) {
      // Calculate index in the OLD weight matrix
      // Row = i_k * m + i_m
      int old_row = (i_k * m_val + i_m) * n_cols;

      // Calculate index in the NEW weight matrix
      // Row = i_m * k + i_k
      int new_row = (i_m * K_dim + i_k) * n_cols;

      // Copy the entire row (size n_cols)
      std::memcpy(W_kmh.data() + new_row, W_mkh + old_row, bytes);
    }
  }
  // Copy the remaining rows (the embedding part) without permutation
  for (int row = K_dim * m_val; row < n_rows; row++) {
    int offset = row * n_cols;
    std::memcpy(W_kmh.data() + offset, W_mkh + offset, bytes);
  }
  // Special treatment for the transposed weights for backward pass.
  // Only first K*m rows are permuted, the rest rows (the embedding part) are
  // not permuted. The embedding part is not needed for backward pass.
  int eff_n_rows = m_val * K_dim;
  for (int row = 0; row < eff_n_rows; row++) {
    for (int col = 0; col < n_cols; col++) {
      W_kmh_T[col * eff_n_rows + row] = W_kmh[row * n_cols + col];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::transpose_W0(std::vector<Scalar>& WT,
                                        const Scalar* W, int n_rows,
                                        int n_cols) {
  int m_val = this->config.max_moment + 1;
  int K_dim = this->config.k_dim;
  int in_feat = m_val * K_dim;
  int out_dims = n_cols;
  for (int row = 0; row < in_feat; row++) {
    for (int col = 0; col < out_dims; col++) {
      WT[col * in_feat + row] = W[row * out_dims + col];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::setup_embedded_biases(cnpypp::npz_t& npz) {
  size_t n_types = this->config.atomic_numbers.size();
  size_t total_size = n_types * (this->ctx.full_sizes[1] + EMBEDDING_SIZE + 1);
  static_pool.grow(total_size);
  static_pool.reset();

  // Flatten the 'Electron_Embeddings' tensor and store it in a 1D vector for
  // easy copying to device memory.
  std::vector<Scalar> flat_embed;
  for (const int Z : this->config.atomic_numbers) {
    if (Z > MAX_ATOMIC_NUMBER) {
      throw std::runtime_error(
        fmt::format("[TensorMDV2] atomic number {} > MAX_ATOMIC_NUMBER.", Z));
    }
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      flat_embed.push_back(Electron_Embeddings<Scalar>[Z][i]);
    }
  }

  // First we copy the electron embeddings to device
  d_embeddings = static_pool.request(flat_embed.size());
  Kernels::Utils::copy_data_to_device(d_embeddings, flat_embed.data(),
                                      flat_embed.size(), this->ctx.device);

  // Copy the 'E0_embedding' tensor, which is the atom-type based bias for the
  // final layer. Note that the shape of this tensor is [Z_max + 1, ].
  std::string eatom_key =
      fmt::format("energy_heads.{}.E0_embedding.weight", config.active_task_id);
  auto it = npz.find(eatom_key);
  if (it == npz.end()) {
    throw std::runtime_error(
        fmt::format("[TensorMDV2] Failed to find the EO embedding tensor."));
  }
  auto* E0 = it->second.data<Scalar>();
  std::vector<Scalar> E0_embed;
  for (const int Z : this->config.atomic_numbers) {
    E0_embed.push_back(E0[Z]);
  }
  d_E0_embed = static_pool.request(E0_embed.size());
  Kernels::Utils::copy_data_to_device(d_E0_embed, E0_embed.data(),
                                      E0_embed.size(), this->ctx.device);

  // Finally we setup the embedded biases for the first layer
  d_h0_embed = static_pool.request(n_types * this->ctx.full_sizes[1]);
  Kernels::V2::compute_first_embedding(
      this->ctx.device, n_types, this->ctx.eff_ndim_in, EMBEDDING_SIZE,
      this->ctx.full_sizes[1], d_embeddings, this->ctx.d_weights[0][0],
      this->ctx.d_biases[0][0], d_h0_embed);

  // Replace the first and the last biases
  this->ctx.d_biases[0][0] = d_h0_embed;
  this->ctx.d_biases[0][this->ctx.n_layers - 1] = d_E0_embed;

  // This model will use atom-type based biases for the first and last layers
  this->ctx.apply_output_bias = true;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::compute(int n, Scalar* G_nkm,
                                   const int* d_tlist, Scalar& h_total,
                                   Scalar* h_outputs, Scalar* dydx) {
  if (use_hybrid_kernel) {
    this->dynamic_pool.grow(n * (this->ctx.full_sizes[1] + 1));
    this->dynamic_pool.reset();
    this->ctx.d_eatom = this->dynamic_pool.request(n);
    this->d_dynamic = this->dynamic_pool.request(n * this->ctx.full_sizes[1]);
    Kernels::Head::nnp_hybrid(&this->ctx, n, G_nkm, d_tlist, h_total, h_outputs,
                              dydx, this->d_dynamic, this->use_W_T);
  } else {
    forward(n, G_nkm, d_tlist, h_total, h_outputs);
    backward(nullptr, dydx);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::forward(int n, Scalar* G_nkm,
                                   const int* d_tlist, Scalar& h_total,
                                   Scalar* h_outputs) {
  EnergyHeadV1<Scalar>::forward(0, n, G_nkm, d_tlist, h_total, h_outputs);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHeadV2<Scalar>::backward(const Scalar* grad_in, Scalar* grad_out) {
  // The new nnp_backward kernel allows us to directly compute the gradient
  // w.r.t. the non-embedding part of the input. We do not need to compute the
  // gradient w.r.t. the expanded input first and then use memcpy to create the
  // gradient for the non-embedding part of the input.
  EnergyHeadV1<Scalar>::backward(0, grad_in, grad_out);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double EnergyHeadV2<Scalar>::get_memory_usage() const {
  double usage = EnergyHeadV1<Scalar>::get_memory_usage();
  usage += static_pool.get_memory_usage();
  usage += dynamic_pool.get_memory_usage();
  return usage;
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class EnergyHeadV2<float>;
template class EnergyHeadV2<double>;

}  // namespace TENSORMD
