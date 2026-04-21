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

#include "mlp_radial_basis_encoder.h"

#include "../kernels/kernels.h"
#include "../kernels/kernels_encoder.h"
#include "../kernels/kernels_spline.h"
#include "../kernels/kernels_utils.h"
#include "../kernels/kernels_v2_ops.h"
#include "../tensormd_constants.h"
#include "../tensormd_memory.h"
#include "../tensormd_types.h"
#include "../tensormd_v2_embeddings.hpp"
#include "cutoff.h"
#include "fmt/format.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
MLPRadialBasisEncoder<Scalar>::MLPRadialBasisEncoder(Device dev,
                                                     ModelConfig& cfg)
    : Encoder<Scalar>(dev, cfg.r_cut),
      mlp(dev),
      config(cfg),
      embed_pool(dev, "MLPRadialBasisEncoder::Pool::embed", true),
      static_pool(dev, "MLPRadialBasisEncoder::Pool::static", false) {
  d_rbf_centers = nullptr;
  d_embeddings = nullptr;
  d_expanded = nullptr;
  rbf_gamma = 0.0;
  this->ctx.num_radial_features = cfg.k_dim;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
MLPRadialBasisEncoder<Scalar>::~MLPRadialBasisEncoder() = default;

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPRadialBasisEncoder<Scalar>::initialize(cnpypp::npz_t& npz) {
  int n_in = 2 * EMBEDDING_SIZE + this->config.mlp.r_feat_dim;
  int nlayers = static_cast<int>(this->config.mlp.hidden_dims.size() + 1);

  // layer_sizes: [hidden_1, ..., hidden_n, n_out]
  std::vector<int> layer_sizes;
  for (const auto& lh : config.mlp.hidden_dims) {
    layer_sizes.push_back(lh);
  }
  layer_sizes.push_back(this->config.k_dim);

  // Setup the weights
  Scalar **weights, **biases;
  weights = new Scalar*[nlayers];
  biases = new Scalar*[nlayers];
  for (int i = 0; i < nlayers; i++) {
    weights[i] = npz.find(fmt::format("radial_encoder.mlp.{}.weight", i * 2))
                     ->second.data<Scalar>();
    biases[i] = npz.find(fmt::format("radial_encoder.mlp.{}.bias", i * 2))
                    ->second.data<Scalar>();
  }

  // Setup the internal MLP model
  mlp.setup(n_in, nlayers, layer_sizes.data(), ActivationType::Silu, &weights,
            &biases, false, true);

  // RBF centers
  auto rbf_centers = npz.find("radial_encoder.centers");
  if (rbf_centers == npz.end()) {
    throw std::runtime_error(
        "MLPRadialBasisEncoder::setup_global::failed to "
        "find the gauss centers!");
  }

  // RBF gamma
  auto rbf_gamma_ptr = npz.find("radial_encoder.gamma");
  if (rbf_gamma_ptr == npz.end()) {
    throw std::runtime_error(
        "MLPRadialBasisEncoder::setup_global::failed to "
        "find the gauss gamma!");
  } else {
    rbf_gamma = *rbf_gamma_ptr->second.data<Scalar>();
  }

  // Flat the embeddings into a contiguous array for efficient copying to device
  std::vector<Scalar> flat_embed;
  for (int Z = 0; Z <= MAX_ATOMIC_NUMBER; Z++) {
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      flat_embed.push_back(Electron_Embeddings<Scalar>[Z][i]);
    }
  }

  // Send RBF centers and embeddings to device memory (using a static pool
  // since they are constant)
  static_pool.grow(rbf_centers->second.num_vals + flat_embed.size());
  d_rbf_centers = static_pool.request(rbf_centers->second.num_vals);
  Kernels::Utils::copy_data_to_device(
      d_rbf_centers, rbf_centers->second.data<Scalar>(),
      rbf_centers->second.num_vals, this->ctx.device);
  d_embeddings = static_pool.request(flat_embed.size());
  Kernels::Utils::copy_data_to_device(d_embeddings, flat_embed.data(),
                                      flat_embed.size(), this->ctx.device);

  // Always use the cosine cutoff function
  this->cutoff = new Cutoff<Scalar>(this->config.r_cut);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPRadialBasisEncoder<Scalar>::radial_forward(TensorData<Scalar>* data) {
  const int n = data->dims.a * data->dims.b * data->dims.c;
  radial_forward(data->R, data->zijlist, n, data->H);
}

template <typename Scalar>
void MLPRadialBasisEncoder<Scalar>::radial_forward(const Scalar* d_R,
                                                   const int* d_zijlist, int n,
                                                   Scalar* d_H) {
  const int n_in = mlp.get_n_in();
  const int n_rbf = config.mlp.r_feat_dim;
  const int n_embed = EMBEDDING_SIZE;

  // Prepare Memory
  embed_pool.grow(n * n_in);
  embed_pool.reset();
  d_expanded = embed_pool.request(n * n_in);

  // Expand the input
  Kernels::V2::expand_encoder_input(this->ctx.device, n, n_rbf, n_embed, d_R,
                                    d_zijlist, d_rbf_centers, rbf_gamma,
                                    d_embeddings, d_expanded);

  // MLP Forward
  Scalar d_tot = 0.0;
  mlp.kernel = TimedKernel::forward;
  mlp.forward(0, n, d_expanded, nullptr, d_tot, d_H);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPRadialBasisEncoder<Scalar>::radial_backward(TensorData<Scalar>* data) {
  const int n = data->dims.a * data->dims.b * data->dims.c;
  const int n_rbf = config.mlp.r_feat_dim;
  const int n_mlp_in = mlp.get_n_in();
  Scalar* grad_out = data->dHdrp;
  mlp.kernel = TimedKernel::backward;
  mlp.backward(0, data->Up, d_expanded);
  Kernels::V2::expand_encoder_grad(this->ctx.device, n, n_rbf, n_mlp_in,
                                   d_expanded, data->R, d_rbf_centers,
                                   rbf_gamma, grad_out);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPRadialBasisEncoder<Scalar>::setup_interpolation_default(
    const std::vector<std::string> species, Scalar rmin, Scalar delta) {
  int n_out = mlp.get_n_out();
  int n = this->ctx.grid_size;

  // Identify unique atom pairs and build map
  std::vector<int> atomic_numbers;
  for (int i = 0; i < species.size(); i++) {
    auto Zi = AtomicSymbolToNumber.find(species[i]);
    if (Zi == AtomicSymbolToNumber.end()) {
      throw std::runtime_error("Unknown specie: " + species[i]);
    } else {
      atomic_numbers.push_back(Zi->second);
    }
  }
  std::vector<int> h_pair_map(PAIR_MAP_SIZE, -1);
  int pair_count = 0;
  for (const int Zi : atomic_numbers) {
    for (const int Zj : atomic_numbers) {
      int key = PAIR_KEY(Zi, Zj);
      if (h_pair_map[key] == -1) {
        h_pair_map[key] = pair_count;
        pair_count++;
      }
    }
  }
  this->ctx.num_pairs = pair_count;

  // Allocate atlas memory on device
  size_t params_per_pair = (size_t)n * 4 * n_out;
  size_t total_atlas_size = pair_count * params_per_pair;
  this->atlas_pool.grow(total_atlas_size);
  this->ctx.d_params_atlas = this->atlas_pool.request(total_atlas_size);

  // Prepare memory for temporary arrays on device
  MemoryPool<Scalar> temp_scalar_pool(
      this->ctx.device, "MLPRadialBasisEncoder::Pool::temp_scalar", false);
  temp_scalar_pool.grow(n + n * n_out + n);
  Scalar* d_x = temp_scalar_pool.request(n);
  Scalar* d_y = temp_scalar_pool.request(n * n_out);
  Scalar* d_s = temp_scalar_pool.request(n);
  MemoryPool<int> temp_int_pool(this->ctx.device,
                                "MLPRadialBasisEncoder::Pool::temp_int", false);
  temp_int_pool.grow(n * 2);
  int* d_zij = temp_int_pool.request(n * 2);

  // Then we calculate coefficients for each pair
  // First we generate the fixed grid on host and copy it to device
  std::vector<Scalar> h_x(n);
  for (int k = 0; k < n; k++) {
    h_x[k] = rmin + static_cast<Scalar>(k) * delta;
  }
  Kernels::Utils::copy_data_to_device(d_x, h_x.data(), n, this->ctx.device);

  // Pre-compute the cutoff values on device
  this->cutoff->compute(this->ctx.device, n, d_x, d_s);

  // Re-iterate to fill atlas
  std::vector<int> h_zij(n * n_out);
  for (const int Zi : atomic_numbers) {
    for (const int Zj : atomic_numbers) {
      int key = PAIR_KEY(Zi, Zj);
      int idx = h_pair_map[key];

      // Prepare Input for MLP on host side
      for (int k = 0; k < n; k++) {
        h_zij[k * 2 + 0] = Zi;
        h_zij[k * 2 + 1] = Zj;
      }

      // Copy zij to device
      Kernels::Utils::copy_data_to_device(d_zij, h_zij.data(), n * 2,
                                          this->ctx.device);

      // Run MLP forward on device
      radial_forward(d_x, d_zij, n, d_y);

      // Apply cutoff to MLP output on device
      Kernels::Encoder::apply_cutoff_forward_only(this->ctx.device, n, n_out,
                                                  d_s, d_y);

      // Calculate coefficients directly into atlas buffer
      Scalar* dest_ptr = this->ctx.d_params_atlas + idx * params_per_pair;
      Kernels::Spline::cubic_setup(this->ctx.device, n, n_out,
                                   this->ctx.grid_dr, d_y, dest_ptr);
    }
  }

  if (TENSORMD_USE_FP32_ATLAS) {
    // Cast the atlas from double to float
    this->atlas_pool_fp32.grow(total_atlas_size);
    this->atlas_pool_fp32.reset();
    this->ctx.d_params_atlas_fp32 =
        this->atlas_pool_fp32.request(total_atlas_size);
    Kernels::Utils::cast_scalar_to_float(
        this->ctx.device, this->ctx.d_params_atlas,
        this->ctx.d_params_atlas_fp32, total_atlas_size);
    // Release the original atlas memory
    // this->atlas_pool.free();
    // this->d_params_atlas = nullptr;
  }

  // Upload pair_map to device
  this->map_pool.grow(PAIR_MAP_SIZE);
  this->map_pool.reset();
  this->ctx.d_pair_map = this->map_pool.request(PAIR_MAP_SIZE);
  Kernels::Utils::copy_data_to_device(this->ctx.d_pair_map, h_pair_map.data(),
                                      PAIR_MAP_SIZE, this->ctx.device);

  this->ctx.ready_to_interpolate = true;

  // Relese memory pools that are no longer needed
  embed_pool.free();

  // Release the dynamic memory used by MLP
  mlp.release_dynamic_memory();
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class MLPRadialBasisEncoder<float>;
template class MLPRadialBasisEncoder<double>;

}  // namespace TENSORMD
