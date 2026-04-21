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

#include "mlp_encoder.h"

#include "../kernels/kernels_encoder.h"
#include "../kernels/kernels_spline.h"
#include "../kernels/kernels_utils.h"
#include "../tensormd_constants.h"
#include "cutoff.h"
#include "fmt/format.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
MLPEncoder<Scalar>::MLPEncoder(Device dev, Scalar rcut)
    : Encoder<Scalar>(dev, rcut), mlp(dev) {
  this->ctx.num_radial_features = 0;
  this->ctx.num_pairs = 1;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
MLPEncoder<Scalar>::~MLPEncoder() = default;

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPEncoder<Scalar>::initialize(cnpypp::npz_t& npz) {
  // Setup the internal MLP model
  int nlayers = *npz.find("fnn::nlayers")->second.data<int>();
  int actfn = *npz.find("fnn::actfn")->second.data<int>();
  int* layer_sizes = npz.find("fnn::layer_sizes")->second.data<int>();
  bool use_resnet_dt = *npz.find("fnn::use_resnet_dt")->second.data<int>() == 1;
  int apply_output_bias =
      *npz.find("fnn::apply_output_bias")->second.data<int>();
  auto** weights = new Scalar*[nlayers];
  auto** biases = new Scalar*[nlayers];
  for (int i = 0; i < nlayers; i++) {
    weights[i] =
        npz.find(fmt::format("fnn::weights_0_{}", i))->second.data<Scalar>();
    if (i < nlayers - 1 || apply_output_bias)
      biases[i] =
          npz.find(fmt::format("fnn::biases_0_{}", i))->second.data<Scalar>();
  }

  ActivationType type = get_activation_type(actfn);
  constexpr int n_in = 1;
  mlp.setup(n_in, nlayers, layer_sizes, type, &weights, &biases, use_resnet_dt,
            apply_output_bias);

  // Setup num_radial_features
  this->ctx.num_radial_features =
      *npz.find("fnn::num_filters")->second.data<int>();

  // Setup the Cutoff function
  if (*npz.find("fctype")->second.data<int>() == 0) {
    this->cutoff = new Cutoff<Scalar>(this->ctx.grid_rmax);
  } else {
    this->cutoff = new Cutoff<Scalar>(this->ctx.grid_rmax, 5.0);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPEncoder<Scalar>::setup_interpolation_default(
    const std::vector<std::string> species, Scalar rmin, Scalar delta) {
  int n = this->ctx.grid_size;
  int n_out = mlp.get_n_out();

  // Generate grid on host
  std::vector<Scalar> h_x(n);
  for (int i = 0; i < n; i++) {
    h_x[i] = rmin + static_cast<Scalar>(i) * delta;
  }

  // Prepare memory on device
  MemoryPool<Scalar> setup_pool(this->ctx.device,
                                "MLPEncoder::Pool::temp_setup", false);
  setup_pool.grow(n + n * n_out + n);
  Scalar* d_x = setup_pool.request(n);
  Scalar* d_y = setup_pool.request(n * n_out);
  Scalar* d_s = setup_pool.request(n);

  // Copy grid to device
  Kernels::Utils::copy_data_to_device(d_x, h_x.data(), n, this->ctx.device);

  // Run mlp.forward on device
  Scalar d_tot = 0.0;
  mlp.forward(0, n, d_x, nullptr, d_tot, d_y);

  // Calculate cutoff on device
  this->cutoff->compute(this->ctx.device, n, d_x, d_s);

  // Apply cutoff to mlp output 'y' on device
  Kernels::Encoder::apply_cutoff_forward_only(this->ctx.device, n, n_out, d_s,
                                              d_y);

  // Setup the interpolation param atlas
  int num_params = n_out * n * 4;
  this->atlas_pool.grow(num_params);
  this->atlas_pool.reset();
  this->ctx.d_params_atlas = this->atlas_pool.request(num_params);
  Kernels::Spline::cubic_setup(this->ctx.device, n, n_out, delta, d_y,
                               this->ctx.d_params_atlas);

  if (TENSORMD_USE_FP32_ATLAS) {
    // Cast the atlas from double to float
    this->atlas_pool_fp32.grow(num_params);
    this->atlas_pool_fp32.reset();
    this->ctx.d_params_atlas_fp32 = this->atlas_pool_fp32.request(num_params);
    Kernels::Utils::cast_scalar_to_float(
        this->ctx.device, this->ctx.d_params_atlas,
        this->ctx.d_params_atlas_fp32, num_params);
    // Release the original atlas memory
    this->atlas_pool.free();
    this->ctx.d_params_atlas = nullptr;
  }

  // Update the flag
  this->ctx.ready_to_interpolate = true;

  // Relese device memory
  this->mlp.release_dynamic_memory();
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void MLPEncoder<Scalar>::radial_forward(TensorData<Scalar>* data) {
  const int n = data->dims.a * data->dims.b * data->dims.c;
  Scalar d_tot = 0.0;
  mlp.kernel = TimedKernel::forward;
  mlp.forward(0, n, data->R, nullptr, d_tot, data->H);
}

template <typename Scalar>
void MLPEncoder<Scalar>::radial_backward(TensorData<Scalar>* data) {
  Scalar* grad_in = data->Up;
  Scalar* grad_out = data->dHdrp;
  mlp.kernel = TimedKernel::backward;
  mlp.backward(0, grad_in, grad_out);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class MLPEncoder<float>;
template class MLPEncoder<double>;

}  // namespace TENSORMD
