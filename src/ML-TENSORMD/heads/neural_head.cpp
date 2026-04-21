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

#include "neural_head.h"

#include "../kernels/kernels_head.h"
#include "../kernels/kernels_utils.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
NeuralHead<Scalar>::NeuralHead(Device dev)
    : ctx(dev), static_pool(dev, "NeuralHead::Pool::static", false) {
  ctx.n_layers = 0;
  ctx.n_species = 0;
  ctx.n_in = 0;
  ctx.n_out = 0;
  ctx.eff_ndim_in = -1;
  ctx.full_sizes.clear();
  ctx.use_resnet_dt = true;
  ctx.apply_output_bias = true;
  ctx.sum_output = true;
  ctx.d_weights = nullptr;
  ctx.d_biases = nullptr;
  ctx.actype = ActivationType::Softplus;

  static_pool.grow(4);
  static_pool.reset();
  ctx.d_output_sum = static_pool.request(1);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
NeuralHead<Scalar>::~NeuralHead() {
  delete static_params_pool;
  for (auto p : mem_pools) {
    delete p;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralHead<Scalar>::setup(int n_species, int n_in, int n_layers,
                               const int* layer_sizes, ActivationType actype,
                               Scalar*** h_weights, Scalar*** h_biases,
                               bool use_resnet_dt, bool apply_output_bias) {
  int i;
  this->ctx.n_species = n_species;
  this->ctx.n_layers = n_layers;
  this->ctx.use_resnet_dt = use_resnet_dt;
  this->ctx.apply_output_bias = apply_output_bias;
  this->ctx.actype = actype;
  this->ctx.full_sizes.push_back(n_in);
  this->ctx.n_in = n_in;
  this->ctx.n_out = layer_sizes[n_layers - 1];
  int n_resnet_blocks = 0;
  for (i = 0; i < n_layers; i++) {
    this->ctx.full_sizes.push_back(layer_sizes[i]);
    if (i > 0 && this->ctx.full_sizes[i] == this->ctx.full_sizes[i + 1]) {
      n_resnet_blocks++;
    }
  }
  if (n_resnet_blocks == 0) {
    this->ctx.use_resnet_dt = false;
  }

  // Initialize the memory pool for static parameters (weights and biases)
  static_params_pool = new MemoryPool<Scalar>(
      this->ctx.device, "NeuralHead::Pool::static_params", false);
  int n_scalars = 0;
  for (int elti = 0; elti < n_species; elti++) {
    for (i = 0; i < n_layers; i++) {
      n_scalars += layer_sizes[i] * ctx.full_sizes[i];  // weights
      if (i < n_layers - 1 || apply_output_bias) {
        n_scalars += layer_sizes[i];  // biases
      }
    }
  }
  static_params_pool->grow(n_species * n_scalars);

  // Copy weights and biases to device
  copy_params_to_device(h_weights, h_biases);
}

template <typename Scalar>
void NeuralHead<Scalar>::copy_params_to_device(Scalar*** h_weights,
                                               Scalar*** h_biases) {
  // Copy weights and biases and set up pointers
  this->ctx.d_weights = new Scalar**[ctx.n_species];
  this->ctx.d_biases = new Scalar**[ctx.n_species];
  for (int elti = 0; elti < this->ctx.n_species; elti++) {
    this->ctx.d_weights[elti] = new Scalar*[this->ctx.n_layers];
    this->ctx.d_biases[elti] = new Scalar*[this->ctx.n_layers];
    for (int i = 0; i < this->ctx.n_layers; i++) {
      int n_rows = this->ctx.full_sizes[i];
      int n_cols = this->ctx.full_sizes[i + 1];
      this->ctx.d_weights[elti][i] =
          static_params_pool->request(n_rows * n_cols);
      Kernels::Utils::copy_data_to_device(this->ctx.d_weights[elti][i],
                                          h_weights[elti][i], n_rows * n_cols,
                                          this->ctx.device);
      if (i < this->ctx.n_layers - 1 || ctx.apply_output_bias) {
        this->ctx.d_biases[elti][i] = static_params_pool->request(n_cols);
        Kernels::Utils::copy_data_to_device(this->ctx.d_biases[elti][i],
                                            h_biases[elti][i], n_cols,
                                            this->ctx.device);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralHead<Scalar>::prepare_step_memory(int elti, int n) {
  // Calculate required size
  // We need 2 ping-pong buffers of size [n * max_layer_width]
  // We need nlayers-1 hidden state buffers of size [n * layer_width]

  // We should also check the final output layer to determine max_width, because
  // the output layer can also be wide (e.g. FiniteTempEnergyHeadV1->H).
  int max_width = 0;
  for (int i = 1; i < ctx.n_layers + 1; ++i) {
    max_width = std::max(max_width, ctx.full_sizes[i]);
  }

  size_t total_scalars = 2 * n * max_width;  // buf1, buf2
  for (int i = 1; i < ctx.n_layers; ++i) {
    total_scalars += n * ctx.full_sizes[i];  // actvals[i]
  }

  // Resize Pool
  if (mem_pools.size() <= elti) {
    mem_pools.resize(ctx.n_species);
    step_data.resize(ctx.n_species);
  }
  if (!mem_pools[elti]) {
    mem_pools[elti] =
        new MemoryPool<Scalar>(this->ctx.device, "NeuralHead::Pool::step");
  }

  mem_pools[elti]->grow(total_scalars);
  mem_pools[elti]->reset();

  // Assign Pointers
  step_data[elti].d_ping = mem_pools[elti]->request(n * max_width);
  step_data[elti].d_pong = mem_pools[elti]->request(n * max_width);

  // Hidden layers need an array of pointers.
  // On CPU this is easy. On GPU, the kernel needs `Scalar**`.
  // Let's store the pointers in a std::vector for the Dispatcher to use
  // The Dispatcher will handle copying `Scalar**` to device if required by the
  // CUDA kernel. For now, we just allocate the data in the pool.
  if (step_data[elti].d_hidden == nullptr) {
    step_data[elti].d_hidden = new Scalar*[ctx.n_layers];
  }
  for (int i = 0; i < ctx.n_layers - 1; ++i) {
    step_data[elti].d_hidden[i] =
        mem_pools[elti]->request(n * ctx.full_sizes[i + 1]);
  }
  step_data[elti].nlocal = n;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralHead<Scalar>::compute(int elti, int n, const Scalar* d_x,
                                 const int* d_tlist, Scalar& h_total,
                                 Scalar* h_outputs, Scalar* d_grad) {
  forward(elti, n, d_x, d_tlist, h_total, h_outputs);
  backward(elti, nullptr, d_grad);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralHead<Scalar>::forward(int elti, int n, const Scalar* d_x,
                                 const int* d_tlist, Scalar& h_total,
                                 Scalar* h_outputs) {
  if (n == 0) return;
  prepare_step_memory(elti, n);

  // Dispatch to Kernel
  // The kernel handles the loop over layers
  TimedKernel kid = this->kernel == TimedKernel::Null
                        ? TimedKernel::head_forward
                        : this->kernel;
  Kernels::Head::nnp_forward(&this->ctx, elti, n, d_x, d_tlist, h_total,
                             h_outputs, this->step_data[elti].d_hidden,
                             this->step_data[elti].d_ping,
                             this->step_data[elti].d_pong, kid);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralHead<Scalar>::backward(int elti, const Scalar* grad_in,
                                  Scalar* grad_out) {
  // Assumes forward was called and step_data is valid
  TimedKernel kid = this->kernel == TimedKernel::Null
                        ? TimedKernel::head_backward
                        : this->kernel;
  Kernels::Head::nnp_backward(&this->ctx, elti, this->step_data[elti].nlocal,
                              grad_in, grad_out, this->step_data[elti].d_hidden,
                              this->step_data[elti].d_ping,
                              this->step_data[elti].d_pong, kid, this->use_W_T);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double NeuralHead<Scalar>::get_memory_usage() const {
  double bytes = 0.0;
  for (const auto& pool : mem_pools) {
    if (pool == nullptr) {
      continue;
    }
    bytes += pool->get_memory_usage();
  }
  if (static_params_pool) {
    bytes += static_params_pool->get_memory_usage();
  }
  return bytes;
}

/* ---------------------------------------------------------------------- */

template class NeuralHead<float>;
template class NeuralHead<double>;

}  // namespace TENSORMD
