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

#ifndef TENSORMD_NEURAL_HEAD_H
#define TENSORMD_NEURAL_HEAD_H

#include "../tensormd_timer.h"
#include "../tensormd_types.h"
#include "cnpy++.hpp"
#include "neural_head_context.h"

namespace TENSORMD {

template <typename Scalar>
class NeuralHead {
 public:
  explicit NeuralHead(Device dev);
  virtual ~NeuralHead();

  const NeuralHeadContext<Scalar>& get_context() const { return ctx; }

  // elti: the i-th set of weights and biases to use.
  // n: number of rows
  // d_x: input descriptors, [n, n_in]
  // d_tlist: the type of each row
  // d_total: sum(d_values)
  // d_outputs: output values, [n, n_out]
  // d_grad: the derivative of `d_total` w.r.t `d_x`
  virtual void compute(int elti, int n, const Scalar* d_x, const int* d_tlist,
                       Scalar& h_total, Scalar* h_outputs, Scalar* d_grad);

  // Exposed for lower-level composition
  virtual void forward(int elti, int n, const Scalar* d_x, const int* tlist,
                       Scalar& h_total, Scalar* h_outputs);
  virtual void backward(int elti, const Scalar* grad_in, Scalar* grad_out);

  // Get the total memory usage of this energy head (in bytes)
  [[nodiscard]] virtual double get_memory_usage() const;

  // Release any dynamic memory (e.g., memory pools) used for the forward and
  // backward pass.
  virtual void release_dynamic_memory() {
    for (auto pool : mem_pools) {
      if (pool != nullptr) pool->free();
    }
  };

  [[nodiscard]] int get_n_in() const { return ctx.n_in; }
  [[nodiscard]] int get_n_out() const { return ctx.n_out; }

  // This flag decides the kernel id for profiling.
  // If Null, defaults will be used (head_forward and head_backward).
  TimedKernel kernel = TimedKernel::Null;

  // The complete setup function
  void setup(int nelt, int num_in, int nlayers, const int* layer_sizes,
             ActivationType act, Scalar*** weights, Scalar*** biases,
             bool use_resnet_dt, bool apply_output_bias);

 protected:
  NeuralHeadContext<Scalar> ctx;

  // Use transposed weights in backward pass?
  bool use_W_T = false;

  // Pool to hold all weights and biases on the device
  MemoryPool<Scalar>* static_params_pool = nullptr;

  // For static and temporary variables
  MemoryPool<Scalar> static_pool;

  // Dynamic Memory (Managed by MemoryPool)
  // Per species pools
  std::vector<MemoryPool<Scalar>*> mem_pools;

  // Pointers into the pool (reset every step)
  struct StepData {
    int nlocal;
    Scalar* d_ping;
    Scalar* d_pong;
    Scalar** d_hidden;  // Array of pointers to hidden layers
  };
  std::vector<StepData> step_data;

  // Helper to ensure pools are large enough
  void prepare_step_memory(int elti, int n_atoms);

  // Helper to copy neural network parameters to device
  virtual void copy_params_to_device(Scalar*** h_weights, Scalar*** h_biases);
};

}  // namespace TENSORMD
#endif