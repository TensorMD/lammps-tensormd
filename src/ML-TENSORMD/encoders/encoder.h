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

#ifndef TENSORMD_ENCODER_H
#define TENSORMD_ENCODER_H

#include "cnpy++.hpp"
#include "cutoff.h"
#include "encoder_context.h"

namespace TENSORMD {

template <typename Scalar>
class Encoder {
 public:
  Encoder(Device dev, Scalar rcut)
      : ctx(dev),
        map_pool(dev, "Encoder::Pool::map", false),
        atlas_pool_fp32(dev, "Encoder::Pool::atlas_fp32", false),
        atlas_pool(dev, "Encoder::Pool::atlas", false) {
    cutoff = nullptr;

    ctx.ready_to_interpolate = false;
    ctx.num_radial_features = 0;
    ctx.grid_dr = 0;
    ctx.grid_size = 0;
    ctx.grid_rmin = 0;
    ctx.grid_rmax = rcut;
    ctx.num_pairs = 0;
    ctx.d_pair_map = nullptr;
    ctx.d_params_atlas = nullptr;
    ctx.d_params_atlas_fp32 = nullptr;
  }

  virtual ~Encoder() { delete cutoff; };

  // Read the npz file to initialize the encoder
  virtual void initialize(cnpypp::npz_t& npz) = 0;

  // Forward propagation: radial(R) * cutoff(R) -> H(R)
  virtual void forward(TensorData<Scalar>* data);

  // Backward propagation
  virtual void backward(TensorData<Scalar>* data);

  // Return the number of radial features (k)
  [[nodiscard]] virtual int get_num_radial_features() const {
    return ctx.num_radial_features;
  };

  // Interpolation
  void setup_interpolation(std::vector<std::string>& species, Scalar rmin,
                           Scalar delta);
  [[nodiscard]] virtual bool is_interpolatable() const {
    return ctx.ready_to_interpolate;
  }
  [[nodiscard]] virtual Scalar get_interp_dr() const { return ctx.grid_dr; }
  [[nodiscard]] virtual Scalar get_interp_r0() const { return ctx.grid_rmin; }
  [[nodiscard]] virtual int get_interp_size() const { return ctx.grid_size; }
  [[nodiscard]] Scalar* get_params_atlas() const { return ctx.d_params_atlas; }
  [[nodiscard]] int* get_pair_map() const { return ctx.d_pair_map; }
  [[nodiscard]] int get_num_pairs() const { return ctx.num_pairs; }

  // Return the memory usage
  [[nodiscard]] virtual double get_memory_usage() const {
    return atlas_pool.get_memory_usage() + map_pool.get_memory_usage();
  };

  // Return the context for kernels
  const EncoderContext<Scalar>& get_context() const { return ctx; }

 protected:
  Cutoff<Scalar>* cutoff;
  EncoderContext<Scalar> ctx;

  // Use MemoryPool to manage device memory automatically
  MemoryPool<Scalar> atlas_pool;
  MemoryPool<float> atlas_pool_fp32;
  MemoryPool<int> map_pool;

  // Calculate the radial interaction tensor: R -> radial(R), dradial(R)
  virtual void radial_forward(TensorData<Scalar>* data) = 0;

  // Backward propagation: grad_in, dHdr -> grad_out
  virtual void radial_backward(TensorData<Scalar>* data) = 0;

  // The internal setup functions for interpolation
  // The default setup function is for rational/cubic interpolation
  virtual void setup_interpolation_default(std::vector<std::string> species,
                                           Scalar rmin, Scalar delta) = 0;
};

}  // namespace TENSORMD

#endif