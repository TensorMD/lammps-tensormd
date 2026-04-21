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

#ifndef LAMMPS_TENSORMD_TIMER_H
#define LAMMPS_TENSORMD_TIMER_H

#include <map>
#include <string>

#include "tensormd_constants.h"
#include "tensormd_device.h"
#include "tensormd_types.h"

namespace TENSORMD {

using Profiling::TimedKernel;

class TensorMDTimer {
 public:
  TensorMDTimer(Device dev, size_t float_precision);
  ~TensorMDTimer();

  // Public attribute for easy turning on/off the timer.
  bool enabled = false;

  // Initialize the constant dimensions
  // This will also update FLOPs_factor for kernels
  void setup(int b, int d, int k, int m, bool use_interpolation = false,
             double dr = 0.0, bool require_vatom = false);

  // Call at each step to update a and c
  // imax, npages and pagesize are saved for analyzing memory usage.
  void step(int a, int c, int imax);

  // Accumulate the recorded times from device timers
  void accumulate();

  // Print statistics
  void collect_and_print(std::vector<FILE*>& c_streams,
                         std::map<std::string, double> memory_usages);

  [[nodiscard]] bool is_initialized() const { return initialized; }

 protected:
  Device device;
  bool use_interpolation;
  bool initialized;
  TensorDims dims;
  int imax;
  int cmax;
  int amax;
  size_t precision;
  double interp_dr;
  bool require_vatom{};

  struct time_record_t {
    TimedKernel parent = TimedKernel::Null;
    double FLOPs_factor = 0.0;
    double accumulated_FLOPs = 0.0;
    double accumulated_time = 0.0;
    double accumulated_memory = 0.0;
  };

  // Initialize the internal time metrics
  std::map<TimedKernel, time_record_t> metrics = {
      {TimedKernel::Setup, {TimedKernel::Null}},
      {TimedKernel::neighbor_list, {TimedKernel::Setup}},
      {TimedKernel::get_exact_cmax, {TimedKernel::Setup}},
      {TimedKernel::sync_lammps_data, {TimedKernel::Setup}},
      {TimedKernel::setup_tensors, {TimedKernel::Setup}},
      {TimedKernel::Descriptor, {TimedKernel::Null}},
      {TimedKernel::interpolate, {TimedKernel::Descriptor}},
      {TimedKernel::calc_sij, {TimedKernel::Descriptor}},
      {TimedKernel::forward, {TimedKernel::Descriptor}},
      {TimedKernel::apply_sij, {TimedKernel::Descriptor}},
      {TimedKernel::v2_embed_input, {TimedKernel::Descriptor}},
      {TimedKernel::v2_embed_grad, {TimedKernel::Descriptor}},
      {TimedKernel::calc_P, {TimedKernel::Descriptor}},
      {TimedKernel::calc_P_fused, {TimedKernel::Descriptor}},
      {TimedKernel::calc_G, {TimedKernel::Descriptor}},
      {TimedKernel::Energy, {TimedKernel::Null}},
      {TimedKernel::head_forward, {TimedKernel::Energy}},
      {TimedKernel::head_backward, {TimedKernel::Energy}},
      {TimedKernel::head_fused, {TimedKernel::Energy}},
      {TimedKernel::head_hybrid, {TimedKernel::Energy}},
      {TimedKernel::v1_head_ops, {TimedKernel::Energy}},
      {TimedKernel::Gradients, {TimedKernel::Null}},
      {TimedKernel::calc_W, {TimedKernel::Gradients}},
      {TimedKernel::fused_forces, {TimedKernel::Gradients}},
      {TimedKernel::RadialForces, {TimedKernel::Null}},
      {TimedKernel::calc_U, {TimedKernel::RadialForces}},
      {TimedKernel::calc_Up, {TimedKernel::RadialForces}},
      {TimedKernel::backward, {TimedKernel::RadialForces}},
      {TimedKernel::calc_Fr, {TimedKernel::RadialForces}},
      {TimedKernel::AngularForces, {TimedKernel::Null}},
      {TimedKernel::calc_V, {TimedKernel::AngularForces}},
      {TimedKernel::calc_Fa, {TimedKernel::AngularForces}},
      {TimedKernel::TheOneKernel, {TimedKernel::Null}},
      {TimedKernel::SumForces, {TimedKernel::Null}},
      {TimedKernel::Overall, {TimedKernel::Null}},
  };

  // Sum up child kernel stats to parent kernel stats
  void process_child_kernel_stats();

  // Average the accumulated times and memory usages across ranks (for MPI runs)
  void average_across_ranks(std::map<std::string, double>& memory_usages);
};

}  // namespace TENSORMD

#endif
