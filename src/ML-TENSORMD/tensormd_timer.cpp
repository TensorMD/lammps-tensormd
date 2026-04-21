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

#include "tensormd_timer.h"

#include <cstdio>

#include "kernels/kernels_utils.h"
#include "mpi.h"
#include "tensormd_strategy.h"

namespace TENSORMD {

inline std::string kernel_to_string(TimedKernel kernel) {
  switch (kernel) {
    case TimedKernel::Setup:
      return "Setup";
    case TimedKernel::get_exact_cmax:
      return "cmax";
    case TimedKernel::neighbor_list:
      return "neighborlist";
    case TimedKernel::sync_lammps_data:
      return "sync";
    case TimedKernel::setup_tensors:
      return "setup";
    case TimedKernel::Descriptor:
      return "Descriptors";
    case TimedKernel::interpolate:
      return "interpolate";
    case TimedKernel::calc_sij:
      return "sij";
    case TimedKernel::apply_sij:
      return "apply_sij";
    case TimedKernel::v2_embed_input:
      return "v2_embed_input";
    case TimedKernel::v2_embed_grad:
      return "v2_embed_grad";
    case TimedKernel::forward:
      return "forward";
    case TimedKernel::calc_P:
      return "P";
    case TimedKernel::calc_P_fused:
      return "P(fused)";
    case TimedKernel::calc_G:
      return "G";
    case TimedKernel::Energy:
      return "Energy";
    case TimedKernel::head_forward:
      return "head_forward";
    case TimedKernel::head_backward:
      return "head_backward";
    case TimedKernel::head_fused:
      return "head_fused";
    case TimedKernel::head_hybrid:
      return "head_hybrid";
    case TimedKernel::v1_head_ops:
      return "v1_head_ops";
    case TimedKernel::Gradients:
      return "Gradients";
    case TimedKernel::calc_W:
      return "W";
    case TimedKernel::fused_forces:
      return "F(fused)";
    case TimedKernel::RadialForces:
      return "F(radial)";
    case TimedKernel::calc_Up:
      return "Up";
    case TimedKernel::calc_U:
      return "U";
    case TimedKernel::backward:
      return "backward";
    case TimedKernel::calc_Fr:
      return "Fr";
    case TimedKernel::AngularForces:
      return "F(angular)";
    case TimedKernel::calc_V:
      return "V";
    case TimedKernel::calc_Fa:
      return "Fa";
    case TimedKernel::SumForces:
      return "F(sum)";
    case TimedKernel::Overall:
      return "Overall";
    case TimedKernel::TheOneKernel:
      return "TheOneKernel";
    default:
      return "Unknown";
  }
}

inline std::string get_FLOPs_symbol(double value) {
  if (value >= 1e21) {
    return "Z";
  } else if (value >= 1e18) {
    return "E";
  } else if (value >= 1e15) {
    return "P";
  } else if (value >= 1e12) {
    return "T";
  } else if (value >= 1e9) {
    return "G";
  } else if (value >= 1e6) {
    return "M";
  } else if (value >= 1e3) {
    return "K";
  } else {
    return " ";
  }
}

inline double get_FLOPs_scale(double value) {
  if (value >= 1e21) {
    return 1e21;
  } else if (value >= 1e18) {
    return 1e18;
  } else if (value >= 1e15) {
    return 1e15;
  } else if (value >= 1e12) {
    return 1e12;
  } else if (value >= 1e9) {
    return 1e9;
  } else if (value >= 1e6) {
    return 1e6;
  } else if (value >= 1e3) {
    return 1e3;
  } else {
    return 1.0;
  }
}

/* ---------------------------------------------------------------------- */

TensorMDTimer::TensorMDTimer(Device dev, size_t float_precision) {
  device = dev;
  initialized = false;
  use_interpolation = false;
  require_vatom = false;
  imax = 0;
  cmax = 0;
  amax = 0;
  interp_dr = 0;
  precision = float_precision;
}

/* ---------------------------------------------------------------------- */

TensorMDTimer::~TensorMDTimer() {
  Kernels::Utils::Profiling::timer_destroy(device);
};

/* ---------------------------------------------------------------------- */

void TensorMDTimer::process_child_kernel_stats() {
  for (auto& kernel : metrics) {
    if (kernel.second.parent != TimedKernel::Null) {
      auto& parent_stat = metrics[kernel.second.parent];
      parent_stat.accumulated_FLOPs += kernel.second.accumulated_FLOPs;
      parent_stat.accumulated_time += kernel.second.accumulated_time;
    }
  }
  std::vector<TimedKernel> remove;
  for (auto& kernel : metrics) {
    if (kernel.second.accumulated_time < 1e-12) {
      if (kernel.first != TimedKernel::Setup &&
          kernel.second.parent != TimedKernel::Setup) {
        remove.push_back(kernel.first);
      }
    }
  }
  for (auto& kernel : remove) {
    metrics.erase(kernel);
  }
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::average_across_ranks(
    std::map<std::string, double>& memory_usages) {
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  double factor = 1.0 / static_cast<double>(nprocs);
  double rank_kernel_FLOPs, rank_kernel_time, rank_memory;
  double accumulated_FLOPs = 0.0;
  double accumulated_time = 0.0;
  double accumulated_memory = 0.0;

  for (auto& kernel : this->metrics) {
    if (kernel.first == TimedKernel::Overall) {
      continue;  // Skip overall for now, will calculate at the end
    }
    MPI_Reduce(&kernel.second.accumulated_FLOPs, &rank_kernel_FLOPs, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&kernel.second.accumulated_time, &rank_kernel_time, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    kernel.second.accumulated_FLOPs = rank_kernel_FLOPs * factor;
    kernel.second.accumulated_time = rank_kernel_time * factor;
    // Only top-level kernels (with parent == Null) contribute to the
    // overall FLOPs and time
    if (kernel.second.parent == TimedKernel::Null) {
      accumulated_FLOPs += kernel.second.accumulated_FLOPs;
      accumulated_time += kernel.second.accumulated_time;
    }
  }
  metrics[TimedKernel::Overall].accumulated_FLOPs = accumulated_FLOPs;
  metrics[TimedKernel::Overall].accumulated_time = accumulated_time;

  for (auto& usage : memory_usages) {
    MPI_Reduce(&usage.second, &rank_memory, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    // Convert to MB
    usage.second = rank_memory * factor / (1024.0 * 1024.0);
    accumulated_memory += usage.second;
  }
  memory_usages["Total"] = accumulated_memory;
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::collect_and_print(
    std::vector<FILE*>& c_streams,
    std::map<std::string, double> memory_usages) {
  if (!enabled || !initialized) return;
  if (c_streams.empty()) return;

  // Collect accumulated times from device (CPU or GPU)
  std::vector<double> collected_times(N_TIMED_KERNELS, 0.0);
  std::vector<double> collected_FLOPs(N_TIMED_KERNELS, 0.0);
  Kernels::Utils::Profiling::timer_collect(device, collected_times.data(),
                                           collected_FLOPs.data());

  // Update the accumulated times in metrics
  for (auto& kernel_stat : metrics) {
    int id = kernel_stat.first;
    if (id < N_TIMED_KERNELS) {
      kernel_stat.second.accumulated_time = collected_times[id];
      kernel_stat.second.accumulated_FLOPs = collected_FLOPs[id];
    }
  }

  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Process the metrics to add up child kernel stats to parent kernel stats
  process_child_kernel_stats();

  // Reduce FLOPs, times and memory usages across ranks, and calculate the
  // total accumulated FLOPs, time and memory usages
  average_across_ranks(memory_usages);

  // Only rank 0 will print the results
  if (rank != 0) return;

  // clang-format off
  for (const auto& c_stream : c_streams) {
    if (c_stream == nullptr) {
      continue;
    }
    fprintf(c_stream, "--------------------- TensorMD Profiling Summary ----------------------\n");
    fprintf(c_stream, "     Strategy: %s\n", strategy_to_string(STRATEGY).c_str());
    fprintf(c_stream, "          All metrics are averaged across %d MPI ranks\n", nprocs);
    fprintf(c_stream, "       Float Precision: %zu bits\n", precision * 8);
    fprintf(c_stream, "                Device: %s\n", device == Device::CPU ? "CPU" : "GPU");
    fprintf(c_stream, "           Tensor Dims: b = %d, d = %d, k = %d, m = %d\n", dims.b, dims.d, dims.k, dims.m);
    fprintf(c_stream, "                        a = %d\n", dims.a);
    fprintf(c_stream, "                     amax = %d\n", amax);
    fprintf(c_stream, "                        c = %d\n", dims.c);
    fprintf(c_stream, "                     cmax = %d\n", cmax);
    fprintf(c_stream, "                     imax = %d\n", imax);
    fprintf(c_stream, "                    vatom = %s\n", require_vatom ? "yes" : "no");
    if (use_interpolation) {
      fprintf(c_stream, "                interp_dr = %.4f\n", interp_dr);
    }
    fprintf(c_stream, "--------------------------- Memory Breakdown --------------------------\n");
    fprintf(c_stream, "              Total Memory  %12.2f MB\n", memory_usages["Total"]);
    fprintf(c_stream, "        Memory Growth Rate  %12.2f %\n", TENSORMD_MEMORY_GROWTH_RATE * 100.0);
    for (const auto& usage : memory_usages) {
      if (usage.first != "Total") {
        fprintf(c_stream, "%26s  %12.2f MB\n", usage.first.c_str(), usage.second);
      }
    }
    fprintf(c_stream, "-----------------------------------------------------------------------\n");

    for (const auto& kernel : metrics) {
      double FLOPs = kernel.second.accumulated_FLOPs;
      auto u1 = get_FLOPs_symbol(FLOPs);
      auto s1 = get_FLOPs_scale(FLOPs);
      double time = kernel.second.accumulated_time;
      double FLOPS = time > 0.0 ? FLOPs / time : 0.0;
      auto u2 = get_FLOPs_symbol(FLOPS);
      double s2 = get_FLOPs_scale(FLOPS);
      FLOPs /= s1;
      FLOPS /= s2;
      std::string name;
      if (kernel.second.parent != TimedKernel::Null) {
        name = " -" + kernel_to_string(kernel.first);
      } else {
        name = kernel_to_string(kernel.first);
      }
      fprintf(c_stream,
              "%-16s  %12.4f seconds, %8.3f %1sFLOPs, %7.3f %1sFLOPS\n",
              name.c_str(), time, FLOPs, u1.c_str(), FLOPS, u2.c_str());
    }

    fprintf(c_stream, "-----------------------------------------------------------------------\n");
  }
  // clang-format on
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::setup(int b, int d, int k, int m, bool use_interp,
                          double dr, bool vatom) {
  this->dims.b = b;
  this->dims.d = d;
  this->dims.k = k;
  this->dims.m = m;
  this->use_interpolation = use_interp;
  this->interp_dr = dr;
  this->require_vatom = vatom;

  Kernels::Utils::Profiling::timer_init(device);
  this->initialized = true;
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::step(int a, int c, int imax_) {
  dims.a = a;
  dims.c = c;
  imax = imax_;
  cmax = std::max(cmax, c);
  amax = std::max(amax, a);
}

void TensorMDTimer::accumulate() {
  if (!enabled || !initialized) return;
  Kernels::Utils::Profiling::timer_accumulate(device);
}

/* ---------------------------------------------------------------------- */

}  // namespace TENSORMD
