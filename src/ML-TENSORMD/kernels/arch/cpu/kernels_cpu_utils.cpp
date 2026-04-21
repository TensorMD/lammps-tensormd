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

#include <chrono>

#include "../../../tensormd_constants.h"
#include "../kernels_internal_utils.h"

namespace TENSORMD::Kernels::Utils::CPU {
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using std::chrono::duration;
using std::chrono::duration_cast;

// Global storage for CPU timings (Per MPI Rank)
static TimePoint start_times[N_TIMED_KERNELS];
static double accumulated_seconds[N_TIMED_KERNELS];
static double accumulated_FLOPs[N_TIMED_KERNELS];
static bool initialized = false;

void timer_init() {
  if (initialized) return;
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    accumulated_seconds[i] = 0.0;
    accumulated_FLOPs[i] = 0.0;
  }
  initialized = true;
}

void timer_destroy() { initialized = false; }

void timer_start(int kernel_id) {
  if (!initialized) return;
  if (kernel_id < N_TIMED_KERNELS) {
    start_times[kernel_id] = Clock::now();
  }
}

void timer_stop(int kernel_id, double FLOPs) {
  if (!initialized) return;
  if (kernel_id < N_TIMED_KERNELS) {
    auto end_time = Clock::now();
    auto elapsed =
        duration_cast<duration<double>>(end_time - start_times[kernel_id])
            .count();
    accumulated_seconds[kernel_id] += elapsed;
    accumulated_FLOPs[kernel_id] += FLOPs;
  }
}

void timer_accumulate() {
  // do nothing because we already accumulate in timer_stop for CPU
}

void timer_collect(double* times, double* FLOPs) {
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    times[i] = accumulated_seconds[i];
    FLOPs[i] = accumulated_FLOPs[i];
  }
}

}  // namespace TENSORMD::Kernels::Utils::CPU