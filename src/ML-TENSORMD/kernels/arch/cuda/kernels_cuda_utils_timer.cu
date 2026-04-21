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

#include "../../../tensormd_constants.h"
#include "../kernels_internal_utils.h"
#include "cuda_common.cuh"

namespace TENSORMD::Kernels::Utils::GPU {

static cudaEvent_t start_events[N_TIMED_KERNELS];
static cudaEvent_t stop_events[N_TIMED_KERNELS];
static double accumulated_seconds[N_TIMED_KERNELS];
static double accumulated_FLOPs[N_TIMED_KERNELS];
static bool event_recorded[N_TIMED_KERNELS];
static bool initialized = false;

void timer_init() {
  if (initialized) return;
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    CHECK(cudaEventCreate(&start_events[i]));
    CHECK(cudaEventCreate(&stop_events[i]));
    accumulated_seconds[i] = 0.0;
    accumulated_FLOPs[i] = 0.0;
    event_recorded[i] = false;
  }
  initialized = true;
}

void timer_destroy() {
  if (!initialized) return;
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    CHECK(cudaEventDestroy(start_events[i]));
    CHECK(cudaEventDestroy(stop_events[i]));
    accumulated_seconds[i] = 0.0;
    accumulated_FLOPs[i] = 0.0;
    event_recorded[i] = false;
  }
  initialized = false;
}

void timer_start(int kernel_id) {
  if (!initialized) return;
  if (kernel_id < N_TIMED_KERNELS) {
    CHECK(cudaEventRecord(start_events[kernel_id]));
  }
}

void timer_stop(int kernel_id, double FLOPs) {
  if (!initialized) return;
  if (kernel_id < N_TIMED_KERNELS) {
    CHECK(cudaEventRecord(stop_events[kernel_id]));
    accumulated_FLOPs[kernel_id] += FLOPs;
    event_recorded[kernel_id] = true;
  }
}

void timer_accumulate() {
  // Read events and update accumulators
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    if (event_recorded[i]) {
      float ms = 0.0f;
      cudaError_t err =
          cudaEventElapsedTime(&ms, start_events[i], stop_events[i]);
      if (err == cudaSuccess) {
        accumulated_seconds[i] += (double)ms / 1000.0;
      }
      event_recorded[i] = false;  // Reset for next step
    }
  }
}

void timer_collect(double* times, double* FLOPs) {
  for (int i = 0; i < N_TIMED_KERNELS; ++i) {
    times[i] = accumulated_seconds[i];
    FLOPs[i] = accumulated_FLOPs[i];
  }
}

}  // namespace TENSORMD::Kernels::Utils::GPU
