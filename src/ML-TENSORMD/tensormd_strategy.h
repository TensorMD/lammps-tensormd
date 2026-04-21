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

#ifndef TENSORMD_STRATEGY_H
#define TENSORMD_STRATEGY_H

#include <string>
#include <vector>
#include <cstring>

#include "tensormd_device.h"

namespace TENSORMD {

inline bool is_var_enabled(const char* var) {
  return (strcmp(var, "1") == 0 || strcmp(var, "yes") == 0 ||
          strcmp(var, "on") == 0 || strcmp(var, "true") == 0);
}

namespace Strategy {
enum STRATEGY {
  Exact = 1,
  Baseline = 2,
  Speed = 3,
  Production = 4,
  Scale = 5,
};
}

// The contraction strategy
extern Strategy::STRATEGY STRATEGY;

// Are tensors P/G/W using K-continuous memory layout
// (AoS with K as the innermost dimension)?
extern bool TENSORMD_PGW_K_CONTINUOUS;

// Whether to use FP32 for the interpolation atlas
extern bool TENSORMD_USE_FP32_ATLAS;

// Whether to group neighbors of the same type during setup
extern bool TENSORMD_GROUP_BY_TYPE;

// The growth rate of the memory pool when resizing is needed. Should be >1.0.
// Default is 1.2
extern double TENSORMD_MEMORY_GROWTH_RATE;

// Verbose the device map for debugging during setup. Default is false.
extern bool TENSORMD_VERBOSE_DEVICE_MAP;

// Check environment variables to update strategy and other global options.
void setup_strategy_from_env(Device dev, std::vector<FILE*>& streams,
                             bool verbose = false);

// Return the string representation of the strategy
std::string strategy_to_string(Strategy::STRATEGY strategy);

}  // namespace TENSORMD

#endif
