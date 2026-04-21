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

#include "tensormd_strategy.h"

#include <cstdarg>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tensormd_device.h"

namespace TENSORMD {

Strategy::STRATEGY STRATEGY = Strategy::STRATEGY::Baseline;

bool TENSORMD_PGW_K_CONTINUOUS = false;
bool TENSORMD_USE_FP32_ATLAS = false;
bool TENSORMD_GROUP_BY_TYPE = false;
double TENSORMD_MEMORY_GROWTH_RATE = 1.2;
bool TENSORMD_VERBOSE_DEVICE_MAP = false;

void setup_strategy_from_env(Device dev, std::vector<FILE*>& streams,
                             bool verbose) {
  auto may_print_log = [&](const char* format, ...) {
    if (!streams.empty() && verbose) {
      for (const auto& stream : streams) {
        if (stream) {
          va_list args;
          va_start(args, format);
          vfprintf(stream, format, args);
          va_end(args);
        }
      }
    }
  };

  char* var = std::getenv("TENSORMD_STRATEGY");
  if (var != nullptr) {
    if (strcmp(var, "Exact") == 0) {
      STRATEGY = Strategy::STRATEGY::Exact;
    } else if (strcmp(var, "Baseline") == 0) {
      STRATEGY = Strategy::STRATEGY::Baseline;
    } else if (strcmp(var, "Speed") == 0) {
      STRATEGY = Strategy::STRATEGY::Speed;
    } else if (strcmp(var, "Production") == 0) {
      STRATEGY = Strategy::STRATEGY::Production;
    } else if (strcmp(var, "Scale") == 0) {
      STRATEGY = Strategy::STRATEGY::Scale;
    } else {
      std::string msg =
          "Invalid value for TENSORMD_STRATEGY: " + std::string(var);
      throw std::runtime_error(msg);
    }
    may_print_log("[TensorMD] Strategy: %s\n", var);
  }
  if (dev == Device::GPU) {
    // GPU uses K-continuous memory of P, G, dEdG and W for memory coalesce.
    TENSORMD_PGW_K_CONTINUOUS = true;
    may_print_log("[TensorMD] P/G/W memory layout: K-continuous (GPU).\n");
  } else {
    var = std::getenv("TENSORMD_PGW_K_CONTINUOUS");
    if (var != nullptr) {
      std::string layout;
      if (is_var_enabled(var)) {
        TENSORMD_PGW_K_CONTINUOUS = true;
        layout = "K-continuous";
      } else if (!is_var_enabled(var)) {
        TENSORMD_PGW_K_CONTINUOUS = false;
        layout = "D/m-continuous";
      } else {
        std::string msg =
            "Invalid value for TENSORMD_PGW_K_CONTINUOUS: " + std::string(var);
        throw std::runtime_error(msg);
      }
      may_print_log("[TensorMD] P/G/W memory layout: %s\n", layout.c_str());
    } else {
      may_print_log("[TensorMD] P/G/W memory layout: D/m-continuous.\n");
    }
  }
  var = std::getenv("TENSORMD_USE_FP32_ATLAS");
  if (var != nullptr) {
    if (is_var_enabled(var)) {
      TENSORMD_USE_FP32_ATLAS = true;
      may_print_log("[TensorMD] Using FP32 for interpolation atlas.\n");
    } else if (!is_var_enabled(var)) {
      TENSORMD_USE_FP32_ATLAS = false;
      may_print_log(
          "[TensorMD] Using native precision for interpolation atlas.\n");
    } else {
      std::string msg =
          "Invalid value for TENSORMD_USE_FP32_ATLAS: " + std::string(var);
      throw std::runtime_error(msg);
    }
  }
  var = std::getenv("TENSORMD_GROUP_BY_TYPE");
  if (var != nullptr) {
    if (is_var_enabled(var)) {
      TENSORMD_GROUP_BY_TYPE = true;
      may_print_log("[TensorMD] Grouping neighbors by type during setup.\n");
    } else if (!is_var_enabled(var)) {
      TENSORMD_GROUP_BY_TYPE = false;
      may_print_log(
          "[TensorMD] Not grouping neighbors by type during setup.\n");
    } else {
      std::string msg =
          "Invalid value for TENSORMD_GROUP_BY_TYPE: " + std::string(var);
      throw std::runtime_error(msg);
    }
  }
  var = std::getenv("TENSORMD_MEMORY_GROWTH_RATE");
  if (var != nullptr) {
    TENSORMD_MEMORY_GROWTH_RATE = std::stod(var);
    if (TENSORMD_MEMORY_GROWTH_RATE <= 1.0) {
      std::string msg = "TENSORMD_MEMORY_GROWTH_RATE should be > 1.0: " +
                        std::to_string(TENSORMD_MEMORY_GROWTH_RATE);
      throw std::runtime_error(msg);
    } else {
      may_print_log("[TensorMD] Memory growth rate: %.2f\n",
                    TENSORMD_MEMORY_GROWTH_RATE);
    }
  }
  var = std::getenv("TENSORMD_VERBOSE_DEVICE_MAP");
  if (var != nullptr) {
    if (is_var_enabled(var)) {
      TENSORMD_VERBOSE_DEVICE_MAP = true;
    } else if (!is_var_enabled(var)) {
      TENSORMD_VERBOSE_DEVICE_MAP = false;
    } else {
      std::string msg =
          "Invalid value for TENSORMD_VERBOSE_DEVICE_MAP: " + std::string(var);
      throw std::runtime_error(msg);
    }
  }
}

std::string strategy_to_string(Strategy::STRATEGY strategy) {
  switch (strategy) {
    case Strategy::STRATEGY::Exact:
      return "Exact (Full MLP Calculation, interpolation is disabled)";
    case Strategy::STRATEGY::Baseline:
      return "Baseline (SC'25 implementation)";
    case Strategy::STRATEGY::Speed:
      return "Speed (Faster simulation with increased memory usage)";
    case Strategy::STRATEGY::Production:
      return "Production (Large scale simulation with reduced memory usage)";
    case Strategy::STRATEGY::Scale:
      return "Scale (Extremely large scale simulation)";
    default:
      return "Unknown";
  }
}

}  // namespace TENSORMD