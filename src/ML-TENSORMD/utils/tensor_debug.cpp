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

#include "tensor_debug.h"

#include <cstdio>
#include <string>
#include <vector>

#include "../../fmt/format.h"
#include "../kernels/kernels.h"
#include "../kernels/kernels_utils.h"

namespace TENSORMD::Debug {

#define GET_DUMP_PREFIX()                                \
  std::string prefix = "dump";                           \
  const char* var = std::getenv("TENSORMD_DUMP_PREFIX"); \
  if (var != nullptr) {                                  \
    prefix = std::string(var);                           \
  }

template <typename Scalar>
void dump_tensor(Device dev, TensorMap<Tensor<Scalar, 4>>& d_tensor,
                 const char* name, int mpiid) {
  if (d_tensor.data() == nullptr) {
    return;
  }
  auto d0 = static_cast<long int>(d_tensor.dimension(0));
  auto d1 = static_cast<long int>(d_tensor.dimension(1));
  auto d2 = static_cast<long int>(d_tensor.dimension(2));
  auto d3 = static_cast<long int>(d_tensor.dimension(3));
  TensorMap<Tensor<Scalar, 4>>* h_tensor;
  Scalar* h_ptr = nullptr;
  std::vector<Scalar> host_data;
  if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    h_ptr = Kernels::Utils::access_data_on_host(
        d_tensor.data(), d_tensor.size(), dev, host_data);
    h_tensor = new TensorMap<Tensor<Scalar, 4>>(h_ptr, d0, d1, d2, d3);
#endif
  } else if (dev == Device::CPU) {
    h_tensor = &d_tensor;
  } else {
    throw std::runtime_error(
        "[TensorMD::dump_tensor] Unknown or unsupported device type.");
  }

  GET_DUMP_PREFIX();

  FILE* fp = nullptr;
  std::string device_str = (dev == Device::CPU) ? "cpu" : "gpu";
  std::string filename = fmt::format("{}_{}_{}_{}.txt", prefix,
                                     std::string(name), device_str, mpiid);
  fp = fopen(filename.c_str(), "w");
  fprintf(fp, "%6ld %6ld %6ld %6ld\n", d0, d1, d2, d3);
  for (long int i = 0; i < d3; i++) {
    for (long int j = 0; j < d2; j++) {
      for (long int k = 0; k < d1; k++) {
        for (long int m = 0; m < d0; m++) {
          if constexpr (std::is_same_v<Scalar, int>) {
            fprintf(fp, "%6ld %6ld %6ld %6ld %6d\n", m, k, j, i,
                    (*h_tensor)(m, k, j, i));
          } else {
            fprintf(fp, "%6ld %6ld %6ld %6ld %20.12e\n", m, k, j, i,
                    (*h_tensor)(m, k, j, i));
          }
        }
      }
    }
  }
  fclose(fp);
  if (dev == Device::GPU) {
    delete h_tensor;
  }
}

template <typename Scalar>
void dump_tensor(Device dev, TensorMap<Tensor<Scalar, 3>>& d_tensor,
                 const char* name, int mpiid) {
  if (d_tensor.data() == nullptr) {
    return;
  }
  auto d0 = static_cast<long int>(d_tensor.dimension(0));
  auto d1 = static_cast<long int>(d_tensor.dimension(1));
  auto d2 = static_cast<long int>(d_tensor.dimension(2));
  TensorMap<Tensor<Scalar, 3>>* h_tensor;
  Scalar* h_ptr = nullptr;
  std::vector<Scalar> host_data;
  if (dev == Device::GPU) {
#if defined(TENSORMD_GPU)
    h_ptr = Kernels::Utils::access_data_on_host(
        d_tensor.data(), d_tensor.size(), dev, host_data);
    h_tensor = new TensorMap<Tensor<Scalar, 3>>(h_ptr, d0, d1, d2);
#endif
  } else if (dev == Device::CPU) {
    h_tensor = &d_tensor;
  } else {
    throw std::runtime_error(
        "[TensorMD::dump_tensor] Unknown or unsupported device type.");
  }

  GET_DUMP_PREFIX();

  FILE* fp = nullptr;
  std::string device_str = (dev == Device::CPU) ? "cpu" : "gpu";
  std::string filename = fmt::format("{}_{}_{}_{}.txt", prefix,
                                     std::string(name), device_str, mpiid);
  fp = fopen(filename.c_str(), "w");
  fprintf(fp, "%6ld %6ld %6ld\n", d0, d1, d2);
  for (long int i = 0; i < d2; i++) {
    for (long int j = 0; j < d1; j++) {
      for (long int k = 0; k < d0; k++) {
        if constexpr (std::is_same_v<Scalar, int>) {
          fprintf(fp, "%6ld %6ld %6ld %6d\n", k, j, i, (*h_tensor)(k, j, i));
        } else {
          fprintf(fp, "%6ld %6ld %6ld %20.12e\n", k, j, i,
                  (*h_tensor)(k, j, i));
        }
      }
    }
  }
  fclose(fp);
  if (dev == Device::GPU) {
    delete h_tensor;
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template void dump_tensor<float>(Device dev, TensorMap<Tensor<float, 4>>& in,
                                 const char* name, int mpiid);
template void dump_tensor<double>(Device dev, TensorMap<Tensor<double, 4>>& in,
                                  const char* name, int mpiid);
template void dump_tensor<int>(Device dev, TensorMap<Tensor<int, 4>>& in,
                               const char* name, int mpiid);

template void dump_tensor<float>(Device dev, TensorMap<Tensor<float, 3>>& in,
                                 const char* name, int mpiid);
template void dump_tensor<double>(Device dev, TensorMap<Tensor<double, 3>>& in,
                                  const char* name, int mpiid);
template void dump_tensor<int>(Device dev, TensorMap<Tensor<int, 3>>& in,
                               const char* name, int mpiid);

}  // namespace TENSORMD::Debug
