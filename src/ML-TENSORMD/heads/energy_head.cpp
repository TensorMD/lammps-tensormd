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

#include "energy_head.h"

#include "../kernels/kernels_utils.h"
#include "../tensormd_strategy.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void EnergyHead<Scalar>::copy_params_to_device(Scalar*** h_weights,
                                               Scalar*** h_biases) {
  // Resize the static params pool
  int n_scalars = 0;
  for (int elti = 0; elti < this->ctx.n_species; elti++) {
    for (int i = 0; i < this->ctx.n_layers; i++) {
      int n_rows = this->ctx.full_sizes[i];
      int n_cols = this->ctx.full_sizes[i + 1];
      n_scalars += n_rows * n_cols * 2;  // weights+weights.T
      if (i < this->ctx.n_layers - 1 || this->ctx.apply_output_bias) {
        n_scalars += n_cols;  // biases
      }
    }
  }
  this->static_params_pool->grow(this->ctx.n_species * n_scalars);
  // Copy weights and biases and set up pointers
  this->ctx.d_weights = new Scalar**[this->ctx.n_species];
  this->ctx.d_weights_T = new Scalar**[this->ctx.n_species];
  this->ctx.d_biases = new Scalar**[this->ctx.n_species];
  for (int elti = 0; elti < this->ctx.n_species; elti++) {
    this->ctx.d_weights[elti] = new Scalar*[this->ctx.n_layers];
    this->ctx.d_weights_T[elti] = new Scalar*[this->ctx.n_layers];
    this->ctx.d_biases[elti] = new Scalar*[this->ctx.n_layers];
    for (int i = 0; i < this->ctx.n_layers; i++) {
      int n_rows = this->ctx.full_sizes[i];
      int n_cols = this->ctx.full_sizes[i + 1];
      this->ctx.d_weights[elti][i] =
          this->static_params_pool->request(n_rows * n_cols);
      this->ctx.d_weights_T[elti][i] =
          this->static_params_pool->request(n_rows * n_cols);
      std::vector<Scalar> h_weights_T(n_rows * n_cols);
      if (i == 0) {  // The first weights matrix may need special treatment
        std::vector<Scalar> h_buffer(n_rows * n_cols);
        if (TENSORMD_PGW_K_CONTINUOUS) {
          // Permute ([mk, h] -> [km, h]) and transpose
          permute_W0(h_buffer, h_weights_T, h_weights[elti][i], n_rows, n_cols);
          Kernels::Utils::copy_data_to_device(this->ctx.d_weights[elti][i],
                                              h_buffer.data(), n_rows * n_cols,
                                              this->ctx.device);
        } else {
          // Only transpose
          transpose_W0(h_weights_T, h_weights[elti][i], n_rows, n_cols);
          Kernels::Utils::copy_data_to_device(
              this->ctx.d_weights[elti][i], h_weights[elti][i], n_rows * n_cols,
              this->ctx.device);
        }
      } else {
        Kernels::Utils::copy_data_to_device(this->ctx.d_weights[elti][i],
                                            h_weights[elti][i], n_rows * n_cols,
                                            this->ctx.device);
        // The standard transpose
        for (int row = 0; row < n_rows; row++) {
          for (int col = 0; col < n_cols; col++) {
            h_weights_T[col * n_rows + row] =
                h_weights[elti][i][row * n_cols + col];
          }
        }
      }
      Kernels::Utils::copy_data_to_device(this->ctx.d_weights_T[elti][i],
                                          h_weights_T.data(), n_rows * n_cols,
                                          this->ctx.device);
      if (i < this->ctx.n_layers - 1 || this->ctx.apply_output_bias) {
        this->ctx.d_biases[elti][i] = this->static_params_pool->request(n_cols);
        Kernels::Utils::copy_data_to_device(this->ctx.d_biases[elti][i],
                                            h_biases[elti][i], n_cols,
                                            this->ctx.device);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template class EnergyHead<float>;
template class EnergyHead<double>;

}  // namespace TENSORMD