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

#include "tensormd_v2.h"

#include <cnpy++.hpp>
#include <vector>

#include "encoders/mlp_radial_basis_encoder.h"
#include "heads/energy_head_v2.h"
#include "kernels/kernels_fusion.h"
#include "tensormd_constants.h"
#include "tensormd_v2_config.hpp"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
TensorMDV2<Scalar>::TensorMDV2(Device dev, int mpiid)
    : TensorMD<Scalar>(dev, mpiid) {}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV2<Scalar>::initialize(cnpypp::npz_t& npz, std::string& desc) {
  this->config = load_config_from_npz(npz);
  this->context->cutforce = config.r_cut;

  for (int i = 0; i < config.atomic_numbers.size(); i++) {
    int number = config.atomic_numbers[i];
    this->numbers.push_back(number);
    this->species[i] = std::string(AtomicNumberToSymbol.find(number)->second);
  }
  for (double mass : config.masses) {
    this->masses.push_back(mass);
  }

  Device dev = this->context->device;

  // Setup the encoder
  this->encoder = new MLPRadialBasisEncoder<Scalar>(dev, config);
  this->encoder->initialize(npz);

  // Setup the energy head
  this->energy_head = new EnergyHeadV2<Scalar>(dev, config);
  this->energy_head->initialize(npz);

  // Initialize the TensorData
  const int b = 1;
  const int k = config.k_dim;
  const int m = config.max_moment + 1;
  const int ntypes = static_cast<int>(config.atomic_numbers.size());
  int T_proj_algo = 1;
  if (!this->config.use_compressed_T) {
    T_proj_algo = 2;
  }
  this->context->data =
      new TensorData<Scalar>(dev, ntypes, b, k, m, T_proj_algo, true);
  char repr[2048];
  snprintf(repr, 2048, "k=%d, m=%d, T=%s", k, m,
           T_proj_algo == 1 ? "compressed" : "accurate");
  desc = std::string(repr);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV2<Scalar>::adjust_strategy(std::string& msg) {
  if (STRATEGY == Strategy::Scale) {
    if (this->context->data->dims.k != 64 && this->context->data->dims.k != 32) {
      msg = "[TensorMD] Fallback to the Production strategy due to K != 64/32\n";
      STRATEGY = Strategy::Production;
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV2<Scalar>::fill_fixed_size_tensors(int* eltij_pos) {
  // Initialize eltij_pos
  const int ntypes = static_cast<int>(config.atomic_numbers.size());
  for (int i = 0; i < ntypes * ntypes; i++) {
    eltij_pos[i] = 0;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV2<Scalar>::compute_energy(const int* typenums,
                                        EnergyResult<double>& result,
                                        double scale) {
  const int n = this->context->data->dims.a;
  if (n == 0) {
    return;
  }

  EnergyResult<Scalar> y;

  // Allocate memory for atomic energies
  if (result.calc_atom()) {
    y.eatom = new Scalar[n];
  }

  // Compute
  auto head = dynamic_cast<EnergyHeadV2<Scalar>*>(this->energy_head);
  head->compute(n, this->context->data->G, this->context->data->tlist, y.toten,
                y.eatom, this->context->data->dEdG);

  // Add the scaled energy
  result.toten += static_cast<double>(y.toten) * scale;

  if (y.eatom != nullptr) {
    for (int a = 0; a < this->context->data->dims.a; a++) {
      result.eatom[a] = static_cast<double>(y.eatom[a]) * scale;
    }
    delete[] y.eatom;
  }
}

template <typename Scalar>
void TensorMDV2<Scalar>::compute_free_energy(const int* typenums, double* etemp,
                                             bool use_atom_etemp,
                                             FreeEnergyResult<double>& tdpe,
                                             double scale) {
  throw std::runtime_error(
      "TensorMDV2::compute_free_energy: is not implemented yet!");
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class TensorMDV2<double>;
template class TensorMDV2<float>;

}  // namespace TENSORMD
