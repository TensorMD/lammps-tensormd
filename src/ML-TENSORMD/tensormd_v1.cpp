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

#include "tensormd_v1.h"

#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "encoders/mlp_encoder.h"
#include "fmt/chrono.h"
#include "heads/energy_head_v1.h"
#include "heads/finite_temp_energy_head_v1.h"
#include "kernels/kernels_utils.h"
#include "kernels/kernels_v1_ops.h"
#include "tensormd_constants.h"

using Eigen::IndexPair;
using Eigen::Tensor;
using Eigen::TensorMap;

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

inline void species_chars_to_symbols(std::map<int, std::string>& symbols,
                                     std::vector<int>& species_chars,
                                     int ntypes) {
  for (int elti = 0; elti < ntypes; elti++) {
    int i = elti * 2;
    char c = static_cast<char>(species_chars[i + 0]);
    std::string s;
    s.push_back(c);
    if (i + 1 < species_chars.size() && species_chars[i + 1] > 0) {
      s.push_back(static_cast<char>(species_chars[i + 1]));
    }
    symbols[elti] = s;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::initialize(cnpypp::npz_t& npz, std::string& desc) {
  this->context->cutforce =
      static_cast<double>(*npz.find("rmax")->second.data<Scalar>());

  // In TensorMDV1, the key 'numbers' in the npz file corresponds to a list of
  // ASCII character codes that represent the element symbols.
  std::vector<int> species_chars;
  for (int i = 0; i < npz.find("numbers")->second.num_vals; i++) {
    species_chars.push_back(npz.find("numbers")->second.data<int>()[i]);
  }

  // Convert the list of ASCII character codes to a map of element symbols
  int n_species = *npz.find("nelt")->second.data<int>();
  species_chars_to_symbols(this->species, species_chars, n_species);

  for (const auto& specie : this->species) {
    auto it = AtomicSymbolToNumber.find(specie.second);
    this->numbers.push_back(it->second);
  }
  for (int i = 0; i < npz.find("masses")->second.num_vals; i++) {
    this->masses.push_back(
        static_cast<double>(npz.find("masses")->second.data<Scalar>()[i]));
  }

  // Setup the Encoder
  if (*npz.find("use_fnn")->second.data<int>() == 1) {
    this->encoder =
        new MLPEncoder<Scalar>(this->context->device, this->context->cutforce);
    this->encoder->initialize(npz);
  } else {
    throw std::runtime_error(
        "[TensorMDV1]: the AnalyticFunctionEncoder is discarded.");
  }

  // Setup the Energy Head
  if (npz.find("tdnp") != npz.end() &&
      npz.find("tdnp")->second.data<int>()[0] == 1) {
    this->energy_head =
        new FiniteTempEnergyHeadV1<Scalar>(this->context->device);
  } else {
    this->energy_head = new EnergyHeadV1<Scalar>(this->context->device);
  }
  this->energy_head->initialize(npz);

  // @deprecated: setup the flag 'is_T_symmetric'
  if (npz.find("is_T_symmetric") != npz.end() &&
      npz.find("is_T_symmetric")->second.data<int>()[0] == 1) {
    use_legacy_T = true;
  }

  // Initialize the TensorData
  const int b = n_species;
  const int k = this->encoder->get_num_radial_features();
  const int m = *npz.find("max_moment")->second.data<int>() + 1;
  const int ntypes = b;
  this->context->data =
      new TensorData<Scalar>(this->context->device, ntypes, b, k, m, false);
  this->context->data->T_proj_algo = this->use_legacy_T ? 0 : 1;

  // Return the model description
  char repr[2048];
  snprintf(repr, 2048, "b=%d, k=%d, m=%d, T=%s", b, k, m,
           this->use_legacy_T ? "legacy" : "compressed");
  desc = std::string(repr);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::adjust_strategy(std::string& msg) {
  if (STRATEGY == Strategy::Speed || STRATEGY == Strategy::Production ||
      STRATEGY == Strategy::Scale) {
    STRATEGY = Strategy::Baseline;
    msg = "[TensorMD]: Fallback to the Baseline strategy for the V1 model.";
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::fill_fixed_size_tensors(int* eltij_pos) {
  const int b = this->context->data->dims.b;

  // Initialize eltij_pos
  // This gives the corresponding 'b' given an atom pair (elti, eltj)
  for (int i = 0; i < b; i++) {
    int k = 1;
    for (int j = 0; j < b; j++) {
      if (i == j) {
        eltij_pos[i * b + j] = 0;
      } else {
        eltij_pos[i * b + j] = k;
        k++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::setup_local(int nlocal, int cmax, const int* typenums,
                                     bool require_vatom) {
  TensorMD<Scalar>::setup_local(nlocal, cmax, typenums, require_vatom);

  elt_count[0].offset = 0;
  elt_count[0].row = 0;

  if (this->context->data->dims.b > 1) {
    std::vector<int> elt_a_offset(this->context->data->dims.b);
    int m = typenums[0];
    for (int elti = 1; elti < this->context->data->dims.b; elti++) {
      elt_count[elti].offset = m;
      elt_count[elti].row = m;
      elt_a_offset[elti] = m;
      m += typenums[elti];
    }
    Kernels::V1::setup_index_maps(elt_a_offset.data(), this->context);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::compute_energy(const int* typenums,
                                        EnergyResult<double>& result,
                                        double scale) {
  Scalar* x_in;
  Scalar* dydx;
  std::vector<Scalar> eatom;
  EnergyResult<Scalar> y;
  const int A = this->context->data->dims.a;
  const int B = this->context->data->dims.b;
  const int K = this->context->data->dims.k;
  const int M = this->context->data->dims.m;
  const int bkm = B * K * M;

  // Allocate memory for atomic energies
  if (result.calc_atom()) {
    eatom.resize(this->context->data->dims.a);
  }

  // Forward propagation: compute E and dEdQ
  for (int b = 0; b < B; b++) {
    if (typenums[b] == 0) {
      continue;
    }

    // Setup the offset
    x_in = &this->context->data->G[this->elt_count[b].offset * bkm];
    if (!eatom.empty()) {
      y.eatom = &eatom.data()[this->elt_count[b].offset];
    } else {
      y.eatom = nullptr;
    }
    y.toten = 0.0;
    dydx = &this->context->data->dEdG[this->elt_count[b].offset * bkm];

    // Compute
    this->energy_head->compute(b, typenums[b], x_in, nullptr, y.toten, y.eatom,
                               dydx);

    // Add the scaled energy
    result.toten += static_cast<double>(y.toten) * scale;
  }

  if (!eatom.empty()) {
    std::vector<int> buffer;
    int* a2i = Kernels::Utils::access_data_on_host(
        this->context->data->a2i, A, this->context->device, buffer);
    for (int a = 0; a < A; a++) {
      if (B == 1) {
        result.eatom[a] = static_cast<double>(eatom[a]) * scale;
      } else {
        result.eatom[a2i[a]] = static_cast<double>(eatom[a]) * scale;
      }
    }
    eatom.clear();
    buffer.clear();
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMDV1<Scalar>::compute_free_energy(const int* typenums, double* etemp,
                                             bool use_atom_etemp,
                                             FreeEnergyResult<double>& tdpe,
                                             double scale) {
  int a, elti;
  Scalar* x_in;
  Scalar *eatom[4], *dfdx, *etemp_atom;
  FreeEnergyResult<Scalar> y;
  auto data = this->context->data;
  const int bkm = data->dims.b * data->dims.k * data->dims.m;

  // Allocate memory for atomic free energies (0), energies (1) and electron
  // entropy (2).
  if (tdpe.calc_atom()) {
    eatom[0] = new Scalar[data->dims.a];
    eatom[1] = new Scalar[data->dims.a];
    eatom[2] = new Scalar[data->dims.a];
  } else {
    eatom[0] = nullptr;
    eatom[1] = nullptr;
    eatom[2] = nullptr;
  }

  if (use_atom_etemp) {
    // Allocate memory for atomic electron temperature (3) if needed
    eatom[3] = new Scalar[data->dims.a];

    // Local to global mapping of atomic indices
#if defined(_OPENMP)
#pragma omp parallel for private(a)
#endif
    for (a = 0; a < data->dims.a; a++) {
      if (data->dims.b == 1) {
        eatom[3][a] = static_cast<Scalar>(etemp[a]);
      } else {
        eatom[3][data->a2i[a]] = static_cast<Scalar>(etemp[a]);
      }
    }
  } else {
    eatom[3] = nullptr;
  }

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < data->dims.b; elti++) {
    if (typenums[elti] == 0) {
      continue;
    }

    // Setup the offset
    x_in = &data->G[this->elt_count[elti].offset * bkm];
    if (tdpe.calc_atom()) {
      y.F.eatom = &eatom[0][this->elt_count[elti].offset];
      y.E.eatom = &eatom[1][this->elt_count[elti].offset];
      y.S.eatom = &eatom[2][this->elt_count[elti].offset];
    } else {
      y.F.eatom = nullptr;
      y.E.eatom = nullptr;
      y.S.eatom = nullptr;
    }
    y.F.toten = 0.0;
    y.E.toten = 0.0;
    y.S.toten = 0.0;
    dfdx = &data->dEdG[this->elt_count[elti].offset * bkm];

    // Compute
    auto head =
        dynamic_cast<FiniteTempEnergyHeadV1<Scalar>*>(this->energy_head);
    if (use_atom_etemp) {
      etemp_atom = &eatom[3][this->elt_count[elti].offset];
    } else {
      auto T = static_cast<Scalar>(etemp[0]);
      etemp_atom = &T;
    }
    head->compute(elti, x_in, typenums[elti], dfdx, use_atom_etemp, etemp_atom,
                  y);

    // Add the scaled values
    tdpe.F.toten += static_cast<double>(y.F.toten) * scale;
    tdpe.E.toten += static_cast<double>(y.E.toten) * scale;
    tdpe.S.toten += static_cast<double>(y.S.toten) * scale;
  }

  if (tdpe.calc_atom()) {
#if defined(_OPENMP)
#pragma omp parallel for private(a)
#endif
    for (a = 0; a < data->dims.a; a++) {
      if (data->dims.b == 1) {
        tdpe.F.eatom[a] = static_cast<double>(eatom[0][a]) * scale;
        tdpe.E.eatom[a] = static_cast<double>(eatom[1][a]) * scale;
        tdpe.S.eatom[a] = static_cast<double>(eatom[2][a]) * scale;
      } else {
        tdpe.F.eatom[data->a2i[a]] = static_cast<double>(eatom[0][a]) * scale;
        tdpe.E.eatom[data->a2i[a]] = static_cast<double>(eatom[1][a]) * scale;
        tdpe.S.eatom[data->a2i[a]] = static_cast<double>(eatom[2][a]) * scale;
      }
    }

    delete[] eatom[0];
    delete[] eatom[1];
    delete[] eatom[2];
  }
  if (use_atom_etemp) {
    delete[] eatom[3];
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class TensorMDV1<float>;
template class TensorMDV1<double>;

}  // namespace TENSORMD
