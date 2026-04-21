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

#include "finite_temp_energy_head_v1.h"

#include "../kernels/kernels_v1_ops.h"
#include "../tensormd_memory.h"
#include "../utils/tensor_debug.h"
#include "energy_head_v1.h"
#include "fmt/format.h"

namespace TENSORMD {

/* ---------------------------------------------------------------------- */

template <typename Scalar>
FiniteTempEnergyHeadV1<Scalar>::FiniteTempEnergyHeadV1(Device dev)
    : EnergyHeadV1<Scalar>(dev) {
  H = nullptr;
  S = nullptr;
  E = nullptr;
  squared_temp = false;
  n_species = 0;
  h_dim = 0;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
FiniteTempEnergyHeadV1<Scalar>::~FiniteTempEnergyHeadV1() {
  for (auto p : mem_pools) delete p;

  delete H;
  delete S;
  delete E;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::prepare_memory(int elti, int n) {
  if (mem_pools.size() <= elti) {
    mem_pools.resize(n_species, nullptr);
    step_data.resize(n_species);
  }
  if (!mem_pools[elti]) {
    mem_pools[elti] = new MemoryPool<Scalar>(
        this->ctx.device, "FiniteTempEnergyHeadV1::Pool::step");
  }

  // We need Ht (input to U and S), dU, dS.
  // Ht size: n * dim_h
  // Ht = [H(G); etemp] where H(G) is the output of H and has dimension dim_h-1.
  size_t size = n * h_dim;

  mem_pools[elti]->grow(3 * size);
  mem_pools[elti]->reset();

  step_data[elti].Ht = mem_pools[elti]->request(size);
  step_data[elti].dE = mem_pools[elti]->request(size);
  step_data[elti].dS = mem_pools[elti]->request(size);
  step_data[elti].nlocal = n;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::initialize(cnpypp::npz_t& npz) {
  this->n_species = npz.find("nelt")->second.data<int>()[0];
  setup_one(npz, "H");
  setup_one(npz, "U");
  setup_one(npz, "S");
}

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::setup_one(cnpypp::npz_t& npz,
                                               const std::string& scope) {
  int num_in;
  int n_rows;

  int nlayers =
      *npz.find(fmt::format("{}::nlayers", scope))->second.data<int>();
  int* layer_sizes =
      npz.find(fmt::format("{}::layer_sizes", scope))->second.data<int>();
  int actfn = *npz.find(fmt::format("{}::actfn", scope))->second.data<int>();
  bool use_resnet_dt =
      *npz.find(fmt::format("{}::use_resnet_dt", scope))->second.data<int>() ==
      1;
  bool apply_output_bias =
      *npz.find(fmt::format("{}::apply_output_bias", scope))
           ->second.data<int>() == 1;

  if (npz.find("tdnp::Sommerfeld") != npz.end()) {
    squared_temp = *npz.find("tdnp::Sommerfeld")->second.data<int>() == 1;
  }

  if (scope == "H") {
    // Reserve a column for electron temperature.
    layer_sizes[nlayers - 1] += 1;
  }

  auto*** weights = new Scalar**[this->n_species];
  auto*** biases = new Scalar**[this->n_species];
  for (int elti = 0; elti < this->n_species; elti++) {
    weights[elti] = new Scalar*[nlayers];
    biases[elti] = new Scalar*[nlayers];
    for (int i = 0; i < nlayers; i++) {
      // Add a column of zeros to the last weight matrix
      if (scope == "H" && i == nlayers - 1) {
        auto orig = npz.find(fmt::format("H::weights_{}_{}", elti, i))
                        ->second.data<Scalar>();
        if (i == 1) {
          n_rows =
              static_cast<int>(npz.find(fmt::format("H::weights_{}_0", elti, i))
                                   ->second.shape[0]);
        } else {
          n_rows = layer_sizes[i - 1];
        }
        int n_cols = layer_sizes[i];
        weights[elti][i] = new Scalar[n_rows * n_cols];
        for (int row = 0; row < n_rows; row++) {
          std::memcpy(&weights[elti][i][row * n_cols],
                      &orig[row * (n_cols - 1)], sizeof(Scalar) * (n_cols - 1));
          weights[elti][i][row * n_cols + n_cols - 1] = 0.0;
        }
      } else {
        weights[elti][i] =
            npz.find(fmt::format("{}::weights_{}_{}", scope, elti, i))
                ->second.data<Scalar>();
      }
      if (i < nlayers - 1 || apply_output_bias) {
        biases[elti][i] =
            npz.find(fmt::format("{}::biases_{}_{}", scope, elti, i))
                ->second.data<Scalar>();
      }
    }
  }

  auto* head = new EnergyHeadV1<Scalar>(this->ctx.device);
  if (scope == "H") {
    this->H = head;
    h_dim = layer_sizes[nlayers - 1];
    num_in = static_cast<int>(npz.find("H::weights_0_0")->second.shape[0]);
  } else if (scope == "U") {
    this->E = head;
    num_in = h_dim;
  } else {
    this->S = head;
    num_in = h_dim;
  }
  ActivationType type = get_activation_type(actfn);
  head->setup(this->n_species, num_in, nlayers, layer_sizes, type, weights,
              biases, use_resnet_dt, apply_output_bias);
  if (scope == "H") {
    for (int elti = 0; elti < this->n_species; elti++) {
      delete[] weights[elti][nlayers - 1];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::compute(int elti, const Scalar* G, int n,
                                             Scalar* dfdx, bool use_atom_etemp,
                                             const Scalar* etemp,
                                             FreeEnergyResult<Scalar>& result) {
  forward(elti, G, n, use_atom_etemp, etemp, result);
  backward(elti, use_atom_etemp, etemp, nullptr, dfdx);
}

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::forward(int elti, const Scalar* G, int n,
                                             bool use_atom_etemp,
                                             const Scalar* etemp,
                                             FreeEnergyResult<Scalar>& result) {
  if (n == 0) return;
  prepare_memory(elti, n);

  Scalar d_tot = 0.0;
  this->H->forward(elti, n, G, nullptr, d_tot, step_data[elti].Ht);

  // Set the last element of each row to `etemp`
  Kernels::V1::tdnp_inject_temperature(this->ctx.device, step_data[elti].nlocal,
                                       h_dim, step_data[elti].Ht,
                                       use_atom_etemp, etemp);

  // Compute electron internal energy and electron entropy
  this->E->forward(elti, n, step_data[elti].Ht, nullptr, result.E.toten,
                   result.E.eatom);
  this->S->forward(elti, n, step_data[elti].Ht, nullptr, result.S.toten,
                   result.S.eatom);

  // Compute the electron free energy: F(T) = E(T) - T * S(T)
  Kernels::V1::tdnp_combine_free_energy(
      this->ctx.device, step_data[elti].nlocal, squared_temp, use_atom_etemp,
      etemp, result.E.eatom, result.E.toten, result.S.eatom, result.S.toten,
      result.F.eatom, result.F.toten);
}

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::backward(int elti, bool use_atom_etemp,
                                              const Scalar* etemp,
                                              const Scalar* grad_in,
                                              Scalar* grad_out) {
  this->E->backward(elti, grad_in, step_data[elti].dE);
  this->S->backward(elti, grad_in, step_data[elti].dS);
  Kernels::V1::tdnp_combine_gradients(
      this->ctx.device, step_data[elti].nlocal, h_dim, squared_temp,
      use_atom_etemp, etemp, step_data[elti].dS, step_data[elti].dE);
  this->H->backward(elti, step_data[elti].dE, grad_out);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double FiniteTempEnergyHeadV1<Scalar>::get_memory_usage() const {
  double bytes = 0.0;
  for (const auto& p : mem_pools) {
    bytes += p->get_memory_usage();
  }
  bytes += H->get_memory_usage();
  bytes += E->get_memory_usage();
  bytes += S->get_memory_usage();
  return bytes;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::compute(int elti, int n, const Scalar* d_x,
                                             const int* d_tlist,
                                             Scalar& h_total, Scalar* h_outputs,
                                             Scalar* d_grad) {
  throw std::runtime_error(
      "FiniteTempEnergyHeadV1::compute without etemp is not implemented");
}

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::forward(int elti, int n, const Scalar* d_x,
                                             const int* d_tlist,
                                             Scalar& h_total,
                                             Scalar* d_outputs) {
  throw std::runtime_error(
      "FiniteTempEnergyHeadV1::forward without etemp is not implemented");
}

template <typename Scalar>
void FiniteTempEnergyHeadV1<Scalar>::backward(int elti, const Scalar* grad_in,
                                              Scalar* grad_out) {
  throw std::runtime_error(
      "FiniteTempEnergyHeadV1::backward without etemp is not implemented");
}

/* ---------------------------------------------------------------------- */

template class FiniteTempEnergyHeadV1<float>;
template class FiniteTempEnergyHeadV1<double>;

}  // namespace TENSORMD
