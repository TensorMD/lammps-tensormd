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

#ifndef TENSORMD_FINITE_TEMP_ENERGY_HEAD_V1_H
#define TENSORMD_FINITE_TEMP_ENERGY_HEAD_V1_H

#include "energy_head_v1.h"

namespace TENSORMD {

template <typename Scalar>
class FiniteTempEnergyHeadV1 : public EnergyHeadV1<Scalar> {
 public:
  FiniteTempEnergyHeadV1(Device dev);
  ~FiniteTempEnergyHeadV1() override;

  void initialize(cnpypp::npz_t& npz) override;

  // The real compute function for finite temperature energy head
  // The electron temperature (scalar or per-atom) is needed.
  void compute(int elti, const Scalar* G, int n, Scalar* dfdx,
               bool use_atom_etemp, const Scalar* etemp,
               FreeEnergyResult<Scalar>& result);
  void forward(int elti, const Scalar* G, int n, bool use_atom_etemp,
               const Scalar* etemp, FreeEnergyResult<Scalar>& result);
  void backward(int elti, bool use_atom_etemp, const Scalar* etemp,
                const Scalar* grad_in, Scalar* grad_out);

  [[nodiscard]] double get_memory_usage() const override;
  [[nodiscard]] bool is_finite_temp_head() const override { return true; }

  void release_dynamic_memory() override {
    for (auto pool : mem_pools) {
      if (pool != nullptr) pool->free();
    }
  }

 protected:
  EnergyHeadV1<Scalar>* H;
  EnergyHeadV1<Scalar>* S;
  EnergyHeadV1<Scalar>* E;

  // The Sommerfeld approach: the entropy contribution is scaled by T^2
  bool squared_temp;

  // The total number of atom species for this model
  int n_species;

  // 'h_dim' is the input dimension of the internal energy model U and electron
  // entropy model S. 'h_dim' equals to the output dimension of the hidden model
  // H plus 1 (for the electron temperature).
  int h_dim;

  // Memory Pools for intermediate vectors (Ht, dE, dS)
  std::vector<MemoryPool<Scalar>*> mem_pools;
  struct StepData {
    int nlocal;
    Scalar *Ht, *dE, *dS;
  };
  std::vector<StepData> step_data;

  void prepare_memory(int elti, int n);
  void setup_one(cnpypp::npz_t& npz, const std::string& scope);

 public:
  // Override the virtual function but do not implement it
  void compute(int elti, int n, const Scalar* d_x, const int* d_tlist,
               Scalar& h_total, Scalar* h_outputs, Scalar* d_grad) override;
  void forward(int elti, int n, const Scalar* d_x, const int* d_tlist,
               Scalar& h_total, Scalar* h_outputs) override;
  void backward(int elti, const Scalar* grad_in, Scalar* grad_out) override;
};

}  // namespace TENSORMD
#endif