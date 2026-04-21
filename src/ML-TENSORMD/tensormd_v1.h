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

#ifndef TENSORMD_V1_H
#define TENSORMD_V1_H

#include <cnpy++.hpp>
#include <string>

#include "tensormd.h"

namespace TENSORMD {

template <typename Scalar>
class TensorMDV1 : public TensorMD<Scalar> {
 public:
  TensorMDV1(Device dev, int mpiid) : TensorMD<Scalar>(dev, mpiid) {}

  void initialize(cnpypp::npz_t& npz, std::string& desc) override;
  void adjust_strategy(std::string& msg) override;
  void fill_fixed_size_tensors(int* eltij_pos) override;
  void setup_local(int nlocal, int cmax, const int* typenums,
                   bool require_vatom) override;

  void compute_energy(const int* typenums, EnergyResult<double>& result,
                      double scale) override;
  void compute_free_energy(const int* typenums, double* etemp,
                           bool use_atom_etemp,
                           FreeEnergyResult<double>& result,
                           double scale) override;

 protected:
  struct _elt_count_t {
    int offset;
    int row;
  } elt_count[32];
  bool use_legacy_T = false;
};

}  // namespace TENSORMD
#endif

// LAMMPS_TENSORMD_H
