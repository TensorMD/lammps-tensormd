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

#ifndef TENSORMD_ENERGY_HEAD_V1_H
#define TENSORMD_ENERGY_HEAD_V1_H

#include "energy_head.h"

namespace TENSORMD {

template <typename Scalar>
class EnergyHeadV1 : public EnergyHead<Scalar> {
 public:
  EnergyHeadV1(Device dev) : EnergyHead<Scalar>(dev) {};
  void initialize(cnpypp::npz_t& npz) override;
  void permute_W0(std::vector<Scalar>& W_kmh, std::vector<Scalar>& W_kmh_T,
                  const Scalar* W_mkh, int n_rows, int n_cols) override;
  void transpose_W0(std::vector<Scalar>& W_mkh_T, const Scalar* W_mkh,
                    int n_rows, int n_cols) override;

 protected:
  int k_dim = 0;
  int b_dim = 0;
  int m_dim = 0;
};

}  // namespace TENSORMD
#endif