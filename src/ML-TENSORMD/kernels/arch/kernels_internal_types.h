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

#ifndef TENSORMD_KERNELS_INTERNAL_TYPES_H
#define TENSORMD_KERNELS_INTERNAL_TYPES_H

namespace TENSORMD::Kernels {

// Wrap device and host pointers into a pure struct.

template <typename Scalar>
struct DeviceArgs {
  // For the core tensors
  int a = 0, b = 0, c = 0, d = 0, k = 0, m = 0;
  Scalar *M = nullptr, *H = nullptr, *R = nullptr,
         *drdrx = nullptr, *dHdr = nullptr, *sij = nullptr, *dsij = nullptr;
  int *ijlist = nullptr, *i2a = nullptr, *a2i = nullptr, *tlist = nullptr,
      *zijlist = nullptr, *numbers = nullptr, *tijlist = nullptr;
  Scalar *P = nullptr, *G = nullptr;
  Scalar *dEdG = nullptr, *W = nullptr, *U = nullptr, *V = nullptr,
         *Fr = nullptr, *Fa = nullptr, *Up = nullptr, *dHdrp = nullptr;
  Scalar scale = 0.0;
  // For neighbor list
  double cutforcesq = 0.0;
  int *ilist = nullptr, *type = nullptr, *fmap = nullptr, *numneigh = nullptr,
      **firstneigh = nullptr, *eltij_pos = nullptr, *elt_a_offset = nullptr;
  int* firstneigh_flat = nullptr;
  int numneigh_max = 0, imax = 0, inum = 0, nghost = 0, nlocal = 0;
  double* x = nullptr;
  double* f = nullptr;
  double* vatom = nullptr;
  // For interpolation
  Scalar* params_atlas = nullptr;
  float* params_atlas_fp32 = nullptr;
  int* pair_map = nullptr;
  int num_pairs = 0;
  int interp_size = 0;
  Scalar interp_dr = 0.0;
  Scalar interp_r0 = 0.0;
  // For NNP
  Scalar** weights = nullptr;
  Scalar** weights_T = nullptr;
  Scalar** biases = nullptr;
  Scalar *etotal = nullptr, *eatoms = nullptr;
  int n_layers = 0;
  int actype = 0;
  // The projection algorithm from P to G
  // 0: V1, legacy
  // 1: V1, compressed
  // 2: V2, accurate
  int T_proj_algo = -1;
};

template <typename Scalar>
struct HostArgs {
  const int *ilist = nullptr, *type = nullptr, *fmap = nullptr,
            *numneigh = nullptr, **firstneigh = nullptr, *eltij_pos = nullptr;
  const int* elt_a_offset = nullptr;
  int numneigh_max = 0, imax = 0, inum = 0, nlocal = 0, nghost = 0;
  double* etotal = nullptr;
  double* eatoms = nullptr;
  double** x = nullptr;
  double** f = nullptr;
  double** vatom = nullptr;

  // TensorMDV2 supports arbitrary atom types, but only a subset of atom types
  // are really included in the training dataset. Those atom types are store in
  // 'numbers'.
  // numbers[elti] gives the atomic number of the elti-th atom type.
  // zlist[i] = numbers[type[i]]
  int* numbers = nullptr;
  // n_numbers indicates the number of atom types that are included in the
  // potential.
  int n_numbers = 0;
  // The number of atom types in the LAMMPS simulation.
  int lmp_ntypes = 0;
};

}  // namespace TENSORMD::Kernels

#endif
