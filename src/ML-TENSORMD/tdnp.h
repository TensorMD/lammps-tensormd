/* ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LAMMPS_TENSORMD_TDNP_H
#define LAMMPS_TENSORMD_TDNP_H

#include "cnpy.h"
#include "nnp.h"

namespace LAMMPS_NS {

class Memory;
class Error;

template <typename Scalar>
class TDNP {

 public:
  TDNP(Memory *mem, Error *err);
  ~TDNP();

  void setup_global(cnpy::npz_t &npz);

  Scalar compute(int elti, Scalar *x_in, int n, Scalar *dfdx, Scalar etemp,
                 Scalar *eatom, Scalar *satom, Scalar &S, Scalar *aatom,
                 Scalar &A);
  Scalar forward(int elti, Scalar *x_in, int n, Scalar etemp, Scalar *eatom,
                 Scalar *satom, Scalar &S, Scalar *aatom, Scalar &a);
  void backward(int elti, Scalar etemp, Scalar *grad_in, Scalar *grad_out);
  double get_flops_per_atom();
  double get_memory_usage();

 protected:
  NeuralNetworkPotential<Scalar> *H;
  NeuralNetworkPotential<Scalar> *S;
  NeuralNetworkPotential<Scalar> *U;

  Memory *memory;
  Error *error;

  bool is_sommrfeld;

  int nelt;
  std::map<int, int> nmax;
  std::map<int, int> nlocal;

  int ydim;
  std::map<int, Scalar *> y;
  std::map<int, Scalar *> dU;
  std::map<int, Scalar *> dS;

   Scalar *y_GPU[4] = {nullptr, nullptr, nullptr, nullptr};
   Scalar *dU_GPU[4] = {nullptr, nullptr, nullptr, nullptr};
   Scalar *dS_GPU[4] = {nullptr, nullptr, nullptr, nullptr};

  void setup_one(cnpy::npz_t &npz, const std::string &scope);
};

}    // namespace LAMMPS_NS

#endif
