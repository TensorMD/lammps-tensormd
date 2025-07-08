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

#ifndef LAMMPS_TENSORMD_CUTOFF_H
#define LAMMPS_TENSORMD_CUTOFF_H

#include "eigen/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;
using Eigen::TensorMap;

namespace LAMMPS_NS {

template <typename Scalar> class Cutoff {

 public:
  explicit Cutoff(Scalar rc)
  {
    rcut = rc;
    cosine = true;
  }

  Cutoff(Scalar rc, Scalar gam)
  {
    rcut = rc;
    gamma = gam;
    cosine = false;
  }

  void compute(const TensorMap<Tensor<Scalar, 3>> &x,
               const TensorMap<Tensor<Scalar, 3>> &mask,
               TensorMap<Tensor<Scalar, 3>> *s,
               TensorMap<Tensor<Scalar, 3>> *ds);

  void compute(const TensorMap<Tensor<Scalar, 3>> &x,
               TensorMap<Tensor<Scalar, 3>> *y);

  bool is_cosine() { return cosine; }

 protected:
  bool cosine;
  Scalar rcut;
  Scalar gamma;
};

}    // namespace LAMMPS_NS

#endif    // LAMMPS_TENSORMD_CUTOFF_H
