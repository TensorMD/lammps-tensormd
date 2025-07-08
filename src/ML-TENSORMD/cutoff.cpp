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

#include "cutoff.h"
#include "math_const.h"

using namespace LAMMPS_NS;

template <typename Scalar>
void Cutoff<Scalar>::compute(const TensorMap<Tensor<Scalar, 3>> &x,
                             const TensorMap<Tensor<Scalar, 3>> &mask,
                             TensorMap<Tensor<Scalar, 3>> *s,
                             TensorMap<Tensor<Scalar, 3>> *ds)
{
  if (cosine) {
    auto Z = (x / rcut).cwiseMin(static_cast<Scalar>(1.0)) *
        static_cast<Scalar>(MathConst::MY_PI);
    *s = mask * (0.5 + 0.5 * Z.cos());
    *ds = mask * (-0.5 * MathConst::MY_PI * Z.sin() / rcut);
  } else {
    auto Z = (x / rcut).cwiseMin(static_cast<Scalar>(MathConst::MY_PI));
    auto Z4 = Z.pow(4);
    auto Z5 = Z4 * Z;
    *s = mask * (1 + gamma * Z5 * Z - (gamma + 1) * Z5);
    *ds = mask * ((gamma + 1) * gamma / rcut * (Z5 - Z4));
  }
}

template <typename Scalar>
void Cutoff<Scalar>::compute(const TensorMap<Tensor<Scalar, 3>> &x,
                             TensorMap<Tensor<Scalar, 3>> *y)
{
  if (cosine) {
    auto Z = (x / rcut).cwiseMin(static_cast<Scalar>(1.0)) *
        static_cast<Scalar>(MathConst::MY_PI);
    *y = 0.5 + 0.5 * Z.cos();
  } else {
    auto Z = (x / rcut).cwiseMin(static_cast<Scalar>(MathConst::MY_PI));
    auto Z4 = Z.pow(4);
    auto Z5 = Z4 * Z;
    *y = 1 + gamma * Z5 * Z - (gamma + 1) * Z5;
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::Cutoff<float>;
template class LAMMPS_NS::Cutoff<double>;
