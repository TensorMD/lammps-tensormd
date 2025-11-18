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

#ifndef LAMMPS_TENSORMD_DATA_TYPES_H
#define LAMMPS_TENSORMD_DATA_TYPES_H

namespace LAMMPS_NS {

// The basic data structure for potential energy.
template <typename Scalar> struct PE {
  Scalar toten;
  Scalar *eatom;

  bool calc_atom() { return eatom != nullptr; }
};

// The basic data structure for temperature-dependent potential energy.
template <typename Scalar> struct TDPE {
  PE<Scalar> F;
  PE<Scalar> E;
  PE<Scalar> S;

  bool calc_atom() { return F.calc_atom() && E.calc_atom() && S.calc_atom(); }
  void set_toten_zero() {
    F.toten = 0;
    E.toten = 0;
    S.toten = 0;
  }
};

}

#endif    //LAMMPS_TENSORMD_DATA_TYPES_H
