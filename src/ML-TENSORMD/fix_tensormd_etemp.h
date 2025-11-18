/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(tensormd/etemp,FixTensorMDElectronTemperature)

#else

#ifndef LMP_FIX_TENSORMD_ETEMP_H
#define LMP_FIX_TENSORMD_ETEMP_H

#include "fix.h"

namespace LAMMPS_NS {

class PairTensorMD;

class FixTensorMDElectronTemperature : public Fix {
 public:
  FixTensorMDElectronTemperature(class LAMMPS *, int, char **);
  ~FixTensorMDElectronTemperature();
  int setmask() override;
  void init() override;
  void end_of_step() override;
  double compute_scalar() override;

 private:
  PairTensorMD *pair;
  double kelvin;

  void change_settings();
};

}    // namespace LAMMPS_NS

#endif
#endif
