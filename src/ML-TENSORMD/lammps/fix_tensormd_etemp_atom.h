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

FixStyle(tensormd/etemp/atom,FixTensorMDAtomicElectronTemperature)

#else

#ifndef LMP_FIX_TENSORMD_ETEMP_ATOM_H
#define LMP_FIX_TENSORMD_ETEMP_ATOM_H

#include "fix.h"

namespace LAMMPS_NS {

class PairTensorMD;

class FixTensorMDAtomicElectronTemperature : public Fix {
 public:
  FixTensorMDAtomicElectronTemperature(class LAMMPS *, int, char **);
  ~FixTensorMDAtomicElectronTemperature();
  int setmask() override;
  void init() override;
  void end_of_step() override;
  void setup(int) override;
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

 private:
  PairTensorMD *pair;
  double *etemp;
  int nmax;
};

}    // namespace LAMMPS_NS

#endif
#endif
