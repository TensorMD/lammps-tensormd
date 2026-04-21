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

#ifdef COMPUTE_CLASS
ComputeStyle(tensormd/density, ComputeTensorMDDensity)
#else

#ifndef LAMMPS_FIX_TENSORMD_H
#define LAMMPS_FIX_TENSORMD_H

#include "../tensormd.h"
#include "compute.h"

namespace LAMMPS_NS {

class PairTensorMD;
class NeighList;

class ComputeTensorMDDensity : public Compute {
 public:
  ComputeTensorMDDensity(class LAMMPS *, int, char **);
  ~ComputeTensorMDDensity() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

 private:

  class NeighList *list;

  int nmax;
  PairTensorMD *pair;
  double **G;

  int coeff(std::vector<std::string> args);
  void read_potential_file(const std::string &globalfile);

  TENSORMD::TensorMD<double> *dpot;
  TENSORMD::TensorMD<float> *fpot;
};

}    // namespace LAMMPS_NS

#endif
#endif