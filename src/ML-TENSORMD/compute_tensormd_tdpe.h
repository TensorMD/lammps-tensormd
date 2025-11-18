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
ComputeStyle(tensormd/tdpe, ComputeTensorMDTDPE)
#else

#ifndef LAMMPS_COMPUTE_TENSORMD_TDPE_H
#define LAMMPS_COMPUTE_TENSORMD_TDPE_H

#include "compute.h"
#include "tensormd.h"

namespace LAMMPS_NS {

class PairTensorMD;
class NeighList;

class ComputeTensorMDTDPE : public Compute {
 public:
  ComputeTensorMDTDPE(class LAMMPS *, int, char **);
  ~ComputeTensorMDTDPE() override;
  void compute_vector() override;
  void init() override {};

 private:
  PairTensorMD *pair;
};

}    // namespace LAMMPS_NS

#endif
#endif