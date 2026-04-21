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
ComputeStyle(tensormd/tdpe/atom, ComputeTensorMDTDPEAtom)
#else

#ifndef LAMMPS_COMPUTE_TENSORMD_TDPE_ATOM_H
#define LAMMPS_COMPUTE_TENSORMD_TDPE_ATOM_H

#include "../tensormd.h"
#include "compute.h"

namespace LAMMPS_NS {

class PairTensorMD;
class NeighList;

class ComputeTensorMDTDPEAtom : public Compute {
public:
 ComputeTensorMDTDPEAtom(class LAMMPS *, int, char **);
 ~ComputeTensorMDTDPEAtom() override;
 void init() override;
 void compute_peratom() override;
 double memory_usage() override;

private:
 int nmax;
 PairTensorMD *pair;
 double **array;
};

}    // namespace LAMMPS_NS

#endif
#endif