/* -------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_tensormd_tdpe.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair_tensormd.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTensorMDTDPE::ComputeTensorMDTDPE(LAMMPS *lmp, int narg,
                                         char **arg) :
    Compute(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR, "Illegal compute tensormd/tdpe command");

  pair = dynamic_cast<PairTensorMD *>(lmp->force->pair);
  if (pair == nullptr)
    error->all(FLERR,
               "pair tensormd must be initialized before command "
               "compute tensormd/tdpe");
  if (pair->tdpe_flag == 0)
    error->all(FLERR,
               "compute tensormd/tdpe requires a temperature-dependent model");

  vector_flag = 1;
  size_vector = 3;
  extvector = 0;
  peatomflag = 1;
  timeflag = 1;
  comm_reverse = 0;

  // internal energy, electron free energy and electron entropy
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeTensorMDTDPE::~ComputeTensorMDTDPE()
{
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDTDPE::compute_vector()
{
  invoked_vector = update->ntimestep;

  double values[3] = {0.0, 0.0, 0.0};

  if (pair && pair->compute_flag && pair->tdpe_flag) {
    values[0] = pair->eng_vdwl;
    values[1] = pair->free_energy;
    values[2] = pair->eentropy;
  }

  MPI_Allreduce(values, vector, size_vector, MPI_DOUBLE, MPI_SUM, world);
}
