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

#include "compute_tensormd_tdpe_atom.h"

#include "../pair_tensormd.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTensorMDTDPEAtom::ComputeTensorMDTDPEAtom(LAMMPS *lmp, int narg,
                                                 char **arg) :
    Compute(lmp, narg, arg), array(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal compute tensormd/tdpe/atom command");

  pair = dynamic_cast<PairTensorMD *>(lmp->force->pair);
  if (pair == nullptr)
    error->all(FLERR,
               "pair tensormd must be initialized before command "
               "compute tensormd/tdpe/atom");

  size_peratom_cols = 3;
  peratom_flag = 1;
  peatomflag = 1;
  timeflag = 1;
  comm_reverse = 0;
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeTensorMDTDPEAtom::~ComputeTensorMDTDPEAtom()
{
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDTDPEAtom::init()
{
  if (modify->get_compute_by_style("tensormd/tdpe/atom").size() > 1 &&
      comm->me == 0)
    error->warning(FLERR, "More than one compute tensormd/tdpe/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDTDPEAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (update->eflag_atom != invoked_peratom)
    error->all(FLERR, "Per-atom energy was not tallied on needed timestep");

  int i, j;

  // grow local energy array if necessary
  // needs to be atom->nmax in length
  if (atom->nmax > nmax) {
    memory->destroy(array);
    nmax = atom->nmax;
    memory->create(array, nmax, size_peratom_cols, "tensormd/tdpe/atom:array");
    array_atom = array;
  }

  // ntotal includes ghosts if either newton flag is set
  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // clear local energy array
  for (i = 0; i < ntotal; i++) {
    for (j = 0; j < size_peratom_cols; j++) { array[i][j] = 0.0; }
  }

  // add in per-atom contributions from each force
  if (pair && pair->compute_flag && pair->tdpe_flag) {
    for (i = 0; i < ntotal; i++) {
      array[i][0] += pair->tdpe.S.eatom[i];
      array[i][1] += pair->tdpe.F.eatom[i];
    }
  }

  // zero energy of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;
  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) {
      array[i][0] = 0.0;
      array[i][1] = 0.0;
    }
}

/* ---------------------------------------------------------------------- */

double ComputeTensorMDTDPEAtom::memory_usage()
{
  double bytes = 0.0;
  bytes += memory->usage(array, nmax, size_peratom_cols);
  return bytes;
}
