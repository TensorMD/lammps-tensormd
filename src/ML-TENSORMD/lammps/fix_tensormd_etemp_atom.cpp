/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_tensormd_etemp_atom.h"

#include <cstring>

#include "../pair_tensormd.h"
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#define eV_to_Kelvin 11604.51812

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixTensorMDAtomicElectronTemperature::FixTensorMDAtomicElectronTemperature(
    LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg), pair(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal fix tensormd/etemp/atom command");
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery < 0) error->all(FLERR, "Illegal fix tensormd/etemp/atom command");

  etemp = nullptr;
  nmax = 0;

  // does not require continuous time stepping
  time_depend = 0;

  // this fix produces either a per-atom vector
  peratom_flag = 1;
  size_peratom_cols = 0;

  // perform initial allocation of atom-based array
  // register with Atom class
  FixTensorMDAtomicElectronTemperature::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  // zero the array since dump may access it on timestep 0
  // zero the array since a variable may access it before first run
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) etemp[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

FixTensorMDAtomicElectronTemperature::~FixTensorMDAtomicElectronTemperature()
{
  atom->delete_callback(id, Atom::GROW);
  memory->destroy(etemp);
}

/* ---------------------------------------------------------------------- */

int FixTensorMDAtomicElectronTemperature::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTensorMDAtomicElectronTemperature::init()
{
  if (strcmp(force->pair_style, "tensormd") != 0) {
    error->all(FLERR, "pair_style must be tensormd for fix {}", style);
  }
  pair = dynamic_cast<PairTensorMD *>(force->pair);
  pair->enable_atom_etemp();

  if (modify->get_fix_by_style(style).size() > 1) {
    error->one(FLERR, "More than one fix {}", style);
  } else if (modify->get_fix_by_style("tensormd/etemp").size() > 1) {
    error->one(FLERR, "Conflicting fixes {} and tensormd/etemp", style);
  }
}

/* ----------------------------------------------------------------------
   only does something if nvalid = current timestep
------------------------------------------------------------------------- */

void FixTensorMDAtomicElectronTemperature::setup(int /*vflag*/)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixTensorMDAtomicElectronTemperature::end_of_step()
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;

  modify->clearstep_compute();

  // compute 'atomic temperature' for each atom in group
  double mvv2e = force->mvv2e;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (rmass)
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        etemp[i] = mvv2e * rmass[i] / 3 *
            (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
      } else
        etemp[i] = 0.0;
    }
  else
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        etemp[i] = mvv2e * mass[type[i]] / 3 *
            (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
      } else
        etemp[i] = 0.0;
    }

  // update the temperature for TensorMD
  pair->update_etemp(etemp);
}

/* ---------------------------------------------------------------------- */

double FixTensorMDAtomicElectronTemperature::memory_usage()
{
  double bytes;
  bytes = (double) atom->nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixTensorMDAtomicElectronTemperature::grow_arrays(int n)
{
  memory->grow(etemp, n, "fix_ave/atom:array");
  vector_atom = etemp;
  array_atom = &etemp;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixTensorMDAtomicElectronTemperature::copy_arrays(int i, int j,
                                                       int /*delflag*/)
{
  etemp[j] = etemp[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixTensorMDAtomicElectronTemperature::pack_exchange(int i, double *buf)
{
  buf[0] = etemp[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixTensorMDAtomicElectronTemperature::unpack_exchange(int nlocal,
                                                          double *buf)
{
  etemp[nlocal] = buf[0];
  return 1;
}
