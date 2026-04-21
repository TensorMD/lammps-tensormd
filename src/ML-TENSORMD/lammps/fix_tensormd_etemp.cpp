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

#include "fix_tensormd_etemp.h"

#include <cstring>

#include "../pair_tensormd.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "modify.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#define eV_to_Kelvin 11604.51812

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixTensorMDElectronTemperature::FixTensorMDElectronTemperature(LAMMPS *lmp,
                                                               int narg,
                                                               char **arg) :
    Fix(lmp, narg, arg), pair(nullptr)
{
  if (narg != 4) error->all(FLERR, "Illegal fix etemp command");
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery < 0) error->all(FLERR, "Illegal fix etemp command");
  kelvin = 0;
  scalar_flag = 1;
}

/* ---------------------------------------------------------------------- */

FixTensorMDElectronTemperature::~FixTensorMDElectronTemperature() = default;

/* ---------------------------------------------------------------------- */

int FixTensorMDElectronTemperature::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTensorMDElectronTemperature::init()
{
  if (strcmp(force->pair_style, "tensormd") != 0) {
    error->all(FLERR, "pair_style must be tensormd for fix tensormd/etemp");
  }

  pair = dynamic_cast<PairTensorMD *>(force->pair);
  if (modify->get_fix_by_style(style).size() > 1) {
    error->one(FLERR, "More than one fix {}", style);
  } else if (modify->get_fix_by_style("tensormd/etemp/atom").size() > 1) {
    error->one(FLERR, "Conflicting fixes {} and tensormd/etemp/atom", style);
  }
  kelvin = pair->get_average_etemp() * eV_to_Kelvin;
}

/* ---------------------------------------------------------------------- */

void FixTensorMDElectronTemperature::end_of_step()
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;
  change_settings();
}

/* ---------------------------------------------------------------------- */

void FixTensorMDElectronTemperature::change_settings()
{
  modify->clearstep_compute();
  int ipe = modify->find_compute("thermo_temp");
  if (ipe == -1) error->all(FLERR, "compute thermo_pe does not exist.");
  Compute *c_temp = modify->compute[ipe];
  kelvin = c_temp->compute_scalar();
  pair->update_etemp(kelvin / eV_to_Kelvin);
  modify->addstep_compute(update->ntimestep + nevery);
}

/* ---------------------------------------------------------------------- */

double FixTensorMDElectronTemperature::compute_scalar()
{
  return kelvin;
}
