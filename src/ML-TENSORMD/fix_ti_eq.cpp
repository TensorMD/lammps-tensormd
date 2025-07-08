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

#include "fix_ti_eq.h"
#include "atom.h"
#include "citeme.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "respa.h"
#include "update.h"
#include <cstring>
#include <mpi.h>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixThermalIntegration::FixThermalIntegration(LAMMPS *lmp, int narg,
                                             char **arg) :
    Fix(lmp, narg, arg)
{
  if (narg < 7 || narg > 8) error->all(FLERR, "Illegal fix ti/eq command");

  // Flags.
  restart_peratom = 1;
  scalar_flag = 1;
  global_freq = 1;
  vector_flag = 0;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  nlevels_respa = 0;

  // disallow resetting the time step, while this fix is defined
  time_depend = 1;

  // set default TI algorithm to Einstein
  algo = Einstein;

  int iarg = -1;
  if (strcmp(arg[3], "einstein") == 0) {
    // Spring constant.
    k = utils::numeric(FLERR, arg[4], false, lmp);
    if (k <= 0.0) error->all(FLERR, "Illegal k for fix ti/eq command");
    iarg = 5;
    algo = Einstein;
  } else if (strcmp(arg[3], "wca") == 0) {
    sigma = utils::numeric(FLERR, arg[4], false, lmp);
    if (sigma <= 0.0) error->all(FLERR, "Illegal sigma for fix ti/eq command");
    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    if (epsilon <= 0.0)
      error->all(FLERR, "Illegal epsilon for fix ti/eq command");
    iarg = 6;
    algo = WCA;
    rc = std::pow(2, 1.0 / 6.0) * sigma;
  } else {
    error->all(FLERR, "Illegal fix ti/eq command");
  }

  if (strcmp(arg[iarg], "lambda") == 0) {
    if (narg != iarg + 2) { error->all(FLERR, "Illegal fix ti/eq command"); }
    lambda = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
    if (lambda < 0.0 || lambda > 1.0) {
      error->all(FLERR, "Illegal lambda for fix ti/eq command");
    }
  }

  // Perform initial allocation of atom-based array
  // Register with Atom class
  xoriginal = nullptr;
  FixThermalIntegration::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // xoriginal = initial unwrapped positions of atoms
  double **x = atom->x;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      domain->unmap(x[i], image[i], xoriginal[i]);
    else
      xoriginal[i][0] = xoriginal[i][1] = xoriginal[i][2] = 0.0;
  }

  espring = 0.0;
}

/* ---------------------------------------------------------------------- */

FixThermalIntegration::~FixThermalIntegration()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id, Atom::GROW);
  atom->delete_callback(id, Atom::RESTART);

  // delete locally stored array
  memory->destroy(xoriginal);
}

/* ---------------------------------------------------------------------- */

int FixThermalIntegration::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::init()
{
  if (utils::strmatch(update->integrate_style, "^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa - 1);
    post_force_respa(vflag, nlevels_respa - 1, 0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa - 1);
  }
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::post_force_wca()
{
  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj;
  double lj1, lj2, lj3, lj4;
  double cutsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;

  double **x = atom->x;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = force->pair->list->inum;
  ilist = force->pair->list->ilist;
  numneigh = force->pair->list->numneigh;
  firstneigh = force->pair->list->firstneigh;

  cutsq = rc * rc;
  lj1 = 48.0 * epsilon * MathSpecial::powint(sigma, 12);
  lj2 = 24.0 * epsilon * MathSpecial::powint(sigma, 6);
  lj3 = 4.0 * epsilon * MathSpecial::powint(sigma, 12);
  lj4 = 4.0 * epsilon * MathSpecial::powint(sigma, 6);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutsq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        forcelj = r6inv * (lj1 * r6inv - lj2);
        fpair = forcelj * r2inv;

        atom->f[i][0] += delx * fpair;
        atom->f[i][1] += dely * fpair;
        atom->f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          atom->f[j][0] -= delx * fpair;
          atom->f[j][1] -= dely * fpair;
          atom->f[j][2] -= delz * fpair;
        }

        evdwl += r6inv * (lj3 * r6inv - lj4) + epsilon;
      }
    }
  }

  espring = evdwl;
}

void FixThermalIntegration::post_force_einstein()
{
  double **x = atom->x;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  double dx, dy, dz, fx, fy, fz;
  double rsq;
  double unwrap[3];

  espring = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      domain->unmap(x[i], image[i], unwrap);
      dx = unwrap[0] - xoriginal[i][0];
      dy = unwrap[1] - xoriginal[i][1];
      dz = unwrap[2] - xoriginal[i][2];
      rsq = dx * dx + dy * dy + dz * dz;
      fx = -k * dx;
      fy = -k * dy;
      fz = -k * dz;
      espring += k * rsq * 0.5;
      atom->f[i][0] = (1 - lambda) * atom->f[i][0] + lambda * fx;
      atom->f[i][1] = (1 - lambda) * atom->f[i][1] + lambda * fy;
      atom->f[i][2] = (1 - lambda) * atom->f[i][2] + lambda * fz;
    }
  }
}

void FixThermalIntegration::post_force(int /*vflag*/)
{
  if (algo == Einstein)
    post_force_einstein();
  else
    post_force_wca();
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::post_force_respa(int vflag, int ilevel,
                                             int /*iloop*/)
{
  if (ilevel == nlevels_respa - 1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixThermalIntegration::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of stretched springs
------------------------------------------------------------------------- */

double FixThermalIntegration::compute_scalar()
{
  double all;
  MPI_Allreduce(&espring, &all, 1, MPI_DOUBLE, MPI_SUM, world);
  return all;
}

/* ----------------------------------------------------------------------
     memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixThermalIntegration::memory_usage()
{
  auto bytes = static_cast<double>(atom->nmax * 3 * sizeof(double));
  return bytes;
}

/* ----------------------------------------------------------------------
     allocate atom-based array
------------------------------------------------------------------------- */

void FixThermalIntegration::grow_arrays(int n)
{
  memory->grow(xoriginal, n, 3, "fix_ti/eq:xoriginal");
}

/* ----------------------------------------------------------------------
     copy values within local atom-based array
------------------------------------------------------------------------- */

void FixThermalIntegration::copy_arrays(int i, int j, int /*delflag*/)
{
  xoriginal[j][0] = xoriginal[i][0];
  xoriginal[j][1] = xoriginal[i][1];
  xoriginal[j][2] = xoriginal[i][2];
}

/* ----------------------------------------------------------------------
    pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixThermalIntegration::pack_exchange(int i, double *buf)
{
  buf[0] = xoriginal[i][0];
  buf[1] = xoriginal[i][1];
  buf[2] = xoriginal[i][2];
  return 3;
}

/* ----------------------------------------------------------------------
    unpack values in local atom-based array from exchange with another proc
 ------------------------------------------------------------------------- */

int FixThermalIntegration::unpack_exchange(int nlocal, double *buf)
{
  xoriginal[nlocal][0] = buf[0];
  xoriginal[nlocal][1] = buf[1];
  xoriginal[nlocal][2] = buf[2];
  return 3;
}

/* ----------------------------------------------------------------------
    pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixThermalIntegration::pack_restart(int i, double *buf)
{
  // pack buf[0] this way because other fixes unpack it
  buf[0] = 4;
  buf[1] = xoriginal[i][0];
  buf[2] = xoriginal[i][1];
  buf[3] = xoriginal[i][2];
  return 4;
}

/* ----------------------------------------------------------------------
    unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixThermalIntegration::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values
  // unpack the Nth first values this way because other fixes pack them

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);
  m++;

  xoriginal[nlocal][0] = extra[nlocal][m++];
  xoriginal[nlocal][1] = extra[nlocal][m++];
  xoriginal[nlocal][2] = extra[nlocal][m];
}

/* ----------------------------------------------------------------------
     maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixThermalIntegration::maxsize_restart()
{
  return 4;
}

/* ----------------------------------------------------------------------
     size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixThermalIntegration::size_restart(int /*nlocal*/)
{
  return 4;
}
