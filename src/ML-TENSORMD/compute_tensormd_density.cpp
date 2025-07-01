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

#include "compute_tensormd_density.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair_tensormd.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTensorMDDensity::ComputeTensorMDDensity(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), list(nullptr), dpot(nullptr), fpot(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal compute tensormd/density command");

  pair = dynamic_cast<PairTensorMD *>(lmp->force->pair);
  if (pair == nullptr)
    error->all(FLERR,
               "pair tensormd must be initialized before command "
               "compute tensormd/density");

  size_peratom_cols = this->coeff(pair->get_coeff_args());
  peratom_flag = 1;
  timeflag = 1;
  comm_reverse = 0;
  nmax = 0;
  G = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeTensorMDDensity::~ComputeTensorMDDensity() {
  delete dpot;
  delete fpot;

  memory->destroy(G);
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDDensity::init()
{
  neighbor->add_request(this,
                        NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  if (modify->get_compute_by_style("tensormd/density").size() > 1 && comm->me == 0)
    error->warning(FLERR, "More than one compute tensormd/density");
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDDensity::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- 
 Coeff functions
 ------------------------------------------------------------------------ */

void ComputeTensorMDDensity::read_potential_file(const std::string &globalfile)
{
  int nelt;
  std::vector<int> numbers;
  std::vector<double> masses;
  auto npz = cnpy::npz_load(globalfile);
  if (npz.find("precision") != npz.end()) {
    if (npz["precision"].data<int>()[0] == 32) {
      fpot = new TensorMD<float>(memory, error, screen, logfile,
                                          comm->me);
      fpot->timer_switch(false);
    }
  }
  if (!fpot) {
    dpot = new TensorMD<double>(memory, error, screen, logfile, comm->me);
    dpot->timer_switch(false);
  }
  if (fpot)
    fpot->read_npz(npz, nelt, numbers, masses);
  else
    dpot->read_npz(npz, nelt, numbers, masses);
}


int ComputeTensorMDDensity::coeff(std::vector<std::string> args)
{
  double dx = 0.0001;
  double cutmax;

  std::string lib_file = utils::get_potential_file_path(args[0]);
  read_potential_file(lib_file);
  if (dpot)
    dpot->setup_global(&cutmax);
  else
    fpot->setup_global(&cutmax);

  // setup fnn interpolation
  if (dpot) {
    dpot->setup_interpolation(dx);
  } else {
    fpot->setup_interpolation(static_cast<float>(dx));
  }

  return dpot ? dpot->get_bkm() : fpot->get_bkm();
}

/* ---------------------------------------------------------------------- */

void ComputeTensorMDDensity::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // Grow G if necessary
  if (atom->nmax > nmax) {
    memory->destroy(G);
    nmax = atom->nmax;
    memory->create(G, nmax, size_peratom_cols, "tensormd/atom:density");
    array_atom = G;
  }

  // invoke full neighbor list (will copy or build if necessary)
  neighbor->build_one(list);

  const int max_neltypes = 8;

  int ntypes = atom->ntypes;
  int numneigh_max[max_neltypes];
  int typenums[max_neltypes];
  int elti, i, j, a, loc, ii, nnl_max;

  // The atom map
  const int* map = pair->get_atom_map();

  // Set zeros
  for (elti = 0; elti < this->atom->ntypes; elti++) {
    typenums[elti] = 0;
    numneigh_max[elti] = 0;
  }

  // Get `numneigh_max` and `ntypes`
  for (ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    elti = map[atom->type[i]];
    typenums[elti]++;
    numneigh_max[elti] = MAX(numneigh_max[elti], list->numneigh[i]);
  }

  // Compute `nnl_max`
  nnl_max = 0;
  for (elti = 0; elti < ntypes; elti++) {
    nnl_max = MAX(nnl_max, static_cast<double>(numneigh_max[elti]));
  }

  // Compute the density
  if (dpot) {
    dpot->setup_local(atom->nlocal, nnl_max, typenums);
    dpot->setup_tensors(list->ilist, atom->type, map, atom->x, list->numneigh, list->firstneigh);
    dpot->compute_radial_density();
    dpot->compute_projected_density();
  } else {
    fpot->setup_local(atom->nlocal, nnl_max, typenums);
    fpot->setup_tensors(list->ilist, atom->type, map, atom->x, list->numneigh, list->firstneigh);
    fpot->compute_radial_density();
    fpot->compute_projected_density();
  }

  int nlocal = atom->nlocal;
  const int *i2a = dpot ? dpot->get_i2a() : fpot->get_i2a();
  for (i = 0; i < nlocal; i++) {
    if (i2a)
      a = i2a[i];
    else
      a = i;
    loc = a * size_peratom_cols;
    for (j = 0; j < size_peratom_cols; j++) {
      if (dpot)
        G[i][j] = dpot->get_G()[loc + j];
      else
        G[i][j] = static_cast<double>(fpot->get_G()[loc + j]);
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeTensorMDDensity::memory_usage()
{
  double bytes = (double)nmax * size_peratom_cols * sizeof(double);
  bytes += dpot ? dpot->memory_usage() : fpot->memory_usage();
  return bytes;
}

