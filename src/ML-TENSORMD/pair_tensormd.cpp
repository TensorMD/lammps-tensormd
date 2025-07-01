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

#include "pair_tensormd.h"

#include <memory>

#include "atom.h"
#include "cnpy.h"
#include "comm.h"
#include "eigen/Eigen/Dense"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "tensormd.h"

using namespace LAMMPS_NS;

#define KB_TO_EV(x) ((x) / 11604.522060401008)

/* ---------------------------------------------------------------------- */

PairTensorMD::PairTensorMD(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  allocated = 0;

  dpot = nullptr;
  fpot = nullptr;
  scale = nullptr;

  tdpe_flag = 0;
  etemp_atom = nullptr;
  etemp = 0.0;
  use_atom_etemp = false;

  cutmax = 0.0;
  nlocal = 0;
}

/* ----------------------------------------------------------------------
free all arrays
check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorMD::~PairTensorMD()
{
  delete dpot;
  delete fpot;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }

  if (tdpe_flag == 1) {
    // eatom is handled by the base `Pair` class.
    memory->destroy(tdpe.F.eatom);
    memory->destroy(tdpe.S.eatom);
  }

  memory->destroy(map);

  if (use_atom_etemp) memory->destroy(etemp_atom);
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::setup_compute(int *typenums)
{
  int elti, i, ii;

  // Set zeros
  for (elti = 0; elti < this->atom->ntypes; elti++) { typenums[elti] = 0; }

  // Get `numneigh_max` and `ntypes`
  for (ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    elti = map[atom->type[i]];
    typenums[elti]++;
  }
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::ev_setup(int eflag, int vflag, int alloc)
{
  Pair::ev_setup(eflag, vflag, alloc);

  tdpe.E.eatom = eatom;

  if (tdpe_flag == 1) {
    // Allocate memory for atomic properties
    if (alloc || use_atom_etemp) {
      int size = comm->nthreads * maxeatom;
      memory->destroy(tdpe.S.eatom);
      memory->create(tdpe.S.eatom, size, "pair:tensormd:tdpe:S");
      memory->destroy(tdpe.F.eatom);
      memory->create(tdpe.F.eatom, size, "pair:tensormd:tdpe:F");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::update_etemp(double scalar)
{
  etemp = scalar;
}

void PairTensorMD::enable_atom_etemp()
{
  use_atom_etemp = true;
  if (etemp_atom) {
    error->one(FLERR, "PairTensorMD: etemp_atom is already allocated.");
  }
  memory->create(etemp_atom, atom->nlocal, "pair:tensormd:etemp");

  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < atom->nlocal; i++) { etemp_atom[i] = etemp; }
}

void PairTensorMD::update_etemp(double *vector)
{
  if (!use_atom_etemp) {
    error->one(
        FLERR,
        "PairTensorMD: atom-specified electron temperature is not enabled.");
  }
  if (atom->nlocal > nlocal) {
    memory->destroy(etemp_atom);
    memory->create(etemp_atom, atom->nlocal, "pair:tensormd:etemp");
    nlocal = atom->nlocal;
  }
  int i;
  etemp = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+:etemp)
#endif
  for (i = 0; i < atom->nlocal; i++) {
    etemp_atom[i] = vector[i];
    etemp += vector[i];
  }
  etemp /= atom->nlocal;
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::compute(int eflag, int vflag)
{
  int numneigh_max;
  auto typenums = new int[atom->ntypes];
  double *et;

  ev_init(eflag, vflag);

  // typenums[elti] is the number of atoms of type `elti`
  setup_compute(typenums);

  if (dpot)
    numneigh_max =
        dpot->get_exact_cmax(list->ilist, list->inum, atom->type, map, atom->x,
                             list->numneigh, list->firstneigh);
  else
    numneigh_max =
        fpot->get_exact_cmax(list->ilist, list->inum, atom->type, map, atom->x,
                             list->numneigh, list->firstneigh);
  if (numneigh_max < 0) {
    error->one(FLERR, "cmax < 0 is unexpected and not permitted.");
  } else if (numneigh_max == 0) {
    // No atoms to compute, skip the computation
    delete[] typenums;
    return;
  }

  // Allocate memory for tensors
  if (dpot)
    dpot->setup_local(atom->nlocal, numneigh_max, typenums);
  else
    fpot->setup_local(atom->nlocal, numneigh_max, typenums);

  // The scale factor for thermodynamics integration. The current implementation
  // suppose all atom types have the same scaling factor.
  double scale_factor = scale[1][1];

  // Calculate density
  tdpe.set_toten_zero();
  et = use_atom_etemp ? etemp_atom : &etemp;

  if (dpot) {
    dpot->compute(list->ilist, atom->type, map, atom->x, list->numneigh,
                  list->firstneigh, typenums, et, use_atom_etemp, tdpe, atom->f,
                  vatom, scale_factor);
  } else {
    fpot->compute(list->ilist, atom->type, map, atom->x, list->numneigh,
                  list->firstneigh, typenums, et, use_atom_etemp, tdpe, atom->f,
                  vatom, scale_factor);
  }

  if (eflag) { eng_vdwl = tdpe.E.toten; }

  // The total virial can be computed
  if (vflag_fdotr) virial_fdotr_compute();

  delete[] typenums;
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  nlocal = atom->nlocal;
  map = new int[n + 1];
  memset(map, 0, sizeof(int) * (n + 1));
}

/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */

void PairTensorMD::settings(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTensorMD::coeff(int narg, char **arg)
{
  int m, n, i;
  double dx;
  double kelvin = 0.0;
  bool read_elt = true;

  if (!allocated) allocate();

  if (narg < 4) error->all(FLERR, "Incorrect args for pair coefficients");

  // insure I,J args are * *
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // check for presence of first potential file
  std::string lib_file = utils::get_potential_file_path(arg[2]);
  if (lib_file.empty())
    error->all(FLERR, fmt::format("Cannot open potential file {}", lib_file));

  // read atom symbols and options
  m = 1;
  i = 3;
  dx = -1.0;
  while (i < narg) {
    auto var = std::string(arg[i]);
    if (var == "interp") {
      dx = std::strtod(arg[i + 1], nullptr);
      i += 2;
      read_elt = false;
    } else if (var == "etemp") {
      kelvin = KB_TO_EV(std::strtod(arg[i + 1], nullptr));
      i += 2;
      read_elt = false;
    } else if (read_elt) {
      eltmap[m] = var;
      m++;
      i++;
    }
  }
  read_potential_file(lib_file);
  if (tdpe_flag == 1) { update_etemp(kelvin); }

  if (screen && comm->me == 0) {
    fprintf(screen, "Read potential file: %s\n", lib_file.c_str());
  }
  if (logfile && comm->me == 0) {
    fprintf(logfile, "Read potential file: %s\n", lib_file.c_str());
  }

  if (dpot)
    dpot->setup_global(&cutmax);
  else
    fpot->setup_global(&cutmax);

  // setup fnn interpolation
  if (dx > 0) {
    if (dpot) {
      dpot->setup_interpolation(dx);
    } else {
      fpot->setup_interpolation(static_cast<float>(dx));
    }
    if (logfile && comm->me == 0) {
      fprintf(logfile, "Use interpolation: dr = %.4f\n", dx);
    }
    if (screen && comm->me == 0) {
      fprintf(screen, "Use interpolation: dr = %.4f\n", dx);
    }
  }

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  // set mass for i,i in atom class

  int count = 0;
  for (i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        if (i == j) atom->set_mass(FLERR, i, mass[map[i]]);
        count++;
      }
      scale[i][j] = 1.0;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

  for (i = 2; i < narg; i++) { this->kwargs.emplace_back(arg[i]); }
}

/* ----------------------------------------------------------------------
init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorMD::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style tensormd requires newton pair on");

  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorMD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];
  return cutmax;
}

/* ----------------------------------------------------------------------
Setup `map`: map[eltype] = i
 eltmap: specified in `pair_coeff`
 model_eltmap: specified in the potential file.
------------------------------------------------------------------------- */

void PairTensorMD::setup_eltmap(int nelt, const int *numbers, int size)
{
  int elti, i, j;
  std::map<int, std::string> model_eltmap;
  for (elti = 0; elti < nelt; elti++) {
    i = elti * 2;
    char c = static_cast<char>(numbers[i + 0]);
    std::string s;
    s.push_back(c);
    if (i + 1 < size && numbers[i + 1] > 0) {
      s.push_back(static_cast<char>(numbers[i + 1]));
    }
    model_eltmap[elti + 1] = s;
  }

  map[0] = -1;
  for (i = 1; i < nelt + 1; i++) {
    for (j = 1; j < nelt + 1; j++) {
      if (eltmap[i] == model_eltmap[j]) { map[i] = j - 1; }
    }
  }
}

/* ----------------------------------------------------------------------
Read the npz potential file
------------------------------------------------------------------------- */

void PairTensorMD::read_potential_file(const std::string &globalfile)
{
  int elti, nelt;
  std::vector<int> numbers;
  std::vector<double> masses;
  auto npz = cnpy::npz_load(globalfile);
  bool is_tdnp;

  if (npz.find("precision") != npz.end()) {
    if (npz["precision"].data<int>()[0] == 32) {
      fpot = new TensorMD<float>(memory, error, screen, logfile, comm->me);
      if (logfile && comm->me == 0)
        fprintf(logfile, "float precision: 32 bits\n");
      if (screen && comm->me == 0)
        fprintf(screen, "float precision: 32 bits\n");
    }
  }
  if (!fpot)
    dpot = new TensorMD<double>(memory, error, screen, logfile, comm->me);
  if (fpot) {
    fpot->read_npz(npz, nelt, numbers, masses);
    is_tdnp = fpot->is_tdnp();
  } else {
    dpot->read_npz(npz, nelt, numbers, masses);
    is_tdnp = dpot->is_tdnp();
  }
  this->setup_eltmap(nelt, numbers.data(), static_cast<int>(numbers.size()));
  for (elti = 0; elti < nelt; elti++) mass.push_back(masses[elti]);

  if (is_tdnp) { tdpe_flag = 1; }
}

/* ----------------------------------------------------------------------
memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorMD::memory_usage()
{
  if (dpot)
    return dpot->memory_usage();
  else
    return fpot->memory_usage();
}

/* ---------------------------------------------------------------------- */

void *PairTensorMD::extract(const char *str, int &dim)
{
  if (strcmp(str, "scale") == 0) {
    dim = 2;
    return (void *) scale;
  } else if (strcmp(str, "etemp") == 0) {
    dim = 0;
    return &etemp;
  }
  return nullptr;
}
