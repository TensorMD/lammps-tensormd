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
#include "cnpy++.hpp"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "tensormd_strategy.h"
#include "tensormd_v1.h"
#include "tensormd_v2.h"

using namespace LAMMPS_NS;

#define KB_TO_EV(x) ((x) / 11604.522060401008)

/* ---------------------------------------------------------------------- */

PairTensorMD::PairTensorMD(LAMMPS* lmp) : Pair(lmp) {
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  allocated = 0;

  dpot = nullptr;
  fpot = nullptr;

  // Select the execution backend (CPU/GPU)
  // TensorMD uses CPU as the default execution backend.
  // There are two ways to specify the preferred execution backend:
  // 1. Set the environment variable TENSORMD_DEVICE to "CPU" or "GPU"
  // 2. Use the pair_coeff command with an additional argument "device CPU/cpu"
  // or "device GPU/gpu". Approach 2 has the higher priority and can override
  // the environment variable setting.
  device = TENSORMD::Device::CPU;
  const char* preferred_device = std::getenv("TENSORMD_DEVICE");
  if (preferred_device != nullptr) {
    std::string device_str(preferred_device);
    if (device_str == "GPU" || device_str == "gpu") {
#if defined(TENSORMD_GPU)
      device = TENSORMD::Device::GPU;
      if (comm->me == 0) {
        utils::logmesg(lmp, "[TensorMD] Using GPU as the execution backend.\n");
      }
#else
      if (comm->me == 0) {
        utils::logmesg(
            lmp,
            "[TensorMD] GPU support is not enabled. Falling back to CPU.\n");
      }
#endif
    } else if (device_str == "CPU" || device_str == "cpu") {
      if (comm->me == 0) {
        utils::logmesg(lmp, "[TensorMD] Using CPU as the execution backend.\n");
      }
    } else {
      error->one(FLERR,
                 "[TensorMD] Invalid TENSORMD_DEVICE environment variable. "
                 "Expected 'CPU/cpu' or 'GPU/gpu'.\n");
    }
  } else {
    if (comm->me == 0) {
      utils::logmesg(lmp,
                     "[TensorMD] No preferred device specified. Using CPU as "
                     "the execution backend.\n");
    }
  }

  scale = nullptr;

  nlocal = 0;
  tdpe_flag = 0;
  etemp_atom = nullptr;
  etemp = 0.0;
  use_atom_etemp = false;
}

/* ----------------------------------------------------------------------
free all arrays
check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorMD::~PairTensorMD() {
  std::vector<FILE*> c_streams = {screen, logfile};
  if (dpot) {
    dpot->print_timer(c_streams);
  } else {
    fpot->print_timer(c_streams);
  }
  delete dpot;
  delete fpot;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }

  // tdpe.E.eatom is handled by the base class Pair
  memory->destroy(tdpe.F.eatom);
  memory->destroy(tdpe.S.eatom);

  if (use_atom_etemp) memory->destroy(etemp_atom);
}

/* ---------------------------------------------------------------------- */

int PairTensorMD::setup_compute(std::vector<int>& typenums) {
  const int ntypes = static_cast<int>(dpot ? dpot->get_species().size()
                                           : fpot->get_species().size());
  for (int elti = 0; elti < ntypes; elti++) {
    typenums[elti] = 0;
  }
  for (int ii = 0; ii < list->inum; ii++) {
    int i = list->ilist[ii];
    int elti = map[atom->type[i]];
    typenums[elti]++;
  }
  int ntotal = 0;
  for (int elti = 0; elti < ntypes; elti++) {
    ntotal += typenums[elti];
  }
  return ntotal;
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::ev_setup(int eflag, int vflag, int alloc) {
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

void PairTensorMD::update_etemp(double scalar) { etemp = scalar; }

void PairTensorMD::enable_atom_etemp() {
  use_atom_etemp = true;
  if (etemp_atom) {
    error->one(FLERR, "[PairTensorMD] etemp_atom is already allocated.");
  }
  memory->create(etemp_atom, atom->nlocal, "pair:tensormd:etemp");

  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < atom->nlocal; i++) {
    etemp_atom[i] = etemp;
  }
}

void PairTensorMD::update_etemp(const double* vector) {
  if (!use_atom_etemp) {
    error->one(
        FLERR,
        "[PairTensorMD] atom-specified electron temperature is not enabled.");
  }
  if (atom->nlocal > nlocal) {
    memory->destroy(etemp_atom);
    memory->create(etemp_atom, atom->nlocal, "pair:tensormd:etemp");
    nlocal = atom->nlocal;
  }
  int i;
  etemp = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+ : etemp)
#endif
  for (i = 0; i < atom->nlocal; i++) {
    etemp_atom[i] = vector[i];
    etemp += vector[i];
  }
  etemp /= atom->nlocal;
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::compute(int eflag, int vflag) {
  // LAMMPS standard ev_init
  ev_init(eflag, vflag);

  // The scale factor for thermodynamics integration. The current implementation
  // suppose all atom types have the same scaling factor.
  double scale_factor = scale[1][1];

  // Wrap the `compute`
  auto compute = [&](auto* model) {

    // typenums[elti] is the number of atoms of type `elti`
    std::vector<int> typenums(model->get_species().size());

    // 1. Determine typenums and numneigh_max
    if (setup_compute(typenums) == 0) {
      return;
    }

    // 2. Sync LAMMPS data to device.
    // This is required for GPU, but we do it for CPU as well for code path
    // consistency.
    model->sync_lammps_data(list->ilist, list->inum, atom->nlocal, atom->nghost,
                            atom->nmax, atom->type, map, atom->x,
                            list->numneigh, list->firstneigh, atom->f, vatom);

    // 3. Build neighbor list if needed (GPU only)
    model->build_neighbor_list(this->neighbor->ago, this->neighbor->cutneighmax,
                               this->domain->sublo, this->domain->subhi);

    // 4. Get the exact 'cmax' for this step
    int cmax = model->get_exact_cmax();
    if (cmax <= 0) {
      return;
    }

    // 5. Setup local tensors if needed
    model->setup_local(atom->nlocal, cmax, typenums.data(), vatom != nullptr);

    // 6. Compute energy, forces, and virial
    tdpe.zero();
    double* etemp_ptr = use_atom_etemp ? etemp_atom : &etemp;
    model->compute(typenums.data(), etemp_ptr, use_atom_etemp, tdpe, atom->f,
                   vatom, scale_factor);
  };

  // Dispatch to the correct model
  if (dpot)
    compute(dpot);
  else
    compute(fpot);

  // LAMMPS standard finalization
  if (eflag) {
    eng_vdwl = tdpe.E.toten;
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::allocate() {
  allocated = 1;

  int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  nlocal = atom->nlocal;
  map = new int[lammps_species.size() + 1];
  for (int i = 0; i < lammps_species.size() + 1; i++) map[i] = 0;
}

/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */

void PairTensorMD::settings(int narg, char** /*arg*/) {
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTensorMD::read_coeff_args(int narg, char** arg, double& grid_dr,
                                   double& grid_rmin, bool& switch_on_timer,
                                   double& kelvin) {
  // Read coeff args: atom species and other optional args
  int itype = 1;
  int iarg = 3;
  bool read_atom_species = true;

  grid_dr = -1.0;
  grid_rmin = 0.0;
  switch_on_timer = false;
  kelvin = 0.0;
  while (iarg < narg) {
    auto var = std::string(arg[iarg]);
    // Check for the interpolation settings
    if (var == "interp") {
      grid_dr = std::strtod(arg[iarg + 1], nullptr);
      iarg += 2;
      read_atom_species = false;
    } else if (var == "rmin") {
      grid_rmin = std::strtod(arg[iarg + 1], nullptr);
      iarg += 2;
      read_atom_species = false;
    }
    // Check for the electron temperature setting
    else if (var == "etemp") {
      kelvin = KB_TO_EV(std::strtod(arg[iarg + 1], nullptr));
      iarg += 2;
      read_atom_species = false;
    }
    // Check for the timer setting
    else if (var == "timer") {
      const char* timer_flag_str = arg[iarg + 1];
      if (strcmp(timer_flag_str, "on") == 0) {
        switch_on_timer = true;
        if (comm->me == 0) {
          utils::logmesg(lmp, "[TensorMD] Timer enabled via pair_coeff.\n");
        }
      } else if (strcmp(timer_flag_str, "off") == 0) {
        if (comm->me == 0) {
          utils::logmesg(lmp, "[TensorMD] Timer disabled via pair_coeff.\n");
        }
      } else {
        error->one(FLERR,
                   "[TensorMD] Invalid timer setting in pair_coeff. "
                   "Expected 'timer on' or 'timer off'.\n");
      }
      iarg += 2;
    }
    // Check for the device setting
    else if (var == "device") {
      auto device_str = std::string(arg[iarg + 1]);
      if (device_str == "GPU" || device_str == "gpu") {
#if defined(TENSORMD_GPU)
        device = TENSORMD::Device::GPU;
        if (comm->me == 0) {
          utils::logmesg(lmp,
                         "[TensorMD] Using GPU as the execution backend (via "
                         "pair_coeff).\n");
        }
#else
        if (comm->me == 0) {
          utils::logmesg(
              lmp,
              "[TensorMD] GPU support is not enabled (via pair_coeff). "
              "Falling back to CPU.\n");
        }
#endif
      } else if (device_str == "CPU" || device_str == "cpu") {
        device = TENSORMD::Device::CPU;
        if (comm->me == 0) {
          utils::logmesg(lmp,
                         "[TensorMD] Using CPU as the execution backend (via "
                         "pair_coeff).\n");
        }
      } else {
        error->one(FLERR,
                   "[TensorMD] Invalid device specified in pair_coeff. "
                   "Expected 'CPU/cpu' or 'GPU/gpu'.\n");
      }
      iarg += 2;
      read_atom_species = false;
    }
    // Read atom species by default
    else if (read_atom_species) {
      lammps_species[itype] = var;
      itype++;
      iarg++;
    }
  }
}

void PairTensorMD::setup_interpolation(double grid_dr, double grid_rmin) {
  // Check interpolation dr
  const char* var_dr_str = std::getenv("TENSORMD_INTERP_DELTA");
  if (var_dr_str != nullptr) {
    grid_dr = std::strtod(var_dr_str, nullptr);
    if (comm->me == 0)
      utils::logmesg(lmp,
                     fmt::format("[TensorMD] Override interpolation dr with "
                                 "TENSORMD_INTERP_DELTA = {}\n",
                                 var_dr_str));
  }

  // Check interpolation r0
  const char* var_r0_str = std::getenv("TENSORMD_INTERP_RMIN");
  if (var_r0_str != nullptr) {
    grid_rmin = std::strtod(var_r0_str, nullptr);
    if (comm->me == 0)
      utils::logmesg(lmp,
                     fmt::format("[TensorMD] Override interpolation rmin with "
                                 "TENSORMD_INTERP_RMIN = {}\n",
                                 var_r0_str));
  }

  if (TENSORMD::STRATEGY != TENSORMD::Strategy::Exact) {
    std::vector<std::string> species;
    for (const auto& spec : lammps_species) {
      species.push_back(spec.second);
    }
    if (dpot) {
      dpot->setup_interpolation(species, grid_rmin, grid_dr);
    } else if (fpot) {
      fpot->setup_interpolation(species, static_cast<float>(grid_rmin),
                                static_cast<float>(grid_dr));
    } else {
      error->all(FLERR, "Unexpected error during TensorMD initialization!");
    }
    if (comm->me == 0)
      utils::logmesg(
          lmp, fmt::format("[TensorMD] Interpolation rmin = {}, dr = {}\n",
                           grid_rmin, grid_dr));
  }
}

void PairTensorMD::setup_timer(bool switch_on_timer) {
  // Check the envionment variable to see if the timer should be enabled
  const char* timer_env_str = std::getenv("TENSORMD_TIMER");
  if (timer_env_str != nullptr) {
    if (strcmp(timer_env_str, "on") == 0) {
      if (dpot) dpot->switch_timer(true);
      if (fpot) fpot->switch_timer(true);
      if (comm->me == 0) {
        utils::logmesg(
            lmp,
            "[TensorMD] Timer enabled via TENSORMD_TIMER environment "
            "variable.\n");
      }
    } else if (strcmp(timer_env_str, "off") == 0) {
      if (dpot) dpot->switch_timer(false);
      if (fpot) fpot->switch_timer(false);
      if (comm->me == 0) {
        utils::logmesg(
            lmp,
            "[TensorMD] Timer disabled via TENSORMD_TIMER environment "
            "variable.\n");
      }
    } else {
      error->one(FLERR,
                 "[TensorMD] Invalid TENSORMD_TIMER environment variable. "
                 "Expected 'on' or 'off'.\n");
    }
  } else if (switch_on_timer) {
    if (dpot) dpot->switch_timer(true);
    if (fpot) fpot->switch_timer(true);
    if (comm->me == 0) {
      utils::logmesg(lmp, "[TensorMD] Timer enabled via pair_coeff.\n");
    }
  }
}

void PairTensorMD::coeff(int narg, char** arg) {
  // Check the number of args
  if (narg < 4) {
    error->all(FLERR, "Incorrect args for pair coefficients");
  }

  // Insure I,J args are * *
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR, "Incorrect args for pair coefficients");
  }

  // Check for presence of first potential file
  std::string lib_file = utils::get_potential_file_path(arg[2]);
  if (lib_file.empty()) {
    error->all(FLERR, fmt::format("Cannot open potential file {}", lib_file));
  }

  double grid_dr, grid_rmin, kelvin;
  bool switch_on_timer;
  read_coeff_args(narg, arg, grid_dr, grid_rmin, switch_on_timer, kelvin);

  // Allocate the memory
  if (!allocated) allocate();

  // Update the strategy first
  std::vector<FILE*> streams{screen, logfile};
  TENSORMD::setup_strategy_from_env(device, streams, comm->me == 0);

  // Read the potential file and initialize a model instance
  if (comm->me == 0) {
    utils::logmesg(
        lmp, fmt::format("[TensorMD] Reading potential file: {}\n", lib_file));
  }
  read_potential_file(lib_file);

  // Setup electron temperature if needed
  if (tdpe_flag == 1) {
    update_etemp(kelvin);
  }

  // Setup the model
  if (dpot) {
    dpot->setup_global(map, atom->ntypes);
  } else if (fpot) {
    fpot->setup_global(map, atom->ntypes);
  } else {
    error->all(FLERR, "Unexpected error during TensorMD initialization!");
  }
  setup_interpolation(grid_dr, grid_rmin);
  setup_timer(switch_on_timer);

  // LAMMPS standard setup
  // clear setflag since coeff() called once with I,J = * *
  int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  // set mass for i,i in atom class
  int count = 0;
  n = atom->ntypes;
  std::vector<double> masses = dpot ? dpot->get_masses() : fpot->get_masses();
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        if (i == j) {
          atom->set_mass(FLERR, i, masses[map[i]]);
        }
        count++;
      }
      scale[i][j] = 1.0;
    }
  }
  if (count == 0) {
    error->all(FLERR, "Incorrect args for pair coefficients");
  }

  // Keep track of the coeff args
  for (int iarg = 2; iarg < narg; iarg++) {
    this->kwargs.emplace_back(arg[iarg]);
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, fmt::format("[TensorMD] Setup done.\n"));
  }
}

/* ----------------------------------------------------------------------
init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorMD::init_style() {
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style tensormd requires newton pair on");

  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorMD::init_one(int i, int j) {
  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];
  return dpot ? dpot->get_rcut() : fpot->get_rcut();
}

/* ----------------------------------------------------------------------
 setup lmp->pair->map
 this array maps LAMMPS pair_coeff atom types to TensorMD atom types.
 ------------------------------------------------------------------------- */
void PairTensorMD::setup_type_map() {
  std::string mesg = "[TensorMD] LAMMPS (coeff) atom types:";
  for (const auto& it : lammps_species) {
    mesg += fmt::format(" {}={}", it.first, it.second);
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, mesg + "\n");
  }
  mesg = "[TensorMD] Internal atom types:";
  auto tensormd_species = dpot ? dpot->get_species() : fpot->get_species();
  for (const auto& it : tensormd_species) {
    mesg += fmt::format(" {}={}", it.first, it.second);
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, mesg + "\n");
  }

  map[0] = -1;
  mesg = "[TensorMD] Atom Type Map (LAMMPS->TensorMD):";
  for (int i = 1; i < lammps_species.size() + 1; i++) {
    for (int j = 0; j < tensormd_species.size(); j++) {
      if (lammps_species[i] == tensormd_species[j]) {
        map[i] = j;
        mesg += fmt::format(" {}={}", i, j);
      }
    }
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, mesg + "\n");
  }
}

/* ----------------------------------------------------------------------
Read the npz potential file
------------------------------------------------------------------------- */

void PairTensorMD::read_potential_file(const std::string& globalfile) {
  int precision, version;
  std::map<int, std::string> species;
  auto npz = cnpypp::npz_load(globalfile);
  bool is_tdnp = false;

  // First, we determin the float precision: 32 bits or 64 bits.
  // This key is required.
  if (npz.find("precision") != npz.end()) {
    precision = npz.find("precision")->second.data<int>()[0];
  } else {
    error->all(FLERR,
               "[TensorMD] Failed to find 'precision' in the potential file!");
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, fmt::format("[TensorMD] Float precision is {} bits.\n",
                                    precision));
  }
  // Then we determine the model version.
  // The potential files of TensorAlloy-1.x does not have this key.
  if (npz.find("__version__major__") != npz.end()) {
    version = npz.find("__version__major__")->second.data<int>()[0];
  } else {
    version = 1;
  }

  // Initialize the potential instance
  int mpiid = comm->me;
  std::string strategy_desc, model_desc;
  if (version == 1 && precision == 32) {
    fpot = new TENSORMD::TensorMDV1<float>(device, mpiid);
    fpot->initialize(npz, model_desc);
    fpot->adjust_strategy(strategy_desc);
    is_tdnp = fpot->is_finite_temperature_model();
  } else if (version == 1 && precision == 64) {
    dpot = new TENSORMD::TensorMDV1<double>(device, mpiid);
    dpot->initialize(npz, model_desc);
    dpot->adjust_strategy(strategy_desc);
    is_tdnp = dpot->is_finite_temperature_model();
  } else if (version == 2 && precision == 32) {
    fpot = new TENSORMD::TensorMDV2<float>(device, mpiid);
    fpot->initialize(npz, model_desc);
    fpot->adjust_strategy(strategy_desc);
  } else {
    dpot = new TENSORMD::TensorMDV2<double>(device, mpiid);
    dpot->initialize(npz, model_desc);
    dpot->adjust_strategy(strategy_desc);
  }
  if (comm->me == 0) {
    auto repr = fmt::format("[TensorMD] V{} model is loaded: {}\n", version,
                            model_desc);
    utils::logmesg(lmp, repr);
    utils::logmesg(lmp, strategy_desc);
  }

  // Setup the index-to-specie map for the simulation
  this->setup_type_map();

  // Setup the tdpe flag
  // TODO: change 'tdpe' to 'free_energy'
  if (is_tdnp) {
    if (comm->me == 0) {
      utils::logmesg(
          lmp, fmt::format("[TensorMD] Finite-temperature model detected.\n"));
    }
    tdpe_flag = 1;
  }
}

/* ----------------------------------------------------------------------
memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorMD::memory_usage() {
  if (dpot) {
    return dpot->get_total_memory_usage();
  }
  if (fpot) {
    return fpot->get_total_memory_usage();
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void* PairTensorMD::extract(const char* str, int& dim) {
  if (strcmp(str, "scale") == 0) {
    dim = 2;
    return (void*)scale;
  }
  if (strcmp(str, "etemp") == 0) {
    dim = 0;
    return &etemp;
  }
  return nullptr;
}
