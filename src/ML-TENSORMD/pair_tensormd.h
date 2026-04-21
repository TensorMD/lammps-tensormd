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

#ifdef PAIR_CLASS

PairStyle(tensormd, PairTensorMD)

#else

#ifndef LMP_PAIR_TENSORMDE_H
#define LMP_PAIR_TENSORMDE_H

#include <map>
#include <string>
#include <vector>

#include "cnpy++.hpp"
#include "eigen/Eigen/Dense"
#include "pair.h"
#include "tensormd.h"
#include "tensormd_types.h"

namespace LAMMPS_NS {

class PairTensorMD : public Pair {
 public:
  explicit PairTensorMD(class LAMMPS*);
  ~PairTensorMD() override;
  void compute(int, int) override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  void init_style() override;
  double init_one(int, int) override;
  void* extract(const char*, int&) override;
  void ev_setup(int, int, int) override;

  double memory_usage() override;

  void update_etemp(double scalar);
  void update_etemp(const double* vector);
  void enable_atom_etemp();
  [[nodiscard]] double get_average_etemp() const { return etemp; }

  // coeff args
  std::vector<std::string> get_coeff_args() { return kwargs; }
  const int* get_atom_map() { return map; }

 private:
  // TODO: use MemoryPool to manage memory for etemp_atom
  int nlocal;

  // Polymorphic pointers to the Model
  // We allow either float or double precision models
  TENSORMD::TensorMD<double>* dpot = nullptr;
  TENSORMD::TensorMD<float>* fpot = nullptr;

  // Execution Backend (CPU/GPU)
  TENSORMD::Device device;

  // Map LAMMPS atom type index (int) to symbol.
  // These are defined by the 'pair_coeff' command.
  std::map<int, std::string> lammps_species;

  // scaling factor for adapt
  double** scale;

  // coeff args
  std::vector<std::string> kwargs;

  void allocate();
  void read_potential_file(const std::string& globalfile);
  void read_coeff_args(int narg, char** arg, double& grid_dr, double& grid_rmin,
                       bool& switch_on_timer, double& kelvin);
  void setup_interpolation(double grid_dr, double grid_rmin);
  void setup_timer(bool switch_on_timer);
  void setup_type_map();
  int setup_compute(std::vector<int>& typenums);

  // The electron temperature for TDNP
  double* etemp_atom;
  double etemp;
  bool use_atom_etemp;

 public:
  // TDNP
  int tdpe_flag;
  TENSORMD::FreeEnergyResult<double> tdpe;
};

}  // namespace LAMMPS_NS

#endif
#endif
