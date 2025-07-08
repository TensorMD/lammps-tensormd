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

#include "cnpy.h"
#include "pair.h"
#include "tensormd.h"
#include <map>
#include <string>
#include <vector>

#include "eigen/Eigen/Dense"

namespace LAMMPS_NS {

class PairTensorMD : public Pair {
 public:
  explicit PairTensorMD(class LAMMPS *);
  ~PairTensorMD() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;
  void ev_setup(int, int, int) override;

  double memory_usage() override;

  // The electron temperature for TDNP
  double etemp;

  // coeff args
  std::vector<std::string> get_coeff_args() {
    return kwargs;
  }
  const int *get_atom_map() { return map; }

 private:

  TensorMD<double> *dpot;
  TensorMD<float> *fpot;
  double cutmax;               // max cutoff for all elements
  std::vector<double> mass;    // mass of library element
  std::map<int, std::string> eltmap;

  double **scale;    // scaling factor for adapt

  std::vector<std::string> kwargs; // coeff args
  
  bool use_exact_cmax;

  void allocate();
  void read_potential_file(const std::string &globalfile);
  void setup_eltmap(int nelt, const int *numbers, int n);
  void setup_compute(int &numneigh_max, int *typenums);

  // TDNP
public:
  int tdpe_flag;
  double eentropy;
  double *satom;
  double free_energy;
  double *aatom;
};

}    // namespace LAMMPS_NS

#endif
#endif
