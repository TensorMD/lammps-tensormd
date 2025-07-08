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

#ifndef LAMMPS_TENSORMD_H
#define LAMMPS_TENSORMD_H

#include <cmath>
#include <map>
#include <string>

#include "cnpy.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "math_const.h"
#include "math_special.h"
#include "ration.hpp"

#define MAX_ELETYPES 5
#define MAX_MOMENT 3

namespace LAMMPS_NS {

class Memory;
class Error;
template <typename Scalar> class NeuralNetworkPotential;
template <typename Scalar> class FeaturePotential;
template <typename Scalar> class TDNP;
template <typename Scalar> class Cutoff;
template <typename Scalar> class Descriptor;
class TensorMDTimer;

template <typename Scalar> class TensorMD {
 public:
  TensorMD(Memory *mem, Error *error, FILE *scr, FILE *log, int mpiid);
  ~TensorMD();

  int get_exact_cmax(int *ilist, int inum, const int *type, const int *fmap,
                     double **x, int *numneigh, int **firstneigh);

  void calc_tensor_density(int *ilist, int inum, const int *type,
                           const int *fmap, double **x, int numneigh_max,
                           int *numneigh, int **firstneigh, int nmax_global,
                           bool use_exact_cmax);

  double memory_usage() const;

  void read_npz(cnpy::npz_t &npz, int &nelt, std::vector<int> &numbers,
                std::vector<double> &masses);
  void setup_interpolation(Scalar delta);
  void setup_global(double *cutmax);
  void setup_local(int nlocal, int numneigh_max, const int *ntypes);
  Scalar run(int *ntypes, double etemp, double *eatom, double **f,
             double **vatom, double scale, double *satom, double &S,
             double *aatom, double &A);

 private:
  Memory *memory;
  Error *error;
  NeuralNetworkPotential<Scalar> *nnp;
  FeaturePotential<Scalar> *fnn;
  TDNP<Scalar> *tdnp;
  Descriptor<Scalar> *descriptor;

  FILE *screen;
  FILE *logfile;
  int mpiid;

  // cutforce = force cutoff
  // cutforcesq = force cutoff squared
  Scalar cutforce, cutforcesq;
  Cutoff<Scalar> *cutoff;

  int neltypes;
  int dims;
  int total_dims;
  int nv_dims;

  struct _eltind_t {
    int offset;
    int row;
  };
  struct _eltind_t eltind[MAX_ELETYPES];
  int **eltij_pos;

  // timer
  TensorMDTimer *timer;
  bool disable_timer;

 protected:
  int vmD_nv[MAX_MOMENT + 1]{};
  int nmax;
  int cmax;
  int alocal;
  int num_filters;
  int max_moment;
  Scalar rmax;
  bool is_T_symmetric;

  // Map `ilocal` to/from row index of `rho`.
  int *i2row;
  int *row2i;

  // The tensor-based algorithm
  Eigen::Tensor<Scalar, 2> T;    // md
  Eigen::Tensor<Scalar, 4> G;    // mkba

  int *ilist_GPU = nullptr;
  double *pos_GPU = nullptr;
  int *numneigh_GPU = nullptr;
  int *firstneigh_GPU = nullptr;
  int *ijlist_GPU = nullptr;
  int *i2row_GPU = nullptr;
  int *eltij_pos_GPU = nullptr;
  int *fmap_type_GPU = nullptr;
  Scalar *R_GPU = nullptr;
  Scalar *G_GPU = nullptr;
  Scalar *dEdG_GPU = nullptr;
  Scalar *T_GPU = nullptr;
  Scalar *P_GPU = nullptr;
  Scalar *dEdS_GPU = nullptr;
  Scalar *M_GPU = nullptr;
  Scalar *H_GPU = nullptr;
  Scalar *V_GPU = nullptr;
  Scalar *dHdr_GPU = nullptr;
  Scalar *drdrx_GPU = nullptr;
  Scalar *F1_GPU = nullptr;
  Scalar *F2_GPU = nullptr;
  Scalar *F_GPU = nullptr;
  Scalar *dMdrx_GPU = nullptr;
  int *mask_GPU = nullptr;
  double *vatom_GPU = nullptr;
  int numneigh_max_GPU = 0;
  int nmax_GPU = 0;

 public:
  void setup_tensors(const int *ilist, int inum, const int *type,
                     const int *fmap, double **x, int numneigh_max,
                     const int *numneigh, int **firstneigh);

  void timer_switch(bool use_timer) { disable_timer = !use_timer; }

  const Scalar *get_G() { return G.data(); }
  const int *get_i2row() { return i2row; }
  int get_total_dim() { return total_dims; }
};

}    // namespace LAMMPS_NS
#endif

// LAMMPS_TENSORMD_H
