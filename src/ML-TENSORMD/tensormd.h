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

  int get_exact_cmax(int *ilist, int inum, const int *type,
                           const int *fmap, double **x,
                           int *numneigh, int **firstneigh);

  void calc_tensor_density(int *ilist, int inum, const int *type,
                           const int *fmap, double **x, int numneigh_max,
                           int *numneigh, int **firstneigh);

  double memory_usage() const;

  void read_npz(cnpy::npz_t &npz, int &nelt, std::vector<int> &numbers,
                std::vector<double> &masses);
  void setup_interpolation(Scalar delta, int algo);
  void setup_global(double *cutmax);
  void setup_local(int nlocal, int numneigh_max, const int *ntypes);
  Scalar run(int *ntypes, double etemp, double *eatom, double **f,
             double **vatom, double scale, double *satom, double &S,
             double *aatom, double &A, int atommax);

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
  int nmax;
  int cmax;
  int alocal;
  int clocal;
  int num_filters;
  int max_moment;
  Scalar rmax;
  bool is_T_symmetric;

  // Map `ilocal` to/from row index of `rho`.
  int *i2row;
  int *row2i;

  double *farray = nullptr;
  double *varray = nullptr;

  // The tensor-based algorithm
  Eigen::Tensor<Scalar, 2> T;        // md  

  int *ijlist = nullptr;      // 2cba  
  int *mask = nullptr;        // cba
  Scalar *dMdrx = nullptr;    // dxcba
  Scalar *R = nullptr;        // cba
  Scalar *sij = nullptr;      // cba
  Scalar *dsij = nullptr;     // cba
  Scalar *M = nullptr;        // dcba 
  Scalar *V = nullptr;        // dcba
  Scalar *H = nullptr;        // kcba
  Scalar *dHdr = nullptr;     // kcba
  Scalar *G = nullptr;        // mkba
  Scalar *dEdG = nullptr;     // mkba
  Scalar *dEdS = nullptr;     // dkba
  Scalar *P = nullptr;        // dkba  
  Scalar *drdrx = nullptr;    // xcba
  Scalar *F1 = nullptr;       // xcba
  Scalar *F2 = nullptr;       // xcba
  Scalar *S = nullptr;     // dkba

#if defined(USE_SW_NNP)
  int block = 4;
  int a_padded = 0;
  int nlayers;
  int use_resnet;  
  int *layer_sizes;
  int *input_sizes;
  Scalar *bias;
  Scalar *kernel_matrix_T;
  Scalar *kernel_matrix_bp;
  Scalar *last_kernel;

  // 512 128 128 128 128 1
  
  // int config_id = 1;
  // int max_col = 32;
  // int max_layer = 128;
  // int kernel_ngroups[4] = {16, 4, 4, 4};
  // int kernel_cpe_id[4][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23},
  //   {24, 25, 26, 27}};
  // int ncols_on_cpe[4][16] = {
  //   {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32}};
  // int kernel_ngroups_bp[4] = {16, 4, 4, 4};
  // int kernel_cpe_id_bp[4][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23},
  //   {24, 25, 26, 27}};
  // int ncols_on_cpe_bp[4][16] = {
  //   {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32}};

  // 128 128 128 128 1 W

  int config_id = 2;
  int max_col = 16;
  int max_layer = 128;
  int kernel_ngroups[3] = {8, 8, 8};
  int kernel_cpe_id[3][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19, 20, 21, 22, 23}};
  int ncols_on_cpe[3][16] = {
    {16, 16, 16, 16, 16, 16, 16, 16},
    {16, 16, 16, 16, 16, 16, 16, 16},
    {16, 16, 16, 16, 16, 16, 16, 16}};
  int kernel_ngroups_bp[3] = {8, 8, 8};
  int kernel_cpe_id_bp[3][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19, 20, 21, 22, 23}};
  int ncols_on_cpe_bp[3][16] = {
    {16, 16, 16, 16, 16, 16, 16, 16},
    {16, 16, 16, 16, 16, 16, 16, 16},
    {16, 16, 16, 16, 16, 16, 16, 16}};

  // 128 128 128 128 1

  // int config_id = 3;
  // int max_col = 32;
  // int max_layer = 128;
  // int kernel_ngroups[3] = {4, 4, 4};
  // int kernel_cpe_id[3][16] = {
  //   {0, 1, 2, 3},
  //   {4, 5, 6, 7},
  //   {8, 9, 10, 11}};
  // int ncols_on_cpe[3][16] = {
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32}};
  // int kernel_ngroups_bp[3] = {4, 4, 4};
  // int kernel_cpe_id_bp[3][16] = {
  //   {0, 1, 2, 3},
  //   {4, 5, 6, 7},
  //   {8, 9, 10, 11}};
  // int ncols_on_cpe_bp[3][16] = {
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32},
  //   {32, 32, 32, 32}};

  // 512 64 64 64 64 1 MoNi

  // int config_id = 4;
  // int max_col = 32;
  // int max_layer = 64;
  // int kernel_ngroups[4] = {16, 4, 4, 4};
  // int kernel_cpe_id[4][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23},
  //   {24, 25, 26, 27}};
  // int ncols_on_cpe[4][16] = {
  //   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16}};
  // int kernel_ngroups_bp[4] = {16, 4, 4, 4};
  // int kernel_cpe_id_bp[4][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23},
  //   {24, 25, 26, 27}};
  // int ncols_on_cpe_bp[4][16] = {
  //   {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16}};

  // 768 64 64 64 1 AlMgSi

  // int config_id = 5;
  // int max_col = 48;
  // int max_layer = 64;
  // int kernel_ngroups[3] = {16, 4, 4};
  // int kernel_cpe_id[3][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23}};
  // int ncols_on_cpe[3][16] = {
  //   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16}};
  // int kernel_ngroups_bp[3] = {16, 4, 4};
  // int kernel_cpe_id_bp[3][16] = {
  //   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  //   {16, 17, 18, 19},
  //   {20, 21, 22, 23}};
  // int ncols_on_cpe_bp[3][16] = {
  //   {48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
  //   {16, 16, 16, 16},
  //   {16, 16, 16, 16}};
#endif

 public:
  void setup_tensors(const int *ilist, int inum, const int *type,
                     const int *fmap, double **x, int numneigh_max,
                     const int *numneigh, int **firstneigh);

  void timer_switch(bool use_timer) {
    disable_timer = !use_timer;
  }

  const Scalar *get_G() { return G; }
  const int *get_i2row() { return i2row; }
  int get_total_dim() { return total_dims; }
};

}    // namespace LAMMPS_NS
#endif

// LAMMPS_TENSORMD_H
