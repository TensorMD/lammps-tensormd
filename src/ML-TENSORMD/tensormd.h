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
#include "tensormd_types.hpp"

namespace LAMMPS_NS {

class Memory;
class Error;
template <typename Scalar> class NeuralNetworkPotential;
template <typename Scalar> class FeaturePotential;
template <typename Scalar> class TDNP;
template <typename Scalar> class Cutoff;
template <typename Scalar> class Descriptor;
class TensorMDTimer;
template <typename Scalar> struct TDPE;

using Eigen::Tensor;
using Eigen::TensorMap;
typedef Eigen::array<Eigen::Index, 2> Shape2d;
typedef Eigen::array<Eigen::Index, 3> Shape3d;
typedef Eigen::array<Eigen::Index, 4> Shape4d;

template <typename Scalar> class TensorMD {

 public:
  TensorMD(Memory *mem, Error *error, FILE *scr, FILE *log, int mpiid);
  ~TensorMD();

  // the total memory usage
  double memory_usage() const;

  // read the model file
  void read_npz(cnpy::npz_t &npz, int &nelt, std::vector<int> &numbers,
                std::vector<double> &masses);

  // setup the fast rational interpolation
  void setup_interpolation(Scalar delta);

  // the global setup, only call this routine once
  void setup_global(double *cutmax);

  // the local setup, call this routine at every step
  void setup_local(int nlocal, int numneigh_max, const int *ntypes);

  // is this a temperature-dependent potential?
  bool is_tdnp() { return tdnp != nullptr; }

  /* ----------------------------------------------------------------------
     Core Kernels.
  ------------------------------------------------------------------------- */

  // 1. setup_tensors
  void setup_tensors(const int *ilist, const int *type, const int *fmap,
                     double **x, int *numneigh, int **firstneigh);

  // 2. calculate the radial density
  void compute_radial_density();

  // 3. calculate projected density (radial && angular)
  void compute_projected_density();

  // 4. calculate the neural network potential
  void compute_nnp(int *typenums, PE<double> &pe, double scale);
  void compute_tdnp(int *typenums, double *etemp, bool use_atom_etemp,
                    TDPE<double> &tdpe, double scale);

  // 5. calculate the gradients
  void compute_gradients();

  // 6. calculate the forces
  void compute_forces(double **f, double **vatom, double scale);

  // The public interface
  void compute(int *ilist, const int *type, const int *fmap, double **pos,
               int *numneigh, int **firstneigh, int *typenums,
               double *etemp, bool use_atom_etemp, TDPE<double> &pe, double **f,
               double **vatom, double scale);

  /* ----------------------------------------------------------------------
     Customized Kernels.
  ------------------------------------------------------------------------- */

  // get the exact cmax for the step
  int get_exact_cmax(const int *ilist, int inum, const int *type,
                     const int *fmap, double **x, int *numneigh,
                     int **firstneigh);

  /* ----------------------------------------------------------------------
     Misc Public APIs
  ------------------------------------------------------------------------- */

  void timer_switch(bool use_timer) { disable_timer = !use_timer; }
  const Scalar *get_G() { return G; }

  // Get the i-to-a mapping, where `i` refers to original LAMMPS atom indices
  // and `a` are rearranged indices of TensorMD.
  const int *get_i2a() { return i2a; }

  // Get the value `b * k * m`, the descriptor size for each atom
  int get_bkm() { return size.m * size.b * size.k; }

 protected:
  Memory *memory;
  Error *error;
  NeuralNetworkPotential<Scalar> *nnp;
  FeaturePotential<Scalar> *fnn;
  TDNP<Scalar> *tdnp;
  Descriptor<Scalar> *descriptor;

  // the screen stream
  FILE *screen;

  // the logfile stream
  FILE *logfile;

  // the MPI rank id
  int mpiid;

  // cutforce = force cutoff
  // cutforcesq = force cutoff squared
  Scalar cutforce, cutforcesq;
  Cutoff<Scalar> *cutoff;

  // the element indices
  struct _eltind_t {
    int offset;
    int row;
  } eltind[95];
  int **eltij_pos;

  // The timer
  TensorMDTimer *timer;
  bool disable_timer;

  // rmax is the maximum cutoff for all elements.
  Scalar rmax;

  // The maximum angular moment
  int max_moment;

  // @deprecated: if the T tensor is symmetric or not.
  bool is_T_symmetric;

  // `i` refers to the original LAMMPS atom indices, while `a` are the
  // rearranged indices of TensorMD. `i2a` and `a2i` are the mappings between
  // the two indices.
  int *i2a;
  int *a2i;

  // The tensor dimension
  // The data structure for the tensor sizes
  struct tensor_size_t {

    int a;               // the number of atoms
    int as;              // the number of atoms at current step
    int b;               // the number of element types
    int c;               // the number of neighbors
    int cs;              // the number of neighbors at current step
    int d;               // the number of unique projected moments
    int k;               // the number of radial basis functions
    int m;               // the number of angular moments (m = max_moment + 1)
    int x;               // the Cartesian directions (x, y, z)
    bool initialized;    // if the tensor sizes are initialized.

    // Convenient helper functions to get the tensor shapes
    Shape3d cba() const { return {cs, b, as}; }
    Shape4d tcba() const { return {2, cs, b, as}; }
    Shape4d xcba() const { return {x, cs, b, as}; }
    Shape4d dkba() const { return {d, k, b, as}; }
    Shape4d dcba() const { return {d, cs, b, as}; }
    Shape4d kcba() const { return {k, cs, b, as}; }
    Shape4d mkba() const { return {m, k, b, as}; }

    void get_memory_space(size_t &nscalars, size_t &nintegers,
                          bool use_sij = false) const
    {
      nintegers = a * b * c * 3;               // ijlist, mask
      size_t n = 1 + 3 + 3 + 3 + k * 2 + d;    // R, drdrx, F1, F2, H, dHdr, M
      if (use_sij) {
        n += 2;    // sij, dsij
      }
      nscalars = n * a * b * c;
      nscalars += a * b * k * m * 2;        // G, dEdG
      nscalars += a * b * k * d;            // dEdS
      nscalars += MAX(k, c) * a * b * d;    // P/V
    }

  } size{0, 0, 0, 0, 0, 0, 0, 0, 3, false};

  // The global memory pools for integers and reals
  struct memory_pool_t {
    Scalar *scalars;
    int *integers;
    size_t ps;
    size_t pi;

    // Get a scalar array of size `n`
    Scalar *fget(size_t n)
    {
      Scalar *ptr = scalars + ps;
      ps += n;
      return ptr;
    }

    // Get an integer array of size `n`
    int *iget(int n)
    {
      int *ptr = integers + pi;
      pi += n;
      return ptr;
    }

    // Reset the memory pool
    void reset()
    {
      ps = 0;
      pi = 0;
    }

  } memory_pool{nullptr, nullptr, 0, 0};

  Tensor<Scalar, 2> T;        // md
  Scalar *R = nullptr;        // cba
  int *ijlist = nullptr;      // 2cba
  Scalar *M = nullptr;        // dcba
  int *mask = nullptr;        // cba
  Scalar *H = nullptr;        // kcba
  Scalar *U = nullptr;        // kcba
  Scalar *dHdr = nullptr;     // kcba
  Scalar *G = nullptr;        // mkba
  Scalar *dEdG = nullptr;     // mkba
  Scalar *dEdS = nullptr;     // dkba
  Scalar *P = nullptr;        // dkba
  Scalar *V = nullptr;        // dcba
  Scalar *drdrx = nullptr;    // xcba
  Scalar *F1 = nullptr;       // xcba
  Scalar *F2 = nullptr;       // xcba
  Scalar *sij = nullptr;      // cba
  Scalar *dsij = nullptr;     // cba
};

}    // namespace LAMMPS_NS
#endif

// LAMMPS_TENSORMD_H
