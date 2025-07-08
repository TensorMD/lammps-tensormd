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

#ifndef LAMMPS_TENSORALLOY_NNP_H
#define LAMMPS_TENSORALLOY_NNP_H

#include <cmath>
#include <map>
#include <vector>

#if defined(EIGEN_USE_MKL_ALL)
#include "mkl.h"
#elif defined(__APPLE__) && defined(AccelerateFramework)
#include <Accelerate/accelerate.h>
#else
#include "cblas.h"
#endif

#include "cnpy.h"

using std::map;
using std::vector;

template <typename Scalar> class Activation;

typedef enum { ReLU, Softplus, Tanh, Squareplus } Activation_fn;

template <typename Scalar>
class NeuralNetworkPotential {
 public:
  NeuralNetworkPotential();
  ~NeuralNetworkPotential();

  void setup_global(cnpy::npz_t &npz);
  void setup_global(int nelt, int num_in, int nlayers, const int *layer_sizes,
                    int actfn, Scalar ***weights, Scalar ***biases,
                    bool use_resnet_dt, bool apply_output_bias);
  Scalar compute(int elti, Scalar *x_in, int n, Scalar *y, Scalar *dydx);
  Scalar forward(int elti, Scalar *x_in, int n, Scalar *y);
  void backward(int elti, Scalar *grad_in, Scalar *grad_out);
  virtual double get_memory_usage();
  double get_flops_per_atom();

  void get_config(int *beta, int *nlayers, int *max_layer);
  void get_layer_sizes(int *layer_sizes, int *input_sizes);
  void get_bias(Scalar *bias);
  void get_last_kernel(Scalar *last_kernel);
  void get_parameter(Scalar *kernel_matrix_on_this_cpe_T,
                     Scalar *kernel_matrix_on_this_cpe_bp, int rank);

 protected:

  int nelt;
  map<int, int> nmax;
  map<int, int> nlocal;

  // Network structure
  int n_in, n_out;
  Scalar **pool;
  int n_pool;
  map<int, Scalar **> dz;

  int nlayers;
  map<int, int> full_sizes;
  int max_size;

  map<int, map<int, Scalar *>> weights;
  map<int, map<int, Scalar *>> biases;

  bool use_resnet_dt;
  bool apply_output_bias;

  // FLOPs
  double forward_flops_per_one, backward_flops_per_one;

  // Activation
  Activation_fn actfn;
  Activation<Scalar> *act;

  // Internal functions
  bool sum_output;
  Scalar forward_resnet(int elti, Scalar *x_in, int n, Scalar *y);
  Scalar forward_dense(int elti, Scalar *x_in, int n, Scalar *y);
  void backward_resnet(int elti, Scalar *grad_in, Scalar *grad_out);
  void backward_dense(int elti, Scalar *grad_in, Scalar *grad_out);
};

#endif    //LAMMPS_TENSORALLOY_NNP_H
