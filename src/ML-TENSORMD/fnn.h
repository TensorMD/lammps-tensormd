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

#ifndef LAMMPS_FNN_H
#define LAMMPS_FNN_H

#include "cnpy.h"
#include "nnp.h"
#include "ration.hpp"

#define ANY_ELEMENT 0

namespace LAMMPS_NS {

template <typename Scalar> class Cutoff;

template <typename Scalar>
class FeaturePotential : public NeuralNetworkPotential<Scalar> {

 public:
  FeaturePotential(Memory *mem, Error *err, int mpiid) :
      NeuralNetworkPotential<Scalar>(mem, err)
  {
    interpolatable = false;
    batch_ration = nullptr;
    rank = mpiid;
    NeuralNetworkPotential<Scalar>::sum_output = true;
  };
  ~FeaturePotential();

  void setup_global(cnpy::npz_t &npz, int &num_filters);
  void setup_global(int nlayers_, const int *layer_sizes, int actfn_,
                    Scalar **weights_, Scalar **biases_, bool use_resnet_dt,
                    bool apply_output_bias);
  void forward(Scalar *x_in, int n, Scalar *y)
  {
    NeuralNetworkPotential<Scalar>::forward(ANY_ELEMENT, x_in, n, y);
  }
  void backward(Scalar *grad_in, Scalar *grad_out)
  {
    NeuralNetworkPotential<Scalar>::backward(ANY_ELEMENT, grad_in, grad_out);
  }

  double get_flops_per_atom(double &forward_flops, double &backward_flops)
  {
    forward_flops = NeuralNetworkPotential<Scalar>::forward_flops_per_one;
    backward_flops = NeuralNetworkPotential<Scalar>::backward_flops_per_one;
    return forward_flops + backward_flops;
  }

  double get_memory_usage()
  {
    double bytes = 0.0;
    if (!interpolatable) {
      return NeuralNetworkPotential<Scalar>::get_memory_usage();
    }
    bytes = batch_ration->get_memory_usage();
    return bytes;
  }

 protected:
  BatchRation1d<Scalar> *batch_ration;
  bool interpolatable;
  int rank;

 public:
  bool is_interpolatable() const { return interpolatable; }
  void setup_ration(Scalar delta, Scalar rmax, Cutoff<Scalar> *cutoff);
  void interpolate(Scalar *x_in, int n, Scalar *y, Scalar *dydx);
};

}    // namespace LAMMPS_NS

#endif    // LAMMPS_FNN_H
