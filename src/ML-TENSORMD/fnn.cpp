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

#include "fnn.h"
#include "cutoff.h"
#include "eigen/Eigen/Dense"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <typename Scalar> FeaturePotential<Scalar>::~FeaturePotential()
{
  delete batch_ration;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FeaturePotential<Scalar>::setup_global(cnpy::npz_t &npz, int &num_filters)
{
  int i;
  int nlayers_, actfn_;
  int *layer_sizes_;
  int use_reset_dt_, apply_output_bias_;
  Scalar **weights_, **biases_;

  nlayers_ = *npz["fnn::nlayers"].data<int>();
  actfn_ = *npz["fnn::actfn"].data<int>();
  layer_sizes_ = npz["fnn::layer_sizes"].data<int>();
  use_reset_dt_ = *npz["fnn::use_resnet_dt"].data<int>();
  apply_output_bias_ = *npz["fnn::apply_output_bias"].data<int>();
  weights_ = new Scalar *[nlayers_];
  biases_ = new Scalar *[nlayers_];
  for (i = 0; i < nlayers_; i++) {
    weights_[i] = npz[fmt::format("fnn::weights_0_{}", i)].data<Scalar>();
    if (i < nlayers_ - 1 || apply_output_bias_)
      biases_[i] = npz[fmt::format("fnn::biases_0_{}", i)].data<Scalar>();
  }
  setup_global(nlayers_, layer_sizes_, actfn_, weights_, biases_, use_reset_dt_,
               apply_output_bias_);
  num_filters = *npz["fnn::num_filters"].data<int>();
}

template <typename Scalar>
void FeaturePotential<Scalar>::setup_global(int nlayers_,
                                            const int *layer_sizes, int actfn_,
                                            Scalar **weights_, Scalar **biases_,
                                            bool use_resnet_dt,
                                            bool apply_output_bias)
{
  if (nlayers_ < 2) {
    NeuralNetworkPotential<Scalar>::error->all(
        FLERR, "FeaturePotential: nlayers >= 2");
  }

  NeuralNetworkPotential<Scalar>::setup_global(
      1, 1, nlayers_, layer_sizes, actfn_, &weights_, &biases_, use_resnet_dt,
      apply_output_bias);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void FeaturePotential<Scalar>::setup_ration(Scalar delta, Scalar rmax,
                                            Cutoff<Scalar> *cutoff)
{
  int i, n;

  n = static_cast<int>(rmax / delta) + 1;
  auto x = new Scalar[n];
  auto n_out = NeuralNetworkPotential<Scalar>::n_out;
  auto error = NeuralNetworkPotential<Scalar>::error;
  for (i = 0; i < n; i++) x[i] = static_cast<Scalar>(i) * delta;

  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  Matrix y{n_out, n};
  forward(x, n, y.data());

  Eigen::RowVectorX<Scalar> sij{n};
  Eigen::TensorMap<Eigen::Tensor<Scalar, 3>> x_{x, {n, 1, 1}};
  Eigen::TensorMap<Eigen::Tensor<Scalar, 3>> s_{sij.data(), {n, 1, 1}};
  cutoff->compute(x_, &s_);
  y.array().rowwise() *= sij.array();
  batch_ration = new BatchRation1d<Scalar>(error);
  batch_ration->setup(n_out, n, delta, y.data());
  delete[] x;
  interpolatable = true;
}

template <typename Scalar>
void FeaturePotential<Scalar>::interpolate(Scalar *x_in, int n, Scalar *y,
                                           Scalar *dydx)
{
  batch_ration->compute(x_in, n, y, dydx);
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::FeaturePotential<float>;
template class LAMMPS_NS::FeaturePotential<double>;
