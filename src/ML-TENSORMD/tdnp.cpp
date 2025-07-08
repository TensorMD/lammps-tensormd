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

#include "tdnp.h"
#include "cppblas.hpp"
#include "error.h"
#include "memory.h"
#include "nnp.h"
#include "nnp.cuh"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <typename Scalar> TDNP<Scalar>::TDNP(Memory *mem, Error *err)
{
  this->H = new NeuralNetworkPotential<Scalar>(mem, err);
  this->U = new NeuralNetworkPotential<Scalar>(mem, err);
  this->S = new NeuralNetworkPotential<Scalar>(mem, err);

  this->memory = mem;
  this->error = err;

  nelt = 0;
  ydim = 0;

  is_sommrfeld = false;
}

template <typename Scalar> TDNP<Scalar>::~TDNP()
{
  int elti;
  for (elti = 0; elti < nelt; elti++) {
    delete[] y[elti];
    delete[] dU[elti];
    delete[] dS[elti];
  }

  y.clear();
  dU.clear();
  dS.clear();

  delete H;
  delete S;
  delete U;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TDNP<Scalar>::setup_one(cnpy::npz_t &npz, const std::string &scope)
{
  int i, row, elti;
  int nlayers, actfn;
  int num_in;
  int n_rows, n_cols;
  int *layer_sizes;
  Scalar ***weights, ***biases;
  bool use_resnet_dt, apply_output_bias;
  nlayers = *npz[fmt::format("{}::nlayers", scope)].data<int>();
  layer_sizes = npz[fmt::format("{}::layer_sizes", scope)].data<int>();
  actfn = *npz[fmt::format("{}::actfn", scope)].data<int>();
  use_resnet_dt =
      *npz[fmt::format("{}::use_resnet_dt", scope)].data<int>() == 1;
  apply_output_bias =
      *npz[fmt::format("{}::apply_output_bias", scope)].data<int>() == 1;
  
  if (npz.find("tdnp::Sommerfeld") != npz.end()) {
    is_sommrfeld = *npz["tdnp::Sommerfeld"].data<int>() == 1;
  }

  if (scope == "H") {
    // Reserve a column for electron temperature.
    layer_sizes[nlayers - 1] += 1;
  }

  weights = new Scalar **[nelt];
  biases = new Scalar **[nelt];
  for (elti = 0; elti < nelt; elti++) {
    weights[elti] = new Scalar *[nlayers];
    biases[elti] = new Scalar *[nlayers];
    for (i = 0; i < nlayers; i++) {
      // Add a column of zeros to the last weight matrix
      if (scope == "H" && i == nlayers - 1) {
        auto orig =
            npz[fmt::format("H::weights_{}_{}", elti, i)].data<Scalar>();
        if (i == 1) {
          n_rows = static_cast<int>(
              npz[fmt::format("H::weights_{}_0", elti, i)].shape[0]);
        } else {
          n_rows = layer_sizes[i - 1];
        }
        n_cols = layer_sizes[i];
        weights[elti][i] = new Scalar[n_rows * n_cols];
        for (row = 0; row < n_rows; row++) {
          cppblas_copy(n_cols - 1, &orig[row * (n_cols - 1)], 1,
                       &weights[elti][i][row * n_cols], 1);
          weights[elti][i][row * n_cols + n_cols - 1] = 0.0;
        }
      } else {
        weights[elti][i] = npz[fmt::format("{}::weights_{}_{}", scope, elti, i)]
                               .data<Scalar>();
      }
      if (i < nlayers - 1 || apply_output_bias) {
        biases[elti][i] =
            npz[fmt::format("{}::biases_{}_{}", scope, elti, i)].data<Scalar>();
      }
    }
  }

  NeuralNetworkPotential<Scalar> *ptr;
  if (scope == "H") {
    ptr = this->H;
    ydim = layer_sizes[nlayers - 1];
    num_in = static_cast<int>(npz["H::weights_0_0"].shape[0]);
  } else if (scope == "U") {
    ptr = this->U;
    num_in = ydim;
  } else {
    ptr = this->S;
    num_in = ydim;
  }
  ptr->setup_global(nelt, num_in, nlayers, layer_sizes, actfn, weights, biases,
                    use_resnet_dt, apply_output_bias);
  if (scope == "H") {
    for (elti = 0; elti < nelt; elti++) { delete[] weights[elti][nlayers - 1]; }
  }
}

template <typename Scalar> void TDNP<Scalar>::setup_global(cnpy::npz_t &npz)
{
  this->nelt = *npz["nelt"].data<int>();
  setup_one(npz, "H");
  setup_one(npz, "U");
  setup_one(npz, "S");

  int elti;
  for (elti = 0; elti < nelt; elti++) {
    this->nmax[elti] = 0;
    this->nlocal[elti] = 0;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar TDNP<Scalar>::compute(int elti, Scalar *x_in, int n, Scalar *dfdx,
                             Scalar etemp, Scalar *eatom, Scalar *satom,
                             Scalar &s, Scalar *aatom, Scalar &a)
{
  Scalar sum;
  sum = forward(elti, x_in, n, etemp, eatom, satom, s, aatom, a);
  backward(elti, etemp, nullptr, dfdx);
  return sum;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar TDNP<Scalar>::forward(int elti, Scalar *x_in, int n, Scalar etemp,
                             Scalar *eatom, Scalar *satom, Scalar &s,
                             Scalar *aatom, Scalar &a)
{
  int i;
  Scalar u;

  nlocal[elti] = n;
  if (n > nmax[elti]) {
    forward_setup_tdnp_GPU(n, &y_GPU[elti], &dU_GPU[elti], &dS_GPU[elti], ydim);
    nmax[elti] = n;
  }

  this->H->forward(elti, x_in, n, y_GPU[elti], false);
  forward_tdnp_GPU(n, ydim, y_GPU[elti], etemp);
  s = this->S->forward(elti, y_GPU[elti], n, satom);
  u = this->U->forward(elti, y_GPU[elti], n, eatom);

  // Compute the electron free energy
  if (this->is_sommrfeld) {
    s *= etemp;
    if (satom) {
      for (i = 0; i < n; i++) { satom[i] *= etemp; }
    }
  }
  if (aatom) {
    cppblas_copy(n, eatom, 1, aatom, 1);
    cppblas_axpy(n, -etemp, satom, 1, aatom, 1);
  }
  a = u - etemp * s;

  // Return the electron internal energy
  return u;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TDNP<Scalar>::backward(int elti, Scalar etemp, Scalar *grad_in,
                            Scalar *grad_out)
{
  int i;
  this->U->backward(elti, grad_in, dU_GPU[elti]);
  this->S->backward(elti, grad_in, dS_GPU[elti]);
  double scale = etemp;
  if (this->is_sommrfeld) scale *= etemp;
  backward_tdnp_GPU(nlocal[elti], ydim, scale, dS_GPU[elti], dU_GPU[elti]);
  this->H->backward(elti, dU_GPU[elti], grad_out);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> double TDNP<Scalar>::get_flops_per_atom()
{
  return H->get_flops_per_atom() + S->get_flops_per_atom() +
      U->get_flops_per_atom();
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> double TDNP<Scalar>::get_memory_usage()
{
  double bytes = 0.0;
  int elti;
  for (elti = 0; elti < nelt; elti ++) {
    bytes += static_cast<double>(nmax[elti] * ydim * 3);
  }
  bytes *= sizeof(Scalar);
  bytes += H->get_memory_usage();
  bytes += U->get_memory_usage();
  bytes += S->get_memory_usage();
  return bytes;
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::TDNP<float>;
template class LAMMPS_NS::TDNP<double>;
