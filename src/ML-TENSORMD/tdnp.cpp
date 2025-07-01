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
#include "tensormd_types.hpp"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <typename Scalar> TDNP<Scalar>::TDNP(Memory *mem, Error *err)
{
  H = new NeuralNetworkPotential<Scalar>(mem, err);
  U = new NeuralNetworkPotential<Scalar>(mem, err);
  S = new NeuralNetworkPotential<Scalar>(mem, err);

  memory = mem;
  error = err;

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
void TDNP<Scalar>::compute(int elti, Scalar *x_in, int n, Scalar *dfdx,
                           Scalar etemp, TDPE<Scalar> &tdpe)
{
  forward(elti, x_in, n, etemp, tdpe);
  backward(elti, etemp, nullptr, dfdx);
}

template <typename Scalar>
void TDNP<Scalar>::compute(int elti, Scalar *x_in, int n, Scalar *dfdx,
                           Scalar *etemp, TDPE<Scalar> &tdpe)
{
  forward(elti, x_in, n, etemp, tdpe);
  backward(elti, etemp, nullptr, dfdx);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TDNP<Scalar>::forward(int elti, Scalar *x_in, int n, Scalar etemp,
                           TDPE<Scalar> &tdpe)
{
  int i;

  nlocal[elti] = n;
  if (n > nmax[elti]) {
    memory->destroy(y[elti]);
    memory->destroy(dU[elti]);
    memory->destroy(dS[elti]);
    nmax[elti] = n;
    memory->create(y[elti], n * ydim, "tensormd::tdnp::y");
    memory->create(dU[elti], n * ydim, "tensormd::tdnp:dU");
    memory->create(dS[elti], n * ydim, "tensormd::tdnp:dS");
  }

  PE<Scalar> h_out{0.0, y[elti]};
  this->H->forward(elti, x_in, n, h_out);

  // Set the last element of each row to `etemp`
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = ydim - 1; i < n * ydim; i += ydim) y[elti][i] = etemp;

  this->U->forward(elti, y[elti], n, tdpe.E);
  this->S->forward(elti, y[elti], n, tdpe.S);

  // Compute the electron free energy: F(T) = E(T) - T * S(T)
  if (this->is_sommrfeld) {
    // For the Sommerfeld model, the electron entropy is quadratic in T.
    // So we should multiply S by T to make unit consistent.
    tdpe.S.toten *= etemp;
    if (tdpe.calc_atom()) {
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
      for (i = 0; i < n; i++) {
        tdpe.S.eatom[i] *= etemp;
      }
    }
  }
  if (tdpe.calc_atom()) {
    cppblas_copy(n, tdpe.E.eatom, 1, tdpe.F.eatom, 1);
    cppblas_axpy(n, -etemp, tdpe.S.eatom, 1, tdpe.F.eatom, 1);
  }
  tdpe.F.toten = tdpe.E.toten - etemp * tdpe.S.toten;
}

template <typename Scalar>
void TDNP<Scalar>::forward(int elti, Scalar *x_in, int n, Scalar *etemp,
                           TDPE<Scalar> &tdpe)
{
  int i;
  Scalar toten;

  nlocal[elti] = n;
  if (n > nmax[elti]) {
    memory->destroy(y[elti]);
    memory->destroy(dU[elti]);
    memory->destroy(dS[elti]);
    nmax[elti] = n;
    memory->create(y[elti], n * ydim, "tensormd::tdnp::y");
    memory->create(dU[elti], n * ydim, "tensormd::tdnp:dU");
    memory->create(dS[elti], n * ydim, "tensormd::tdnp:dS");
  }

  PE<Scalar> h_out{0.0, y[elti]};
  this->H->forward(elti, x_in, n, h_out);

  // Set the last element of each row to `etemp`
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < n; i++) {
    y[elti][(i + 1) * ydim - 1] = etemp[i];
  }

  this->U->forward(elti, y[elti], n, tdpe.E);
  this->S->forward(elti, y[elti], n, tdpe.S);

  // Compute the electron free energy: F(T) = E(T) - T * S(T)
  if (this->is_sommrfeld) {
    // For the Sommerfeld model, the electron entropy is quadratic in T.
    // So we should multiply S by T to make unit consistent.
    toten = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+ : toten)
#endif
    for (i = 0; i < n; i++) {
      tdpe.S.eatom[i] *= etemp[i];
      toten += tdpe.S.eatom[i];
    }
    tdpe.S.toten = toten;
  }
  toten = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for private(i) reduction(+ : toten)
#endif
  for (i = 0; i < n; i++) {
    tdpe.F.eatom[i] = tdpe.E.eatom[i] - etemp[i] * tdpe.S.eatom[i];
    toten += tdpe.F.eatom[i];
  }
  tdpe.F.toten = toten;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TDNP<Scalar>::backward(int elti, Scalar etemp, Scalar *grad_in,
                            Scalar *grad_out)
{
  int i;
  this->U->backward(elti, grad_in, dU[elti]);
  this->S->backward(elti, grad_in, dS[elti]);
  double scale = etemp;
  if (this->is_sommrfeld) scale *= etemp;
  cppblas_axpy(ydim * nlocal[elti], -scale, dS[elti], 1, dU[elti], 1);
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = ydim - 1; i < nlocal[elti] * ydim; i += ydim) dU[elti][i] = 0.0;
  this->H->backward(elti, dU[elti], grad_out);
}

template <typename Scalar>
void TDNP<Scalar>::backward(int elti, Scalar *etemp, Scalar *grad_in,
                            Scalar *grad_out)
{
  int i;
  this->U->backward(elti, grad_in, dU[elti]);
  this->S->backward(elti, grad_in, dS[elti]);

#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < ydim * nlocal[elti]; i++) {
    if (this->is_sommrfeld) {
      dU[elti][i] -= etemp[i / ydim] * etemp[i / ydim] * dS[elti][i];
    } else {
      dU[elti][i] -= etemp[i / ydim] * dS[elti][i];
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = ydim - 1; i < nlocal[elti] * ydim; i += ydim) dU[elti][i] = 0.0;
  this->H->backward(elti, dU[elti], grad_out);
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
