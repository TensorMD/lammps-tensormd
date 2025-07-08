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

#include "descriptor.h"
#include "cutoff.h"
#include "eigen/Eigen/Dense"

using namespace LAMMPS_NS;
using Eigen::Tensor;
using Eigen::TensorMap;

enum { DEFAULT, BATCH, BATCH_GHOST, BATCH_DIRECT };

/* ---------------------------------------------------------------------- */

template <typename Scalar> Descriptor<Scalar>::~Descriptor()
{
  memory->destroy(params);

  int k;
  for (k = 0; k < this->d1; k++) delete ration[k];
  ration.clear();

  delete batch_ration;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void Descriptor<Scalar>::compute(TensorMap<Tensor<Scalar, 3>> &R,
                                 TensorMap<Tensor<Scalar, 3>> &sij,
                                 TensorMap<Tensor<Scalar, 3>> &dsij,
                                 TensorMap<Tensor<Scalar, 4>> *H,
                                 TensorMap<Tensor<Scalar, 4>> *dHdr)
{
  if (method == PowerExp) {
    pexp(R, sij, dsij, H, dHdr);
  } else if (method == Morse) {
    morse(R, sij, dsij, H, dHdr);
  } else if (method == Density) {
    density(R, sij, dsij, H, dHdr);
  } else {
    sf(R, sij, dsij, H, dHdr);
  }
}

template <typename Scalar>
void Descriptor<Scalar>::pexp(TensorMap<Tensor<Scalar, 3>> &R,
                              TensorMap<Tensor<Scalar, 3>> &sij,
                              TensorMap<Tensor<Scalar, 3>> &dsij,
                              TensorMap<Tensor<Scalar, 4>> *H,
                              TensorMap<Tensor<Scalar, 4>> *dHdr)
{
  // f(x) = exp(-(x / rl)**pl)
  int k;
  Scalar *rlist = params[0];
  Scalar *plist = params[1];

  for (k = 0; k < this->d1; k++) {
    auto aij = R / rlist[k];
    auto fij = (-aij.pow(plist[k])).exp().eval();
    auto dfij = (-fij * plist[k] * aij.pow(plist[k] - 1.0) / rlist[k]).eval();
    (*H).chip(k, 0) = sij * fij;
    if (dHdr) (*dHdr).chip(k, 0) = dsij * fij + sij * dfij;
  }
}

template <typename Scalar>
void Descriptor<Scalar>::morse(TensorMap<Tensor<Scalar, 3>> &R,
                               TensorMap<Tensor<Scalar, 3>> &sij,
                               TensorMap<Tensor<Scalar, 3>> &dsij,
                               TensorMap<Tensor<Scalar, 4>> *H,
                               TensorMap<Tensor<Scalar, 4>> *dHdr)
{
  // f(x) = d * [ exp(-2 * gamma * (r - r0)) - 2 * exp(-gamma * (r - r0)) ]
  int k;
  Scalar *D = params[0];
  Scalar *gamma = params[1];
  Scalar *r0 = params[2];

  for (k = 0; k < this->d1; k++) {
    auto gdr = gamma[k] * (R - r0[k]);
    auto expgdr = (-gdr).exp();
    auto exp2gdr = (-2 * gdr).exp();
    auto fij = (D[k] * (exp2gdr - 2 * expgdr)).eval();
    auto dfij = -D[k] * gamma[k] * 2 * (exp2gdr - expgdr);
    (*H).chip(k, 0) = sij * fij;
    if (dHdr) (*dHdr).chip(k, 0) = dsij * fij + sij * dfij;
  }
}

template <typename Scalar>
void Descriptor<Scalar>::density(TensorMap<Tensor<Scalar, 3>> &R,
                                 TensorMap<Tensor<Scalar, 3>> &sij,
                                 TensorMap<Tensor<Scalar, 3>> &dsij,
                                 TensorMap<Tensor<Scalar, 4>> *H,
                                 TensorMap<Tensor<Scalar, 4>> *dHdr)
{
  // f(r) = a * exp(-b * (r / re - 1))
  int k;
  Scalar *A = params[0];
  Scalar *beta = params[1];
  Scalar *re = params[2];

  for (k = 0; k < this->d1; k++) {
    auto z = -beta[k] * ((R - re[k]) / re[k]);
    auto expz = z.exp();
    auto fij = A[k] * expz;
    auto dfij = -fij * beta[k] / re[k];
    (*H).chip(k, 0) = sij * fij;
    if (dHdr) (*dHdr).chip(k, 0) = dsij * fij + sij * dfij;
  }
}

template <typename Scalar>
void Descriptor<Scalar>::sf(TensorMap<Tensor<Scalar, 3>> &R,
                            TensorMap<Tensor<Scalar, 3>> &sij,
                            TensorMap<Tensor<Scalar, 3>> &dsij,
                            TensorMap<Tensor<Scalar, 4>> *H,
                            TensorMap<Tensor<Scalar, 4>> *dHdr)
{
  // f(x) = exp(-(x - omega)**2 * eta)
  int k;
  Scalar *eta = params[0];
  Scalar *omega = params[1];
  Scalar rc2 = rmax * rmax;

  for (k = 0; k < this->d1; k++) {
    auto dx = R - omega[k];
    auto fij = (-eta[k] * dx.square() / rc2).exp();
    auto dfij = -2.0 * eta[k] / rc2 * dx * fij;
    (*H).chip(k, 0) = sij * fij;
    if (dHdr) (*dHdr).chip(k, 0) = dsij * fij + sij * dfij;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void Descriptor<Scalar>::setup_ration(Scalar delta, Scalar rcut,
                                      int ration_algo, Cutoff<Scalar> *cutoff)
{
  int i, n;
  this->algo = ration_algo;

  n = static_cast<int>(rcut / delta) + 1;
  if (algo == BATCH_GHOST) { n += 4; }
  auto x = new Scalar[n];
  auto n_out = this->d1;
  if (algo == BATCH_GHOST)
    for (i = 0; i < n; i++) x[i] = static_cast<Scalar>(i - 2) * delta;
  else
    for (i = 0; i < n; i++) x[i] = static_cast<Scalar>(i) * delta;
  
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  Matrix y{n_out, n};
  auto sij = new Scalar[n];

  TensorMap<Tensor<Scalar, 3>> R{x, {n, 1, 1}};
  TensorMap<Tensor<Scalar, 3>> s{sij, {n, 1, 1}};
  cutoff->compute(R, &s);

  TensorMap<Tensor<Scalar, 4>> H{y.data(), {n_out, n, 1, 1}};
  compute(R, s, s, &H, nullptr);

  if (algo != DEFAULT) {
    batch_ration = new BatchRation1d<Scalar>(error);
    if (algo == BATCH_DIRECT) {
      batch_ration->setup_direct(n_out, n, delta, y.data());
    } else {
      batch_ration->setup(n_out, n, delta, y.data());
    }
  } else {
    y.transposeInPlace();
    for (i = 0; i < n_out; i++) {
      ration[i] = new Ration1d<Scalar>(error);
      ration[i]->setup(n, delta, &y.data()[i * n]);
    }
  }

  interpolatable = true;

  delete [] x;
  delete [] sij;
}

template <typename Scalar>
void Descriptor<Scalar>::interpolate(Scalar *x_in, int n, Scalar *y,
                                     Scalar *dydx)
{
  if (algo == BATCH_GHOST) {
    batch_ration->compute_ghost(x_in, n, y, dydx);
  } else if (algo == BATCH) {
    batch_ration->compute(x_in, n, y, dydx);
  } else if (algo == BATCH_DIRECT) {
    batch_ration->compute_direct(x_in, n, y, dydx);
  } else {
    int k;
    for (k = 0; k < this->d1; k++) {
      ration[k]->compute(x_in, n, &y[k], &dydx[k], this->d1);
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::Descriptor<float>;
template class LAMMPS_NS::Descriptor<double>;
