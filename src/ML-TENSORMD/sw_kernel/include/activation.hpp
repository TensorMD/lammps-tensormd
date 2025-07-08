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

#ifndef LAMMPS_TENSORALLOY_ACTIVATION_H
#define LAMMPS_TENSORALLOY_ACTIVATION_H

#include "cppblas.hpp"
#include <cmath>

template <typename Scalar> class Activation {

 public:
  Activation() = default;
  ~Activation() = default;

  void activate(int actfn, Scalar *x, int n, Scalar *dx)
  {
    switch (actfn) {
      case 0:
        relu(x, n, dx);
        break;
      case 1:
        softplus(x, n, dx);
        break;
      case 2:
        tanh(x, n, dx);
        break;
      case 3:
        squareplus(x, n, dx);
        break;
      default:
        break;
    }
  }

  void activate(int actfn, Scalar *x, int n)
  {
    switch (actfn) {
      case 0:
        relu(x, n);
        break;
      case 1:
        softplus(x, n);
        break;
      case 2:
        tanh(x, n);
        break;
      case 3:
        squareplus(x, n);
        break;
      default:
        break;
    }
  }

  void grad(int actfn, Scalar *x, int n)
  {
    switch (actfn) {
      case 0:
        relu_grad(x, n);
        break;
      case 1:
        softplus_grad(x, n);
        break;
      case 2:
        tanh_grad(x, n);
        break;
      case 3:
        squareplus_grad(x, n);
        break;
      default:
        break;
    }
  }

  Scalar *ones_like(int n)
  {
    grow_ones(n);
    return ones;
  }

  double get_memory_usage()
  {
    return static_cast<double>(ones_size * sizeof(Scalar));
  }

 protected:
  /* ---------------------------------------------------------------------- */

  Scalar *ones;
  size_t ones_size;

  void grow_ones(int n)
  {
    if (n > ones_size) {
      ones_size = n;
      ones = (Scalar *) realloc(ones, ones_size * sizeof(Scalar));
      for (int i = 0; i < n; ++i) ones[i] = 1.0;
    }
  }

  /* ---------------------------------------------------------------------- */

  void softplus(Scalar *x, int n, Scalar *dx)
  {
    static const Scalar threshold = -20.0;
    Scalar exp;
    int i;
    for (i = 0; i < n; ++i) {
      exp = std::exp(x[i]);
      if (x[i] < -threshold) { x[i] = std::log1p(exp); }
      dx[i] = 1.0 / (1.0 + 1.0 / exp);
    }
  }

  void softplus(Scalar *x, int n)
  {
    static const Scalar threshold = -20.0;
    int i;
    Scalar exp;
    for (i = 0; i < n; ++i) {
      exp = std::exp(x[i]);
      if (x[i] < -threshold) { x[i] = std::log1p(exp); }
    }
  }

  void softplus_grad(Scalar *x, int n)
  {
    static const Scalar threshold = -20.0;
    Scalar exp;
    for (int i = 0; i < n; ++i) {
      if (x[i] >= -threshold) {
        exp = x[i];
      } else {
        exp = std::exp(x[i]) - 1.0;
      }
      x[i] = 1.0 / (1.0 + 1.0 / exp);
    }
  }

  /* ---------------------------------------------------------------------- */

  void relu(Scalar *x, int n, Scalar *dx)
  {
    int i;
    for (i = 0; i < n; ++i) {
      x[i] = x[i] >= 0 ? x[i] : 0.0;
      dx[i] = x[i] >= 0 ? 1.0 : 0.0;
    }
  }

  void relu(Scalar *x, int n)
  {
    int i;
    for (i = 0; i < n; ++i) { x[i] = x[i] >= 0 ? x[i] : 0.0; }
  }

  void relu_grad(Scalar *x, int n)
  {
    int i;
    for (i = 0; i < n; ++i) { x[i] = x[i] >= 0 ? 1.0 : 0.0; }
  }

  /* ---------------------------------------------------------------------- */

  void squareplus(Scalar *x, int n, Scalar *dx)
  {
    Scalar z;
    for (int i = 0; i < n; i++) {
      z = sqrt(x[i] * x[i] + 4.0);
      x[i] = 0.5 * (x[i] + z);
      dx[i] = x[i] / z;
    }
  }

  void squareplus(Scalar *x, int n)
  {
    for (int i = 0; i < n; i++) {
      x[i] = 0.5 * (x[i] + sqrt(x[i] * x[i] + 4.0));
    }
  }

  void squareplus_grad(Scalar *x, int n)
  {
    Scalar z;
    for (int i = 0; i < n; i++) {
      z = sqrt(x[i] * x[i] + 4.0);
      x[i] = 0.5 * (1 + x[i] / z);
    }
  }

  /* ---------------------------------------------------------------------- */

  void tanh(Scalar *x, int n, Scalar *dx)
  {
    int i;
    for (i = 0; i < n; ++i) {
      x[i] = std::tanh(x[i]);
      dx[i] = 1.0 - x[i] * x[i];
    }
  }

  void tanh(Scalar *x, int n)
  {
    int i;
    for (i = 0; i < n; ++i) { x[i] = std::tanh(x[i]); }
  }

  void tanh_grad(Scalar *x, int n)
  {
    int i;
    for (i = 0; i < n; i++) { x[i] = 1.0 - x[i] * x[i]; }
  }

  /* ---------------------------------------------------------------------- */
};

#endif    //LAMMPS_TENSORALLOY_ACTIVATION_H
