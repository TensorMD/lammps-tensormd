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

#ifndef LAMMPS_TENSOR_DEBUG_H
#define LAMMPS_TENSOR_DEBUG_H

#include "eigen/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;
using Eigen::TensorMap;

template <typename Scalar>
bool has_inf_or_nan(TensorMap<Tensor<Scalar, 3>> &in, const char *name,
                    int mpiid, bool write_to_file);

template <typename Scalar>
bool has_inf_or_nan(TensorMap<Tensor<Scalar, 4>> &in, const char *name,
                    int mpiid, bool write_to_file);

template <typename Scalar>
bool has_inf_or_nan(Tensor<Scalar, 3> &in, const char *name, int mpiid,
                    bool write_to_file);

template <typename Scalar>
bool has_inf_or_nan(Tensor<Scalar, 4> &in, const char *name, int mpiid,
                    bool write_to_file);

template <typename Scalar>
void dump_tensor(TensorMap<Tensor<Scalar, 3>> &in, const char *name, int mpiid);

template <typename Scalar>
void dump_tensor(Tensor<Scalar, 3> &in, const char *name, int mpiid);

template <typename Scalar>
void dump_tensor(TensorMap<Tensor<Scalar, 4>> &in, const char *name, int mpiid);

template <typename Scalar>
void dump_tensor(Tensor<Scalar, 4> &in, const char *name, int mpiid);

template <typename Scalar>
void dump_sum_forces_data(int amax, int bmax, int cmax, int imax,
                          Tensor<Scalar, 4> &F1, Tensor<Scalar, 4> &F2,
                          Tensor<int, 4> &ijlist, double **f);

#endif
