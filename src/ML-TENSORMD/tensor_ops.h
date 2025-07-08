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

#ifndef LAMMPS_TENSOR_OPS_H
#define LAMMPS_TENSOR_OPS_H

#include "eigen/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;
using Eigen::TensorMap;

template <typename Scalar>
void batch_matmul(TensorMap<Tensor<Scalar, 3>> &in_x,
                  TensorMap<Tensor<Scalar, 3>> &in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 3>> *out, int start, int limit);

template <typename Scalar>
void batch_matmul(Tensor<Scalar, 4> &in_x, Tensor<Scalar, 4> &in_y, bool adj_x,
                  bool adj_y, Tensor<Scalar, 4> *out);

template <typename Scalar>
void batch_matmul(TensorMap<Tensor<Scalar, 4>> &in_x,
                  TensorMap<Tensor<Scalar, 4>> &in_y, bool adj_x, bool adj_y,
                  TensorMap<Tensor<Scalar, 4>> *out);

template <typename Scalar>
void batch_matmul(Tensor<Scalar, 4> &in_x, TensorMap<Tensor<Scalar, 4>> &in_y,
                  bool adj_x, bool adj_y, Tensor<Scalar, 4> *out);

// F1_xcba = U_kcba * dHdr_kcba * drdrx_xcba
template <typename Scalar>
void kernel_F1(Tensor<Scalar, 4> &U, TensorMap<Tensor<Scalar, 4>> &dHdr,
               TensorMap<Tensor<Scalar, 4>> &drdrx, 
               TensorMap<Tensor<Scalar, 4>> *F1);

template <typename Scalar>
void kernel_F1(Tensor<Scalar, 4> &U, TensorMap<Tensor<Scalar, 4>> &dHdr,
               TensorMap<Tensor<Scalar, 4>> &drdrx,
               TensorMap<Tensor<int, 3>> &mask,
               TensorMap<Tensor<Scalar, 4>> *F1);

// F2_xcba = dMdrx_xdcba * V_dcba
template <typename Scalar>
void kernel_F2(TensorMap<Tensor<Scalar, 3>> &dMdrx,
               TensorMap<Tensor<Scalar, 3>> &V,
               TensorMap<Tensor<Scalar, 3>> *F2);

template <typename Scalar>
void kernel_F2(TensorMap<Tensor<Scalar, 5>> &dMdrx,
               TensorMap<Tensor<Scalar, 4>> &V,
               TensorMap<Tensor<int, 3>> &mask,
               Tensor<Scalar, 4> *F2);

template <typename Scalar>
void kernel_fused_F2(int max_moment, TensorMap<Tensor<Scalar, 3>> &V,
                     TensorMap<Tensor<Scalar, 2>> &R,
                     TensorMap<Tensor<Scalar, 3>> &drdrx,
                     TensorMap<Tensor<Scalar, 3>> *F2);

template <typename Scalar>
void kernel_fused_F2(int max_moment, TensorMap<Tensor<Scalar, 4>> &V,
                     TensorMap<Tensor<Scalar, 3>> &R,
                     TensorMap<Tensor<Scalar, 4>> &drdrx,
                     TensorMap<Tensor<int, 3>> &mask,
                     Tensor<Scalar, 4> *F2);

// U_kcba = dEdS_dkba * M_dcba
template <typename Scalar>
void kernel_U(TensorMap<Tensor<Scalar, 4>> &dEdS,
              TensorMap<Tensor<Scalar, 4>> &M,
              TensorMap<Tensor<Scalar, 4>> *U);

// V_dcba = dEdS_dkba * H_kcba
template <typename Scalar>
void kernel_V(TensorMap<Tensor<Scalar, 4>> &dEdS,
              TensorMap<Tensor<Scalar, 4>> &H,
              TensorMap<Tensor<Scalar, 4>> *V);

// P_dkba = M_dcba * H_kcba
template <typename Scalar>
void kernel_P(TensorMap<Tensor<Scalar, 4>> &M, TensorMap<Tensor<Scalar, 4>> &H,
              TensorMap<Tensor<Scalar, 4>> *P);

// dEdS = 2 * T_md * dEdG_mkba * P_dkba
template <typename Scalar>
void kernel_dEdS(Tensor<Scalar, 2> &T, TensorMap<Tensor<Scalar, 4>> &dEdG, 
                 TensorMap<Tensor<Scalar, 4>> &P, 
                 TensorMap<Tensor<Scalar, 4>> *dEdS);

template <typename Scalar>
void kernel_dEdG(TensorMap<Tensor<Scalar, 4>> &G, 
                 TensorMap<Tensor<Scalar, 4>> *dEdG);

#endif
