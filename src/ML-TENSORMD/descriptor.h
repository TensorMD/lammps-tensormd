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

#ifndef LAMMPS_TENSORMD_DESCRIPTOR_H
#define LAMMPS_TENSORMD_DESCRIPTOR_H

#include "cnpy.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "memory.h"
#include "error.h"
#include "ration.hpp"
#include <map>

namespace LAMMPS_NS {

typedef enum { PowerExp, Morse, Density, SF } Descriptor_fn;

using Eigen::Tensor;
using Eigen::TensorMap;
using std::map;

template <typename Scalar> class Cutoff;

template <typename Scalar> class Descriptor {

 public:
  Descriptor(Memory *mem, Error *err, cnpy::npz_t &npz)
  {
    batch_ration = nullptr;
    
    memory = mem;
    error = err;
    interpolatable = false;
    algo = PowerExp;

    method = PowerExp;
    if (npz.find("descriptor::method") != npz.end()) {
      method =
          static_cast<Descriptor_fn>(*npz["descriptor::method"].data<int>());
    }

    if (method == PowerExp) {
      d0 = 2;
      d1 = static_cast<int>(npz["descriptor::rl"].num_vals);
      memory->create(params, d0, d1, "descriptor::pexp::params");
      for (int i = 0; i < d1; i++) {
        params[0][i] = npz["descriptor::rl"].data<Scalar>()[i];
        params[1][i] = npz["descriptor::pl"].data<Scalar>()[i];
      }
    } else if (method == Morse) {
      d0 = 3;
      d1 = static_cast<int>(npz["descriptor::D"].num_vals);
      memory->create(params, d0, d1, "descriptor::morse::params");
      for (int i = 0; i < d1; i++) {
        params[0][i] = npz["descriptor::D"].data<Scalar>()[i];
        params[1][i] = npz["descriptor::gamma"].data<Scalar>()[i];
        params[2][i] = npz["descriptor::r0"].data<Scalar>()[i];
      }
    } else if (method == Density) {
      d0 = 3;
      d1 = static_cast<int>(npz["descriptor::A"].num_vals);
      memory->create(params, d0, d1, "descriptor::density::params");
      for (int i = 0; i < d1; i++) {
        params[0][i] = npz["descriptor::A"].data<Scalar>()[i];
        params[1][i] = npz["descriptor::beta"].data<Scalar>()[i];
        params[2][i] = npz["descriptor::re"].data<Scalar>()[i];
      }
    } else if (method == SF) {
      d0 = 2;
      d1 = static_cast<int>(npz["descriptor::eta"].num_vals);
      memory->create(params, d0, d1, "descriptor::sf::params");
      for (int i = 0; i < d1; i++) {
        params[0][i] = npz["descriptor::eta"].data<Scalar>()[i];
        params[1][i] = npz["descriptor::omega"].data<Scalar>()[i];
      }
    }

    rmax = *npz["rmax"].data<Scalar>();
  }

  ~Descriptor();

  // Number of radial filters
  int get_num_filters() { return d1; }

  // The direct compute routine
  void compute(TensorMap<Tensor<Scalar, 3>> &R,
               TensorMap<Tensor<Scalar, 3>> &sij,
               TensorMap<Tensor<Scalar, 3>> &dsij,
               TensorMap<Tensor<Scalar, 4>> *H,
               TensorMap<Tensor<Scalar, 4>> *dHdr);

  // Setup ration1d interpolation
  void setup_ration(Scalar delta, Scalar rmax, int ration_algo,
                    Cutoff<Scalar> *cutoff);

  // The interpolation routine
  void interpolate(Scalar *x_in, int n, Scalar *y, Scalar *dydx);

  bool is_interpolatable() const { return interpolatable; }

 protected:
  Memory *memory;
  Error *error;

  Descriptor_fn method;

  // params is a two-dimensional array with shape [d0, d1]
  Scalar **params;
  // d0 represents the number of parameters.
  int d0;
  // d1 is the number of parameter sets, or the number of radial filters.
  int d1;

  // the cutoff radius
  Scalar rmax;

  // Ration1D interpolation algorithm.
  int algo;
  // Interpolation status
  bool interpolatable;

  map<int, Ration1d<Scalar> *> ration;
  BatchRation1d<Scalar> *batch_ration;

  // Morse function
  void morse(TensorMap<Tensor<Scalar, 3>> &R, TensorMap<Tensor<Scalar, 3>> &sij,
             TensorMap<Tensor<Scalar, 3>> &dsij,
             TensorMap<Tensor<Scalar, 4>> *H,
             TensorMap<Tensor<Scalar, 4>> *dHdr);

  // Power expoenential function
  void pexp(TensorMap<Tensor<Scalar, 3>> &R, TensorMap<Tensor<Scalar, 3>> &sij,
            TensorMap<Tensor<Scalar, 3>> &dsij, TensorMap<Tensor<Scalar, 4>> *H,
            TensorMap<Tensor<Scalar, 4>> *dHdr);

  // The exponential density function
  void density(TensorMap<Tensor<Scalar, 3>> &R,
               TensorMap<Tensor<Scalar, 3>> &sij,
               TensorMap<Tensor<Scalar, 3>> &dsij,
               TensorMap<Tensor<Scalar, 4>> *H,
               TensorMap<Tensor<Scalar, 4>> *dHdr);

  // The symmetry function
  void sf(TensorMap<Tensor<Scalar, 3>> &R, TensorMap<Tensor<Scalar, 3>> &sij,
          TensorMap<Tensor<Scalar, 3>> &dsij, TensorMap<Tensor<Scalar, 4>> *H,
          TensorMap<Tensor<Scalar, 4>> *dHdr);

};    // namespace LAMMPS_NS

}    // namespace LAMMPS_NS

#endif    // LAMMPS_TENSORALLOY_DESCRIPTOR_H
