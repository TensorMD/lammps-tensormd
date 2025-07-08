/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
    Contributing authors:
             Rodrigo Freitas (UC Berkeley) - rodrigof@berkeley.edu
             Mark Asta (UC Berkeley) - mdasta@berkeley.edu
             Maurice de Koning (Unicamp/Brazil) - dekoning@ifi.unicamp.br
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(ti/eq, FixThermalIntegration)

#else

#ifndef LMP_FIX_TI_EQ_H
#define LMP_FIX_TI_EQ_H

#include "fix.h"

namespace LAMMPS_NS {

typedef enum { Einstein, WCA } TI_ALGO;

class FixThermalIntegration : public Fix {
 public:
  FixThermalIntegration(class LAMMPS *, int, char **);
  ~FixThermalIntegration() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void post_force_wca();
  void post_force_einstein();
  void min_post_force(int) override;
  double compute_scalar() override;

  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int size_restart(int) override;
  int maxsize_restart() override;

 private:
  double k;          // Spring constant.
  double espring;    // Springs energies.
  double sigma;      // WCA parameter
  double epsilon;    // WCA parameter
  double rc;
  double **xoriginal;    // Original coords of atoms.
  double lambda;         // Coupling parameter.
  TI_ALGO algo;          // Thermal integration algorithm.
  int nlevels_respa;
};

}    // namespace LAMMPS_NS

#endif
#endif

    /* ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    E: Illegal fix ti/eq switching function

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    */
