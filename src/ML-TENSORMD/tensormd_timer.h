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

#include <chrono>
#include <map>
#include <cstdio>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

namespace LAMMPS_NS {

namespace TIMER {
  enum Routine {
    ALLOC,           // allocate memory
    SETUP,           // setup_tensors()
    CUTOFF,          // sij and dsij
    FNN_FORWARD,     // fnn->forward()
    FNN_INTERP,      // fnn->interpolate()
    DESCRIPTOR,      // H_kcba
    P,               // P_dkba = H_kcba * M_dcba
    Q,               // Q = T_dm * (P_dkba)**2
    G,               // G.chip(0, 0) = G.chip(0, 0).sqrt()
    Kernel1,         // P + Q + G
    NN_COMPUTE,      // nnp->compute() or tdnp->compute()
    DEDP,            // dEdP = 2 * T_dm.contract(dEdG_mkba) * P_dkba
    U,               // U
    V,               // V    
    FNN_BACKWARD,    // fnn->backward()
    F1,              // F1
    F2,              // F2_xcba = dMdrx_xdcba * V_dcba
    Kernel4,         // U + F1
    FORCES,          // F = F1 + F2, scatter_sum(F)
    UNKNOWN
  };
}

class TensorMDTimer {

 public:
  TensorMDTimer(FILE *scr, FILE *log);
  ~TensorMDTimer();

  void setup_fnn(double forward_flops, double backward_flops)
  {
    this->fnn_forward_flops_per_one = forward_flops;
    this->fnn_backward_flops_per_one = backward_flops;
  }
  void setup_nn(double flops_per_atom)
  {
    this->nn_flops_per_atom = flops_per_atom;
  };
  void setup_tdnp(double flops_per_atom)
  {
    this->nn_flops_per_atom = flops_per_atom;
    this->is_tdnp = true;
  };
  void setup(int b_, int d_, int m_, int k_, bool cosine_cutoff);
  void update(int a_, int c_);

  // Start the timer
  void begin() { started = true; }

  // Tic
  void tic() { t_start = high_resolution_clock::now(); };

  // Record the time and flops of this routine
  void record(TIMER::Routine routine);

  // Print statistics
  void print();

 protected:
  FILE *screen;
  FILE *logfile;

  bool started;

  // the descriptor algorithm
  TIMER::Routine algo;

  // Basic parameters
  // a: the number of local atoms
  // b: the number of element types
  // c: the maximum number of neighbors
  // d: the dimension of unique angular moments
  // k: the number of descriptors
  // m: the maximum angular moment (m <= 3)
  // x: the number of cartesian directions, x is always 3 (X/Y/Z)
  double a, b, c, d, k, m, x;

  // the cutoff function
  bool use_cosine_cutoff;

  // NN & FNN flops for each atom
  double nn_flops_per_atom;
  double fnn_forward_flops_per_one;
  double fnn_backward_flops_per_one;
  bool is_tdnp;

  // the internal timer
  high_resolution_clock::time_point t_start;

  struct time_record_t {
    double flops;
    double secs;
  };

  void record_alloc(double secs);
  void record_setup(double secs);
  void record_cutoff(double secs);
  void record_fnn_forward(double secs);
  void record_fnn_interp(double secs);
  void record_descriptor(double secs);
  void record_P(double secs);
  void record_Q(double secs);
  void record_G(double secs);
  void record_Kernel1(double secs);
  void record_Kernel4(double secs);
  void record_nn_compute(double secs);
  void record_dEdP(double secs);
  void record_fnn_backward(double secs);
  void record_F1(double secs);
  void record_F2(double secs);
  void record_U(double secs);
  void record_V(double secs);
  void record_forces(double secs);

  std::map<TIMER::Routine, struct time_record_t> timer;
};

}    // namespace LAMMPS_NS