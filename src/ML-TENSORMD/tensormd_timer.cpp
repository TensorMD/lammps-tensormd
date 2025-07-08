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

#include "tensormd_timer.h"
#include "mpi.h"
#include <cstdio>

using namespace LAMMPS_NS;

inline const char *routine2string(TIMER::Routine routine)
{
  switch (routine) {
    case TIMER::ALLOC:
      return "alloc";
    case TIMER::SETUP:
      return "setup";
    case TIMER::CUTOFF:
      return "cutoff";
    case TIMER::FNN_FORWARD:
      return "fnn->forward";
    case TIMER::FNN_INTERP:
      return "fnn->interp";
    case TIMER::DESCRIPTOR:
      return "descriptor";
    case TIMER::P:
      return "P";
    case TIMER::Q:
      return "Q";
    case TIMER::G:
      return "G";
    case TIMER::Kernel1:
      return "Kernel1";
    case TIMER::NN_COMPUTE:
      return "nn->compute";
    case TIMER::DEDP:
      return "dEdP";
    case TIMER::FNN_BACKWARD:
      return "fnn->backward";
    case TIMER::F1:
      return "F1";
    case TIMER::F2:
      return "F2";
    case TIMER::U:
      return "U";
    case TIMER::V:
      return "V";
    case TIMER::Kernel3:
      return "Kernel3";
    case TIMER::Kernel4:
      return "Kernel4";
    case TIMER::FORCES:
      return "forces";
    default:
      return "Unknown";
  }
}

/* ---------------------------------------------------------------------- */

TensorMDTimer::TensorMDTimer(FILE *scr, FILE *log)
{
  this->b = 0;
  this->d = 0;
  this->m = 0;
  this->k = 0;
  this->a = 0;
  this->c = 0;
  this->x = 3;
  this->use_cosine_cutoff = true;

  for (int val = TIMER::ALLOC; val != TIMER::UNKNOWN; val++) {
    auto routine = static_cast<TIMER::Routine>(val);
    timer[routine] = time_record_t{0.0, 0.0};
  }

  this->screen = scr;
  this->logfile = log;
  this->started = false;

  this->nn_flops_per_atom = 0.0;
  this->fnn_forward_flops_per_one = 0.0;
  this->fnn_backward_flops_per_one = 0.0;
  this->is_tdnp = false;

  this->algo = TIMER::DESCRIPTOR;
}

TensorMDTimer::~TensorMDTimer() = default;

/* ---------------------------------------------------------------------- */

void TensorMDTimer::print()
{
  if (this->algo == TIMER::FNN_FORWARD) {
    timer.erase(TIMER::FNN_INTERP);
    timer.erase(TIMER::F1);
  } else if (this->algo == TIMER::FNN_INTERP) {
    timer.erase(TIMER::FNN_FORWARD);
    timer.erase(TIMER::FNN_BACKWARD);
    timer.erase(TIMER::CUTOFF);
    timer.erase(TIMER::DESCRIPTOR);
  } else {
    timer.erase(TIMER::FNN_FORWARD);
    timer.erase(TIMER::FNN_INTERP);
    timer.erase(TIMER::FNN_BACKWARD);
  }

  timer.erase(TIMER::P);
  timer.erase(TIMER::Q);
  timer.erase(TIMER::G);

#if defined(BUILD_BASELINE)
  timer.erase(TIMER::Kernel3);
  timer.erase(TIMER::Kernel4);
#else
  timer.erase(TIMER::V);
  timer.erase(TIMER::F2);
  timer.erase(TIMER::U);
  timer.erase(TIMER::F1);  
#endif

  double total_flops = 0.0;
  double total_secs = 0.0;
  double flops, secs;

  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  double factor = 1.0 / static_cast<double>(nprocs);

  for (auto &routine : timer) {
    MPI_Reduce(&routine.second.flops, &flops, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&routine.second.secs, &secs, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    routine.second.flops = flops * factor;
    routine.second.secs = secs * factor;
    total_flops += routine.second.flops;
    total_secs += routine.second.secs;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;

  char unit;
  double scale;
  if (total_flops >= 1e15) {
    unit = 'M';
    scale = 1e15;
  } else if (total_flops >= 1e12) {
    unit = 'T';
    scale = 1e12;
  } else if (total_flops >= 1e9) {
    unit = 'G';
    scale = 1e9;
  } else {
    unit = 'M';
    scale = 1e6;
  }
  total_flops /= scale;

  char buf[1024];
  if (logfile) {
    fprintf(logfile, "\n");
    fprintf(logfile,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(logfile, "\n");
    fprintf(logfile, "                               FLOPs Report\n");
    fprintf(logfile, "\n");
    fprintf(logfile,
            "------------------------------------------------------------------"
            "--------\n");
  }
  if (screen) {
    fprintf(screen, "\n");
    fprintf(screen,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(screen, "\n");
    fprintf(screen, "                               FLOPs Report\n");
    fprintf(screen, "\n");
    fprintf(screen,
            "------------------------------------------------------------------"
            "--------\n");
  }
  for (const auto &routine : timer) {
    const char *key = routine2string(routine.first);
    sprintf(buf, "%14s: %10.4f %cFlops, %12.4f secs, %12.4f %cFLOP/s", key,
            routine.second.flops / scale, unit, routine.second.secs,
            routine.second.flops / routine.second.secs / scale, unit);
    if (logfile) fprintf(logfile, "%s\n", buf);
    if (screen) fprintf(screen, "%s\n", buf);
  }
  sprintf(buf, "%14s: %10.4f %cFlops, %12.4f secs, %12.4f %cFLOP/s", "Overall",
          total_flops, unit, total_secs, total_flops / total_secs, unit);
  if (logfile) {
    fprintf(logfile,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(logfile, "%s\n", buf);
    fprintf(logfile,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(logfile, "\n");
  }
  if (screen) {
    fprintf(screen,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(screen, "%s\n", buf);
    fprintf(screen,
            "------------------------------------------------------------------"
            "--------\n");
    fprintf(screen, "\n");
  }
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::update(int a_, int c_)
{
  this->a = static_cast<double>(a_);
  this->c = static_cast<double>(c_);
}

void TensorMDTimer::setup(int b_, int d_, int m_, int k_, bool cosine_cutoff)
{
  this->b = static_cast<double>(b_);
  this->d = static_cast<double>(d_);
  this->m = static_cast<double>(m_);
  this->k = static_cast<double>(k_);
  this->use_cosine_cutoff = cosine_cutoff;
}

/* ---------------------------------------------------------------------- */

void TensorMDTimer::record(TIMER::Routine routine)
{
  if (!started) return;
  auto secs =
      duration_cast<duration<double>>(high_resolution_clock::now() - t_start)
          .count();

  switch (routine) {
    case TIMER::ALLOC:
      record_alloc(secs);
      break;
    case TIMER::SETUP:
      record_setup(secs);
      break;
    case TIMER::CUTOFF:
      record_cutoff(secs);
      break;
    case TIMER::FNN_FORWARD:
      record_fnn_forward(secs);
      break;
    case TIMER::FNN_INTERP:
      record_fnn_interp(secs);
      break;
    case TIMER::DESCRIPTOR:
      record_descriptor(secs);
      break;
    case TIMER::P:
      record_P(secs);
      break;
    case TIMER::Q:
      record_Q(secs);
      break;
    case TIMER::G:
      record_G(secs);
      break;
    case TIMER::Kernel1:
      record_Kernel1(secs);
      break;
    case TIMER::NN_COMPUTE:
      record_nn_compute(secs);
      break;
    case TIMER::DEDP:
      record_dEdP(secs);
      break;
    case TIMER::FNN_BACKWARD:
      record_fnn_backward(secs);
      break;
    case TIMER::F1:
      record_F1(secs);
      break;
    case TIMER::F2:
      record_F2(secs);
      break;
    case TIMER::U:
      record_U(secs);
      break;
    case TIMER::V:
      record_V(secs);
      break;
    case TIMER::Kernel3:
      record_Kernel3(secs);
      break;
    case TIMER::Kernel4:
      record_Kernel4(secs);
      break;
    case TIMER::FORCES:
      record_forces(secs);
      break;
    default:
      break;
  }
  t_start = {};
}

void TensorMDTimer::record_alloc(double secs)
{
  timer[TIMER::ALLOC].secs += secs;
}

void TensorMDTimer::record_setup(double secs)
{
  timer[TIMER::SETUP].secs += secs;
}

void TensorMDTimer::record_cutoff(double secs)
{
  if (use_cosine_cutoff)
    timer[TIMER::CUTOFF].flops += 10 * a * b * c;
  else
    timer[TIMER::CUTOFF].flops += 12 * a * b * c;
  timer[TIMER::CUTOFF].secs += secs;
}

void TensorMDTimer::record_fnn_forward(double secs)
{
  timer[TIMER::FNN_FORWARD].flops += fnn_forward_flops_per_one * a * b * c;
  timer[TIMER::FNN_FORWARD].secs += secs;
  algo = TIMER::FNN_FORWARD;
}

void TensorMDTimer::record_fnn_interp(double secs)
{
  // Approx 19 ops for each ration1d call
  timer[TIMER::FNN_INTERP].flops += a * b * c * k * 19;
  timer[TIMER::FNN_INTERP].secs += secs;
  algo = TIMER::FNN_INTERP;
}

void TensorMDTimer::record_descriptor(double secs)
{
  if (algo == TIMER::FNN_FORWARD)
    timer[TIMER::DESCRIPTOR].flops += a * b * c * k * 2;
  else if (algo == TIMER::FNN_INTERP)
    timer[TIMER::DESCRIPTOR].flops += a * b * c * k * 4;
  else
    timer[TIMER::DESCRIPTOR].flops += a * b * c * k * 10;
  timer[TIMER::DESCRIPTOR].secs += secs;
}

void TensorMDTimer::record_P(double secs)
{
  // P_dkba = M_dcba * H_kcba
  timer[TIMER::P].flops += a * b * c * d * k * 2;
  timer[TIMER::P].secs += secs;
}

void TensorMDTimer::record_Q(double secs)
{
  // Q_mkba = T_md * (P_dkba)**2
  timer[TIMER::Q].flops += (2 * (m + 1) + 1) * d * k * b * a;
  timer[TIMER::Q].secs += secs;
}

void TensorMDTimer::record_G(double secs)
{
  // G_mkba
  timer[TIMER::G].flops += a * b * k;
  timer[TIMER::G].secs += secs;
}

void TensorMDTimer::record_Kernel1(double secs)
{
  timer[TIMER::Kernel1].flops += a * b * k + a * b * c * d * k * 2 + (2 * (m + 1) + 1) * d * k * b * a;
  timer[TIMER::Kernel1].secs += secs;
}

void TensorMDTimer::record_nn_compute(double secs)
{
  timer[TIMER::NN_COMPUTE].flops += a * nn_flops_per_atom;
  timer[TIMER::NN_COMPUTE].secs += secs;
}

void TensorMDTimer::record_dEdP(double secs)
{
  // dEdG_mkba = dEdG_mkba * dGdQ_mkba
  // dGdQ.chip(0, 0) = 0.5 / G.chip(0, 0)
  // dEdP_dkba = 2 * T_md * dGdQ_mkba * P_dkba
  timer[TIMER::DEDP].flops += (2 * (m + 1) + 1) * d * k * b * a + k * b * a * 2;
  timer[TIMER::DEDP].secs += secs;
}

void TensorMDTimer::record_fnn_backward(double secs)
{
  timer[TIMER::FNN_BACKWARD].flops += fnn_backward_flops_per_one * a * b * c;
  timer[TIMER::FNN_BACKWARD].secs += secs;
}

void TensorMDTimer::record_F1(double secs)
{
  timer[TIMER::F1].flops += a * b * c * (k * 2 + 3);
  timer[TIMER::F1].secs += secs;
}

void TensorMDTimer::record_F2(double secs)
{
  timer[TIMER::F2].flops += a * b * c * (d * 2 + 3);
  timer[TIMER::F2].secs += secs;
}

void TensorMDTimer::record_U(double secs)
{
  timer[TIMER::U].flops += a * b * d * k * c * 2;
  timer[TIMER::U].secs += secs;
}

void TensorMDTimer::record_V(double secs)
{
  timer[TIMER::V].flops += a * b * d * k * c * 2;
  timer[TIMER::V].secs += secs;
}

void TensorMDTimer::record_Kernel3(double secs)
{
  timer[TIMER::Kernel3].flops += a * b * d * k * c * 2 + a * b * c * (d * 2 + 3);
  timer[TIMER::Kernel3].secs += secs;
}

void TensorMDTimer::record_Kernel4(double secs)
{
  timer[TIMER::Kernel4].flops += a * b * d * k * c * 2 + a * b * c * (k * 2 + 3);
  timer[TIMER::Kernel4].secs += secs;
}

void TensorMDTimer::record_forces(double secs)
{
  timer[TIMER::FORCES].flops += 4 * a * b * c * x;
  timer[TIMER::FORCES].secs += secs;
}
