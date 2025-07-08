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

#include "tensor_debug.h"
#include <cmath>
#include <cstdio>

template <typename Scalar>
bool has_inf_or_nan(TensorMap<Tensor<Scalar, 3>> &in, const char *name,
                    int mpiid, bool write_to_file)
{
  FILE *fp = nullptr;
  char filename[256];
  if (write_to_file) {
    snprintf(filename, 256, "mpi%d_%s.txt", mpiid, name);
    fp = fopen(filename, "w");
  }

  bool has_nan = false;
  bool has_inf = false;
  for (int i = 0; i < in.dimension(2); i++) {
    for (int j = 0; j < in.dimension(1); j++) {
      for (int k = 0; k < in.dimension(0); k++) {
        if (std::isnan(in(k, j, i))) has_nan = true;
        if (std::isinf(in(k, j, i))) has_inf = true;
        if (has_nan || has_inf) {
          if (write_to_file) {
            fprintf(fp, "(%d, %d, %d)\n", k, j, i);
          } else {
            printf("mpi%d, %s, (%d, %d, %d)\n", mpiid, name, k, j, i);
          }
        };
      }
    }
  }

  if (write_to_file) { 
    fclose(fp); 
    if (!(has_nan || has_inf)) {
      remove(filename);
    }
  }

  return (has_nan || has_inf);
}

template <typename Scalar>
bool has_inf_or_nan(Tensor<Scalar, 3> &in, const char *name, int mpiid,
                    bool write_to_file)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  TensorMap<Tensor<Scalar, 3>> in_map{in.data(), {d0, d1, d2}};
  return has_inf_or_nan(in_map, name, mpiid, write_to_file);
}

template <typename Scalar>
bool has_inf_or_nan(TensorMap<Tensor<Scalar, 4>> &in, const char *name,
                    int mpiid, bool write_to_file)
{
  FILE *fp = nullptr;
  char filename[256];
  if (write_to_file) {
    snprintf(filename, 256, "mpi%d_%s.txt", mpiid, name);
    fp = fopen(filename, "w");
  }

  bool has_nan = false;
  bool has_inf = false;
  for (int i = 0; i < in.dimension(3); i++) {
    for (int j = 0; j < in.dimension(2); j++) {
      for (int k = 0; k < in.dimension(1); k++) {
        for (int m = 0; m < in.dimension(0); m++) {
          if (std::isnan(in(m, k, j, i))) has_nan = true;
          if (std::isinf(in(m, k, j, i))) has_inf = true;
          if (has_nan || has_inf) {
            if (write_to_file) {
              fprintf(fp, "(%d, %d, %d, %d)\n", m, k, j, i);
            } else {
              printf("mpi%d, %s, (%d, %d, %d, %d)\n", mpiid, name, m, k, j, i);
            }
          };
        }
      }
    }
  }

  if (write_to_file) { 
    fclose(fp); 
    if (!(has_inf || has_nan)) {
      remove(filename);
    }
  }

  return (has_nan || has_inf);
}

template <typename Scalar>
bool has_inf_or_nan(Tensor<Scalar, 4> &in, const char *name, int mpiid,
                    bool write_to_file)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  long d3 = static_cast<long>(in.dimension(3));
  TensorMap<Tensor<Scalar, 4>> in_map{in.data(), {d0, d1, d2, d3}};
  return has_inf_or_nan(in_map, name, mpiid, write_to_file);
}

template <typename Scalar>
void dump_tensor(TensorMap<Tensor<Scalar, 3>> &in, const char *name, int mpiid)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  FILE *fp = nullptr;
  char filename[256];
  snprintf(filename, 256, "dump_mpi%d_%s.txt", mpiid, name);
  fp = fopen(filename, "w");
  fprintf(fp, "%ld, %ld, %ld\n", d0, d1, d2);
  for (int i = 0; i < in.dimension(2); i++) {
    for (int j = 0; j < in.dimension(1); j++) {
      for (int k = 0; k < in.dimension(0); k++) {
        fprintf(fp, "%d %d %d %lf\n", k, j, i, in(k, j, i));
      }
    }
  }
  fclose(fp);
}

template <typename Scalar>
void dump_tensor(Tensor<Scalar, 3> &in, const char *name, int mpiid)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  TensorMap<Tensor<Scalar, 3>> in_map{in.data(), {d0, d1, d2}};
  dump_tensor(in_map, name, mpiid);
}

template <typename Scalar>
void dump_tensor(TensorMap<Tensor<Scalar, 4>> &in, const char *name, int mpiid)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  long d3 = static_cast<long>(in.dimension(3));
  FILE *fp = nullptr;
  char filename[256];
  snprintf(filename, 256, "dump_mpi%d_%s.txt", mpiid, name);
  fp = fopen(filename, "w");
  fprintf(fp, "%ld, %ld, %ld, %ld\n", d0, d1, d2, d3);
  for (int i = 0; i < in.dimension(3); i++) {
    for (int j = 0; j < in.dimension(2); j++) {
      for (int k = 0; k < in.dimension(1); k++) {
        for (int m = 0; m < in.dimension(0); m++) {
          fprintf(fp, "%d %d %d %d %lf\n", m, k, j, i, in(m, k, j, i));
        }
      }
    }
  }
  fclose(fp);
}

template <typename Scalar>
void dump_tensor(Tensor<Scalar, 4> &in, const char *name, int mpiid)
{
  long d0 = static_cast<long>(in.dimension(0));
  long d1 = static_cast<long>(in.dimension(1));
  long d2 = static_cast<long>(in.dimension(2));
  long d3 = static_cast<long>(in.dimension(3));
  TensorMap<Tensor<Scalar, 4>> in_map{in.data(), {d0, d1, d2, d3}};
  dump_tensor(in_map, name, mpiid);
}

template <typename Scalar>
void dump_sum_forces_data(int amax, int bmax, int cmax, int imax,
                          Tensor<Scalar, 4> &F1, Tensor<Scalar, 4> &F2,
                          Tensor<int, 4> &ijlist, double **f)
{
  int a, b, c, i, j, x;

  FILE *fp = fopen("sum_forces.dat", "w");
  fprintf(fp, "%d\n", amax);
  fprintf(fp, "%d\n", bmax);
  fprintf(fp, "%d\n", cmax);

  for (a = 0; a < amax; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        i = ijlist(0, c, b, a);
        j = ijlist(1, c, b, a);
        fprintf(fp, "%5d %1d %3d %5d %5d ", a, b, c, i, j);
        for (x = 0; x < 3; x++) {
          fprintf(fp, "% 16.10lf ", F1(x, c, b, a));
        }
        for (x = 0; x < 3; x++) {
          fprintf(fp, "% 16.10lf ", F2(x, c, b, a));
        }
        fprintf(fp, "\n");
      }
    }
  }

  fprintf(fp, "%d\n", imax);
  for (i = 0; i < imax; i++) {
    fprintf(fp, "% 16.10lf % 16.10lf % 16.10lf\n", f[i][0], f[i][1], f[i][2]);
  }

  fclose(fp);
}

template bool has_inf_or_nan(TensorMap<Tensor<double, 3>> &in, const char *name,
                             int mpiid, bool write_to_file);
template bool has_inf_or_nan(TensorMap<Tensor<double, 4>> &in, const char *name,
                             int mpiid, bool write_to_file);
template bool has_inf_or_nan(Tensor<double, 3> &in, const char *name, int mpiid,
                             bool write_to_file);
template bool has_inf_or_nan(Tensor<double, 4> &in, const char *name, int mpiid,
                             bool write_to_file);

template bool has_inf_or_nan(TensorMap<Tensor<float, 3>> &in, const char *name,
                             int mpiid, bool write_to_file);
template bool has_inf_or_nan(TensorMap<Tensor<float, 4>> &in, const char *name,
                             int mpiid, bool write_to_file);
template bool has_inf_or_nan(Tensor<float, 3> &in, const char *name, int mpiid,
                             bool write_to_file);
template bool has_inf_or_nan(Tensor<float, 4> &in, const char *name, int mpiid,
                             bool write_to_file);

template void dump_tensor(TensorMap<Tensor<double, 3>> &in, const char *name,
                          int mpiid);
template void dump_tensor(TensorMap<Tensor<double, 4>> &in, const char *name, int mpiid);
template void dump_tensor(Tensor<double, 3> &in, const char *name, int mpiid);
template void dump_tensor(Tensor<double, 4> &in, const char *name, int mpiid);

template void dump_tensor(TensorMap<Tensor<float, 3>> &in, const char *name,
                          int mpiid);
template void dump_tensor(TensorMap<Tensor<float, 4>> &in, const char *name, int mpiid);
template void dump_tensor(Tensor<float, 3> &in, const char *name, int mpiid);
template void dump_tensor(Tensor<float, 4> &in, const char *name, int mpiid);

template void dump_sum_forces_data(int amax, int bmax, int cmax, int imax,
                                   Tensor<double, 4> &F1, Tensor<double, 4> &F2,
                                   Tensor<int, 4> &ijlist, double **f);
template void dump_sum_forces_data(int amax, int bmax, int cmax, int imax,
                                   Tensor<float, 4> &F1, Tensor<float, 4> &F2,
                                   Tensor<int, 4> &ijlist, double **f);
