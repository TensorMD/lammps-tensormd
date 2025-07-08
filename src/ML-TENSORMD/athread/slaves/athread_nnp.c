#include <slave.h>
#include <simd.h>
#include "common.h"

void axpy(void **args)
{
  int global_size = *(int *)args[0];
  int n = *(int *)args[1];
  double *x = (double *)args[2];
  double *y = (double *)args[3];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    for (int ii = 0; ii < n; ii++) {
      y[gid * n + ii] = y[gid * n + ii] + x[ii];
    }
  }
}

void vmul(void **args)
{
  int global_size = *(int *)args[0];
  double *x = (double *)args[1];
  double *y = (double *)args[2];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    y[gid] = y[gid] * x[gid];
  }
}

void softplus(void **args)
{
  int global_size = *(int *)args[0];
  double *x = (double *)args[1];
  double *y = (double *)args[2];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    double xi = x[gid];
    double exp_ = exp(xi);
    double threshold = log(2.22045e-16) + 2;
    if (xi < -threshold) { xi = log1p(exp_); }
    y[gid] = 1.0 / (1.0 + 1.0 / exp_);
    x[gid] = xi;    
  }
}

void squareplus(void **args)
{
  int global_size = *(int *)args[0];
  double *x = (double *)args[1];
  double *y = (double *)args[2];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    double xi = x[gid];
    double z = sqrt(xi * xi + 4.0);
    xi = 0.5 * (xi + z);
    y[gid] = xi / z;
    x[gid] = xi;    
  }
}

void axpy_softplus(void **args)
{
  int res = *(int *)args[0];
  int global_size = *(int *)args[1];
  int n = *(int *)args[2];
  double *old_x = (double *)args[3];
  double *x = (double *)args[4];
  double *y = (double *)args[5];
  double *bias = (double *)args[6];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    for (int ii = 0; ii < n; ii++) {
      double xi = x[gid * n + ii];
      xi += bias[ii];
      double exp_ = exp(xi);
      double threshold = log(2.22045e-16) + 2;
      if (xi < -threshold) { xi = log1p(exp_); }
      y[gid * n + ii] = 1.0 / (1.0 + 1.0 / exp_);
      if (res) { xi += old_x[gid * n + ii]; }
      x[gid * n + ii] = xi;
    } 
  }
}

void axpy_squareplus(void **args)
{
  int res = *(int *)args[0];
  int global_size = *(int *)args[1];
  int n = *(int *)args[2];
  double *old_x = (double *)args[3];
  double *x = (double *)args[4];
  double *y = (double *)args[5];
  double *bias = (double *)args[6];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    for (int ii = 0; ii < n; ii++) {
      double xi = x[gid * n + ii];
      xi += bias[ii];
      double z = sqrt(xi * xi + 4.0);
      xi = 0.5 * (xi + z);
      y[gid * n + ii] = xi / z;
      if (res) { xi += old_x[gid * n + ii]; }
      x[gid * n + ii] = xi;
    } 
  }
}

void axpy_squareplus_simd(void **args)
{
  int res = *(int *)args[0];
  int global_size = *(int *)args[1];
  int n = *(int *)args[2];
  double *old_x = (double *)args[3];
  double *x = (double *)args[4];
  double *y = (double *)args[5];
  double *bias = (double *)args[6];
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);

  doublev8 vec8d_xi, vec8d_bias, vec8d_z, vec8d_4, vec8d_05;
  vec8d_4 = simd_vcpyfd(4.0);
  vec8d_05 = simd_vcpyfd(0.5);
  for (int gid = global_id_start; gid < global_id_end; gid++) {
    for (int ii = 0; ii < n; ii += 8) {
      simd_load(vec8d_xi, &x[gid * n + ii]);
      simd_load(vec8d_bias, &bias[ii]);
      vec8d_xi = simd_vaddd(vec8d_xi, vec8d_bias);
      vec8d_z = simd_vsqrtd(simd_vmad(vec8d_xi, vec8d_xi, vec8d_4));
      vec8d_xi = simd_vmuld(vec8d_05, simd_vaddd(vec8d_xi, vec8d_z));
      simd_store(simd_vdivd(vec8d_xi, vec8d_z), &y[gid * n + ii]);
      if (res) { 
        doublev8 vec8d_old_x;
        simd_load(vec8d_old_x, &old_x[gid * n + ii]);
        vec8d_xi = simd_vaddd(vec8d_xi, vec8d_old_x);
      }
      simd_store(vec8d_xi, &x[gid * n + ii]);
    } 
  }
}