#include <math.h>
#include <slave.h>
#include "common.h"


void athread_get_task_range_special(int global_size, int *start, int *end, int num_threads)
{
  int athread_rows, athread_start, athread_end, group;
  athread_rows = global_size / num_threads + 1;

  if (_MYID % num_threads == 0) {
    group = _MYID / num_threads;
    athread_start = athread_rows * group;
    athread_end = athread_rows * (group + 1);
    if (athread_end > global_size) {
      athread_end = global_size;
    }
    *start = athread_start;
    *end = athread_end;
  } else {
    *start = 0;
    *end = 0;
  }
}

void sum_forces_virial(void **args)
{
  int amax = *(int *) args[0];
  int bmax = *(int *) args[1];
  int cmax = *(int *) args[2];
  int *ijlist = (int *) args[3];
  double *F1 = (double *) args[4];
  double *F2 = (double *) args[5];
  double *drdrx = (double *) args[6];
  double *R = (double *) args[7];
  int natoms = *(int *) args[8];
  int vatom_flag = *(int *) args[9];
  double *farray = (double *) args[10];
  double *varray = (double *) args[11];
  int num_threads = *(int *) args[12];

  int a, b, c, x, i, j, loc;
  double df[3], virial[6];

#define ijlist(d, c, b, a) ijlist[d + 2 * (c + cmax * (b + bmax * a))]
#define F1(x, c, b, a) F1[x + 3 * (c + cmax * (b + bmax * a))]
#define F2(x, c, b, a) F2[x + 3 * (c + cmax * (b + bmax * a))]
#define drdrx(x, c, b, a) drdrx[x + 3 * (c + cmax * (b + bmax * a))]
#define R(c, b, a) R[c + cmax * (b + bmax * a)]

  int start = 0, end = amax;
  athread_get_task_range_special(amax, &start, &end, num_threads);

  double *f_cid = NULL;
  double *v_cid = NULL;
  if (end > 0) {
    f_cid = farray + (_MYID / num_threads) * natoms * 3;
    if (vatom_flag) {
      v_cid = varray + (_MYID / num_threads) * natoms * 6;
    }
  }

  for (a = start; a < end; a++) {
    for (b = 0; b < bmax; b++) {
      for (c = 0; c < cmax; c++) {
        j = ijlist(1, c, b, a);
        if (j >= 0) {
          i = ijlist(0, c, b, a);
          for (x = 0; x < 3; x++) {
            df[x] = F1(x, c, b, a) + F2(x, c, b, a);
            f_cid[i * 3 + x] += df[x];
            f_cid[j * 3 + x] -= df[x];
          }
          if (vatom_flag) {
            virial[0] = -drdrx(0, c, b, a) * df[0] * R(c, b, a);
            virial[1] = -drdrx(1, c, b, a) * df[1] * R(c, b, a);
            virial[2] = -drdrx(2, c, b, a) * df[2] * R(c, b, a);
            virial[3] = -drdrx(0, c, b, a) * df[1] * R(c, b, a);
            virial[4] = -drdrx(0, c, b, a) * df[2] * R(c, b, a);
            virial[5] = -drdrx(1, c, b, a) * df[2] * R(c, b, a);
            v_cid[i * 6 + 0] += 0.5 * virial[0];
            v_cid[i * 6 + 1] += 0.5 * virial[1];
            v_cid[i * 6 + 2] += 0.5 * virial[2];
            v_cid[i * 6 + 3] += 0.5 * virial[3];
            v_cid[i * 6 + 4] += 0.5 * virial[4];
            v_cid[i * 6 + 5] += 0.5 * virial[5];
            v_cid[j * 6 + 0] += 0.5 * virial[0];
            v_cid[j * 6 + 1] += 0.5 * virial[1];
            v_cid[j * 6 + 2] += 0.5 * virial[2];
            v_cid[j * 6 + 3] += 0.5 * virial[3];
            v_cid[j * 6 + 4] += 0.5 * virial[4];
            v_cid[j * 6 + 5] += 0.5 * virial[5];
          }
        }
        else {
          break;
        }
      }
    }
  }
#undef ijlist
#undef F1
#undef F2
#undef drdrx
#undef R
}

void reduce_forces_virial(void **args)
{
  int natoms = *(int *) args[0];
  double *farray = (double *) args[1];
  double *varray = (double *) args[2];
  double **f = (double **) args[3];
  double **vatom = (double **) args[4];
  double *ptr;
  double value;
  int vatom_flag = *(int *) args[5];
  int num_duplicates = *(int *) args[6];
  int num_threads = *(int *) args[7];

  int start = 0, end = natoms;
  athread_get_task_range(natoms, &start, &end, num_threads);

  for (int it = 0; it < num_duplicates; it++) {
    for (int i = start; i < end; i++) {
      ptr = farray + (it * natoms + i) * 3;
      for (int x = 0; x < 3; x++) {
        f[i][x] += ptr[x];
      }
      if (vatom_flag) {
        ptr = varray + (it * natoms + i) * 6;
        for (int x = 0; x < 6; x++) {
          vatom[i][x] += ptr[x];
        }
      }
    }
  }
}

void initialize_array_1d(void **args)
{
  int size = *(int *) args[0];
  double *array = (double *) args[1];
  double value = *(double *) args[2];
  int num_threads = *(int *) args[3];

  int start, end;
  athread_get_task_range(size, &start, &end, num_threads);

  for (int i = start; i < end; i++) {
    array[i] = value;
  }
}
