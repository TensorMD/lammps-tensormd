#include <slave.h>

#include "common.h"

void batch_dgemv_F2(union athread_args *args)
{
  int global_size = args[0].i;
  const int x = 3;
  int d = args[2].i;
  double *A = args[3].p;
  double *B = args[4].p;
  double *C = args[5].p;

  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);
  if (global_id_start >= global_id_end) return;

  int global_id = global_id_start;
  for (; global_id < global_id_end; global_id++) {
    double *a = &A[d * x * global_id];
    double *b = &B[d * global_id];
    double *c = &C[x * global_id];
    for (int j = 0; j < x; j++) c[j] = 0;
    for (int i = 0; i < x; i++)
      for (int j = 0; j < d; j++) c[i] += b[j] * a[i * d + j];
  }
}
