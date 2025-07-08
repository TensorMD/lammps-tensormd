#include <slave.h>

#include "common.h"

// F1[1:3] = ddot(U[1:k], dHdr[1:k]) * drdrx[3]
void batch_ddot_F1(union athread_args *args)
{
  int global_size = args[0].i;
  int k = args[1].i;
  const int x = 3;
  double *A = args[3].p;    // F1[x]
  double *B = args[4].p;    // U[k]
  double *C = args[5].p;    // dHdr[k]
  double *D = args[6].p;    // drdrx[x]

  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);
  if (global_id_start >= global_id_end) return;

  int gid = global_id_start;
  for (; gid < global_id_end; gid++) {
    double *a = &A[x * gid];
    double *b = &B[k * gid];
    double *c = &C[k * gid];
    double *d = &D[x * gid];
    double ddot = 0;
    for (int i = 0; i < k; i++) ddot += b[i] * c[i];
    for (int i = 0; i < x; i++) a[i] = ddot * d[i];
  }
}
