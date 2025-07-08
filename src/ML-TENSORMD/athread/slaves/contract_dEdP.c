#include <slave.h>

#include "common.h"

void contract_dEdP(union athread_args *args)
{
  int size = args[0].i;
  int m = args[1].i;
  double *dEdS = args[2].p;
  double *dEdG = args[3].p;
  double *P = args[4].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  int gid = global_id_start;
  if (m == 1) {
    for (; gid < global_id_end; gid++)
      // dEdS[gid] = 2 * dEdG[gid] * P[gid];
      dEdS[gid] = dEdG[gid];
  } else if (m == 2) {
    const int d = 4;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] = dEdG[0 + m * gid];
      dEdS[1 + d * gid] = 2 * dEdG[1 + m * gid] * P[1 + d * gid];
      dEdS[2 + d * gid] = 2 * dEdG[1 + m * gid] * P[2 + d * gid];
      dEdS[3 + d * gid] = 2 * dEdG[1 + m * gid] * P[3 + d * gid];
    }
  } else if (m == 3) {
    const int d = 10;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * 1 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] = dEdG[0 + m * gid];
      dEdS[1 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[1 + d * gid];
      dEdS[2 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[2 + d * gid];
      dEdS[3 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[3 + d * gid];
      dEdS[4 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[4 + d * gid];
      dEdS[5 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[5 + d * gid];
      dEdS[6 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[6 + d * gid];
      dEdS[7 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[7 + d * gid];
      dEdS[8 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[8 + d * gid];
      dEdS[9 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[9 + d * gid];
    }
  } else if (m == 4) {
    const int d = 20;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * 1 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] = 2 * 1 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[1 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[1 + d * gid];
      dEdS[2 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[2 + d * gid];
      dEdS[3 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[3 + d * gid];
      dEdS[4 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[4 + d * gid];
      dEdS[5 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[5 + d * gid];
      dEdS[6 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[6 + d * gid];
      dEdS[7 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[7 + d * gid];
      dEdS[8 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[8 + d * gid];
      dEdS[9 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[9 + d * gid];
      dEdS[10 + d * gid] = 2 * 1 * dEdG[3 + m * gid] * P[10 + d * gid];
      dEdS[11 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[11 + d * gid];
      dEdS[12 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[12 + d * gid];
      dEdS[13 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[13 + d * gid];
      dEdS[14 + d * gid] = 2 * 6 * dEdG[3 + m * gid] * P[14 + d * gid];
      dEdS[15 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[15 + d * gid];
      dEdS[16 + d * gid] = 2 * 1 * dEdG[3 + m * gid] * P[16 + d * gid];
      dEdS[17 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[17 + d * gid];
      dEdS[18 + d * gid] = 2 * 3 * dEdG[3 + m * gid] * P[18 + d * gid];
      dEdS[19 + d * gid] = 2 * 1 * dEdG[3 + m * gid] * P[19 + d * gid];
    }
  }
}

void dEdS_P(union athread_args *args)
{
  int size = args[0].i;
  int m = args[1].i;
  double *dEdS = args[2].p;
  double *dEdG = args[3].p;
  double *P = args[4].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  int gid = global_id_start;
  if (m == 1) {
    for (; gid < global_id_end; gid++)
      // dEdS[gid] = 2 * dEdG[gid] * P[gid];
      dEdS[gid] = dEdG[gid];
  } else if (m == 2) {
    const int d = 4;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] = dEdG[0 + m * gid];
      dEdS[1 + d * gid] = 2 * dEdG[1 + m * gid] * P[1 + d * gid];
      dEdS[2 + d * gid] = 2 * dEdG[1 + m * gid] * P[2 + d * gid];
      dEdS[3 + d * gid] = 2 * dEdG[1 + m * gid] * P[3 + d * gid];
    }
  } else if (m == 3) {
    const int d = 10;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * 1 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] = dEdG[0 + m * gid];
      dEdS[1 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[1 + d * gid];
      dEdS[2 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[2 + d * gid];
      dEdS[3 + d * gid] = 2 * 1 * dEdG[1 + m * gid] * P[3 + d * gid];
      dEdS[4 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[4 + d * gid];
      dEdS[5 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[5 + d * gid];
      dEdS[6 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[6 + d * gid];
      dEdS[7 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[7 + d * gid];
      dEdS[8 + d * gid] = 2 * 2 * dEdG[2 + m * gid] * P[8 + d * gid];
      dEdS[9 + d * gid] = 2 * 1 * dEdG[2 + m * gid] * P[9 + d * gid];
    }
  } else if (m == 4) {
    const int d = 20;
    for (; gid < global_id_end; gid++) {
      // dEdS[0 + d * gid] = 2 * 1 * dEdG[0 + m * gid] * P[0 + d * gid];
      dEdS[0 + d * gid] *= P[0 + d * gid];
      dEdS[1 + d * gid] *= P[1 + d * gid];
      dEdS[2 + d * gid] *= P[2 + d * gid];
      dEdS[3 + d * gid] *= P[3 + d * gid];
      dEdS[4 + d * gid] *= P[4 + d * gid];
      dEdS[5 + d * gid] *= P[5 + d * gid];
      dEdS[6 + d * gid] *= P[6 + d * gid];
      dEdS[7 + d * gid] *= P[7 + d * gid];
      dEdS[8 + d * gid] *= P[8 + d * gid];
      dEdS[9 + d * gid] *= P[9 + d * gid];
      dEdS[10 + d * gid] *= P[10 + d * gid];
      dEdS[11 + d * gid] *= P[11 + d * gid];
      dEdS[12 + d * gid] *= P[12 + d * gid];
      dEdS[13 + d * gid] *= P[13 + d * gid];
      dEdS[14 + d * gid] *= P[14 + d * gid];
      dEdS[15 + d * gid] *= P[15 + d * gid];
      dEdS[16 + d * gid] *= P[16 + d * gid];
      dEdS[17 + d * gid] *= P[17 + d * gid];
      dEdS[18 + d * gid] *= P[18 + d * gid];
      dEdS[19 + d * gid] *= P[19 + d * gid];
    }
  }
}
