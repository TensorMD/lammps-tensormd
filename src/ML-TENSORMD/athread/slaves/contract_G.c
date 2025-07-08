#include <slave.h>

#include "common.h"

void contract_G(union athread_args *args)
{
  int size = args[0].i;
  int m = args[1].i;
  double *G = args[2].p;
  double *P = args[3].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  // Q_mkba = T_md * (P_dkba)**2
  int gid = global_id_start;
  if (m == 1) {
    for (; gid < global_id_end; gid++) G[gid] = P[gid];
  } else if (m == 2) {
    const int d = 4;
    for (; gid < global_id_end; gid++) {
      double P0 = P[0 + d * gid];
      double P1 = P[1 + d * gid] * P[1 + d * gid];
      double P2 = P[2 + d * gid] * P[2 + d * gid];
      double P3 = P[3 + d * gid] * P[3 + d * gid];
      G[0 + m * gid] = P0;
      G[1 + m * gid] = P1 + P2 + P3;
    }
  } else if (m == 3) {
    const int d = 10;
    for (; gid < global_id_end; gid++) {
      double P0 = 1 * P[0 + d * gid];
      double P1 = 1 * P[1 + d * gid] * P[1 + d * gid];
      double P2 = 1 * P[2 + d * gid] * P[2 + d * gid];
      double P3 = 1 * P[3 + d * gid] * P[3 + d * gid];
      double P4 = 1 * P[4 + d * gid] * P[4 + d * gid];
      double P5 = 2 * P[5 + d * gid] * P[5 + d * gid];
      double P6 = 2 * P[6 + d * gid] * P[6 + d * gid];
      double P7 = 1 * P[7 + d * gid] * P[7 + d * gid];
      double P8 = 2 * P[8 + d * gid] * P[8 + d * gid];
      double P9 = 1 * P[9 + d * gid] * P[9 + d * gid];
      G[0 + m * gid] = P0;
      G[1 + m * gid] = P1 + P2 + P3;
      G[2 + m * gid] = P4 + P5 + P6 + P7 + P8 + P9;
    }
  } else if (m == 4) {
    const int d = 20;
    for (; gid < global_id_end; gid++) {
      double P0 = 1 * P[0 + d * gid];
      double P1 = 1 * P[1 + d * gid] * P[1 + d * gid];
      double P2 = 1 * P[2 + d * gid] * P[2 + d * gid];
      double P3 = 1 * P[3 + d * gid] * P[3 + d * gid];
      double P4 = 1 * P[4 + d * gid] * P[4 + d * gid];
      double P5 = 2 * P[5 + d * gid] * P[5 + d * gid];
      double P6 = 2 * P[6 + d * gid] * P[6 + d * gid];
      double P7 = 1 * P[7 + d * gid] * P[7 + d * gid];
      double P8 = 2 * P[8 + d * gid] * P[8 + d * gid];
      double P9 = 1 * P[9 + d * gid] * P[9 + d * gid];
      double P10 = 1 * P[10 + d * gid] * P[10 + d * gid];
      double P11 = 3 * P[11 + d * gid] * P[11 + d * gid];
      double P12 = 3 * P[12 + d * gid] * P[12 + d * gid];
      double P13 = 3 * P[13 + d * gid] * P[13 + d * gid];
      double P14 = 6 * P[14 + d * gid] * P[14 + d * gid];
      double P15 = 3 * P[15 + d * gid] * P[15 + d * gid];
      double P16 = 1 * P[16 + d * gid] * P[16 + d * gid];
      double P17 = 3 * P[17 + d * gid] * P[17 + d * gid];
      double P18 = 3 * P[18 + d * gid] * P[18 + d * gid];
      double P19 = 1 * P[19 + d * gid] * P[19 + d * gid];
      G[0 + m * gid] = P0;
      G[1 + m * gid] = P1 + P2 + P3;
      G[2 + m * gid] = P4 + P5 + P6 + P7 + P8 + P9;
      G[3 + m * gid] = P10 + P11 + P12 + P13 + P14 + P15 + P16 + P17 + P18 + P19;
    }
  }
}

void P_2(union athread_args *args)
{
  int size = args[0].i;
  int m = args[1].i;
  double *S = args[2].p;
  double *P = args[3].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  // Q_mkba = T_md * (P_dkba)**2
  for(int gid = global_id_start; gid < global_id_end; gid++) {
    S[gid * 20 + 0] = P[gid * 20 + 0];
    for (int d = 1; d < 20; d++) {
      S[gid * 20 + d] = P[gid * 20 + d] * P[gid * 20 + d];
    }
  }
}
