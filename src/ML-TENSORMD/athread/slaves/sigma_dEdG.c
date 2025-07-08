#include <slave.h>
#include <crts.h>
#include <math.h>
#include "common.h"

void sigma_dEdG(void **args) {
    int size = *(int *)args[0];
    int m = *(int *)args[1];
    double *G = (double *)args[2];
    double *dEdG = (double *)args[3];

    int global_id_start, global_id_end;
    athread_get_task_range(size, &global_id_start, &global_id_end, 64);
    if (global_id_start >= global_id_end) return;

    for (int gid = global_id_start; gid < global_id_end; gid++) {
        if (fabs(G[gid * m]) < 1e-8) dEdG[gid * m] = 0.0;
        else dEdG[gid * m] = dEdG[gid * m] / G[gid * m] * 0.5;
    }
}