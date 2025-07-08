#include <math.h>
#include <slave.h>

#include "common.h"

void r_cutoff(union athread_args *args)
{
  int fctype = args[0].i;
  int size = args[1].i;
  double cutforce = args[2].d;
  double *mask = args[3].p;
  double *R = args[4].p;
  double *sij = args[5].p;
  double *dsij = args[6].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  const double MY_PI = 3.14159265358979323846;
  const double gamma = 5.0;

  int gid = global_id_start;
  if (fctype) {
    for (; gid < global_id_end; gid++) {
      if (mask[gid] == 0) {
        sij[gid] = 0;
        dsij[gid] = 0;
      } else {
        double z = fmin(R[gid] / cutforce, 1.0) * MY_PI;
        sij[gid] = 0.5 + 0.5 * cos(z);
        dsij[gid] = -0.5 * MY_PI * sin(z) / cutforce;
      }
    }
  } else {
    for (; gid < global_id_end; gid++) {
      if (mask[gid] == 0) {
        sij[gid] = 0;
        dsij[gid] = 0;
      } else {
        double z = fmin(R[gid] / cutforce, MY_PI);
        double z4 = (z * z) * (z * z);
        double z5 = z4 * z;
        sij[gid] = 1 + gamma * z5 * z - (gamma + 1) * z5;
        dsij[gid] = (gamma + 1) * gamma / cutforce * (z5 - z4);
      }
    }
  }
}
