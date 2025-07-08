#include <slave.h>

#include "common.h"

void descriptor(union athread_args *args)
{
  int interp = args[0].i;
  int size = args[1].i;
  int num_filters = args[2].i;
  double *H = args[3].p;
  double *dHdr = args[4].p;
  double *sij = args[5].p;
  double *dsij = args[6].p;

  int global_id_start, global_id_end;
  athread_get_task_range(size, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  int gid = global_id_start;
  for (; gid < global_id_end; gid++) {
    if (interp) {
      for (int k = 0; k < num_filters; k++) {
        dHdr[k + num_filters * gid] = dsij[gid] * H[k + num_filters * gid] +
            sij[gid] * dHdr[k + num_filters * gid];
        H[k + num_filters * gid] = sij[gid] * H[k + num_filters * gid];
      }
    } else {
      for (int k = 0; k < num_filters; k++) {
        dHdr[k + num_filters * gid] = dsij[gid] * H[k + num_filters * gid];
        H[k + num_filters * gid] = sij[gid] * H[k + num_filters * gid];
      }
    }
  }
}
