#include <math.h>
#include <slave.h>
#include "common.h"

#define NEIGHMASK 0x1FFFFFFF
#define min(x, y)        \
  ({                     \
    typeof(x) _x = (x);  \
    typeof(y) _y = (y);  \
    (void) (&_x == &_y); \
    _x < _y ? _x : _y;   \
  })

void get_cmax(void **args)
{
  int inum = *(int *) args[0];
  const int *type = (int *) args[1];
  const int *fmap = (int *) args[2];
  double **pos = (double **) args[3];
  const int *numneigh = (int *) args[4];
  int **firstneigh = (int **) args[5];
  int neltypes = *(int *) args[6];
  double cutforcesq = *(double *) args[7];
  int **eltij_pos = (int **) args[8];
	int *ilist = (int *) args[9];
	int *cmax_array = (int *) args[10];

	int i_start, i_end;
  athread_get_task_range(inum, &i_start, &i_end, 64);
  int cmax_ = 0;
  int ii, elti, eltj, i, j, b, jn, neigh[neltypes];
  double xtmp, ytmp, ztmp, dij[3], rij2;
  for (ii = i_start; ii < i_end; ii++) {
    for (elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    i = ilist[ii];
    elti = fmap[type[i]];
    xtmp = pos[i][0];
    ytmp = pos[i][1];
    ztmp = pos[i][2];
    for (jn = 0; jn < numneigh[i]; jn++) {
      j = firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = pos[j][0] - xtmp;
      dij[1] = pos[j][1] - ytmp;
      dij[2] = pos[j][2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < cutforcesq && rij2 > 1e-10) {
        eltj = fmap[type[j]];
        b = eltij_pos[elti][eltj];
        neigh[b]++;
      }
    }
    for (b = 0; b < neltypes; b++) {
      if (neigh[b] > cmax_) cmax_ = neigh[b];
    }
  }
	cmax_array[_MYID] = cmax_;
}
