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
void setup_tensors(void **args)
{
  // global size
  int global_size = *(int *) args[0];
  // row to index
  int *i2row = (int *) args[1];
  // atom
  const int *type = (int *) args[2];
  const int *fmap = (int *) args[3];
  double **pos = (double **) args[4];
  // neighbor list
  const int *numneigh = (int *) args[5];
  int **firstneigh = (int **) args[6];
  // this
  int neltypes = *(int *) args[7];
  double cutforcesq = *(double *) args[8];
  int *ilist = (int *) args[9];
  int **eltij_pos = (int **) args[10];
  // size
  int size_c = *(int *) args[11];    // numneigh_max
  int size_d = *(int *) args[12];
  // outputs
  int *ijlist = (int *) args[13];
  double *R = (double *) args[14];
  int *mask = (int *) args[15];
  double *M = (double *) args[16];
  double *drdrx = (double *) args[17];
  double *dMdrx = (double *) args[18];

#define ijlist(d, c, b, a) ijlist[d + 2 * (c + size_c * (b + neltypes * a))]
#define R(c, b, a) R[c + size_c * (b + neltypes * a)]
#define mask(c, b, a) mask[c + size_c * (b + neltypes * a)]
#define M(d, c, b, a) M[d + size_d * (c + size_c * (b + neltypes * a))]
#define drdrx(d, c, b, a) drdrx[d + 3 * (c + size_c * (b + neltypes * a))]
#define dMdrx(d, e, c, b, a) \
  dMdrx[d + size_d * (e + 3 * (c + size_c * (b + neltypes * a)))]

  int i, j, a, b, c;
  int ii, jn, elti, eltj, numneighi;
  int neigh[neltypes];
  double xtmp, ytmp, ztmp;
  double rijinv, rij, rij2, dij[3];
  double x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz;
  double yyy, yyz, yzz, zzz, xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz;
  double xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz;
  double xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz;
  double xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz, yyzzz, yzzzz;
  double zzzzz;
  double xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz, xxxyzz;
  double xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy, xyyyyz, xyyyzz;
  double xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz, yyyzzz, yyzzzz, yzzzzz;
  double zzzzzz;
  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);
  
  for (int ii = global_id_start; ii < global_id_end; ii++) {
    for (elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    i = ilist[ii];
    if (neltypes > 1) {
      a = i2row[i];
    } else {
      a = i;
    }
    elti = fmap[type[i]];
    xtmp = pos[i][0];
    ytmp = pos[i][1];
    ztmp = pos[i][2];
    numneighi = numneigh[i];
    for (jn = 0; jn < numneighi; jn++) {
      j = firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = pos[j][0] - xtmp;
      dij[1] = pos[j][1] - ytmp;
      dij[2] = pos[j][2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < cutforcesq && rij2 > 1e-10) {
        eltj = fmap[type[j]];
        rij = sqrt(rij2);
        rijinv = 1.0 / rij;
        b = eltij_pos[elti][eltj];
        c = neigh[b];
        if (c < size_c) {
          x = dij[0] * rijinv;
          y = dij[1] * rijinv;
          z = dij[2] * rijinv;
          R(c, b, a) = rij;
          mask(c, b, a) = 1.0;
          M(0, c, b, a) = 1.0;
#if defined(BUILD_BASELINE)
          dMdrx(0, 0, c, b, a) = 0.0;
          dMdrx(0, 1, c, b, a) = 0.0;
          dMdrx(0, 2, c, b, a) = 0.0;
#endif
          drdrx(0, c, b, a) = x;
          drdrx(1, c, b, a) = y;
          drdrx(2, c, b, a) = z;
          ijlist(0, c, b, a) = i;
          ijlist(1, c, b, a) = j;
          if (size_d > 1) {
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M(1, c, b, a) = x;
            M(2, c, b, a) = y;
            M(3, c, b, a) = z;
#if defined(BUILD_BASELINE)
            dMdrx(1, 0, c, b, a) = -rijinv * (xx - 1);
            dMdrx(1, 1, c, b, a) = -rijinv * xy;
            dMdrx(1, 2, c, b, a) = -rijinv * xz;
            dMdrx(2, 0, c, b, a) = -rijinv * xy;
            dMdrx(2, 1, c, b, a) = -rijinv * (yy - 1);
            dMdrx(2, 2, c, b, a) = -rijinv * yz;
            dMdrx(3, 0, c, b, a) = -rijinv * xz;
            dMdrx(3, 1, c, b, a) = -rijinv * yz;
            dMdrx(3, 2, c, b, a) = -rijinv * (zz - 1);
#endif
            if (size_d > 4) {
              xxx = xx * x;
              xxy = xx * y;
              xxz = xx * z;
              xyy = xy * y;
              xyz = xy * z;
              xzz = xz * z;
              yyy = yy * y;
              yyz = yy * z;
              yzz = yz * z;
              zzz = zz * z;
              M(4, c, b, a) = xx;
              M(5, c, b, a) = xy;
              M(6, c, b, a) = xz;
              M(7, c, b, a) = yy;
              M(8, c, b, a) = yz;
              M(9, c, b, a) = zz;
#if defined(BUILD_BASELINE)
              dMdrx(4, 0, c, b, a) = -rijinv * (2 * xxx - 2 * x);
              dMdrx(4, 1, c, b, a) = -rijinv * 2 * xxy;
              dMdrx(4, 2, c, b, a) = -rijinv * 2 * xxz;
              dMdrx(5, 0, c, b, a) = -rijinv * (2 * xxy - y);
              dMdrx(5, 1, c, b, a) = -rijinv * (2 * xyy - x);
              dMdrx(5, 2, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(6, 0, c, b, a) = -rijinv * (2 * xxz - z);
              dMdrx(6, 1, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(6, 2, c, b, a) = -rijinv * (2 * xzz - x);
              dMdrx(7, 0, c, b, a) = -rijinv * 2 * xyy;
              dMdrx(7, 1, c, b, a) = -rijinv * (2 * yyy - 2 * y);
              dMdrx(7, 2, c, b, a) = -rijinv * 2 * yyz;
              dMdrx(8, 0, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(8, 1, c, b, a) = -rijinv * (2 * yyz - z);
              dMdrx(8, 2, c, b, a) = -rijinv * (2 * yzz - y);
              dMdrx(9, 0, c, b, a) = -rijinv * 2 * xzz;
              dMdrx(9, 1, c, b, a) = -rijinv * 2 * yzz;
              dMdrx(9, 2, c, b, a) = -rijinv * (2 * zzz - 2 * z);
#endif
              if (size_d > 10) {
                xxxx = xxx * x;
                xxxy = xxx * y;
                xxxz = xxx * z;
                xxyy = xxy * y;
                xxyz = xxy * z;
                xxzz = xxz * z;
                xyyy = xyy * y;
                xyyz = xyy * z;
                xyzz = xyz * z;
                xzzz = xzz * z;
                yyyy = yyy * y;
                yyyz = yyy * z;
                yyzz = yyz * z;
                yzzz = yzz * z;
                zzzz = zzz * z;
                M(10, c, b, a) = xxx;
                M(11, c, b, a) = xxy;
                M(12, c, b, a) = xxz;
                M(13, c, b, a) = xyy;
                M(14, c, b, a) = xyz;
                M(15, c, b, a) = xzz;
                M(16, c, b, a) = yyy;
                M(17, c, b, a) = yyz;
                M(18, c, b, a) = yzz;
                M(19, c, b, a) = zzz;
#if defined(BUILD_BASELINE)
                dMdrx(10, 0, c, b, a) = -rijinv * (3 * xxxx - 3 * xx);
                dMdrx(10, 1, c, b, a) = -rijinv * 3 * xxxy;
                dMdrx(10, 2, c, b, a) = -rijinv * 3 * xxxz;
                dMdrx(11, 0, c, b, a) = -rijinv * (3 * xxxy - 2 * xy);
                dMdrx(11, 1, c, b, a) = -rijinv * (3 * xxyy - xx);
                dMdrx(11, 2, c, b, a) = -rijinv * 3 * xxyz;
                dMdrx(12, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                dMdrx(12, 1, c, b, a) = -rijinv * 3 * xxyz;
                dMdrx(12, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                dMdrx(13, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                dMdrx(13, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                dMdrx(13, 2, c, b, a) = -rijinv * 3 * xyyz;
                dMdrx(14, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                dMdrx(14, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                dMdrx(14, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
                dMdrx(15, 0, c, b, a) = -rijinv * (3 * xxzz - zz);
                dMdrx(15, 1, c, b, a) = -rijinv * 3 * xyzz;
                dMdrx(15, 2, c, b, a) = -rijinv * (3 * xzzz - 2 * xz);
                dMdrx(16, 0, c, b, a) = -rijinv * 3 * xyyy;
                dMdrx(16, 1, c, b, a) = -rijinv * (3 * yyyy - 3 * yy);
                dMdrx(16, 2, c, b, a) = -rijinv * 3 * yyyz;
                dMdrx(17, 0, c, b, a) = -rijinv * 3 * xyyz;
                dMdrx(17, 1, c, b, a) = -rijinv * (3 * yyyz - 2 * yz);
                dMdrx(17, 2, c, b, a) = -rijinv * (3 * yyzz - yy);
                dMdrx(18, 0, c, b, a) = -rijinv * 3 * xyzz;
                dMdrx(18, 1, c, b, a) = -rijinv * (3 * yyzz - zz);
                dMdrx(18, 2, c, b, a) = -rijinv * (3 * yzzz - 2 * yz);
                dMdrx(19, 0, c, b, a) = -rijinv * 3 * xzzz;
                dMdrx(19, 1, c, b, a) = -rijinv * 3 * yzzz;
                dMdrx(19, 2, c, b, a) = -rijinv * (3 * zzzz - 3 * zz);
#endif
                if (size_d > 20) {
                  xxxxx = xxxx * x;
                  xxxxy = xxxx * y;
                  xxxxz = xxxx * z;
                  xxxyy = xxxy * y;
                  xxxyz = xxxy * z;
                  xxxzz = xxxz * z;
                  xxyyy = xxyy * y;
                  xxyyz = xxyy * z;
                  xxyzz = xxyz * z;
                  xxzzz = xxzz * z;
                  xyyyy = xyyy * y;
                  xyyyz = xyyy * z;
                  xyyzz = xyyz * z;
                  xyzzz = xyzz * z;
                  xzzzz = xzzz * z;
                  yyyyy = yyyy * y;
                  yyyyz = yyyy * z;
                  yyyzz = yyyz * z;
                  yyzzz = yyzz * z;
                  yzzzz = yzzz * z;
                  zzzzz = zzzz * z;
                  M(20, c, b, a) = xxxx;
                  M(21, c, b, a) = xxxy;
                  M(22, c, b, a) = xxxz;
                  M(23, c, b, a) = xxyy;
                  M(24, c, b, a) = xxyz;
                  M(25, c, b, a) = xxzz;
                  M(26, c, b, a) = xyyy;
                  M(27, c, b, a) = xyyz;
                  M(28, c, b, a) = xyzz;
                  M(29, c, b, a) = xzzz;
                  M(30, c, b, a) = yyyy;
                  M(31, c, b, a) = yyyz;
                  M(32, c, b, a) = yyzz;
                  M(33, c, b, a) = yzzz;
                  M(33, c, b, a) = yzzz;
                  M(34, c, b, a) = zzzz;
#if defined(BUILD_BASELINE)
                  dMdrx(20, 0, c, b, a) = -rijinv * (4 * xxxxx - 4 * xxx);
                  dMdrx(20, 1, c, b, a) = -rijinv * 4 * xxxxy;
                  dMdrx(20, 2, c, b, a) = -rijinv * 4 * xxxxz;
                  dMdrx(21, 0, c, b, a) = -rijinv * (4 * xxxxy - 3 * xxy);
                  dMdrx(21, 1, c, b, a) = -rijinv * (4 * xxxyy - xxx);
                  dMdrx(21, 2, c, b, a) = -rijinv * 4 * xxxyz;
                  dMdrx(22, 0, c, b, a) = -rijinv * (4 * xxxxz - 3 * xxz);
                  dMdrx(22, 1, c, b, a) = -rijinv * 4 * xxxyz;
                  dMdrx(22, 2, c, b, a) = -rijinv * (4 * xxxzz - xxx);
                  dMdrx(23, 0, c, b, a) = -rijinv * (4 * xxxyy - 2 * xyy);
                  dMdrx(23, 1, c, b, a) = -rijinv * (4 * xxyyy - 2 * xxy);
                  dMdrx(23, 2, c, b, a) = -rijinv * 4 * xxyyz;
                  dMdrx(24, 0, c, b, a) = -rijinv * (4 * xxxyz - 2 * xyz);
                  dMdrx(24, 1, c, b, a) = -rijinv * (4 * xxyyz - xxz);
                  dMdrx(24, 2, c, b, a) = -rijinv * (4 * xxyzz - xxy);
                  dMdrx(25, 0, c, b, a) = -rijinv * (4 * xxxzz - 2 * xzz);
                  dMdrx(25, 1, c, b, a) = -rijinv * 4 * xxyzz;
                  dMdrx(25, 2, c, b, a) = -rijinv * (4 * xxzzz - 2 * xxz);
                  dMdrx(26, 0, c, b, a) = -rijinv * (4 * xxyyy - yyy);
                  dMdrx(26, 1, c, b, a) = -rijinv * (4 * xyyyy - 3 * xyy);
                  dMdrx(26, 2, c, b, a) = -rijinv * 4 * xyyyz;
                  dMdrx(27, 0, c, b, a) = -rijinv * (4 * xxyyz - yyz);
                  dMdrx(27, 1, c, b, a) = -rijinv * (4 * xyyyz - 2 * xyz);
                  dMdrx(27, 2, c, b, a) = -rijinv * (4 * xyyzz - xyy);
                  dMdrx(28, 0, c, b, a) = -rijinv * (4 * xxyzz - yzz);
                  dMdrx(28, 1, c, b, a) = -rijinv * (4 * xyyzz - xzz);
                  dMdrx(28, 2, c, b, a) = -rijinv * (4 * xyzzz - 2 * xyz);
                  dMdrx(29, 0, c, b, a) = -rijinv * (4 * xxzzz - zzz);
                  dMdrx(29, 1, c, b, a) = -rijinv * 4 * xyzzz;
                  dMdrx(29, 2, c, b, a) = -rijinv * (4 * xzzzz - 3 * xzz);
                  dMdrx(30, 0, c, b, a) = -rijinv * 4 * xyyyy;
                  dMdrx(30, 1, c, b, a) = -rijinv * (4 * yyyyy - 4 * yyy);
                  dMdrx(30, 2, c, b, a) = -rijinv * 4 * yyyyz;
                  dMdrx(31, 0, c, b, a) = -rijinv * 4 * xyyyz;
                  dMdrx(31, 1, c, b, a) = -rijinv * (4 * yyyyz - 3 * yyz);
                  dMdrx(31, 2, c, b, a) = -rijinv * (4 * yyyzz - yyy);
                  dMdrx(32, 0, c, b, a) = -rijinv * 4 * xyyzz;
                  dMdrx(32, 1, c, b, a) = -rijinv * (4 * yyyzz - 2 * yzz);
                  dMdrx(32, 2, c, b, a) = -rijinv * (4 * yyzzz - 2 * yyz);
                  dMdrx(33, 0, c, b, a) = -rijinv * 4 * xyzzz;
                  dMdrx(33, 1, c, b, a) = -rijinv * (4 * yyzzz - zzz);
                  dMdrx(33, 2, c, b, a) = -rijinv * (4 * yzzzz - 3 * yzz);
                  dMdrx(34, 0, c, b, a) = -rijinv * 4 * xzzzz;
                  dMdrx(34, 1, c, b, a) = -rijinv * 4 * yzzzz;
                  dMdrx(34, 2, c, b, a) = -rijinv * (4 * zzzzz - 4 * zzz);
#endif
                  if (size_d > 35) {
                    xxxxxx = xxxxx * x;
                    xxxxxy = xxxxx * y;
                    xxxxxz = xxxxx * z;
                    xxxxyy = xxxxy * y;
                    xxxxyz = xxxxy * z;
                    xxxxzz = xxxxz * z;
                    xxxyyy = xxxyy * y;
                    xxxyyz = xxxyy * z;
                    xxxyzz = xxxyz * z;
                    xxxzzz = xxxzz * z;
                    xxyyyy = xxyyy * y;
                    xxyyyz = xxyyy * z;
                    xxyyzz = xxyyz * z;
                    xxyzzz = xxyzz * z;
                    xxzzzz = xxzzz * z;
                    xyyyyy = xyyyy * y;
                    xyyyyz = xyyyy * z;
                    xyyyzz = xyyyz * z;
                    xyyzzz = xyyzz * z;
                    xyzzzz = xyzzz * z;
                    xzzzzz = xzzzz * z;
                    yyyyyy = yyyyy * y;
                    yyyyyz = yyyyy * z;
                    yyyyzz = yyyyz * z;
                    yyyzzz = yyyzz * z;
                    yyzzzz = yyzzz * z;
                    yzzzzz = yzzzz * z;
                    zzzzzz = zzzzz * z;
                    M(35, c, b, a) = xxxxx;
                    M(36, c, b, a) = xxxxy;
                    M(37, c, b, a) = xxxxz;
                    M(38, c, b, a) = xxxyy;
                    M(39, c, b, a) = xxxyz;
                    M(40, c, b, a) = xxxzz;
                    M(41, c, b, a) = xxyyy;
                    M(42, c, b, a) = xxyyz;
                    M(43, c, b, a) = xxyzz;
                    M(44, c, b, a) = xxzzz;
                    M(45, c, b, a) = xyyyy;
                    M(46, c, b, a) = xyyyz;
                    M(47, c, b, a) = xyyzz;
                    M(48, c, b, a) = xyzzz;
                    M(49, c, b, a) = xzzzz;
                    M(50, c, b, a) = yyyyy;
                    M(51, c, b, a) = yyyyz;
                    M(52, c, b, a) = yyyzz;
                    M(53, c, b, a) = yyzzz;
                    M(54, c, b, a) = yzzzz;
                    M(55, c, b, a) = zzzzz;
#if defined(BUILD_BASELINE)
                    dMdrx(35, 0, c, b, a) = -rijinv * (5 * xxxxxx - 5 * xxxx);
                    dMdrx(35, 1, c, b, a) = -rijinv * 5 * xxxxxy;
                    dMdrx(35, 2, c, b, a) = -rijinv * 5 * xxxxxz;
                    dMdrx(36, 0, c, b, a) = -rijinv * (5 * xxxxxy - 4 * xxxy);
                    dMdrx(36, 1, c, b, a) = -rijinv * (5 * xxxxyy - xxxx);
                    dMdrx(36, 2, c, b, a) = -rijinv * 5 * xxxxyz;
                    dMdrx(37, 0, c, b, a) = -rijinv * (5 * xxxxxz - 4 * xxxz);
                    dMdrx(37, 1, c, b, a) = -rijinv * 5 * xxxxyz;
                    dMdrx(37, 2, c, b, a) = -rijinv * (5 * xxxxzz - xxxx);
                    dMdrx(38, 0, c, b, a) = -rijinv * (5 * xxxxyy - 3 * xxyy);
                    dMdrx(38, 1, c, b, a) = -rijinv * (5 * xxxyyy - 2 * xxxy);
                    dMdrx(38, 2, c, b, a) = -rijinv * 5 * xxxyyz;
                    dMdrx(39, 0, c, b, a) = -rijinv * (5 * xxxxyz - 3 * xxyz);
                    dMdrx(39, 1, c, b, a) = -rijinv * (5 * xxxyyz - xxxz);
                    dMdrx(39, 2, c, b, a) = -rijinv * (5 * xxxyzz - xxxy);
                    dMdrx(40, 0, c, b, a) = -rijinv * (5 * xxxxzz - 3 * xxzz);
                    dMdrx(40, 1, c, b, a) = -rijinv * 5 * xxxyzz;
                    dMdrx(40, 2, c, b, a) = -rijinv * (5 * xxxzzz - 2 * xxxz);
                    dMdrx(41, 0, c, b, a) = -rijinv * (5 * xxxyyy - 2 * xyyy);
                    dMdrx(41, 1, c, b, a) = -rijinv * (5 * xxyyyy - 3 * xxyy);
                    dMdrx(41, 2, c, b, a) = -rijinv * 5 * xxyyyz;
                    dMdrx(42, 0, c, b, a) = -rijinv * (5 * xxxyyz - 2 * xyyz);
                    dMdrx(42, 1, c, b, a) = -rijinv * (5 * xxyyyz - 2 * xxyz);
                    dMdrx(42, 2, c, b, a) = -rijinv * (5 * xxyyzz - xxyy);
                    dMdrx(43, 0, c, b, a) = -rijinv * (5 * xxxyzz - 2 * xyzz);
                    dMdrx(43, 1, c, b, a) = -rijinv * (5 * xxyyzz - xxzz);
                    dMdrx(43, 2, c, b, a) = -rijinv * (5 * xxyzzz - 2 * xxyz);
                    dMdrx(44, 0, c, b, a) = -rijinv * (5 * xxxzzz - 2 * xzzz);
                    dMdrx(44, 1, c, b, a) = -rijinv * 5 * xxyzzz;
                    dMdrx(44, 2, c, b, a) = -rijinv * (5 * xxzzzz - 3 * xxzz);
                    dMdrx(45, 0, c, b, a) = -rijinv * (5 * xxyyyy - yyyy);
                    dMdrx(45, 1, c, b, a) = -rijinv * (5 * xyyyyy - 4 * xyyy);
                    dMdrx(45, 2, c, b, a) = -rijinv * 5 * xyyyyz;
                    dMdrx(46, 0, c, b, a) = -rijinv * (5 * xxyyyz - yyyz);
                    dMdrx(46, 1, c, b, a) = -rijinv * (5 * xyyyyz - 3 * xyyz);
                    dMdrx(46, 2, c, b, a) = -rijinv * (5 * xyyyzz - xyyy);
                    dMdrx(47, 0, c, b, a) = -rijinv * (5 * xxyyzz - yyzz);
                    dMdrx(47, 1, c, b, a) = -rijinv * (5 * xyyyzz - 2 * xyzz);
                    dMdrx(47, 2, c, b, a) = -rijinv * (5 * xyyzzz - 2 * xyyz);
                    dMdrx(48, 0, c, b, a) = -rijinv * (5 * xxyzzz - yzzz);
                    dMdrx(48, 1, c, b, a) = -rijinv * (5 * xyyzzz - xzzz);
                    dMdrx(48, 2, c, b, a) = -rijinv * (5 * xyzzzz - 3 * xyzz);
                    dMdrx(49, 0, c, b, a) = -rijinv * (5 * xxzzzz - zzzz);
                    dMdrx(49, 1, c, b, a) = -rijinv * 5 * xyzzzz;
                    dMdrx(49, 2, c, b, a) = -rijinv * (5 * xzzzzz - 4 * xzzz);
                    dMdrx(50, 0, c, b, a) = -rijinv * 5 * xyyyyy;
                    dMdrx(50, 1, c, b, a) = -rijinv * (5 * yyyyyy - 5 * yyyy);
                    dMdrx(50, 2, c, b, a) = -rijinv * 5 * yyyyyz;
                    dMdrx(51, 0, c, b, a) = -rijinv * 5 * xyyyyz;
                    dMdrx(51, 1, c, b, a) = -rijinv * (5 * yyyyyz - 4 * yyyz);
                    dMdrx(51, 2, c, b, a) = -rijinv * (5 * yyyyzz - yyyy);
                    dMdrx(52, 0, c, b, a) = -rijinv * 5 * xyyyzz;
                    dMdrx(52, 1, c, b, a) = -rijinv * (5 * yyyyzz - 3 * yyzz);
                    dMdrx(52, 2, c, b, a) = -rijinv * (5 * yyyzzz - 2 * yyyz);
                    dMdrx(53, 0, c, b, a) = -rijinv * 5 * xyyzzz;
                    dMdrx(53, 1, c, b, a) = -rijinv * (5 * yyyzzz - 2 * yzzz);
                    dMdrx(53, 2, c, b, a) = -rijinv * (5 * yyzzzz - 3 * yyzz);
                    dMdrx(54, 0, c, b, a) = -rijinv * 5 * xyzzzz;
                    dMdrx(54, 1, c, b, a) = -rijinv * (5 * yyzzzz - zzzz);
                    dMdrx(54, 2, c, b, a) = -rijinv * (5 * yzzzzz - 4 * yzzz);
                    dMdrx(55, 0, c, b, a) = -rijinv * 5 * xzzzzz;
                    dMdrx(55, 1, c, b, a) = -rijinv * 5 * yzzzzz;
                    dMdrx(55, 2, c, b, a) = -rijinv * (5 * zzzzzz - 5 * zzzz);
#endif
                  }
                }
              }
            }
          }
        }
        neigh[b]++;
      }
    }

    for (b = 0; b < neltypes; b++) {
      for (c = neigh[b]; c < size_c; c++) {
        ijlist(0, c, b, a) = -1;
        ijlist(1, c, b, a) = -1;
        mask(c, b, a) = 0;
        R(c, b, a) = 0.0;
        drdrx(0, c, b, a) = 0.0;
        drdrx(1, c, b, a) = 0.0;
        drdrx(2, c, b, a) = 0.0;
        for (int d = 0; d < size_d; d++) { M(d, c, b, a) = 0.0; }
#if defined(BUILD_BASELINE)
        for (int d = 0; d < size_d; d++) {
          dMdrx(d, 0, c, b, a) = 0.0;
          dMdrx(d, 1, c, b, a) = 0.0;
          dMdrx(d, 2, c, b, a) = 0.0;
        }
#endif
      }
    }
  }
#undef ijlist
#undef drdrx
#undef M
#undef mask
#undef R
#undef dMdrx
}
