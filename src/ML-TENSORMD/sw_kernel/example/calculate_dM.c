//
// Created by Xin Chen on 2023/2/23.
//

#include "calculate_dM.h"

void calculate_dM_s(double *dM)
{
  dM[0] = 0.0;
  dM[1] = 0.0;
  dM[2] = 0.0;
}

void calculate_dM_p(double rijinv, double *drdrx, double *dM)
{
  double x = drdrx[0];
  double y = drdrx[1];
  double z = drdrx[2];
  double xx, xy, xz, yy, yz, zz;

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = 0.0;
  dM[5] = -rijinv * xy;
  dM[6] = -rijinv * (yy - 1);
  dM[7] = -rijinv * yz;
  dM[8] = 0.0;
  dM[9] = -rijinv * xz;
  dM[10] = -rijinv * yz;
  dM[11] = -rijinv * (zz - 1);
}

void calculate_dM_d(double rijinv, double *drdrx, double *dM)
{
  double x = drdrx[0];
  double y = drdrx[1];
  double z = drdrx[2];
  double xx, xy, xz, yy, yz, zz;
  double xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = -rijinv * (2 * xxx - 2 * x);
  dM[5] = -rijinv * (2 * xxy - y);
  dM[6] = -rijinv * (2 * xxz - z);
  dM[7] = -rijinv * 2 * xyy;
  dM[8] = -rijinv * 2 * xyz;
  dM[9] = -rijinv * 2 * xzz;
  dM[10] = 0.0;
  dM[11] = -rijinv * xy;
  dM[12] = -rijinv * (yy - 1);
  dM[13] = -rijinv * yz;
  dM[14] = -rijinv * 2 * xxy;
  dM[15] = -rijinv * (2 * xyy - x);
  dM[16] = -rijinv * 2 * xyz;
  dM[17] = -rijinv * (2 * yyy - 2 * y);
  dM[18] = -rijinv * (2 * yyz - z);
  dM[19] = -rijinv * 2 * yzz;
  dM[20] = 0.0;
  dM[21] = -rijinv * xz;
  dM[22] = -rijinv * yz;
  dM[23] = -rijinv * (zz - 1);
  dM[24] = -rijinv * 2 * xxz;
  dM[25] = -rijinv * 2 * xyz;
  dM[26] = -rijinv * (2 * xzz - x);
  dM[27] = -rijinv * 2 * yyz;
  dM[28] = -rijinv * (2 * yzz - y);
  dM[29] = -rijinv * (2 * zzz - 2 * z);
}

void calculate_dM_f(double rijinv, double *drdrx, double *dM)
{
  double x = drdrx[0];
  double y = drdrx[1];
  double z = drdrx[2];
  double xx, xy, xz, yy, yz, zz;
  double xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  double xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  double yyyy, yyyz, yyzz, yzzz, zzzz;

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;
  xxx = x * xx;
  xxy = x * xy;
  xxz = x * xz;
  xyy = y * xy;
  xyz = y * xz;
  xzz = z * xz;
  yyy = y * yy;
  yyz = y * yz;
  yzz = z * yz;
  zzz = z * zz;
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

  dM[0] = 0.0;
  dM[1] = -rijinv * (xx - 1);
  dM[2] = -rijinv * xy;
  dM[3] = -rijinv * xz;
  dM[4] = -rijinv * (2 * xxx - 2 * x);
  dM[5] = -rijinv * (2 * xxy - y);
  dM[6] = -rijinv * (2 * xxz - z);
  dM[7] = -rijinv * 2 * xyy;
  dM[8] = -rijinv * 2 * xyz;
  dM[9] = -rijinv * 2 * xzz;
  dM[10] = -rijinv * (3 * xxxx - 3 * xx);
  dM[11] = -rijinv * (3 * xxxy - 2 * xy);
  dM[12] = -rijinv * (3 * xxxz - 2 * xz);
  dM[13] = -rijinv * (3 * xxyy - yy);
  dM[14] = -rijinv * (3 * xxyz - yz);
  dM[15] = -rijinv * (3 * xxzz - zz);
  dM[16] = -rijinv * 3 * xyyy;
  dM[17] = -rijinv * 3 * xyyz;
  dM[18] = -rijinv * 3 * xyzz;
  dM[19] = -rijinv * 3 * xzzz;

  dM[20] = 0.0;
  dM[21] = -rijinv * xy;
  dM[22] = -rijinv * (yy - 1);
  dM[23] = -rijinv * yz;
  dM[24] = -rijinv * 2 * xxy;
  dM[25] = -rijinv * (2 * xyy - x);
  dM[26] = -rijinv * 2 * xyz;
  dM[27] = -rijinv * (2 * yyy - 2 * y);
  dM[28] = -rijinv * (2 * yyz - z);
  dM[29] = -rijinv * 2 * yzz;
  dM[30] = -rijinv * 3 * xxxy;
  dM[31] = -rijinv * (3 * xxyy - xx);
  dM[32] = -rijinv * 3 * xxyz;
  dM[33] = -rijinv * (3 * xyyy - 2 * xy);
  dM[34] = -rijinv * (3 * xyyz - xz);
  dM[35] = -rijinv * 3 * xyzz;
  dM[36] = -rijinv * (3 * yyyy - 3 * yy);
  dM[37] = -rijinv * (3 * yyyz - 2 * yz);
  dM[38] = -rijinv * (3 * yyzz - zz);
  dM[39] = -rijinv * 3 * yzzz;

  dM[40] = 0.0;
  dM[41] = -rijinv * xz;
  dM[42] = -rijinv * yz;
  dM[43] = -rijinv * (zz - 1);
  dM[44] = -rijinv * 2 * xxz;
  dM[45] = -rijinv * 2 * xyz;
  dM[46] = -rijinv * (2 * xzz - x);
  dM[47] = -rijinv * 2 * yyz;
  dM[48] = -rijinv * (2 * yzz - y);
  dM[49] = -rijinv * (2 * zzz - 2 * z);
  dM[50] = -rijinv * 3 * xxxz;
  dM[51] = -rijinv * 3 * xxyz;
  dM[52] = -rijinv * (3 * xxzz - xx);
  dM[53] = -rijinv * 3 * xyyz;
  dM[54] = -rijinv * (3 * xyzz - xy);
  dM[55] = -rijinv * (3 * xzzz - 2 * xz);
  dM[56] = -rijinv * 3 * yyyz;
  dM[57] = -rijinv * (3 * yyzz - yy);
  dM[58] = -rijinv * (3 * yzzz - 2 * yz);
  dM[59] = -rijinv * (3 * zzzz - 3 * zz);
}

void calculate_dM(int max_moment, double rijinv, double *drdrx, double *dM)
{
  if (max_moment == 0)
    calculate_dM_s(dM);
  else if (max_moment == 1)
    calculate_dM_p(rijinv, drdrx, dM);
  else if (max_moment == 2)
    calculate_dM_d(rijinv, drdrx, dM);
  else
    calculate_dM_f(rijinv, drdrx, dM);
}
