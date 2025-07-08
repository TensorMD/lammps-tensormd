#include "hip/hip_runtime.h"
#include "hipcub/hipcub.hpp"
#include "math.cuh"
#include <stdio.h>

#define NEIGHMASK 0x1FFFFFFF
#define DD 20
#define MM 4
#define K1_CC 300
#define F1_blockdim 32
#define F2_blockdim 128
#define K1_blockdim 128
#define ST_blockdim 4
#define GEC_blockdim 16
#define SF_blockdim 256

// Modify this value to control the additional reserved space size during memory allocation
#define MALLOC_BUFF 12

int *firstneigh_host;
int *fmap_type_host;
int *numneigh_max_device;
void *globalPool;
int max_alocal;
int max_inum;
int max_numneigh_max;
int numneigh_max_global;
int max_old_numneigh_max;
int max_nmax;
int maxThread;
hipblasHandle_t handle;

void memory_GPU(int mpiid, FILE *screen, FILE *logfile)
{
#if defined(GPU_DEBUG)
  size_t total, free;
  hipMemGetInfo(&free, &total);

  if (mpiid == 0) {
    if (screen) {
      fprintf(screen, "Use device memory %.2fGB / %.2fGB\n",
              (double) (total - free) / 1024 / 1024 / 1024,
              (double) total / 1024 / 1024 / 1024);
    }
    if (logfile) {
      fprintf(logfile, "Use device memory %.2fGB / %.2fGB\n",
              (double) (total - free) / 1024 / 1024 / 1024,
              (double) total / 1024 / 1024 / 1024);
    }
  }
#endif
}

void setup_device_GPU(int mpiid, FILE *screen, FILE *logfile)
{
  hipDeviceProp_t prop;
  int deviceCount, deviceId;
  hipGetDeviceCount(&deviceCount);
  deviceId = mpiid % deviceCount;
  hipSetDevice(deviceId);
  hipGetDeviceProperties(&prop, deviceId);
  maxThread = 256;

  if (mpiid == 0) {
    if (screen) { fprintf(screen, "Use %d GPU(s) per node\n", deviceCount); }
    if (logfile) { fprintf(logfile, "Use %d GPU(s) per node\n", deviceCount); }
  }
}

template <typename Scalar>
__device__ void calculate_dM_s(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  dM[ix0 + 0] = 0.0;
  dM[iy0 + 0] = 0.0;
  dM[iz0 + 0] = 0.0;
}

template <typename Scalar>
__device__ void calculate_dM_p(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  Scalar xx, xy, xz, yy, yz, zz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_s(rijinv, x, y, z, dM, offset);

  xx = x * x;
  xy = x * y;
  xz = x * z;
  yy = y * y;
  yz = z * y;
  zz = z * z;

  dM[ix0 + 1] = -rijinv * (xx - 1);
  dM[ix0 + 2] = -rijinv * xy;
  dM[ix0 + 3] = -rijinv * xz;

  dM[iy0 + 1] = -rijinv * xy;
  dM[iy0 + 2] = -rijinv * (yy - 1);
  dM[iy0 + 3] = -rijinv * yz;

  dM[iz0 + 1] = -rijinv * xz;
  dM[iz0 + 2] = -rijinv * yz;
  dM[iz0 + 3] = -rijinv * (zz - 1);
}

template <typename Scalar>
__device__ void calculate_dM_d(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_p(rijinv, x, y, z, dM, offset);

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

  dM[ix0 + 4] = -rijinv * (2 * xxx - 2 * x);
  dM[ix0 + 5] = -rijinv * (2 * xxy - y);
  dM[ix0 + 6] = -rijinv * (2 * xxz - z);
  dM[ix0 + 7] = -rijinv * 2 * xyy;
  dM[ix0 + 8] = -rijinv * 2 * xyz;
  dM[ix0 + 9] = -rijinv * 2 * xzz;

  dM[iy0 + 4] = -rijinv * 2 * xxy;
  dM[iy0 + 5] = -rijinv * (2 * xyy - x);
  dM[iy0 + 6] = -rijinv * 2 * xyz;
  dM[iy0 + 7] = -rijinv * (2 * yyy - 2 * y);
  dM[iy0 + 8] = -rijinv * (2 * yyz - z);
  dM[iy0 + 9] = -rijinv * 2 * yzz;

  dM[iz0 + 4] = -rijinv * 2 * xxz;
  dM[iz0 + 5] = -rijinv * 2 * xyz;
  dM[iz0 + 6] = -rijinv * (2 * xzz - x);
  dM[iz0 + 7] = -rijinv * 2 * yyz;
  dM[iz0 + 8] = -rijinv * (2 * yzz - y);
  dM[iz0 + 9] = -rijinv * (2 * zzz - 2 * z);
}

template <typename Scalar>
__device__ void calculate_dM_f(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_d(rijinv, x, y, z, dM, offset);

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

  dM[ix0 + 10] = -rijinv * (3 * xxxx - 3 * xx);
  dM[ix0 + 11] = -rijinv * (3 * xxxy - 2 * xy);
  dM[ix0 + 12] = -rijinv * (3 * xxxz - 2 * xz);
  dM[ix0 + 13] = -rijinv * (3 * xxyy - yy);
  dM[ix0 + 14] = -rijinv * (3 * xxyz - yz);
  dM[ix0 + 15] = -rijinv * (3 * xxzz - zz);
  dM[ix0 + 16] = -rijinv * 3 * xyyy;
  dM[ix0 + 17] = -rijinv * 3 * xyyz;
  dM[ix0 + 18] = -rijinv * 3 * xyzz;
  dM[ix0 + 19] = -rijinv * 3 * xzzz;

  dM[iy0 + 10] = -rijinv * 3 * xxxy;
  dM[iy0 + 11] = -rijinv * (3 * xxyy - xx);
  dM[iy0 + 12] = -rijinv * 3 * xxyz;
  dM[iy0 + 13] = -rijinv * (3 * xyyy - 2 * xy);
  dM[iy0 + 14] = -rijinv * (3 * xyyz - xz);
  dM[iy0 + 15] = -rijinv * 3 * xyzz;
  dM[iy0 + 16] = -rijinv * (3 * yyyy - 3 * yy);
  dM[iy0 + 17] = -rijinv * (3 * yyyz - 2 * yz);
  dM[iy0 + 18] = -rijinv * (3 * yyzz - zz);
  dM[iy0 + 19] = -rijinv * 3 * yzzz;

  dM[iz0 + 10] = -rijinv * 3 * xxxz;
  dM[iz0 + 11] = -rijinv * 3 * xxyz;
  dM[iz0 + 12] = -rijinv * (3 * xxzz - xx);
  dM[iz0 + 13] = -rijinv * 3 * xyyz;
  dM[iz0 + 14] = -rijinv * (3 * xyzz - xy);
  dM[iz0 + 15] = -rijinv * (3 * xzzz - 2 * xz);
  dM[iz0 + 16] = -rijinv * 3 * yyyz;
  dM[iz0 + 17] = -rijinv * (3 * yyzz - yy);
  dM[iz0 + 18] = -rijinv * (3 * yzzz - 2 * yz);
  dM[iz0 + 19] = -rijinv * (3 * zzzz - 3 * zz);
}

template <typename Scalar>
__device__ void calculate_dM_g(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  Scalar xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz;
  Scalar xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz;
  Scalar yyzzz, yzzzz, zzzzz;

  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_f(rijinv, x, y, z, dM, offset);

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

  dM[ix0 + 20] = -rijinv * (4 * xxxxx - 4 * xxx);
  dM[ix0 + 21] = -rijinv * (4 * xxxxy - 3 * xxy);
  dM[ix0 + 22] = -rijinv * (4 * xxxxz - 3 * xxz);
  dM[ix0 + 23] = -rijinv * (4 * xxxyy - 2 * xyy);
  dM[ix0 + 24] = -rijinv * (4 * xxxyz - 2 * xyz);
  dM[ix0 + 25] = -rijinv * (4 * xxxzz - 2 * xzz);
  dM[ix0 + 26] = -rijinv * (4 * xxyyy - yyy);
  dM[ix0 + 27] = -rijinv * (4 * xxyyz - yyz);
  dM[ix0 + 28] = -rijinv * (4 * xxyzz - yzz);
  dM[ix0 + 29] = -rijinv * (4 * xxzzz - zzz);
  dM[ix0 + 30] = -rijinv * 4 * xyyyy;
  dM[ix0 + 31] = -rijinv * 4 * xyyyz;
  dM[ix0 + 32] = -rijinv * 4 * xyyzz;
  dM[ix0 + 33] = -rijinv * 4 * xyzzz;
  dM[ix0 + 34] = -rijinv * 4 * xzzzz;

  dM[iy0 + 20] = -rijinv * 4 * xxxxy;
  dM[iy0 + 21] = -rijinv * (4 * xxxyy - xxx);
  dM[iy0 + 22] = -rijinv * 4 * xxxyz;
  dM[iy0 + 23] = -rijinv * (4 * xxyyy - 2 * xxy);
  dM[iy0 + 24] = -rijinv * (4 * xxyyz - xxz);
  dM[iy0 + 25] = -rijinv * 4 * xxyzz;
  dM[iy0 + 26] = -rijinv * (4 * xyyyy - 3 * xyy);
  dM[iy0 + 27] = -rijinv * (4 * xyyyz - 2 * xyz);
  dM[iy0 + 28] = -rijinv * (4 * xyyzz - xzz);
  dM[iy0 + 29] = -rijinv * 4 * xyzzz;
  dM[iy0 + 30] = -rijinv * (4 * yyyyy - 4 * yyy);
  dM[iy0 + 31] = -rijinv * (4 * yyyyz - 3 * yyz);
  dM[iy0 + 32] = -rijinv * (4 * yyyzz - 2 * yzz);
  dM[iy0 + 33] = -rijinv * (4 * yyzzz - zzz);
  dM[iy0 + 34] = -rijinv * 4 * yzzzz;

  dM[iz0 + 20] = -rijinv * 4 * xxxxz;
  dM[iz0 + 21] = -rijinv * 4 * xxxyz;
  dM[iz0 + 22] = -rijinv * (4 * xxxzz - xxx);
  dM[iz0 + 23] = -rijinv * 4 * xxyyz;
  dM[iz0 + 24] = -rijinv * (4 * xxyzz - xxy);
  dM[iz0 + 25] = -rijinv * (4 * xxzzz - 2 * xxz);
  dM[iz0 + 26] = -rijinv * 4 * xyyyz;
  dM[iz0 + 27] = -rijinv * (4 * xyyzz - xyy);
  dM[iz0 + 28] = -rijinv * (4 * xyzzz - 2 * xyz);
  dM[iz0 + 29] = -rijinv * (4 * xzzzz - 3 * xzz);
  dM[iz0 + 30] = -rijinv * 4 * yyyyz;
  dM[iz0 + 31] = -rijinv * (4 * yyyzz - yyy);
  dM[iz0 + 32] = -rijinv * (4 * yyzzz - 2 * yyz);
  dM[iz0 + 33] = -rijinv * (4 * yzzzz - 3 * yzz);
  dM[iz0 + 34] = -rijinv * (4 * zzzzz - 4 * zzz);
}

template <typename Scalar>
__device__ void calculate_dM_h(Scalar rijinv, Scalar x, Scalar y, Scalar z,
                               Scalar *dM, int offset)
{
  Scalar xx, xy, xz, yy, yz, zz;
  Scalar xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz;
  Scalar yyyy, yyyz, yyzz, yzzz, zzzz;
  Scalar xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz;
  Scalar xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz;
  Scalar yyzzz, yzzzz, zzzzz;
  Scalar xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz;
  Scalar xxxyzz, xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy;
  Scalar xyyyyz, xyyyzz, xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz;
  Scalar yyyzzz, yyzzzz, yzzzzz, zzzzzz;

  int ix0 = 0;
  int iy0 = offset;
  int iz0 = offset + offset;

  calculate_dM_g(rijinv, x, y, z, dM, offset);

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

  dM[ix0 + 35] = -rijinv * (5 * xxxxxx - 5 * xxxx);
  dM[ix0 + 36] = -rijinv * (5 * xxxxxy - 4 * xxxy);
  dM[ix0 + 37] = -rijinv * (5 * xxxxxz - 4 * xxxz);
  dM[ix0 + 38] = -rijinv * (5 * xxxxyy - 3 * xxyy);
  dM[ix0 + 39] = -rijinv * (5 * xxxxyz - 3 * xxyz);
  dM[ix0 + 40] = -rijinv * (5 * xxxxzz - 3 * xxzz);
  dM[ix0 + 41] = -rijinv * (5 * xxxyyy - 2 * xyyy);
  dM[ix0 + 42] = -rijinv * (5 * xxxyyz - 2 * xyyz);
  dM[ix0 + 43] = -rijinv * (5 * xxxyzz - 2 * xyzz);
  dM[ix0 + 44] = -rijinv * (5 * xxxzzz - 2 * xzzz);
  dM[ix0 + 45] = -rijinv * (5 * xxyyyy - yyyy);
  dM[ix0 + 46] = -rijinv * (5 * xxyyyz - yyyz);
  dM[ix0 + 47] = -rijinv * (5 * xxyyzz - yyzz);
  dM[ix0 + 48] = -rijinv * (5 * xxyzzz - yzzz);
  dM[ix0 + 49] = -rijinv * (5 * xxzzzz - zzzz);
  dM[ix0 + 50] = -rijinv * 5 * xyyyyy;
  dM[ix0 + 51] = -rijinv * 5 * xyyyyz;
  dM[ix0 + 52] = -rijinv * 5 * xyyyzz;
  dM[ix0 + 53] = -rijinv * 5 * xyyzzz;
  dM[ix0 + 54] = -rijinv * 5 * xyzzzz;
  dM[ix0 + 55] = -rijinv * 5 * xzzzzz;

  dM[iy0 + 35] = -rijinv * 5 * xxxxxy;
  dM[iy0 + 36] = -rijinv * (5 * xxxxyy - xxxx);
  dM[iy0 + 37] = -rijinv * 5 * xxxxyz;
  dM[iy0 + 38] = -rijinv * (5 * xxxyyy - 2 * xxxy);
  dM[iy0 + 39] = -rijinv * (5 * xxxyyz - xxxz);
  dM[iy0 + 40] = -rijinv * 5 * xxxyzz;
  dM[iy0 + 41] = -rijinv * (5 * xxyyyy - 3 * xxyy);
  dM[iy0 + 42] = -rijinv * (5 * xxyyyz - 2 * xxyz);
  dM[iy0 + 43] = -rijinv * (5 * xxyyzz - xxzz);
  dM[iy0 + 44] = -rijinv * 5 * xxyzzz;
  dM[iy0 + 45] = -rijinv * (5 * xyyyyy - 4 * xyyy);
  dM[iy0 + 46] = -rijinv * (5 * xyyyyz - 3 * xyyz);
  dM[iy0 + 47] = -rijinv * (5 * xyyyzz - 2 * xyzz);
  dM[iy0 + 48] = -rijinv * (5 * xyyzzz - xzzz);
  dM[iy0 + 49] = -rijinv * 5 * xyzzzz;
  dM[iy0 + 50] = -rijinv * (5 * yyyyyy - 5 * yyyy);
  dM[iy0 + 51] = -rijinv * (5 * yyyyyz - 4 * yyyz);
  dM[iy0 + 52] = -rijinv * (5 * yyyyzz - 3 * yyzz);
  dM[iy0 + 53] = -rijinv * (5 * yyyzzz - 2 * yzzz);
  dM[iy0 + 54] = -rijinv * (5 * yyzzzz - zzzz);
  dM[iy0 + 55] = -rijinv * 5 * yzzzzz;

  dM[iz0 + 35] = -rijinv * 5 * xxxxxz;
  dM[iz0 + 36] = -rijinv * 5 * xxxxyz;
  dM[iz0 + 37] = -rijinv * (5 * xxxxzz - xxxx);
  dM[iz0 + 38] = -rijinv * 5 * xxxyyz;
  dM[iz0 + 39] = -rijinv * (5 * xxxyzz - xxxy);
  dM[iz0 + 40] = -rijinv * (5 * xxxzzz - 2 * xxxz);
  dM[iz0 + 41] = -rijinv * 5 * xxyyyz;
  dM[iz0 + 42] = -rijinv * (5 * xxyyzz - xxyy);
  dM[iz0 + 43] = -rijinv * (5 * xxyzzz - 2 * xxyz);
  dM[iz0 + 44] = -rijinv * (5 * xxzzzz - 3 * xxzz);
  dM[iz0 + 45] = -rijinv * 5 * xyyyyz;
  dM[iz0 + 46] = -rijinv * (5 * xyyyzz - xyyy);
  dM[iz0 + 47] = -rijinv * (5 * xyyzzz - 2 * xyyz);
  dM[iz0 + 48] = -rijinv * (5 * xyzzzz - 3 * xyzz);
  dM[iz0 + 49] = -rijinv * (5 * xzzzzz - 4 * xzzz);
  dM[iz0 + 50] = -rijinv * 5 * yyyyyz;
  dM[iz0 + 51] = -rijinv * (5 * yyyyzz - yyyy);
  dM[iz0 + 52] = -rijinv * (5 * yyyzzz - 2 * yyyz);
  dM[iz0 + 53] = -rijinv * (5 * yyzzzz - 3 * yyzz);
  dM[iz0 + 54] = -rijinv * (5 * yzzzzz - 4 * yzzz);
  dM[iz0 + 55] = -rijinv * (5 * zzzzzz - 5 * zzzz);
}

template <typename Scalar>
__device__ void calculate_dM(int max_moment, Scalar rijinv, Scalar x, Scalar y,
                             Scalar z, Scalar *dM)
{
  if (max_moment == 0)
    calculate_dM_s<Scalar>(rijinv, x, y, z, dM, 1);
  else if (max_moment == 1)
    calculate_dM_p<Scalar>(rijinv, x, y, z, dM, 4);
  else if (max_moment == 2)
    calculate_dM_d<Scalar>(rijinv, x, y, z, dM, 10);
  else if (max_moment == 3)
    calculate_dM_f<Scalar>(rijinv, x, y, z, dM, 20);
  else if (max_moment == 4)
    calculate_dM_g<Scalar>(rijinv, x, y, z, dM, 35);
  else
    calculate_dM_h<Scalar>(rijinv, x, y, z, dM, 56);
}

template <typename Scalar>
__global__ void kernel_F1_device(Scalar *U, Scalar *dHdr, Scalar *drdrx,
                                 Scalar *F1, int *mask, int n, int d)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  Scalar tmp = 0;
  if (i < n && mask[i] != 0) {
    for (int j = 0; j < d; j++) { tmp += U[i * d + j] * dHdr[i * d + j]; }
    F1[i * 3 + 0] = drdrx[i * 3 + 0] * tmp;
    F1[i * 3 + 1] = drdrx[i * 3 + 1] * tmp;
    F1[i * 3 + 2] = drdrx[i * 3 + 2] * tmp;
  }
}

template <typename Scalar>
void kernel_F1_GPU(Scalar *U, Scalar *M, Scalar *drdrx, Scalar *F1, int *mask,
                   int a, int b, int c, int d)
{
  kernel_F1_device<<<(a * b * c + F1_blockdim - 1) / F1_blockdim,
                     F1_blockdim>>>(U, M, drdrx, F1, mask, a * b * c, d);
  hipDeviceSynchronize();
}

template <typename Scalar>
__global__ void kernel_fused_F2_device(int m, Scalar *V, Scalar *R,
                                       Scalar *drdrx, Scalar *F2, int n, int d)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  Scalar dM[168];
  Scalar V_reg[56];
  Scalar rijinv;
  Scalar rij = R[i];
  if (i < n && rij != 0) {
    rijinv = 1.0 / rij;
    calculate_dM(m - 1, rijinv, drdrx[i * 3], drdrx[i * 3 + 1],
                 drdrx[i * 3 + 2], dM);
    Scalar *V_ = V + i * d;
    Scalar y1 = 0, y2 = 0, y3 = 0;
    for (int j = 0; j < d; j++) {
      y1 += V_[j] * dM[0 * d + j];
      y2 += V_[j] * dM[1 * d + j];
      y3 += V_[j] * dM[2 * d + j];
    }
    F2[i * 3 + 0] = y1;
    F2[i * 3 + 1] = y2;
    F2[i * 3 + 2] = y3;
  }
}

template <typename Scalar>
__global__ void kernel_fused_F2_device(Scalar *V, Scalar *R, Scalar *drdrx,
                                       Scalar *F2, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  Scalar dM[3 * DD];
  Scalar V_reg[DD];
  Scalar rijinv;
  Scalar rij = R[i];
  if (i < n && rij != 0) {
    rijinv = 1.0 / rij;
    calculate_dM(3, rijinv, drdrx[i * 3], drdrx[i * 3 + 1], drdrx[i * 3 + 2],
                 dM);
    Scalar *V_ = V + i * DD;
    Scalar y1 = 0, y2 = 0, y3 = 0;
    for (int j = 0; j < DD; j++) {
      y1 += V_[j] * dM[0 * DD + j];
      y2 += V_[j] * dM[1 * DD + j];
      y3 += V_[j] * dM[2 * DD + j];
    }
    F2[i * 3 + 0] = y1;
    F2[i * 3 + 1] = y2;
    F2[i * 3 + 2] = y3;
  }
}

template <typename Scalar>
void kernel_fused_F2_GPU(Scalar *V, Scalar *R, Scalar *drdrx, Scalar *F2, int n,
                         int d, int m)
{
  if (m == MM) {
    hipLaunchKernelGGL(kernel_fused_F2_device,
                       dim3((n + F2_blockdim - 1) / F2_blockdim),
                       dim3(F2_blockdim), 0, 0, V, R, drdrx, F2, n);
  } else {
    hipLaunchKernelGGL(kernel_fused_F2_device,
                       dim3((n + F2_blockdim - 1) / F2_blockdim),
                       dim3(F2_blockdim), 0, 0, m, V, R, drdrx, F2, n, d);
  }

  hipDeviceSynchronize();
}

template <typename Scalar>
__global__ void kernel_dEdG_device(Scalar *src, Scalar *dst, int m, int size)
{
  int i = (threadIdx.x + blockDim.x * blockIdx.x) * m;
  if (i < size) {
    if (fabs(src[i]) < 1e-8)
      dst[i] = 0.0;
    else
      dst[i] = dst[i] / src[i] * 0.5;
  }
}

__global__ void kernel2_device(double *G, double *T, double *dEdG, double *P,
                               double *dEdS, int m, int d, int k)
{
  extern __shared__ double sd[];
  double *T_shm = sd;
  double *P_shm = sd + DD * MM;
  double dEdG_reg[MM] = {0};
  double dEdS_reg[DD] = {0};
  double *P_ = P + blockIdx.x * k * d;
  double *G_ = G + blockIdx.x * k * m;
  double *dEdG_ = dEdG + blockIdx.x * k * m;
  double *dEdS_ = dEdS + blockIdx.x * k * d;
  double temp;
  for (int ii = threadIdx.x; ii < k * 5; ii += blockDim.x) {
    reinterpret_cast<double4 *>(P_shm)[ii] =
        reinterpret_cast<double4 *>(P_)[ii];
  }
  for (int ii = threadIdx.x; ii < DD * MM; ii += blockDim.x) {
    T_shm[ii] = 2 * T[ii];
  }
  __syncthreads();

  reinterpret_cast<double4 *>(dEdG_reg)[0] =
      reinterpret_cast<double4 *>(dEdG_)[threadIdx.x];
  temp = G_[threadIdx.x * MM];
  if (fabs(temp) < 1e-8) {
    dEdG_reg[0] = 0.0;
  } else {
    dEdG_reg[0] = dEdG_reg[0] / temp * 0.5;
  }

  for (int ii = 0; ii < DD; ii++) {
#pragma unroll
    for (int jj = 0; jj < MM; jj++) {
      dEdS_reg[ii] += dEdG_reg[jj] * T_shm[ii * MM + jj];
    }
    dEdS_reg[ii] *= P_shm[threadIdx.x * d + ii];
  }
#pragma unroll
  for (int ii = 0; ii < 5; ii++) {
    reinterpret_cast<double4 *>(dEdS_)[threadIdx.x * 5 + ii] =
        reinterpret_cast<double4 *>(dEdS_reg)[ii];
  }
}

__global__ void kernel2_device(float *G, float *T, float *dEdG, float *P,
                               float *dEdS, int m, int d, int k)
{
  extern __shared__ float ss[];
  float *T_shm = ss;
  float *P_shm = ss + DD * MM;
  float dEdG_reg[MM] = {0};
  float dEdS_reg[DD] = {0};
  float *P_ = P + blockIdx.x * k * d;
  float *G_ = G + blockIdx.x * k * m;
  float *dEdG_ = dEdG + blockIdx.x * k * m;
  float *dEdS_ = dEdS + blockIdx.x * k * d;
  float temp;
  for (int ii = threadIdx.x; ii < k * 5; ii += blockDim.x) {
    reinterpret_cast<float4 *>(P_shm)[ii] = reinterpret_cast<float4 *>(P_)[ii];
  }
  for (int ii = threadIdx.x; ii < DD * MM; ii += blockDim.x) {
    T_shm[ii] = 2 * T[ii];
  }
  __syncthreads();

  reinterpret_cast<float4 *>(dEdG_reg)[0] =
      reinterpret_cast<float4 *>(dEdG_)[threadIdx.x];
  temp = G_[threadIdx.x * MM];
  if (fabs(temp) < 1e-8) {
    dEdG_reg[0] = 0.0;
  } else {
    dEdG_reg[0] = dEdG_reg[0] / temp * 0.5;
  }

  for (int ii = 0; ii < DD; ii++) {
#pragma unroll
    for (int jj = 0; jj < MM; jj++) {
      dEdS_reg[ii] += dEdG_reg[jj] * T_shm[ii * m + jj];
    }
    dEdS_reg[ii] *= P_shm[threadIdx.x * d + ii];
  }
#pragma unroll
  for (int ii = 0; ii < 5; ii++) {
    reinterpret_cast<float4 *>(dEdS_)[threadIdx.x * 5 + ii] =
        reinterpret_cast<float4 *>(dEdS_reg)[ii];
  }
}

template <typename Scalar>
void kernel2_GPU(Scalar *G, Scalar *T, Scalar *dEdG, Scalar *P, Scalar *dEdS,
                 int m, int d, int k, int b, int a)
{
  if (m == MM && d == DD) {
    int shared_size = (DD * MM + k * DD) * sizeof(Scalar);
    hipLaunchKernelGGL(kernel2_device, dim3(a * b), dim3(k), shared_size, 0, G,
                       T, dEdG, P, dEdS, m, d, k);
  } else {
    Scalar alpha = 2.0, beta = 0.0;
    hipLaunchKernelGGL(kernel_dEdG_device,
                       dim3((a * b * k + maxThread - 1) / maxThread),
                       dim3(maxThread), 0, 0, G, dEdG, m, a * b * k * m);
    gpublasgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, d, a * b * k, m, &alpha, T,
                m, dEdG, m, &beta, dEdS, d);
    hipLaunchKernelGGL(vmul_device,
                       dim3((a * b * k * d + maxThread - 1) / maxThread),
                       dim3(maxThread), 0, 0, dEdS, P, a * b * k * d);
  }
  hipDeviceSynchronize();
}

template <typename Scalar>
void kernel_U_GPU(Scalar *dEdS, Scalar *dHdr, Scalar *U, int a, int b, int d,
                  int c, int k)
{
  const int lda = d;
  const int ldb = k;
  const int ldc = d;
  const int stridea = d * k;
  const int strideb = c * k;
  const int stridec = d * c;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, d, c, k, &alpha,
                            dEdS, lda, stridea, dHdr, ldb, strideb, &beta, U,
                            ldc, stridec, a * b);
  hipDeviceSynchronize();
}

__global__ void kernel4_device(double *dEdS, double *M, double *dHdr,
                               double *F1, int *mask, int a, int b, int d,
                               int c, int k)
{
  extern __shared__ double sd[];
  double *dEdS_shm = sd;
  double *M_shm = dEdS_shm + k * d;
  double *U_shm = M_shm + 8 * d;
  double *dEdS_ = dEdS + blockIdx.x * d * k;
  double *M_ = M + blockIdx.x * c * d;
  double *dHdr_ = dHdr + blockIdx.x * c * k;
  double *F1_ = F1 + blockIdx.x * c * 3;
  int tid_y, tid_x, c_thread, c_index = 0;
  double M1_reg[20];
  double M2_reg[20];
  double dEdS_reg[20];
  int cmax = mask[blockIdx.x * c];

  for (int ii = threadIdx.x; ii < k * 5; ii += blockDim.x) {
    reinterpret_cast<double4 *>(dEdS_shm)[ii] =
        reinterpret_cast<double4 *>(dEdS_)[ii];
  }
  __syncthreads();

  while (c_index < cmax) {
    for (int ii = threadIdx.x; (ii < 8 * d) && (c_index * d + ii < c * d);
         ii += blockDim.x) {
      M_shm[ii] = M_[c_index * d + ii];
    }
    __syncthreads();

    tid_y = threadIdx.x / k;
    tid_x = threadIdx.x % k;
    double sum1 = 0, sum2 = 0;
#pragma unroll
    for (int ii = 0; ii < 5; ii++) {
      reinterpret_cast<double4 *>(M1_reg)[ii] =
          reinterpret_cast<double4 *>(M_shm)[tid_y * 5 + ii];
      reinterpret_cast<double4 *>(M2_reg)[ii] =
          reinterpret_cast<double4 *>(M_shm)[(tid_y + 4) * 5 + ii];
      reinterpret_cast<double4 *>(dEdS_reg)[ii] =
          reinterpret_cast<double4 *>(dEdS_shm)[tid_x * 5 + ii];
    }
#pragma unroll
    for (int ii = 0; ii < DD; ii++) {
      sum1 += M1_reg[ii] * dEdS_reg[ii];
      sum2 += M2_reg[ii] * dEdS_reg[ii];
    }
    c_thread = c_index + tid_y;
    U_shm[tid_y * k + tid_x] = sum1 * dHdr_[c_thread * k + tid_x];
    U_shm[(tid_y + 4) * k + tid_x] = sum2 * dHdr_[(c_thread + 4) * k + tid_x];
    __syncthreads();

    int k_reduce = k / 2;
    tid_y = threadIdx.x / k_reduce;
    tid_x = threadIdx.x % k_reduce;
    while (k_reduce != 0) {
      if (tid_x < k_reduce)
        U_shm[tid_y * k + tid_x] += U_shm[tid_y * k + tid_x + k_reduce];
      __syncthreads();
      k_reduce /= 2;
    }

    tid_y = threadIdx.x / 3;
    tid_x = threadIdx.x % 3;
    c_thread = c_index + tid_y;
    if (tid_y < 8 && c_thread < c) {
      F1_[c_thread * 3 + tid_x] =
          U_shm[tid_y * k + 0] * M_shm[tid_y * d + tid_x + 1];
    }
    __syncthreads();
    c_index += 8;
  }
}

__global__ void kernel4_device(float *dEdS, float *M, float *dHdr, float *F1,
                               int *mask, int a, int b, int d, int c, int k)
{
}

template <typename Scalar>
void kernel4_GPU(Scalar *dEdS, Scalar *M, Scalar *U, Scalar *dHdr,
                 Scalar *drdrx, Scalar *F1, int *mask, int a, int b, int d,
                 int c, int k)
{
  if (d == DD && k <= 64 && sizeof(Scalar) == 8) {
    int shared_size = (8 * d + k * d + 8 * k) * sizeof(Scalar);
    kernel4_device<<<a * b, 32 * k / 8, shared_size>>>(dEdS, M, dHdr, F1, mask,
                                                       a, b, d, c, k);
  } else {
    kernel_U_GPU(dEdS, dHdr, U, a, b, d, c, k);
    kernel_F1_GPU(U, M, drdrx, F1, mask, a, b, c, d);
  }
  hipDeviceSynchronize();
}

template <typename Scalar>
void kernel_V_GPU(Scalar *dEdS, Scalar *H, Scalar *V, int a, int b, int d,
                  int c, int k)
{
  const int lda = d;
  const int ldb = k;
  const int ldc = d;
  const int stridea = d * k;
  const int strideb = c * k;
  const int stridec = d * c;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, d, c, k, &alpha,
                            dEdS, lda, stridea, H, ldb, strideb, &beta, V, ldc,
                            stridec, a * b);
  hipDeviceSynchronize();
}

template <typename Scalar>
__global__ void get_sign_square_device(Scalar *P, Scalar *S, int *sign, int n,
                                       int d)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  Scalar p;
  if (i < n) {
    p = P[i];
    S[i] = p * p;
    if (i % d == 0) { sign[i / d] = p < 0 ? -1 : 1; }
  }
}

template <typename Scalar>
__global__ void sqrt_sign_device(Scalar *G, int *sign, int n, int m)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) { G[i * m] = sqrt(G[i * m]) * sign[i]; }
}

__global__ void kernel_1_fused_device(double *P, double *T, double *G, int c,
                                      int d, int k, int m)
{
  __shared__ double T_shm[DD * MM];
  double *P_ = P + blockIdx.x * k * d;
  double *G_ = G + blockIdx.x * k * m;
  double temp;
  double P_reg[DD] = {0};
  double G_reg[MM] = {0};
  int sign;
  for (int ii = threadIdx.x; ii < DD * MM; ii += blockDim.x) {
    T_shm[ii] = T[ii];
  }
  __syncthreads();

#pragma unroll
  for (int ii = 0; ii < 5; ii++) {
    reinterpret_cast<double4 *>(P_reg)[ii] =
        reinterpret_cast<double4 *>(P_)[threadIdx.x * 5 + ii];
  }
  sign = P_reg[0] < 0 ? -1 : 1;

#pragma unroll
  for (int ii = 0; ii < DD; ii++) { P_reg[ii] *= P_reg[ii]; }
  for (int ii = 0; ii < m; ii++) {
    temp = 0;
#pragma unroll
    for (int jj = 0; jj < DD; jj++) { temp += P_reg[jj] * T_shm[jj * m + ii]; }
    G_reg[ii] = temp;
  }
  G_reg[0] = sqrt(G_reg[0]) * sign;
  reinterpret_cast<double4 *>(G_)[threadIdx.x] =
      reinterpret_cast<double4 *>(G_reg)[0];
}

__global__ void kernel_1_fused_device(float *P, float *T, float *G, int c,
                                      int d, int k, int m)
{
  __shared__ float T_shm[DD * MM];
  float *P_ = P + blockIdx.x * k * d;
  float *G_ = G + blockIdx.x * k * m;
  float temp;
  float P_reg[DD] = {0};
  float G_reg[MM] = {0};
  int sign;
  for (int ii = threadIdx.x; ii < DD * MM; ii += blockDim.x) {
    T_shm[ii] = T[ii];
  }
  __syncthreads();

#pragma unroll
  for (int ii = 0; ii < 5; ii++) {
    reinterpret_cast<float4 *>(P_reg)[ii] =
        reinterpret_cast<float4 *>(P_)[threadIdx.x * 5 + ii];
  }
  sign = P_reg[0] < 0 ? -1 : 1;

#pragma unroll
  for (int ii = 0; ii < DD; ii++) { P_reg[ii] *= P_reg[ii]; }
  for (int ii = 0; ii < m; ii++) {
    temp = 0;
#pragma unroll
    for (int jj = 0; jj < DD; jj++) { temp += P_reg[jj] * T_shm[jj * m + ii]; }
    G_reg[ii] = temp;
  }
  G_reg[0] = sqrt(G_reg[0]) * sign;
  reinterpret_cast<float4 *>(G_)[threadIdx.x] =
      reinterpret_cast<float4 *>(G_reg)[0];
}

template <typename Scalar>
void kernel_1_GPU(Scalar *M, Scalar *H, Scalar *P, Scalar *T, Scalar *G,
                  Scalar *S, int *sign, int a, int b, int c, int d, int k,
                  int m)
{
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, HIPBLAS_OP_N, HIPBLAS_OP_T, d, k, c, &alpha,
                            M, d, d * c, H, k, c * k, &beta, P, d, d * k,
                            a * b);
  if (m == MM && d == DD) {
    hipLaunchKernelGGL(kernel_1_fused_device, dim3(a * b), dim3(k), 0, 0, P, T,
                       G, c, d, k, m);
  } else {
    hipLaunchKernelGGL(get_sign_square_device,
                       dim3((a * b * k * d + K1_blockdim - 1) / K1_blockdim),
                       dim3(K1_blockdim), 0, 0, P, S, sign, a * b * k * d, d);
    gpublasgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, a * b * k, d, &alpha, T,
                m, S, d, &beta, G, m);
    hipLaunchKernelGGL(sqrt_sign_device,
                       dim3((a * b * k + maxThread - 1) / maxThread),
                       dim3(maxThread), 0, 0, G, sign, a * b * k, m);
  }
  hipDeviceSynchronize();
}

template <typename Scalar>
__global__ void
setup_tensors_device(const int *ilist, const int inum, int *fmap_type,
                     double *pos, int numneigh_max, int old_numneigh_max,
                     const int *numneigh, int *firstneigh, Scalar cutforcesq,
                     int *eltij_pos, int *i2row, Scalar *R, Scalar *drdrx,
                     Scalar *M, int *mask, int *ijlist, int alocal,
                     int neltypes, int max_moment, int d)
{
  int i, j, a, b, c;
  int jn, elti, eltj, index, numneigh_i;
  int neigh[8];
  double3 tmp, dij;
  double rijinv, rij, rij2;
  double x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz;
  double yyy, yyz, yzz, zzz;
  int ii = threadIdx.x + blockDim.x * blockIdx.x;

  if (ii < inum) {
    for (elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    i = ilist[ii];
    if (neltypes == 1) {
      a = i;
    } else {
      a = i2row[i];
    }
    elti = fmap_type[i];
    tmp = reinterpret_cast<double3 *>(pos)[i];
    numneigh_i = numneigh[i];
    for (jn = 0; jn < numneigh_i; jn++) {
      j = firstneigh[i * old_numneigh_max + jn];
      j &= NEIGHMASK;
      eltj = fmap_type[j];
      b = eltij_pos[elti * neltypes + eltj];
      c = neigh[b];
      if (c < numneigh_max) {
        dij = reinterpret_cast<double3 *>(pos)[j];
        dij.x -= tmp.x;
        dij.y -= tmp.y;
        dij.z -= tmp.z;
        rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
        if (rij2 < cutforcesq && rij2 > 1e-10) {
          rij = sqrt(rij2);
          index = a * neltypes * numneigh_max + b * numneigh_max + c;
          R[index] = rij;
          mask[index] = 1;
          M[index * d + 0] = 1.0;
          rijinv = 1.0 / rij;
          x = dij.x * rijinv;
          y = dij.y * rijinv;
          z = dij.z * rijinv;
          drdrx[index * 3 + 0] = x;
          drdrx[index * 3 + 1] = y;
          drdrx[index * 3 + 2] = z;
          reinterpret_cast<int2 *>(ijlist)[index] = make_int2(i, j);
          if (max_moment == 1) {
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
          } else if (max_moment == 2) {
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M[index * d + 4] = xx;
            M[index * d + 5] = xy;
            M[index * d + 6] = xz;
            M[index * d + 7] = yy;
            M[index * d + 8] = yz;
            M[index * d + 9] = zz;
          } else if (max_moment == 3) {
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M[index * d + 4] = xx;
            M[index * d + 5] = xy;
            M[index * d + 6] = xz;
            M[index * d + 7] = yy;
            M[index * d + 8] = yz;
            M[index * d + 9] = zz;
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
            M[index * d + 10] = xxx;
            M[index * d + 11] = xxy;
            M[index * d + 12] = xxz;
            M[index * d + 13] = xyy;
            M[index * d + 14] = xyz;
            M[index * d + 15] = xzz;
            M[index * d + 16] = yyy;
            M[index * d + 17] = yyz;
            M[index * d + 18] = yzz;
            M[index * d + 19] = zzz;
          } else if (max_moment == 4) {
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M[index * d + 4] = xx;
            M[index * d + 5] = xy;
            M[index * d + 6] = xz;
            M[index * d + 7] = yy;
            M[index * d + 8] = yz;
            M[index * d + 9] = zz;
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
            M[index * d + 10] = xxx;
            M[index * d + 11] = xxy;
            M[index * d + 12] = xxz;
            M[index * d + 13] = xyy;
            M[index * d + 14] = xyz;
            M[index * d + 15] = xzz;
            M[index * d + 16] = yyy;
            M[index * d + 17] = yyz;
            M[index * d + 18] = yzz;
            M[index * d + 19] = zzz;
            double xxxx = xxx * x;
            double xxxy = xxx * y;
            double xxxz = xxx * z;
            double xxyy = xxy * y;
            double xxyz = xxy * z;
            double xxzz = xxz * z;
            double xyyy = xyy * y;
            double xyyz = xyy * z;
            double xyzz = xyz * z;
            double xzzz = xzz * z;
            double yyyy = yyy * y;
            double yyyz = yyy * z;
            double yyzz = yyz * z;
            double yzzz = yzz * z;
            double zzzz = zzz * z;
            M[index * d + 20] = xxxx;
            M[index * d + 21] = xxxy;
            M[index * d + 22] = xxxz;
            M[index * d + 23] = xxyy;
            M[index * d + 24] = xxyz;
            M[index * d + 25] = xxzz;
            M[index * d + 26] = xyyy;
            M[index * d + 27] = xyyz;
            M[index * d + 28] = xyzz;
            M[index * d + 29] = xzzz;
            M[index * d + 30] = yyyy;
            M[index * d + 31] = yyyz;
            M[index * d + 32] = yyzz;
            M[index * d + 33] = yzzz;
            M[index * d + 34] = zzzz;
          } else if (max_moment == 5) {
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M[index * d + 4] = xx;
            M[index * d + 5] = xy;
            M[index * d + 6] = xz;
            M[index * d + 7] = yy;
            M[index * d + 8] = yz;
            M[index * d + 9] = zz;
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
            M[index * d + 10] = xxx;
            M[index * d + 11] = xxy;
            M[index * d + 12] = xxz;
            M[index * d + 13] = xyy;
            M[index * d + 14] = xyz;
            M[index * d + 15] = xzz;
            M[index * d + 16] = yyy;
            M[index * d + 17] = yyz;
            M[index * d + 18] = yzz;
            M[index * d + 19] = zzz;
            double xxxx = xxx * x;
            double xxxy = xxx * y;
            double xxxz = xxx * z;
            double xxyy = xxy * y;
            double xxyz = xxy * z;
            double xxzz = xxz * z;
            double xyyy = xyy * y;
            double xyyz = xyy * z;
            double xyzz = xyz * z;
            double xzzz = xzz * z;
            double yyyy = yyy * y;
            double yyyz = yyy * z;
            double yyzz = yyz * z;
            double yzzz = yzz * z;
            double zzzz = zzz * z;
            M[index * d + 20] = xxxx;
            M[index * d + 21] = xxxy;
            M[index * d + 22] = xxxz;
            M[index * d + 23] = xxyy;
            M[index * d + 24] = xxyz;
            M[index * d + 25] = xxzz;
            M[index * d + 26] = xyyy;
            M[index * d + 27] = xyyz;
            M[index * d + 28] = xyzz;
            M[index * d + 29] = xzzz;
            M[index * d + 30] = yyyy;
            M[index * d + 31] = yyyz;
            M[index * d + 32] = yyzz;
            M[index * d + 33] = yzzz;
            M[index * d + 34] = zzzz;
            double xxxxx = xxxx * x;
            double xxxxy = xxxx * y;
            double xxxxz = xxxx * z;
            double xxxyy = xxxy * y;
            double xxxyz = xxxy * z;
            double xxxzz = xxxz * z;
            double xxyyy = xxyy * y;
            double xxyyz = xxyy * z;
            double xxyzz = xxyz * z;
            double xxzzz = xxzz * z;
            double xyyyy = xyyy * y;
            double xyyyz = xyyy * z;
            double xyyzz = xyyz * z;
            double xyzzz = xyzz * z;
            double xzzzz = xzzz * z;
            double yyyyy = yyyy * y;
            double yyyyz = yyyy * z;
            double yyyzz = yyyz * z;
            double yyzzz = yyzz * z;
            double yzzzz = yzzz * z;
            double zzzzz = zzzz * z;
            M[index * d + 35] = xxxxx;
            M[index * d + 36] = xxxxy;
            M[index * d + 37] = xxxxz;
            M[index * d + 38] = xxxyy;
            M[index * d + 39] = xxxyz;
            M[index * d + 40] = xxxzz;
            M[index * d + 41] = xxyyy;
            M[index * d + 42] = xxyyz;
            M[index * d + 43] = xxyzz;
            M[index * d + 44] = xxzzz;
            M[index * d + 45] = xyyyy;
            M[index * d + 46] = xyyyz;
            M[index * d + 47] = xyyzz;
            M[index * d + 48] = xyzzz;
            M[index * d + 49] = xzzzz;
            M[index * d + 50] = yyyyy;
            M[index * d + 51] = yyyyz;
            M[index * d + 52] = yyyzz;
            M[index * d + 53] = yyzzz;
            M[index * d + 54] = yzzzz;
            M[index * d + 55] = zzzzz;
          }
          neigh[b]++;
        }
      }
    }
    for (int bb = 0; bb < neltypes; bb++) {
      mask[a * neltypes * numneigh_max + bb * numneigh_max] = neigh[bb];
    }
  }
}

template <typename Scalar>
__global__ void get_exact_cmax_device(int *ilist, int inum, int *fmap_type,
                                      double *pos, int *numneigh,
                                      int *firstneigh, int neltypes,
                                      Scalar cutforcesq, int old_numneigh_max,
                                      int *eltij_pos, int *numneigh_max_device)
{
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii < inum) {
    double3 tmp, dij;
    int neigh[8] = {0};
    for (int elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    int i = ilist[ii];
    int elti = fmap_type[i];
    tmp = reinterpret_cast<double3 *>(pos)[i];
    for (int jn = 0; jn < numneigh[i]; jn++) {
      int j = firstneigh[i * old_numneigh_max + jn];
      j &= NEIGHMASK;
      dij = reinterpret_cast<double3 *>(pos)[j];
      dij.x -= tmp.x;
      dij.y -= tmp.y;
      dij.z -= tmp.z;
      double rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
      if (rij2 < cutforcesq && rij2 > 1e-10) {
        int eltj = fmap_type[j];
        int b = eltij_pos[elti * neltypes + eltj];
        neigh[b]++;
      }
    }
    int tmp_max = 0;
    for (int b = 0; b < neltypes; b++) {
      if (neigh[b] > tmp_max) tmp_max = neigh[b];
    }
    numneigh_max_device[ii] = tmp_max;
  }
}

template <typename Scalar>
int setup_tensors_GPU(int *ilist, int inum, const int *type, const int *fmap,
                      double *pos, int old_numneigh_max, int *numneigh,
                      int **firstneigh, int *eltij_pos, int *i2row,
                      Scalar **R_GPU, Scalar **G_GPU, Scalar **dEdG_GPU,
                      Scalar **P_GPU, Scalar **dEdS_GPU, Scalar **M_GPU,
                      Scalar **H_GPU, Scalar **V_GPU, Scalar **dHdr_GPU,
                      Scalar **drdrx_GPU, Scalar **F1_GPU, Scalar **F2_GPU,
                      int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU,
                      double **pos_GPU, int **numneigh_GPU,
                      int **firstneigh_GPU, int **ijlist_GPU, int **i2row_GPU,
                      Scalar cutforcesq, int nmax, int alocal, int neltypes,
                      int d, int k, int m, bool use_exact_cmax)
{
  int numneigh_max = old_numneigh_max;

  size_t host_size;
  size_t gmem1_size;
  if (max_inum < inum || max_old_numneigh_max < old_numneigh_max ||
      max_nmax < nmax) {
    hipHostFree(firstneigh_host);
    hipFree(*firstneigh_GPU);
    hipFree(*pos_GPU);
    if (use_exact_cmax) { hipFree(numneigh_max_device); }

    max_inum = inum + MALLOC_BUFF;
    max_old_numneigh_max = old_numneigh_max;
    max_nmax = nmax;
    host_size = max_inum * old_numneigh_max + nmax;
    gmem1_size = max_inum * old_numneigh_max + nmax + max_inum * 3;

    hipHostMalloc(&firstneigh_host, sizeof(int) * host_size);
    fmap_type_host = firstneigh_host + max_inum * old_numneigh_max;

    hipMalloc(firstneigh_GPU, sizeof(int) * gmem1_size);
    *fmap_type_GPU = *firstneigh_GPU + max_inum * old_numneigh_max;
    *ilist_GPU = *fmap_type_GPU + nmax;
    *numneigh_GPU = *ilist_GPU + max_inum;
    *i2row_GPU = *numneigh_GPU + max_inum;
    hipMalloc(pos_GPU, sizeof(double) * nmax * 3);
    if (use_exact_cmax) {
      hipMalloc(&numneigh_max_device, sizeof(int) * (max_inum + 1));
    }
  }

#pragma omp parallel for
  for (int ii = 0; ii < nmax; ii++) {
    int a = type[ii];
    if (a <= neltypes && a >= 0) { fmap_type_host[ii] = fmap[a]; }
  }
#pragma omp parallel for
  for (int ii = 0; ii < inum; ii++) {
    memcpy(firstneigh_host + ii * old_numneigh_max, firstneigh[ii],
           sizeof(int) * numneigh[ii]);
  }
  hipMemcpy(*firstneigh_GPU, firstneigh_host, sizeof(int) * host_size,
                  hipMemcpyHostToDevice);
  
      hipMemcpy(*ilist_GPU, ilist, sizeof(int) * inum, hipMemcpyHostToDevice);
  hipMemcpy(*numneigh_GPU, numneigh, sizeof(int) * inum,
                  hipMemcpyHostToDevice);
  hipMemcpy(*pos_GPU, pos, sizeof(double) * nmax * 3,
                  hipMemcpyHostToDevice);
  if (neltypes > 1) {
    hipMemcpy(*i2row_GPU, i2row, sizeof(int) * inum,
                    hipMemcpyHostToDevice);
  }

  if (use_exact_cmax) {
    hipLaunchKernelGGL(
        get_exact_cmax_device, dim3((inum + GEC_blockdim - 1) / GEC_blockdim),
        dim3(GEC_blockdim), 0, 0, *ilist_GPU, inum, *fmap_type_GPU, *pos_GPU,
        *numneigh_GPU, *firstneigh_GPU, neltypes, cutforcesq, old_numneigh_max,
        eltij_pos, numneigh_max_device);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    hipcub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                              numneigh_max_device, numneigh_max_device + inum,
                              inum);
    hipMalloc(&d_temp_storage, temp_storage_bytes);
    hipcub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                              numneigh_max_device, numneigh_max_device + inum,
                              inum);
    hipMemcpy(&numneigh_max, numneigh_max_device + inum, sizeof(int) * 1,
                    hipMemcpyDeviceToHost);
    hipFree(d_temp_storage);
  }

  numneigh_max_global = numneigh_max;

  if (max_alocal < alocal || max_numneigh_max < numneigh_max) {
    hipFree(*R_GPU);
    hipFree(*mask_GPU);

    max_alocal = alocal + MALLOC_BUFF;
    max_numneigh_max = numneigh_max + MALLOC_BUFF;
    size_t R_size = max_alocal * neltypes * max_numneigh_max;
    size_t dEdS_size = max_alocal * neltypes * k * d;
    size_t M_size = max_alocal * neltypes * max_numneigh_max * d;
    size_t H_size = max_alocal * neltypes * max_numneigh_max * k;
    size_t P_size = dEdS_size;
    size_t G_size = max_alocal * neltypes * k * m;
    size_t V_size = max(M_size, P_size + G_size * 2);
    size_t dHdr_size = H_size;
    size_t drdrx_size = R_size * 3;
    size_t F_size = drdrx_size;
    size_t gmem2_size = R_size + M_size + dEdS_size + H_size + V_size +
        dHdr_size + drdrx_size + 2 * F_size;
    size_t ijlist_size = R_size * 2;
    size_t mask_size = R_size;
    size_t gmem3_size = ijlist_size + mask_size;

    hipMalloc(R_GPU, sizeof(Scalar) * gmem2_size);
    *M_GPU = *R_GPU + R_size;
    *dEdS_GPU = *M_GPU + M_size;
    globalPool = *dEdS_GPU;
    *H_GPU = *dEdS_GPU + dEdS_size;
    *V_GPU = *H_GPU + H_size;
    *P_GPU = *V_GPU;
    *G_GPU = *P_GPU + P_size;
    *dEdG_GPU = *G_GPU + G_size;
    *dHdr_GPU = *V_GPU + V_size;
    *drdrx_GPU = *dHdr_GPU + dHdr_size;
    *F1_GPU = *drdrx_GPU + drdrx_size;
    *F2_GPU = *F1_GPU + F_size;
    hipMalloc(mask_GPU, sizeof(int) * gmem3_size);
    *ijlist_GPU = *mask_GPU + mask_size;
  }
  hipMemset(*mask_GPU, 0, sizeof(int) * alocal * neltypes * numneigh_max);
  hipMemset(*R_GPU, 0, sizeof(Scalar) * alocal * neltypes * numneigh_max * (1 + d));

  setup_tensors_device<<<(inum + ST_blockdim - 1) / ST_blockdim, ST_blockdim>>>(
      *ilist_GPU, inum, *fmap_type_GPU, *pos_GPU, numneigh_max,
      old_numneigh_max, *numneigh_GPU, *firstneigh_GPU, cutforcesq, eltij_pos,
      *i2row_GPU, *R_GPU, *drdrx_GPU, *M_GPU, *mask_GPU, *ijlist_GPU, alocal,
      neltypes, m - 1, d);
  hipDeviceSynchronize();
  return numneigh_max;
}

template <typename Scalar>
void debug_array_GPU(int cdim, Scalar *G_GPU, Scalar *H_GPU, Scalar *M_GPU,
                     Scalar *R_GPU, int *mask_GPU, int *ijlist_GPU,
                     Scalar *dEdG_GPU)
{
  const int bdim = 1;
  const int kdim = 32;
  const int mdim = 4;
  const int ddim = 20;

  Scalar *storage = (Scalar *) malloc(sizeof(Scalar) * bdim * cdim * kdim);
  int n;

  n = 0;
  hipMemcpy(storage, G_GPU, sizeof(Scalar) * bdim * kdim * mdim,
            hipMemcpyDeviceToHost);
  FILE *fp = fopen("G.txt", "w");
  for (int i = 0; i < bdim * kdim * mdim / 8; i++) {
    for (int j = 0; j < 8; j++) {
      fprintf(fp, "% 16.8f ", storage[n]);
      n++;
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  n = 0;
  hipMemcpy(storage, dEdG_GPU, sizeof(Scalar) * bdim * kdim * mdim,
            hipMemcpyDeviceToHost);
  fp = fopen("dEdG.txt", "w");
  for (int i = 0; i < bdim * kdim * mdim / 8; i++) {
    for (int j = 0; j < 8; j++) {
      fprintf(fp, "% 16.8f ", storage[n]);
      n++;
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  n = 0;
  hipMemcpy(storage, H_GPU, sizeof(Scalar) * bdim * cdim * kdim,
            hipMemcpyDeviceToHost);
  fp = fopen("H.txt", "w");
  for (int i = 0; i < bdim * cdim * kdim / 8; i++) {
    for (int j = 0; j < 8; j++) {
      fprintf(fp, "% 16.8f ", storage[n]);
      n++;
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  n = 0;
  hipMemcpy(storage, M_GPU, sizeof(Scalar) * bdim * cdim * ddim,
            hipMemcpyDeviceToHost);
  fp = fopen("M.txt", "w");
  for (int i = 0; i < bdim * cdim * ddim / 8; i++) {
    for (int j = 0; j < 8; j++) {
      fprintf(fp, "% 16.8f ", storage[n]);
      n++;
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  n = 0;
  hipMemcpy(storage, R_GPU, sizeof(Scalar) * bdim * cdim,
            hipMemcpyDeviceToHost);
  int *istorage = (int *) malloc(sizeof(int) * bdim * cdim * 3);
  hipMemcpy(istorage, mask_GPU, sizeof(int) * bdim * cdim,
            hipMemcpyDeviceToHost);
  hipMemcpy(istorage + bdim * cdim, ijlist_GPU, sizeof(int) * bdim * cdim * 2,
            hipMemcpyDeviceToHost);
  fp = fopen("R.txt", "w");
  for (int i = 0; i < bdim; i++) {
    for (int j = 0; j < cdim; j++) {
      fprintf(fp, "b=%1d c=%2d R=% 16.8f mask=%d i=%5d j=%5d\n", i, j,
              storage[n], istorage[n], istorage[bdim * cdim + n * 2 + 0],
              istorage[bdim * cdim + n * 2 + 1]);
      n++;
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  free(istorage);
  free(storage);
}

template <typename Scalar>
void free_tensors_GPU(int *ilist_GPU, double *pos_GPU, int *numneigh_GPU,
                      int *firstneigh_GPU, int *ijlist_GPU, int *i2row_GPU,
                      int *eltij_pos_GPU, int *fmap_type_GPU, Scalar *R_GPU,
                      Scalar *G_GPU, Scalar *dEdG_GPU, Scalar *T_GPU,
                      Scalar *P_GPU, Scalar *dEdS_GPU, Scalar *M_GPU,
                      Scalar *H_GPU, Scalar *V_GPU, Scalar *dHdr_GPU,
                      Scalar *drdrx_GPU, int *mask_GPU, double *vatom_GPU)
{
  hipFree(pos_GPU);
  hipFree(firstneigh_GPU);
  hipFree(eltij_pos_GPU);
  hipFree(R_GPU);
  hipFree(mask_GPU);
  hipFree(vatom_GPU);
  hipFree(T_GPU);
  hipFree(firstneigh_host);
  hipFree(numneigh_max_device);
}

template <typename Scalar>
void setup_global_GPU(Scalar **T_GPU, Scalar *T, int **eltij_pos_GPU,
                      int *eltij_pos, int neltypes, int d, int m)
{
  hipMalloc(T_GPU, sizeof(Scalar) * d * m);
  hipMemcpy(*T_GPU, T, sizeof(Scalar) * d * m, hipMemcpyHostToDevice);
  hipMalloc(eltij_pos_GPU, sizeof(int) * neltypes * neltypes);
  hipMemcpy(*eltij_pos_GPU, eltij_pos, sizeof(int) * neltypes * neltypes,
                  hipMemcpyHostToDevice);
  hipblasCreate(&handle);
  firstneigh_host = nullptr;
  fmap_type_host = nullptr;
  numneigh_max_device = nullptr;
  globalPool = nullptr;
  max_alocal = 0;
  max_inum = 0;
  max_numneigh_max = 0;
  max_old_numneigh_max = 0;
  max_nmax = 0;
}

template <typename Scalar>
__global__ void sum_forces_reduce_device(int n, double *f, double *vatom,
                                         Scalar *F1, Scalar *F2, Scalar *R,
                                         Scalar *drdrx, int *ijlist, int *mask,
                                         double scale, int blockmax)
{
  typedef hipcub::BlockReduce<double, SF_blockdim> blockreduce;
  int ii = threadIdx.x + blockmax * blockIdx.x;
  int i, j, x;
  double df[3] = {0}, virial[6] = {0};
  double fi[3] = {0}, vi[6] = {0};
  __shared__ typename blockreduce::TempStorage temp_storage1;
  __shared__ typename blockreduce::TempStorage temp_storage2;
  __shared__ typename blockreduce::TempStorage temp_storage3;
  __shared__ typename blockreduce::TempStorage temp_storage4;
  __shared__ typename blockreduce::TempStorage temp_storage5;
  __shared__ typename blockreduce::TempStorage temp_storage6;
  __shared__ typename blockreduce::TempStorage temp_storage7;
  __shared__ typename blockreduce::TempStorage temp_storage8;
  __shared__ typename blockreduce::TempStorage temp_storage9;

  if (threadIdx.x < blockmax && mask[ii] != 0) {
    j = ijlist[ii * 2 + 1];
    for (x = 0; x < 3; x++) {
      df[x] = (F1[ii * 3 + x] + F2[ii * 3 + x]) * scale;
      atomicAdd(f + j * 3 + x, -df[x]);
    }
    if (vatom) {
      Scalar r = R[ii];
      Scalar dr[3];
      dr[0] = drdrx[ii * 3];
      dr[1] = drdrx[ii * 3 + 1];
      dr[2] = drdrx[ii * 3 + 2];
      virial[0] = -dr[0] * df[0] * r;
      virial[1] = -dr[1] * df[1] * r;
      virial[2] = -dr[2] * df[2] * r;
      virial[3] = -dr[0] * df[1] * r;
      virial[4] = -dr[0] * df[2] * r;
      virial[5] = -dr[1] * df[2] * r;
      atomicAdd(vatom + j * 6, 0.5 * virial[0]);
      atomicAdd(vatom + j * 6 + 1, 0.5 * virial[1]);
      atomicAdd(vatom + j * 6 + 2, 0.5 * virial[2]);
      atomicAdd(vatom + j * 6 + 3, 0.5 * virial[3]);
      atomicAdd(vatom + j * 6 + 4, 0.5 * virial[4]);
      atomicAdd(vatom + j * 6 + 5, 0.5 * virial[5]);
    }
  }

  fi[0] = blockreduce(temp_storage1).Reduce(df[0], hipcub::Sum());
  fi[1] = blockreduce(temp_storage2).Reduce(df[1], hipcub::Sum());
  fi[2] = blockreduce(temp_storage3).Reduce(df[2], hipcub::Sum());
  if (threadIdx.x == 0) {
    i = ijlist[ii * 2];
    atomicAdd(f + i * 3, fi[0]);
    atomicAdd(f + i * 3 + 1, fi[1]);
    atomicAdd(f + i * 3 + 2, fi[2]);
  }
  if (vatom) {
    vi[0] = blockreduce(temp_storage4).Reduce(virial[0], hipcub::Sum());
    vi[1] = blockreduce(temp_storage5).Reduce(virial[1], hipcub::Sum());
    vi[2] = blockreduce(temp_storage6).Reduce(virial[2], hipcub::Sum());
    vi[3] = blockreduce(temp_storage7).Reduce(virial[3], hipcub::Sum());
    vi[4] = blockreduce(temp_storage8).Reduce(virial[4], hipcub::Sum());
    vi[5] = blockreduce(temp_storage9).Reduce(virial[5], hipcub::Sum());
    if (threadIdx.x == 0) {
      atomicAdd(vatom + i * 6, 0.5 * vi[0]);
      atomicAdd(vatom + i * 6 + 1, 0.5 * vi[1]);
      atomicAdd(vatom + i * 6 + 2, 0.5 * vi[2]);
      atomicAdd(vatom + i * 6 + 3, 0.5 * vi[3]);
      atomicAdd(vatom + i * 6 + 4, 0.5 * vi[4]);
      atomicAdd(vatom + i * 6 + 5, 0.5 * vi[5]);
    }
  }
}

template <typename Scalar>
__global__ void sum_forces_device(int n, double *f, double *vatom, Scalar *F1,
                                  Scalar *F2, Scalar *R, Scalar *drdrx,
                                  int *ijlist, int *mask, double scale)
{
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j, x;
  double df[3], virial[6];
  if (ii < n && mask[ii] != 0) {
    i = ijlist[ii * 2];
    j = ijlist[ii * 2 + 1];
    if (i >= 0 && j >= 0) {
      for (x = 0; x < 3; x++) {
        df[x] = (F1[ii * 3 + x] + F2[ii * 3 + x]) * scale;
        atomicAdd(f + i * 3 + x, df[x]);
        atomicAdd(f + j * 3 + x, -df[x]);
      }
      if (vatom) {
        Scalar r = R[ii];
        Scalar dr[3];
        dr[0] = drdrx[ii * 3];
        dr[1] = drdrx[ii * 3 + 1];
        dr[2] = drdrx[ii * 3 + 2];
        virial[0] = -dr[0] * df[0] * r;
        virial[1] = -dr[1] * df[1] * r;
        virial[2] = -dr[2] * df[2] * r;
        virial[3] = -dr[0] * df[1] * r;
        virial[4] = -dr[0] * df[2] * r;
        virial[5] = -dr[1] * df[2] * r;
        atomicAdd(vatom + i * 6, 0.5 * virial[0]);
        atomicAdd(vatom + i * 6 + 1, 0.5 * virial[1]);
        atomicAdd(vatom + i * 6 + 2, 0.5 * virial[2]);
        atomicAdd(vatom + i * 6 + 3, 0.5 * virial[3]);
        atomicAdd(vatom + i * 6 + 4, 0.5 * virial[4]);
        atomicAdd(vatom + i * 6 + 5, 0.5 * virial[5]);
        atomicAdd(vatom + j * 6, 0.5 * virial[0]);
        atomicAdd(vatom + j * 6 + 1, 0.5 * virial[1]);
        atomicAdd(vatom + j * 6 + 2, 0.5 * virial[2]);
        atomicAdd(vatom + j * 6 + 3, 0.5 * virial[3]);
        atomicAdd(vatom + j * 6 + 4, 0.5 * virial[4]);
        atomicAdd(vatom + j * 6 + 5, 0.5 * virial[5]);
      }
    }
  }
}

template <typename Scalar>
void sum_forces_GPU(int n, Scalar *F1_GPU, Scalar *F2_GPU, Scalar *R_GPU,
                    Scalar *drdrx_GPU, int *ijlist_GPU, int *mask_GPU,
                    double *f_GPU, double **vatom_GPU, double *f,
                    double **vatom, double scale, int nmax, int nelt, int cmax)
{
  hipMemset(f_GPU, 0, sizeof(double) * nmax * 3);
  if (vatom) {
    hipFree(*vatom_GPU);
    hipMalloc(vatom_GPU, sizeof(double) * nmax * 6);
  }
  if (nelt * cmax <= SF_blockdim) {
    hipLaunchKernelGGL(sum_forces_reduce_device, dim3(n), dim3(SF_blockdim), 0,
                       0, n, f_GPU, *vatom_GPU, F1_GPU, F2_GPU, R_GPU,
                       drdrx_GPU, ijlist_GPU, mask_GPU, scale, nelt * cmax);
  } else {
    hipLaunchKernelGGL(sum_forces_device, dim3((n * nelt * cmax + 7) / 8),
                       dim3(8), 0, 0, n * nelt * cmax, f_GPU, *vatom_GPU,
                       F1_GPU, F2_GPU, R_GPU, drdrx_GPU, ijlist_GPU, mask_GPU,
                       scale);
  }
  hipDeviceSynchronize();
  hipMemcpy(f, f_GPU, sizeof(double) * nmax * 3, hipMemcpyDeviceToHost);
  if (vatom) {
    hipMemcpy(*vatom, *vatom_GPU, sizeof(double) * nmax * 6,
              hipMemcpyDeviceToHost);
  }
}

template void kernel_fused_F2_GPU<double>(double *V, double *R, double *drdrx,
                                          double *F2, int n, int d, int m);
template void kernel_fused_F2_GPU<float>(float *V, float *R, float *drdrx,
                                         float *F2, int n, int d, int m);
template void kernel2_GPU<double>(double *G, double *T, double *dEdG, double *P,
                                  double *dEdS, int m, int d, int k, int b,
                                  int a);
template void kernel2_GPU<float>(float *G, float *T, float *dEdG, float *P,
                                 float *dEdS, int m, int d, int k, int b,
                                 int a);
template void kernel4_GPU<float>(float *dEdS, float *M, float *U, float *dHdr,
                                 float *drdrx, float *F1, int *mask, int a,
                                 int b, int d, int c, int k);
template void kernel4_GPU<double>(double *dEdS, double *M, double *U,
                                  double *dHdr, double *drdrx, double *F1,
                                  int *mask, int a, int b, int d, int c, int k);
template void kernel_V_GPU<double>(double *dEdS, double *H, double *V, int a,
                                   int b, int d, int c, int k);
template void kernel_V_GPU<float>(float *dEdS, float *H, float *V, int a, int b,
                                  int d, int c, int k);
template void kernel_1_GPU<double>(double *M, double *H, double *P, double *T,
                                   double *G, double *S, int *sign, int a,
                                   int b, int c, int d, int k, int m);
template void kernel_1_GPU<float>(float *M, float *H, float *P, float *T,
                                  float *G, float *S, int *sign, int a, int b,
                                  int c, int d, int k, int m);
template int setup_tensors_GPU<double>(
    int *ilist, int inum, const int *type, const int *fmap, double *pos,
    int old_numneigh_max, int *numneigh, int **firstneigh, int *eltij_pos,
    int *i2row, double **R_GPU, double **G_GPU, double **dEdG_GPU,
    double **P_GPU, double **dEdS_GPU, double **M_GPU, double **H_GPU,
    double **V_GPU, double **dHdr_GPU, double **drdrx_GPU, double **F1_GPU,
    double **F2_GPU, int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU,
    double **pos_GPU, int **numneigh_GPU, int **firstneigh_GPU,
    int **ijlist_GPU, int **i2row_GPU, double cutforcesq, int nmax, int alocal,
    int neltypes, int d, int k, int m, bool use_exact_cmax);
template int setup_tensors_GPU<float>(
    int *ilist, int inum, const int *type, const int *fmap, double *pos,
    int old_numneigh_max, int *numneigh, int **firstneigh, int *eltij_pos,
    int *i2row, float **R_GPU, float **G_GPU, float **dEdG_GPU, float **P_GPU,
    float **dEdS_GPU, float **M_GPU, float **H_GPU, float **V_GPU,
    float **dHdr_GPU, float **drdrx_GPU, float **F1_GPU, float **F2_GPU,
    int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU, double **pos_GPU,
    int **numneigh_GPU, int **firstneigh_GPU, int **ijlist_GPU, int **i2row_GPU,
    float cutforcesq, int nmax, int alocal, int neltypes, int d, int k, int m,
    bool use_exact_cmax);
template void
free_tensors_GPU<double>(int *ilist_GPU, double *pos_GPU, int *numneigh_GPU,
                         int *firstneigh_GPU, int *ijlist_GPU, int *i2row_GPU,
                         int *eltij_pos_GPU, int *fmap_type_GPU, double *R_GPU,
                         double *G_GPU, double *dEdG_GPU, double *T_GPU,
                         double *P_GPU, double *dEdS_GPU, double *M_GPU,
                         double *H_GPU, double *V_GPU, double *dHdr_GPU,
                         double *drdrx_GPU, int *mask_GPU, double *vatom_GPU);
template void free_tensors_GPU<float>(
    int *ilist_GPU, double *pos_GPU, int *numneigh_GPU, int *firstneigh_GPU,
    int *ijlist_GPU, int *i2row_GPU, int *eltij_pos_GPU, int *fmap_type_GPU,
    float *R_GPU, float *G_GPU, float *dEdG_GPU, float *T_GPU, float *P_GPU,
    float *dEdS_GPU, float *M_GPU, float *H_GPU, float *V_GPU, float *dHdr_GPU,
    float *drdrx_GPU, int *mask_GPU, double *vatom_GPU);
template void setup_global_GPU<float>(float **T_GPU, float *T,
                                      int **eltij_pos_GPU, int *eltij_pos,
                                      int neltypes, int d, int m);
template void setup_global_GPU<double>(double **T_GPU, double *T,
                                       int **eltij_pos_GPU, int *eltij_pos,
                                       int neltypes, int d, int m);
template void sum_forces_GPU<float>(int n, float *F1_GPU, float *F2_GPU,
                                    float *R_GPU, float *drdrx_GPU,
                                    int *ijlist_GPU, int *mask_GPU,
                                    double *f_GPU, double **vatom_GPU,
                                    double *f, double **vatom, double scale,
                                    int nmax, int nelt, int cmax);
template void sum_forces_GPU<double>(int n, double *F1_GPU, double *F2_GPU,
                                     double *R_GPU, double *drdrx_GPU,
                                     int *ijlist_GPU, int *mask_GPU,
                                     double *f_GPU, double **vatom_GPU,
                                     double *f, double **vatom, double scale,
                                     int nmax, int nelt, int cmax);

template void debug_array_GPU<double>(int cdim, double *G_GPU, double *H_GPU,
                                      double *M_GPU, double *R_GPU,
                                      int *mask_GPU, int *ijlist_GPU,
                                      double *dEdG_GPU);
template void debug_array_GPU<float>(int cdim, float *G_GPU, float *H_GPU,
                                     float *M_GPU, float *R_GPU, int *mask_GPU,
                                     int *ijlist_GPU, float *dEdG_GPU);
