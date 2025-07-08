#include "common.cuh"
#include "cub/cub.cuh"
#include "math.cuh"
#include <stdio.h>

#define NEIGHMASK 0x1FFFFFFF
#define KK 128
#define DD 20
#define MM 4
#define CC 300
#define F1_blockdim 32
#define F2_blockdim 128
#define K1_blockdim 128
#define ST_blockdim 4
#define GEC_blockdim 16
#define SF_blockdim 256

int *firstneigh_global;
int *fmap_type_global;
int *numneigh_max_device;
void *globalPool;
int max_alocal;
int max_inum;
int max_numneigh_max;
int numneigh_max_global;
int max_old_numneigh_max;
int max_nmax;
int maxThread;
cublasHandle_t handle;

void memory_GPU(int mpiid, FILE *screen, FILE *logfile)
{
#if defined(GPU_DEBUG)
  size_t total, free;
  CHECK(cudaMemGetInfo(&free, &total));

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
  cudaDeviceProp prop;
  int deviceId, deviceCount;

  CHECK(cudaGetDeviceCount(&deviceCount));
  deviceId = (mpiid + 1) % deviceCount;
  CHECK(cudaSetDevice(deviceId));
  CHECK(cudaGetDeviceProperties(&prop, deviceId));
  maxThread = prop.maxThreadsDim[0];

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
                                 Scalar *F1, int *mask, int n, int k)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n && mask[i] != 0) {
    Scalar tmp = 0;
    for (int j = 0; j < k; j++) { tmp += U[i * k + j] * dHdr[i * k + j]; }
    F1[i * 3 + 0] += drdrx[i * 3 + 0] * tmp;
    F1[i * 3 + 1] += drdrx[i * 3 + 1] * tmp;
    F1[i * 3 + 2] += drdrx[i * 3 + 2] * tmp;
  }
}

template <typename Scalar>
void kernel_F1_GPU(Scalar *U, Scalar *dHdr, Scalar *drdrx, Scalar *F1,
                   int *mask, int a, int b, int c, int k)
{
  kernel_F1_device<<<(a * b * c + F1_blockdim - 1) / F1_blockdim,
                      F1_blockdim>>>(U, dHdr, drdrx, F1, mask, a * b * c, k);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

template <typename Scalar>
__global__ void kernel_F2_device(Scalar *V, Scalar *R, Scalar *dMdrx,
                                 Scalar *F2, int n, int d)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  Scalar rij;
  Scalar rijinv;
  if (i < n && R[i] != 0) {
    rij = R[i];
    rijinv = 1.0 / rij;
    Scalar *V_ = V + i * d;
    Scalar *dM = dMdrx + i * d * 3;
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
void kernel_fused_F2_GPU(Scalar *V, Scalar *R, Scalar *drdrx, Scalar *dMdrx,
                         Scalar *F2, int n, int d, int m)
{
  kernel_F2_device<<<(n + F2_blockdim - 1) / F2_blockdim, F2_blockdim>>>(
      V, R, dMdrx, F2, n, d);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
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

template <typename Scalar>
void kernel2_GPU(Scalar *G, Scalar *T, Scalar *dEdG, Scalar *P, Scalar *dEdS,
                 int m, int d, int k, int b, int a)
{
  Scalar alpha = 2.0, beta = 0.0;
  kernel_dEdG_device<<<(a * b * k + maxThread - 1) / maxThread, maxThread>>>(
      G, dEdG, m, a * b * k * m);
  gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d, a * b * k, m, &alpha, T, m,
              dEdG, m, &beta, dEdS, d);
  vmul_device<<<(a * b * k * d + maxThread - 1) / maxThread, maxThread>>>(
      dEdS, P, a * b * k * d);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

template <typename Scalar>
void kernel_U_GPU(Scalar *dEdS, Scalar *M, Scalar *U, int a, int b, int d, int c,
                  int k)
{
  const int lda = d;
  const int ldb = d;
  const int ldc = k;
  const int stridea = d * k;
  const int strideb = c * d;
  const int stridec = k * c;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, c, d, &alpha,
                            dEdS, lda, stridea, M, ldb, strideb, &beta, U, ldc,
                            stridec, a * b);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

template <typename Scalar>
void kernel_V_GPU(Scalar *dEdS, Scalar *H, Scalar *V, int a, int b, int d, int c,
                  int k)
{
  const int lda = d;
  const int ldb = k;
  const int ldc = d;
  const int stridea = d * k;
  const int strideb = c * k;
  const int stridec = d * c;
  Scalar alpha = 1.0;
  Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, c, k, &alpha,
                            dEdS, lda, stridea, H, ldb, strideb, &beta, V, ldc,
                            stridec, a * b);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
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

template <typename Scalar>
void kernel_1_GPU(Scalar *M, Scalar *H, Scalar *P, Scalar *T, Scalar *G,
                  Scalar *S, int *sign, int a, int b, int c, int d, int k,
                  int m)
{
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  gpublasgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, k, c, &alpha,
                            M, d, d * c, H, k, c * k, &beta, P, d, d * k,
                            a * b);
  get_sign_square_device<<<(a * b * k * d + K1_blockdim - 1) / K1_blockdim,
                           K1_blockdim>>>(P, S, sign, a * b * k * d, d);
  gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, a * b * k, d, &alpha, T, m,
              S, d, &beta, G, m);
  sqrt_sign_device<<<(a * b * k + maxThread - 1) / maxThread, maxThread>>>(
      G, sign, a * b * k, m);

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

template <typename Scalar>
__global__ void
setup_tensors_device(const int *ilist, const int inum, int *fmap_type,
                     double *pos, int numneigh_max, int old_numneigh_max,
                     const int *numneigh, int *firstneigh, Scalar cutforcesq,
                     int *eltij_pos, int *i2row, Scalar *R, Scalar *drdrx,
                     Scalar *M, Scalar *dMdrx, int *mask, int *ijlist,
                     int alocal, int neltypes, int max_moment, int d)
{
  int i, j, a, b, c;
  int jn, elti, eltj, index, numneigh_i;
  int neigh[8];
  double3 tmp, dij;
  double rijinv, rij, rij2;
  Scalar x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz;
  Scalar yyy, yyz, yzz, zzz;
  Scalar xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy,
         yyyz, yyzz, yzzz, zzzz;
  Scalar xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz,
         xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz, yyzzz, yzzzz,
         zzzzz;
  Scalar xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz,
         xxxyzz, xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy,
         xyyyyz, xyyyzz, xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz,
         yyyzzz, yyzzzz, yzzzzz, zzzzzz;
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
      dij = reinterpret_cast<double3 *>(pos)[j];
      dij.x -= tmp.x;
      dij.y -= tmp.y;
      dij.z -= tmp.z;
      rij2 = dij.x * dij.x + dij.y * dij.y + dij.z * dij.z;
      if (rij2 < cutforcesq && rij2 > 1e-10) {
        eltj = fmap_type[j];
        b = eltij_pos[elti * neltypes + eltj];
        c = neigh[b];
        if (c < numneigh_max) {
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
          dMdrx[index * 3 * d + 0 * d + 0] = 0;
          dMdrx[index * 3 * d + 1 * d + 0] = 0;
          dMdrx[index * 3 * d + 2 * d + 0] = 0;
          reinterpret_cast<int2 *>(ijlist)[index] = make_int2(i, j);
          if (max_moment > 0) {
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M[index * d + 1] = x;
            M[index * d + 2] = y;
            M[index * d + 3] = z;
            dMdrx[index * 3 * d + 0 * d + 1] = -rijinv * (xx - 1);
            dMdrx[index * 3 * d + 1 * d + 1] = -rijinv * xy;
            dMdrx[index * 3 * d + 2 * d + 1] = -rijinv * xz;
            dMdrx[index * 3 * d + 0 * d + 2] = -rijinv * xy;
            dMdrx[index * 3 * d + 1 * d + 2] = -rijinv * (yy - 1);
            dMdrx[index * 3 * d + 2 * d + 2] = -rijinv * yz;
            dMdrx[index * 3 * d + 0 * d + 3] = -rijinv * xz;
            dMdrx[index * 3 * d + 1 * d + 3] = -rijinv * yz;
            dMdrx[index * 3 * d + 2 * d + 3] = -rijinv * (zz - 1);
            if (max_moment > 1) {
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
              M[index * d + 4] = xx;
              M[index * d + 5] = xy;
              M[index * d + 6] = xz;
              M[index * d + 7] = yy;
              M[index * d + 8] = yz;
              M[index * d + 9] = zz;
              dMdrx[index * 3 * d + 0 * d + 4] = -rijinv * (2 * xxx - 2 * x);
              dMdrx[index * 3 * d + 1 * d + 4] = -rijinv * 2 * xxy;
              dMdrx[index * 3 * d + 2 * d + 4] = -rijinv * 2 * xxz;
              dMdrx[index * 3 * d + 0 * d + 5] = -rijinv * (2 * xxy - y);
              dMdrx[index * 3 * d + 1 * d + 5] = -rijinv * (2 * xyy - x);
              dMdrx[index * 3 * d + 2 * d + 5] = -rijinv * 2 * xyz;
              dMdrx[index * 3 * d + 0 * d + 6] = -rijinv * (2 * xxz - z);
              dMdrx[index * 3 * d + 1 * d + 6] = -rijinv * 2 * xyz;
              dMdrx[index * 3 * d + 2 * d + 6] = -rijinv * (2 * xzz - x);
              dMdrx[index * 3 * d + 0 * d + 7] = -rijinv * 2 * xyy;
              dMdrx[index * 3 * d + 1 * d + 7] = -rijinv * (2 * yyy - 2 * y);
              dMdrx[index * 3 * d + 2 * d + 7] = -rijinv * 2 * yyz;
              dMdrx[index * 3 * d + 0 * d + 8] = -rijinv * 2 * xyz;
              dMdrx[index * 3 * d + 1 * d + 8] = -rijinv * (2 * yyz - z);
              dMdrx[index * 3 * d + 2 * d + 8] = -rijinv * (2 * yzz - y);
              dMdrx[index * 3 * d + 0 * d + 9] = -rijinv * 2 * xzz;
              dMdrx[index * 3 * d + 1 * d + 9] = -rijinv * 2 * yzz;
              dMdrx[index * 3 * d + 2 * d + 9] = -rijinv * (2 * zzz - 2 * z);
              if (max_moment > 2) {
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
                dMdrx[index * 3 * d + 0 * d + 10] = -rijinv * (3 * xxxx - 3 * xx);
                dMdrx[index * 3 * d + 1 * d + 10] = -rijinv * 3 * xxxy;
                dMdrx[index * 3 * d + 2 * d + 10] = -rijinv * 3 * xxxz;
                dMdrx[index * 3 * d + 0 * d + 11] = -rijinv * (3 * xxxy - 2 * xy);
                dMdrx[index * 3 * d + 1 * d + 11] = -rijinv * (3 * xxyy - xx);
                dMdrx[index * 3 * d + 2 * d + 11] = -rijinv * 3 * xxyz;
                dMdrx[index * 3 * d + 0 * d + 12] = -rijinv * (3 * xxxz - 2 * xz);
                dMdrx[index * 3 * d + 1 * d + 12] = -rijinv * 3 * xxyz;
                dMdrx[index * 3 * d + 2 * d + 12] = -rijinv * (3 * xxzz - xx);
                dMdrx[index * 3 * d + 0 * d + 13] = -rijinv * (3 * xxyy - yy);
                dMdrx[index * 3 * d + 1 * d + 13] = -rijinv * (3 * xyyy - 2 * xy);
                dMdrx[index * 3 * d + 2 * d + 13] = -rijinv * 3 * xyyz;
                dMdrx[index * 3 * d + 0 * d + 14] = -rijinv * (3 * xxyz - yz);
                dMdrx[index * 3 * d + 1 * d + 14] = -rijinv * (3 * xyyz - xz);
                dMdrx[index * 3 * d + 2 * d + 14] = -rijinv * (3 * xyzz - xy);
                dMdrx[index * 3 * d + 0 * d + 15] = -rijinv * (3 * xxzz - zz);
                dMdrx[index * 3 * d + 1 * d + 15] = -rijinv * 3 * xyzz;
                dMdrx[index * 3 * d + 2 * d + 15] = -rijinv * (3 * xzzz - 2 * xz);
                dMdrx[index * 3 * d + 0 * d + 16] = -rijinv * 3 * xyyy;
                dMdrx[index * 3 * d + 1 * d + 16] = -rijinv * (3 * yyyy - 3 * yy);
                dMdrx[index * 3 * d + 2 * d + 16] = -rijinv * 3 * yyyz;
                dMdrx[index * 3 * d + 0 * d + 17] = -rijinv * 3 * xyyz;
                dMdrx[index * 3 * d + 1 * d + 17] = -rijinv * (3 * yyyz - 2 * yz);
                dMdrx[index * 3 * d + 2 * d + 17] = -rijinv * (3 * yyzz - yy);
                dMdrx[index * 3 * d + 0 * d + 18] = -rijinv * 3 * xyzz;
                dMdrx[index * 3 * d + 1 * d + 18] = -rijinv * (3 * yyzz - zz);
                dMdrx[index * 3 * d + 2 * d + 18] = -rijinv * (3 * yzzz - 2 * yz);
                dMdrx[index * 3 * d + 0 * d + 19] = -rijinv * 3 * xzzz;
                dMdrx[index * 3 * d + 1 * d + 19] = -rijinv * 3 * yzzz;
                dMdrx[index * 3 * d + 2 * d + 19] = -rijinv * (3 * zzzz - 3 * zz);
                if (max_moment > 3) {
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
                  dMdrx[index * 3 * d + 0 * d + 20] = -rijinv * (4 * xxxxx - 4 * xxx);
                  dMdrx[index * 3 * d + 1 * d + 20] = -rijinv * 4 * xxxxy;
                  dMdrx[index * 3 * d + 2 * d + 20] = -rijinv * 4 * xxxxz;
                  dMdrx[index * 3 * d + 0 * d + 21] = -rijinv * (4 * xxxxy - 3 * xxy);
                  dMdrx[index * 3 * d + 1 * d + 21] = -rijinv * (4 * xxxyy - xxx);
                  dMdrx[index * 3 * d + 2 * d + 21] = -rijinv * 4 * xxxyz;
                  dMdrx[index * 3 * d + 0 * d + 22] = -rijinv * (4 * xxxxz - 3 * xxz);
                  dMdrx[index * 3 * d + 1 * d + 22] = -rijinv * 4 * xxxyz;
                  dMdrx[index * 3 * d + 2 * d + 22] = -rijinv * (4 * xxxzz - xxx);
                  dMdrx[index * 3 * d + 0 * d + 23] = -rijinv * (4 * xxxyy - 2 * xyy);
                  dMdrx[index * 3 * d + 1 * d + 23] = -rijinv * (4 * xxyyy - 2 * xxy);
                  dMdrx[index * 3 * d + 2 * d + 23] = -rijinv * 4 * xxyyz;
                  dMdrx[index * 3 * d + 0 * d + 24] = -rijinv * (4 * xxxyz - 2 * xyz);
                  dMdrx[index * 3 * d + 1 * d + 24] = -rijinv * (4 * xxyyz - xxz);
                  dMdrx[index * 3 * d + 2 * d + 24] = -rijinv * (4 * xxyzz - xxy);
                  dMdrx[index * 3 * d + 0 * d + 25] = -rijinv * (4 * xxxzz - 2 * xzz);
                  dMdrx[index * 3 * d + 1 * d + 25] = -rijinv * 4 * xxyzz;
                  dMdrx[index * 3 * d + 2 * d + 25] = -rijinv * (4 * xxzzz - 2 * xxz);
                  dMdrx[index * 3 * d + 0 * d + 26] = -rijinv * (4 * xxyyy - yyy);
                  dMdrx[index * 3 * d + 1 * d + 26] = -rijinv * (4 * xyyyy - 3 * xyy);
                  dMdrx[index * 3 * d + 2 * d + 26] = -rijinv * 4 * xyyyz;
                  dMdrx[index * 3 * d + 0 * d + 27] = -rijinv * (4 * xxyyz - yyz);
                  dMdrx[index * 3 * d + 1 * d + 27] = -rijinv * (4 * xyyyz - 2 * xyz);
                  dMdrx[index * 3 * d + 2 * d + 27] = -rijinv * (4 * xyyzz - xyy);
                  dMdrx[index * 3 * d + 0 * d + 28] = -rijinv * (4 * xxyzz - yzz);
                  dMdrx[index * 3 * d + 1 * d + 28] = -rijinv * (4 * xyyzz - xzz);
                  dMdrx[index * 3 * d + 2 * d + 28] = -rijinv * (4 * xyzzz - 2 * xyz);
                  dMdrx[index * 3 * d + 0 * d + 29] = -rijinv * (4 * xxzzz - zzz);
                  dMdrx[index * 3 * d + 1 * d + 29] = -rijinv * 4 * xyzzz;
                  dMdrx[index * 3 * d + 2 * d + 29] = -rijinv * (4 * xzzzz - 3 * xzz);
                  dMdrx[index * 3 * d + 0 * d + 30] = -rijinv * 4 * xyyyy;
                  dMdrx[index * 3 * d + 1 * d + 30] = -rijinv * (4 * yyyyy - 4 * yyy);
                  dMdrx[index * 3 * d + 2 * d + 30] = -rijinv * 4 * yyyyz;
                  dMdrx[index * 3 * d + 0 * d + 31] = -rijinv * 4 * xyyyz;
                  dMdrx[index * 3 * d + 1 * d + 31] = -rijinv * (4 * yyyyz - 3 * yyz);
                  dMdrx[index * 3 * d + 2 * d + 31] = -rijinv * (4 * yyyzz - yyy);
                  dMdrx[index * 3 * d + 0 * d + 32] = -rijinv * 4 * xyyzz;
                  dMdrx[index * 3 * d + 1 * d + 32] = -rijinv * (4 * yyyzz - 2 * yzz);
                  dMdrx[index * 3 * d + 2 * d + 32] = -rijinv * (4 * yyzzz - 2 * yyz);
                  dMdrx[index * 3 * d + 0 * d + 33] = -rijinv * 4 * xyzzz;
                  dMdrx[index * 3 * d + 1 * d + 33] = -rijinv * (4 * yyzzz - zzz);
                  dMdrx[index * 3 * d + 2 * d + 33] = -rijinv * (4 * yzzzz - 3 * yzz);
                  dMdrx[index * 3 * d + 0 * d + 34] = -rijinv * 4 * xzzzz;
                  dMdrx[index * 3 * d + 1 * d + 34] = -rijinv * 4 * yzzzz;
                  dMdrx[index * 3 * d + 2 * d + 34] = -rijinv * (4 * zzzzz - 4 * zzz);
                  if (max_moment > 4) {
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
                    dMdrx[index * 3 * d + 0 * d + 35] = -rijinv * (5 * xxxxxx - 5 * xxxx);
                    dMdrx[index * 3 * d + 0 * d + 36] = -rijinv * (5 * xxxxxy - 4 * xxxy);
                    dMdrx[index * 3 * d + 0 * d + 37] = -rijinv * (5 * xxxxxz - 4 * xxxz);
                    dMdrx[index * 3 * d + 0 * d + 38] = -rijinv * (5 * xxxxyy - 3 * xxyy);
                    dMdrx[index * 3 * d + 0 * d + 39] = -rijinv * (5 * xxxxyz - 3 * xxyz);
                    dMdrx[index * 3 * d + 0 * d + 40] = -rijinv * (5 * xxxxzz - 3 * xxzz);
                    dMdrx[index * 3 * d + 0 * d + 41] = -rijinv * (5 * xxxyyy - 2 * xyyy);
                    dMdrx[index * 3 * d + 0 * d + 42] = -rijinv * (5 * xxxyyz - 2 * xyyz);
                    dMdrx[index * 3 * d + 0 * d + 43] = -rijinv * (5 * xxxyzz - 2 * xyzz);
                    dMdrx[index * 3 * d + 0 * d + 44] = -rijinv * (5 * xxxzzz - 2 * xzzz);
                    dMdrx[index * 3 * d + 0 * d + 45] = -rijinv * (5 * xxyyyy - yyyy);
                    dMdrx[index * 3 * d + 0 * d + 46] = -rijinv * (5 * xxyyyz - yyyz);
                    dMdrx[index * 3 * d + 0 * d + 47] = -rijinv * (5 * xxyyzz - yyzz);
                    dMdrx[index * 3 * d + 0 * d + 48] = -rijinv * (5 * xxyzzz - yzzz);
                    dMdrx[index * 3 * d + 0 * d + 49] = -rijinv * (5 * xxzzzz - zzzz);
                    dMdrx[index * 3 * d + 0 * d + 50] = -rijinv * 5 * xyyyyy;
                    dMdrx[index * 3 * d + 0 * d + 51] = -rijinv * 5 * xyyyyz;
                    dMdrx[index * 3 * d + 0 * d + 52] = -rijinv * 5 * xyyyzz;
                    dMdrx[index * 3 * d + 0 * d + 53] = -rijinv * 5 * xyyzzz;
                    dMdrx[index * 3 * d + 0 * d + 54] = -rijinv * 5 * xyzzzz;
                    dMdrx[index * 3 * d + 0 * d + 55] = -rijinv * 5 * xzzzzz;

                    dMdrx[index * 3 * d + 1 * d + 35] = -rijinv * 5 * xxxxxy;
                    dMdrx[index * 3 * d + 1 * d + 36] = -rijinv * (5 * xxxxyy - xxxx);
                    dMdrx[index * 3 * d + 1 * d + 37] = -rijinv * 5 * xxxxyz;
                    dMdrx[index * 3 * d + 1 * d + 38] = -rijinv * (5 * xxxyyy - 2 * xxxy);
                    dMdrx[index * 3 * d + 1 * d + 39] = -rijinv * (5 * xxxyyz - xxxz);
                    dMdrx[index * 3 * d + 1 * d + 40] = -rijinv * 5 * xxxyzz;
                    dMdrx[index * 3 * d + 1 * d + 41] = -rijinv * (5 * xxyyyy - 3 * xxyy);
                    dMdrx[index * 3 * d + 1 * d + 42] = -rijinv * (5 * xxyyyz - 2 * xxyz);
                    dMdrx[index * 3 * d + 1 * d + 43] = -rijinv * (5 * xxyyzz - xxzz);
                    dMdrx[index * 3 * d + 1 * d + 44] = -rijinv * 5 * xxyzzz;
                    dMdrx[index * 3 * d + 1 * d + 45] = -rijinv * (5 * xyyyyy - 4 * xyyy);
                    dMdrx[index * 3 * d + 1 * d + 46] = -rijinv * (5 * xyyyyz - 3 * xyyz);
                    dMdrx[index * 3 * d + 1 * d + 47] = -rijinv * (5 * xyyyzz - 2 * xyzz);
                    dMdrx[index * 3 * d + 1 * d + 48] = -rijinv * (5 * xyyzzz - xzzz);
                    dMdrx[index * 3 * d + 1 * d + 49] = -rijinv * 5 * xyzzzz;
                    dMdrx[index * 3 * d + 1 * d + 50] = -rijinv * (5 * yyyyyy - 5 * yyyy);
                    dMdrx[index * 3 * d + 1 * d + 51] = -rijinv * (5 * yyyyyz - 4 * yyyz);
                    dMdrx[index * 3 * d + 1 * d + 52] = -rijinv * (5 * yyyyzz - 3 * yyzz);
                    dMdrx[index * 3 * d + 1 * d + 53] = -rijinv * (5 * yyyzzz - 2 * yzzz);
                    dMdrx[index * 3 * d + 1 * d + 54] = -rijinv * (5 * yyzzzz - zzzz);
                    dMdrx[index * 3 * d + 1 * d + 55] = -rijinv * 5 * yzzzzz;

                    dMdrx[index * 3 * d + 2 * d + 35] = -rijinv * 5 * xxxxxz;
                    dMdrx[index * 3 * d + 2 * d + 36] = -rijinv * 5 * xxxxyz;
                    dMdrx[index * 3 * d + 2 * d + 37] = -rijinv * (5 * xxxxzz - xxxx);
                    dMdrx[index * 3 * d + 2 * d + 38] = -rijinv * 5 * xxxyyz;
                    dMdrx[index * 3 * d + 2 * d + 39] = -rijinv * (5 * xxxyzz - xxxy);
                    dMdrx[index * 3 * d + 2 * d + 40] = -rijinv * (5 * xxxzzz - 2 * xxxz);
                    dMdrx[index * 3 * d + 2 * d + 41] = -rijinv * 5 * xxyyyz;
                    dMdrx[index * 3 * d + 2 * d + 42] = -rijinv * (5 * xxyyzz - xxyy);
                    dMdrx[index * 3 * d + 2 * d + 43] = -rijinv * (5 * xxyzzz - 2 * xxyz);
                    dMdrx[index * 3 * d + 2 * d + 44] = -rijinv * (5 * xxzzzz - 3 * xxzz);
                    dMdrx[index * 3 * d + 2 * d + 45] = -rijinv * 5 * xyyyyz;
                    dMdrx[index * 3 * d + 2 * d + 46] = -rijinv * (5 * xyyyzz - xyyy);
                    dMdrx[index * 3 * d + 2 * d + 47] = -rijinv * (5 * xyyzzz - 2 * xyyz);
                    dMdrx[index * 3 * d + 2 * d + 48] = -rijinv * (5 * xyzzzz - 3 * xyzz);
                    dMdrx[index * 3 * d + 2 * d + 49] = -rijinv * (5 * xzzzzz - 4 * xzzz);
                    dMdrx[index * 3 * d + 2 * d + 50] = -rijinv * 5 * yyyyyz;
                    dMdrx[index * 3 * d + 2 * d + 51] = -rijinv * (5 * yyyyzz - yyyy);
                    dMdrx[index * 3 * d + 2 * d + 52] = -rijinv * (5 * yyyzzz - 2 * yyyz);
                    dMdrx[index * 3 * d + 2 * d + 53] = -rijinv * (5 * yyzzzz - 3 * yyzz);
                    dMdrx[index * 3 * d + 2 * d + 54] = -rijinv * (5 * yzzzzz - 4 * yzzz);
                    dMdrx[index * 3 * d + 2 * d + 55] = -rijinv * (5 * zzzzzz - 5 * zzzz);
                  }
                }
              }
            }
          } 
        }
        neigh[b]++;
      }
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
                      Scalar **drdrx_GPU, Scalar **F_GPU, Scalar **dMdrx_GPU,
                      int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU,
                      double **pos_GPU, int **numneigh_GPU,
                      int **firstneigh_GPU, int **ijlist_GPU, int **i2row_GPU,
                      Scalar cutforcesq, int nmax, int alocal, int neltypes,
                      int d, int k, int m, bool use_exact_cmax)
{
  int numneigh_max = old_numneigh_max;
  size_t firstneigh_size = inum * old_numneigh_max;
  size_t global_size = firstneigh_size + nmax;
  size_t int_size = firstneigh_size + nmax + inum * 3;
  if (max_inum < inum || max_old_numneigh_max < old_numneigh_max
                      || max_nmax < nmax) {
    CHECK(cudaFreeHost(firstneigh_global));
    CHECK(cudaFree(*firstneigh_GPU));
    CHECK(cudaFree(*pos_GPU));
    if (use_exact_cmax) { CHECK(cudaFree(numneigh_max_device)); }
    max_inum = inum;
    max_old_numneigh_max = old_numneigh_max;
    max_nmax = nmax;
    CHECK(cudaMallocHost(&firstneigh_global, sizeof(int) * global_size));
    fmap_type_global = firstneigh_global + firstneigh_size;
    CHECK(cudaMalloc(firstneigh_GPU, sizeof(int) * int_size));
    *fmap_type_GPU = *firstneigh_GPU + firstneigh_size;
    *ilist_GPU = *fmap_type_GPU + nmax;
    *numneigh_GPU = *ilist_GPU + inum;
    *i2row_GPU = *numneigh_GPU + inum;
    CHECK(cudaMalloc(pos_GPU, sizeof(double) * nmax * 3));
    if (use_exact_cmax) {
      CHECK(cudaMalloc(&numneigh_max_device, sizeof(int) * (inum + 1)));
    }
  }
#pragma omp parallel for
  for (int ii = 0; ii < nmax; ii++) {
    int a = type[ii];
    if (a <= neltypes && a >= 0) { fmap_type_global[ii] = fmap[a]; }
  }
#pragma omp parallel for
  for (int ii = 0; ii < inum; ii++) {
    memcpy(firstneigh_global + ii * old_numneigh_max, firstneigh[ii],
           sizeof(int) * numneigh[ii]);
  }
  CHECK(cudaMemcpy(*firstneigh_GPU, firstneigh_global,
                   sizeof(int) * global_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(*ilist_GPU, ilist, sizeof(int) * inum,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(*numneigh_GPU, numneigh, sizeof(int) * inum,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(*pos_GPU, pos, sizeof(double) * nmax * 3,
                   cudaMemcpyHostToDevice));                   
  if (neltypes > 1) {
    CHECK(cudaMemcpy(*i2row_GPU, i2row, sizeof(int) * inum,
                     cudaMemcpyHostToDevice));
  }

  if (use_exact_cmax) {
    get_exact_cmax_device<<<(inum + GEC_blockdim - 1) / GEC_blockdim,
                            GEC_blockdim>>>(
        *ilist_GPU, inum, *fmap_type_GPU, *pos_GPU, *numneigh_GPU,
        *firstneigh_GPU, neltypes, cutforcesq, old_numneigh_max, eltij_pos,
        numneigh_max_device);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           numneigh_max_device, numneigh_max_device + inum,
                           inum);
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           numneigh_max_device, numneigh_max_device + inum,
                           inum);
    CHECK(cudaMemcpy(&numneigh_max, numneigh_max_device + inum, sizeof(int) * 1,
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_temp_storage));
  }

  numneigh_max_global = numneigh_max;
  size_t R_size = alocal * neltypes * numneigh_max;
  size_t dEdS_size = alocal * neltypes * k * d;
  size_t M_size = alocal * neltypes * numneigh_max * d;
  size_t H_size = alocal * neltypes * numneigh_max * k;
  size_t P_size = dEdS_size;
  size_t G_size = alocal * neltypes * k * m;    
  size_t V_size = max(M_size, P_size + G_size * 2);
  size_t dHdr_size = H_size;
  size_t drdrx_size = R_size * 3;
  size_t F_size = drdrx_size;
  size_t scalar_size = R_size + M_size + dEdS_size + H_size + V_size + dHdr_size +
                drdrx_size + F_size;
  size_t ijlist_size = R_size * 2;
  size_t mask_size = R_size;
  int_size = ijlist_size + mask_size;
  if (max_alocal < alocal || max_numneigh_max < numneigh_max) {
    CHECK(cudaFree(*R_GPU));
    CHECK(cudaFree(*mask_GPU));
    CHECK(cudaFree(*dMdrx_GPU));
    max_alocal = alocal;
    max_numneigh_max = numneigh_max;
    CHECK(cudaMalloc(
        dMdrx_GPU, sizeof(Scalar) * alocal * neltypes * numneigh_max * d * 3));
    CHECK(cudaMalloc(R_GPU, sizeof(Scalar) * scalar_size));
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
    *F_GPU = *drdrx_GPU + drdrx_size;
    CHECK(cudaMalloc(mask_GPU, sizeof(int) * int_size));
    *ijlist_GPU = *mask_GPU + mask_size;
  }
  CHECK(cudaMemset(*mask_GPU, 0, sizeof(int) * mask_size));
  CHECK(cudaMemset(*R_GPU, 0, sizeof(Scalar) * (R_size + M_size)));

  setup_tensors_device<<<(inum + ST_blockdim - 1) / ST_blockdim, ST_blockdim>>>(
      *ilist_GPU, inum, *fmap_type_GPU, *pos_GPU, numneigh_max,
      old_numneigh_max, *numneigh_GPU, *firstneigh_GPU, cutforcesq, eltij_pos,
      *i2row_GPU, *R_GPU, *drdrx_GPU, *M_GPU, *dMdrx_GPU, *mask_GPU,
      *ijlist_GPU, alocal, neltypes, m - 1, d);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  return numneigh_max;
}

template <typename Scalar>
void free_tensors_GPU(int *ilist_GPU, double *pos_GPU, int *numneigh_GPU,
                      int *firstneigh_GPU, int *ijlist_GPU, int *i2row_GPU,
                      int *eltij_pos_GPU, int *fmap_type_GPU, Scalar *R_GPU,
                      Scalar *G_GPU, Scalar *dEdG_GPU, Scalar *T_GPU,
                      Scalar *P_GPU, Scalar *dEdS_GPU, Scalar *M_GPU,
                      Scalar *H_GPU, Scalar *V_GPU, Scalar *dHdr_GPU,
                      Scalar *drdrx_GPU, Scalar *F_GPU, Scalar *dMdrx_GPU,
                      int *mask_GPU, double *vatom_GPU)
{
  cudaFree(pos_GPU);
  cudaFree(firstneigh_GPU);
  cudaFree(eltij_pos_GPU);
  cudaFree(R_GPU);
  cudaFree(dMdrx_GPU);
  cudaFree(mask_GPU);
  cudaFree(vatom_GPU);
  cudaFree(T_GPU);
  cudaFree(firstneigh_global);
  cudaFree(numneigh_max_device);
}

template <typename Scalar>
void setup_global_GPU(Scalar **T_GPU, Scalar *T, int **eltij_pos_GPU,
                      int *eltij_pos, int neltypes, int d, int m)
{
  CHECK(cudaMalloc(T_GPU, sizeof(Scalar) * d * m));
  CHECK(cudaMemcpy(*T_GPU, T, sizeof(Scalar) * d * m, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(eltij_pos_GPU, sizeof(int) * neltypes * neltypes));
  CHECK(cudaMemcpy(*eltij_pos_GPU, eltij_pos, sizeof(int) * neltypes * neltypes,
                   cudaMemcpyHostToDevice));
  cublasCreate(&handle);
  firstneigh_global = nullptr;
  fmap_type_global = nullptr;
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
                                         Scalar *F, Scalar *R, Scalar *drdrx,
                                         int *ijlist, int *mask, double scale,
                                         int blockmax)
{
  typedef cub::BlockReduce<Scalar, SF_blockdim> blockreduce;
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
      df[x] = F[ii * 3 + x] * scale;
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

  fi[0] = blockreduce(temp_storage1).Reduce(df[0], cub::Sum());
  fi[1] = blockreduce(temp_storage2).Reduce(df[1], cub::Sum());
  fi[2] = blockreduce(temp_storage3).Reduce(df[2], cub::Sum());
  if (threadIdx.x == 0) {
    i = ijlist[ii * 2];
    atomicAdd(f + i * 3, fi[0]);
    atomicAdd(f + i * 3 + 1, fi[1]);
    atomicAdd(f + i * 3 + 2, fi[2]);
  }
  if (vatom) {
    vi[0] = blockreduce(temp_storage4).Reduce(virial[0], cub::Sum());
    vi[1] = blockreduce(temp_storage5).Reduce(virial[1], cub::Sum());
    vi[2] = blockreduce(temp_storage6).Reduce(virial[2], cub::Sum());
    vi[3] = blockreduce(temp_storage7).Reduce(virial[3], cub::Sum());
    vi[4] = blockreduce(temp_storage8).Reduce(virial[4], cub::Sum());
    vi[5] = blockreduce(temp_storage9).Reduce(virial[5], cub::Sum());
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
__global__ void sum_forces_device(int n, double *f, double *vatom, Scalar *F,
                                  Scalar *R, Scalar *drdrx, int *ijlist,
                                  int *mask, double scale)
{
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j, x;
  double df[3], virial[6];
  if (ii < n && mask[ii] != 0) {
    i = ijlist[ii * 2];
    j = ijlist[ii * 2 + 1];
    if (i >= 0 && j >= 0) {
      for (x = 0; x < 3; x++) {
        df[x] = F[ii * 3 + x] * scale;
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
void sum_forces_GPU(int n, Scalar *F_GPU, Scalar *R_GPU, Scalar *drdrx_GPU,
                    int *ijlist_GPU, int *mask_GPU, double *f_GPU,
                    double **vatom_GPU, double *f, double *vatom, double scale,
                    int nmax, int nelt, int cmax)
{
  CHECK(cudaMemset(f_GPU, 0, sizeof(double) * nmax * 3));
  if (vatom) {
    CHECK(cudaFree(*vatom_GPU));
    CHECK(cudaMalloc(vatom_GPU, sizeof(double) * nmax * 6));
  }
  if (nelt * cmax <= SF_blockdim) {
    sum_forces_reduce_device<<<n, SF_blockdim>>>(n, f_GPU, *vatom_GPU, F_GPU,
                                                 R_GPU, drdrx_GPU, ijlist_GPU,
                                                 mask_GPU, scale, nelt * cmax);
  } else {
    sum_forces_device<<<(n * nelt * cmax + maxThread - 1) / maxThread,
                        maxThread>>>(n * nelt * cmax, f_GPU, *vatom_GPU, F_GPU,
                                     R_GPU, drdrx_GPU, ijlist_GPU, mask_GPU,
                                     scale);
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(
      cudaMemcpy(f, f_GPU, sizeof(double) * nmax * 3, cudaMemcpyDeviceToHost));
  if (vatom) {
    CHECK(cudaMemcpy(vatom, *vatom_GPU, sizeof(double) * nmax * 6,
                     cudaMemcpyDeviceToHost));
  }
}

template void kernel_F1_GPU<double>(double *U, double *dHdr, double *drdrx,
                                    double *F1, int *mask, int a, int b, int c,
                                    int k);
template void kernel_F1_GPU<float>(float *U, float *dHdr, float *drdrx,
                                   float *F1, int *mask, int a, int b, int c,
                                   int k);
template void kernel_fused_F2_GPU<double>(double *V, double *R, double *drdrx,
                                          double *dMdrx, double *F2, int n,
                                          int d, int m);
template void kernel_fused_F2_GPU<float>(float *V, float *R, float *drdrx,
                                         float *dMdrx, float *F2, int n, int d,
                                         int m);
template void kernel2_GPU<double>(double *G, double *T, double *dEdG, double *P,
                                  double *dEdS, int m, int d, int k, int b,
                                  int a);
template void kernel2_GPU<float>(float *G, float *T, float *dEdG, float *P,
                                 float *dEdS, int m, int d, int k, int b,
                                 int a);
template void kernel_U_GPU<double>(double *dEdS, double *M, double *U,
                                   int a, int b, int d, int c, int k);
template void kernel_U_GPU<float>(float *dEdS, float *M, float *U, int a,
                                  int b, int d, int c, int k);
template void kernel_V_GPU<double>(double *dEdS, double *H, double *V,
                                   int a, int b, int d, int c, int k);
template void kernel_V_GPU<float>(float *dEdS, float *H, float *V, int a,
                                  int b, int d, int c, int k);
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
    double **V_GPU, double **dHdr_GPU, double **drdrx_GPU, double **F_GPU,
    double **dMdrx_GPU, int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU,
    double **pos_GPU, int **numneigh_GPU, int **firstneigh_GPU,
    int **ijlist_GPU, int **i2row_GPU, double cutforcesq, int nmax, int alocal,
    int neltypes, int d, int k, int m, bool use_exact_cmax);
template int setup_tensors_GPU<float>(
    int *ilist, int inum, const int *type, const int *fmap, double *pos,
    int old_numneigh_max, int *numneigh, int **firstneigh, int *eltij_pos,
    int *i2row, float **R_GPU, float **G_GPU, float **dEdG_GPU, float **P_GPU,
    float **dEdS_GPU, float **M_GPU, float **H_GPU, float **V_GPU,
    float **dHdr_GPU, float **drdrx_GPU, float **F_GPU, float **dMdrx_GPU,
    int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU, double **pos_GPU,
    int **numneigh_GPU, int **firstneigh_GPU, int **ijlist_GPU, int **i2row_GPU,
    float cutforcesq, int nmax, int alocal, int neltypes, int d, int k, int m,
    bool use_exact_cmax);
template void free_tensors_GPU<double>(
    int *ilist_GPU, double *pos_GPU, int *numneigh_GPU, int *firstneigh_GPU,
    int *ijlist_GPU, int *i2row_GPU, int *eltij_pos_GPU, int *fmap_type_GPU,
    double *R_GPU, double *G_GPU, double *dEdG_GPU, double *T_GPU,
    double *P_GPU, double *dEdS_GPU, double *M_GPU, double *H_GPU,
    double *V_GPU, double *dHdr_GPU, double *drdrx_GPU, double *F_GPU,
    double *dMdrx_GPU, int *mask_GPU, double *vatom_GPU);
template void free_tensors_GPU<float>(
    int *ilist_GPU, double *pos_GPU, int *numneigh_GPU, int *firstneigh_GPU,
    int *ijlist_GPU, int *i2row_GPU, int *eltij_pos_GPU, int *fmap_type_GPU,
    float *R_GPU, float *G_GPU, float *dEdG_GPU, float *T_GPU, float *P_GPU,
    float *dEdS_GPU, float *M_GPU, float *H_GPU, float *V_GPU, float *dHdr_GPU,
    float *drdrx_GPU, float *F_GPU, float *dMdrx_GPU, int *mask_GPU,
    double *vatom_GPU);
template void setup_global_GPU<float>(float **T_GPU, float *T,
                                      int **eltij_pos_GPU, int *eltij_pos,
                                      int neltypes, int d, int m);
template void setup_global_GPU<double>(double **T_GPU, double *T,
                                       int **eltij_pos_GPU, int *eltij_pos,
                                       int neltypes, int d, int m);
template void sum_forces_GPU<float>(int n, float *F_GPU, float *R_GPU,
                                    float *drdrx_GPU, int *ijlist_GPU,
                                    int *mask_GPU, double *f_GPU,
                                    double **vatom_GPU, double *f,
                                    double *vatom, double scale, int nmax,
                                    int nelt, int cmax);
template void sum_forces_GPU<double>(int n, double *F_GPU, double *R_GPU,
                                     double *drdrx_GPU, int *ijlist_GPU,
                                     int *mask_GPU, double *f_GPU,
                                     double **vatom_GPU, double *f,
                                     double *vatom, double scale, int nmax,
                                     int nelt, int cmax);