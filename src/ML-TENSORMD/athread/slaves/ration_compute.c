#include <crts.h>
#include <math.h>
#include <slave.h>
#include "common.h"
#include <simd.h>

#define RESULT_CACHE 64
#define PARAMS_CACHE 4
#define min(x, y)        \
  ({                     \
    typeof(x) _x = (x);  \
    typeof(y) _y = (y);  \
    (void) (&_x == &_y); \
    _x < _y ? _x : _y;   \
  })

#define swap(x, y)     \
  {                    \
    (void) (&x == &y); \
    typeof(x) z = x;   \
    x = y;             \
    y = z;             \
  }

void batch_ration_compute(void **args)
{
  int n = *(int *)args[0];
  int n_out = *(int *)args[1];
  int size = *(int *)args[2];
  double dx = *(double *)args[3];
  double *params = (double *)args[4];
  double *x = (double *)args[5];
  double *f = (double *)args[6];
  double *df = (double *)args[7];

  int k_start, k_end;
  athread_get_task_range(n, &k_start, &k_end, 64);

  if (k_start >= k_end) return;

  double xdx = 1.0 / dx;

  doublev8 vec8d_dx = dx;
  doublev8 vec8d_s;
  doublev8 vec8d_c1;
  doublev8 vec8d_c2;
  doublev8 vec8d_c3;
  doublev8 vec8d_c4;
  doublev8 vec8d_r;
  doublev8 vec8d_q;
  doublev8 vec8d_y;
  doublev8 vec8d_p_df;
  doublev8 temp;
  doublev8 zero;
  int i;
  double q, r;
  double *f_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
  double *df_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
  double *p_f = f_ldm, *p_df = df_ldm;
  
  int kmod = (k_end - k_start) % RESULT_CACHE;
  for (int k = k_start; k + RESULT_CACHE <= k_end; k+=RESULT_CACHE) {  
    for (int kk = 0; kk < RESULT_CACHE; kk++) {
      i = (int) (x[k + kk] * xdx);
      i = min(size - 2, i);

      if (i > 0 && i < size - 2) {
        q = x[k + kk] - ((double) i) * dx;
        r = dx - q;
        double *p_y = &params[(i * 4 + 0) * n_out];
        double *p_s = &params[(i * 4 + 1) * n_out];
        double *p_c1 = &params[(i * 4 + 2) * n_out];
        double *p_c2 = &params[(i * 4 + 3) * n_out]; 
        vec8d_r = r;
        vec8d_q = q;
        for (int j = 0; j < n_out; j += 8) {
          // load for 64 aligned
          // uload for unaligned
          simd_load(vec8d_c1, p_c1 + j);
          simd_load(vec8d_c2, p_c2 + j);
          simd_load(vec8d_s, p_s + j);
          simd_load(vec8d_y, p_y + j);
          // c3 = fabs(c2 * r)
          vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
          vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
          // c4 = c3 + abs(c1 * q)
          vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
          vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
          vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
          // if (c4 > 0.0) c3 /= c4
          temp = simd_vdivd(vec8d_c3, vec8d_c4);
          vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
          // c4 = c2 + c3 * (c1 - c2)
          temp = simd_vsubd(vec8d_c1, vec8d_c2);
          vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
          // p_f = p_y + q * (s - r * c4)
          temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
          temp = simd_vmad(vec8d_q, temp, vec8d_y);
          // f
          simd_store(temp, p_f + kk * n_out + j);
          // p_df = s + dx * (c4 - c2) * (1 - c3)
          temp = simd_vsubd(vec8d_c4, vec8d_c2);
          temp = simd_vmuld(vec8d_dx, temp);
          vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
          vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
          // p_df = p_df + (q - r) * c4;
          temp = simd_vsubd(vec8d_q, vec8d_r);
          vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
          //df
          simd_store(vec8d_p_df, p_df + kk * n_out + j);
        }
      } else {
        for (int j = 0; j < n_out; j+=8) {
          simd_store(zero, p_f + kk * n_out + j);
          simd_store(zero, p_df + kk * n_out + j);
        }
      }
    }
    CRTS_dma_put(f + k * n_out, p_f, n_out * RESULT_CACHE * sizeof(double));
    CRTS_dma_put(df + k * n_out, p_df, n_out * RESULT_CACHE * sizeof(double));
  }

  for (int k = k_end - kmod; k < k_end; k++) {
    i = (int) (x[k] * xdx);
    i = min(size - 2, i);
    q = x[k] - ((double) i) * dx;
    r = dx - q;

    double *p_y = &params[(i * 4 + 0) * n_out];
    double *p_s = &params[(i * 4 + 1) * n_out];
    double *p_c1 = &params[(i * 4 + 2) * n_out];
    double *p_c2 = &params[(i * 4 + 3) * n_out];

    if (i > 0 && i < size - 2) {
      vec8d_r = r;
      vec8d_q = q;
      for (int j = 0; j < n_out; j += 8) {
        // load for 64 aligned
        // uload for unaligned
        simd_load(vec8d_c1, p_c1 + j);
        simd_load(vec8d_c2, p_c2 + j);
        simd_load(vec8d_s, p_s + j);
        simd_load(vec8d_y, p_y + j);
        // c3 = fabs(c2 * r)
        vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
        vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
        // c4 = c3 + abs(c1 * q)
        vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
        vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
        vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
        // if (c4 > 0.0) c3 /= c4
        temp = simd_vdivd(vec8d_c3, vec8d_c4);
        vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
        // c4 = c2 + c3 * (c1 - c2)
        temp = simd_vsubd(vec8d_c1, vec8d_c2);
        vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
        // p_f = p_y + q * (s - r * c4)
        temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
        temp = simd_vmad(vec8d_q, temp, vec8d_y);
        // f
        simd_store(temp, p_f + j);
        // p_df = s + dx * (c4 - c2) * (1 - c3)
        temp = simd_vsubd(vec8d_c4, vec8d_c2);
        temp = simd_vmuld(vec8d_dx, temp);
        vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
        vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
        // p_df = p_df + (q - r) * c4;
        temp = simd_vsubd(vec8d_q, vec8d_r);
        vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
        //df
        simd_store(vec8d_p_df, p_df + j);
      }
    } else {
      for (int j = 0; j < n_out; j+=8) {
        simd_store(zero, p_f + j);
        simd_store(zero, p_df + j);
      }
    }
    CRTS_dma_put(f + k * n_out, p_f, n_out * sizeof(double));
    CRTS_dma_put(df + k * n_out, p_df, n_out * sizeof(double));    
  }

  CRTS_pldm_free(f_ldm, n_out * RESULT_CACHE * sizeof(double));
  CRTS_pldm_free(df_ldm, n_out * RESULT_CACHE * sizeof(double));
}

void batch_ration_computef(void **args)
{
  int n = *(int *)args[0];
  int n_out = *(int *)args[1];
  int size = *(int *)args[2];
  double dx = *(double *)args[3];
  double *params = (double *)args[4];
  double *x = (double *)args[5];
  double *f = (double *)args[6];
  double *df = (double *)args[7];

  int k_start, k_end;
  athread_get_task_range(n, &k_start, &k_end, 64);

  if (k_start >= k_end) return;

  double xdx = 1.0 / dx;

  doublev8 vec8d_dx = dx;
  doublev8 vec8d_s;
  doublev8 vec8d_c1;
  doublev8 vec8d_c2;
  doublev8 vec8d_c3;
  doublev8 vec8d_c4;
  doublev8 vec8d_r;
  doublev8 vec8d_q;
  doublev8 vec8d_y;
  doublev8 vec8d_p_df;
  doublev8 temp;
  doublev8 zero;
  int i;
  double q, r;
  double *f_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
  double *p_f = f_ldm;
  
  int kmod = (k_end - k_start) % RESULT_CACHE;
  for (int k = k_start; k + RESULT_CACHE <= k_end; k+=RESULT_CACHE) {  
    for (int kk = 0; kk < RESULT_CACHE; kk++) {
      i = (int) (x[k + kk] * xdx);
      i = min(size - 2, i);

      if (i > 0 && i < size - 2) {
        q = x[k + kk] - ((double) i) * dx;
        r = dx - q;
        double *p_y = &params[(i * 4 + 0) * n_out];
        double *p_s = &params[(i * 4 + 1) * n_out];
        double *p_c1 = &params[(i * 4 + 2) * n_out];
        double *p_c2 = &params[(i * 4 + 3) * n_out]; 
        vec8d_r = r;
        vec8d_q = q;
        for (int j = 0; j < n_out; j += 8) {
          // load for 64 aligned
          // uload for unaligned
          simd_load(vec8d_c1, p_c1 + j);
          simd_load(vec8d_c2, p_c2 + j);
          simd_load(vec8d_s, p_s + j);
          simd_load(vec8d_y, p_y + j);
          // c3 = fabs(c2 * r)
          vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
          vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
          // c4 = c3 + abs(c1 * q)
          vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
          vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
          vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
          // if (c4 > 0.0) c3 /= c4
          temp = simd_vdivd(vec8d_c3, vec8d_c4);
          vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
          // c4 = c2 + c3 * (c1 - c2)
          temp = simd_vsubd(vec8d_c1, vec8d_c2);
          vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
          // p_f = p_y + q * (s - r * c4)
          temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
          temp = simd_vmad(vec8d_q, temp, vec8d_y);
          // f
          simd_store(temp, p_f + kk * n_out + j);
        }
      } else {
        for (int j = 0; j < n_out; j+=8) {
          simd_store(zero, p_f + kk * n_out + j);
        }
      }
    }
    CRTS_dma_put(f + k * n_out, p_f, n_out * RESULT_CACHE * sizeof(double));
  }

  for (int k = k_end - kmod; k < k_end; k++) {
    i = (int) (x[k] * xdx);
    i = min(size - 2, i);
    q = x[k] - ((double) i) * dx;
    r = dx - q;

    double *p_y = &params[(i * 4 + 0) * n_out];
    double *p_s = &params[(i * 4 + 1) * n_out];
    double *p_c1 = &params[(i * 4 + 2) * n_out];
    double *p_c2 = &params[(i * 4 + 3) * n_out];

    if (i > 0 && i < size - 2) {
      vec8d_r = r;
      vec8d_q = q;
      for (int j = 0; j < n_out; j += 8) {
        // load for 64 aligned
        // uload for unaligned
        simd_load(vec8d_c1, p_c1 + j);
        simd_load(vec8d_c2, p_c2 + j);
        simd_load(vec8d_s, p_s + j);
        simd_load(vec8d_y, p_y + j);
        // c3 = fabs(c2 * r)
        vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
        vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
        // c4 = c3 + abs(c1 * q)
        vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
        vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
        vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
        // if (c4 > 0.0) c3 /= c4
        temp = simd_vdivd(vec8d_c3, vec8d_c4);
        vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
        // c4 = c2 + c3 * (c1 - c2)
        temp = simd_vsubd(vec8d_c1, vec8d_c2);
        vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
        // p_f = p_y + q * (s - r * c4)
        temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
        temp = simd_vmad(vec8d_q, temp, vec8d_y);
        // f
        simd_store(temp, p_f + j);
      }
    } else {
      for (int j = 0; j < n_out; j+=8) {
        simd_store(zero, p_f + j);
      }
    }
    CRTS_dma_put(f + k * n_out, p_f, n_out * sizeof(double));
  }

  CRTS_pldm_free(f_ldm, n_out * RESULT_CACHE * sizeof(double));
}

void batch_ration_computedf(void **args)
{
  int n = *(int *)args[0];
  int n_out = *(int *)args[1];
  int size = *(int *)args[2];
  double dx = *(double *)args[3];
  double *params = (double *)args[4];
  double *x = (double *)args[5];
  double *f = (double *)args[6];
  double *df = (double *)args[7];

  int k_start, k_end;
  athread_get_task_range(n, &k_start, &k_end, 64);

  if (k_start >= k_end) return;

  double xdx = 1.0 / dx;

  doublev8 vec8d_dx = dx;
  doublev8 vec8d_s;
  doublev8 vec8d_c1;
  doublev8 vec8d_c2;
  doublev8 vec8d_c3;
  doublev8 vec8d_c4;
  doublev8 vec8d_r;
  doublev8 vec8d_q;
  doublev8 vec8d_y;
  doublev8 vec8d_p_df;
  doublev8 temp;
  doublev8 zero;
  int i;
  double q, r;
  double *df_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
  double *p_df = df_ldm;
  
  int kmod = (k_end - k_start) % RESULT_CACHE;
  for (int k = k_start; k + RESULT_CACHE <= k_end; k+=RESULT_CACHE) {  
    for (int kk = 0; kk < RESULT_CACHE; kk++) {
      i = (int) (x[k + kk] * xdx);
      i = min(size - 2, i);

      if (i > 0 && i < size - 2) {
        q = x[k + kk] - ((double) i) * dx;
        r = dx - q;
        double *p_y = &params[(i * 4 + 0) * n_out];
        double *p_s = &params[(i * 4 + 1) * n_out];
        double *p_c1 = &params[(i * 4 + 2) * n_out];
        double *p_c2 = &params[(i * 4 + 3) * n_out]; 
        vec8d_r = r;
        vec8d_q = q;
        for (int j = 0; j < n_out; j += 8) {
          // load for 64 aligned
          // uload for unaligned
          simd_load(vec8d_c1, p_c1 + j);
          simd_load(vec8d_c2, p_c2 + j);
          simd_load(vec8d_s, p_s + j);
          simd_load(vec8d_y, p_y + j);
          // c3 = fabs(c2 * r)
          vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
          vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
          // c4 = c3 + abs(c1 * q)
          vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
          vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
          vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
          // if (c4 > 0.0) c3 /= c4
          temp = simd_vdivd(vec8d_c3, vec8d_c4);
          vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
          // c4 = c2 + c3 * (c1 - c2)
          temp = simd_vsubd(vec8d_c1, vec8d_c2);
          vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
          // p_df = s + dx * (c4 - c2) * (1 - c3)
          temp = simd_vsubd(vec8d_c4, vec8d_c2);
          temp = simd_vmuld(vec8d_dx, temp);
          vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
          vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
          // p_df = p_df + (q - r) * c4;
          temp = simd_vsubd(vec8d_q, vec8d_r);
          vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
          //df
          simd_store(vec8d_p_df, p_df + kk * n_out + j);
        }
      } else {
        for (int j = 0; j < n_out; j+=8) {
          simd_store(zero, p_df + kk * n_out + j);
        }
      }
    }
    CRTS_dma_put(df + k * n_out, p_df, n_out * RESULT_CACHE * sizeof(double));
  }

  for (int k = k_end - kmod; k < k_end; k++) {
    i = (int) (x[k] * xdx);
    i = min(size - 2, i);
    q = x[k] - ((double) i) * dx;
    r = dx - q;

    double *p_y = &params[(i * 4 + 0) * n_out];
    double *p_s = &params[(i * 4 + 1) * n_out];
    double *p_c1 = &params[(i * 4 + 2) * n_out];
    double *p_c2 = &params[(i * 4 + 3) * n_out];

    if (i > 0 && i < size - 2) {
      vec8d_r = r;
      vec8d_q = q;
      for (int j = 0; j < n_out; j += 8) {
        // load for 64 aligned
        // uload for unaligned
        simd_load(vec8d_c1, p_c1 + j);
        simd_load(vec8d_c2, p_c2 + j);
        simd_load(vec8d_s, p_s + j);
        simd_load(vec8d_y, p_y + j);
        // c3 = fabs(c2 * r)
        vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
        vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
        // c4 = c3 + abs(c1 * q)
        vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
        vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
        vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
        // if (c4 > 0.0) c3 /= c4
        temp = simd_vdivd(vec8d_c3, vec8d_c4);
        vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
        // c4 = c2 + c3 * (c1 - c2)
        temp = simd_vsubd(vec8d_c1, vec8d_c2);
        vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
        // p_df = s + dx * (c4 - c2) * (1 - c3)
        temp = simd_vsubd(vec8d_c4, vec8d_c2);
        temp = simd_vmuld(vec8d_dx, temp);
        vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
        vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
        // p_df = p_df + (q - r) * c4;
        temp = simd_vsubd(vec8d_q, vec8d_r);
        vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
        //df
        simd_store(vec8d_p_df, p_df + j);
      }
    } else {
      for (int j = 0; j < n_out; j+=8) {
        simd_store(zero, p_df + j);
      }
    }
    CRTS_dma_put(df + k * n_out, p_df, n_out * sizeof(double));    
  }
  CRTS_pldm_free(df_ldm, n_out * RESULT_CACHE * sizeof(double));
}

// void batch_ration_compute(void **args)
// {
//   int n = *(int *)args[0];
//   int n_out = *(int *)args[1];
//   int size = *(int *)args[2];
//   double dx = *(double *)args[3];
//   double *params = (double *)args[4];
//   double *x = (double *)args[5];
//   double *f = (double *)args[6];
//   double *df = (double *)args[7];

//   int k_start, k_end;
//   athread_get_task_range(n, &k_start, &k_end, 64);

//   if (k_start >= k_end) return;

//   double xdx = 1.0 / dx;

//   doublev8 vec8d_dx = dx;
//   doublev8 vec8d_s;
//   doublev8 vec8d_c1;
//   doublev8 vec8d_c2;
//   doublev8 vec8d_c3;
//   doublev8 vec8d_c4;
//   doublev8 vec8d_r;
//   doublev8 vec8d_q;
//   doublev8 vec8d_y;
//   doublev8 vec8d_p_df;
//   doublev8 temp;
//   doublev8 zero;
//   int i;
//   double q, r;
//   crts_rply_t reply = 0;
//   double *f_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
//   double *df_ldm = (double *)CRTS_pldm_malloc(n_out * RESULT_CACHE * sizeof(double));
//   double *params_ldm = (double *)CRTS_pldm_malloc(8 * 4 * PARAMS_CACHE * 2 * sizeof(double));
//   double *p_f = f_ldm, *p_df = df_ldm;
  
//   int kmod = (k_end - k_start) % RESULT_CACHE;
//   for (int k = k_start; k + RESULT_CACHE <= k_end; k+=RESULT_CACHE) { 
//     for (int kk = 0; kk < RESULT_CACHE; kk++) {
//       i = (int) (x[k + kk] * xdx);
//       i = min(size - 2, i);

//       if (i > 0 && i < size - 2) {
//         q = x[k + kk] - ((double) i) * dx;
//         r = dx - q;
//         vec8d_r = r;
//         vec8d_q = q;
//         CRTS_dma_iget(params_ldm, &params[(i * 4) * n_out], 8 * PARAMS_CACHE * 4 * sizeof(double), &reply);
//         for (int j = 0, flag = 0; j < n_out; j += 8 * PARAMS_CACHE) {
//           // load for 64 aligned
//           // uload for unaligned
//           CRTS_dma_wait_value(&reply, 1);
//           reply = 0;
//           int offset = flag * 8 * PARAMS_CACHE * 4;
//           flag++;
//           flag = flag % 2;
//           if (j + 8 * PARAMS_CACHE < n_out) {
//             CRTS_dma_iget(params_ldm + flag * 8 * PARAMS_CACHE * 4, &params[(i * 4) * n_out + (j + 8 * PARAMS_CACHE) * 4], 8 * PARAMS_CACHE * 4 * sizeof(double), &reply);
//           }
          
//           for (int jj = 0; jj < PARAMS_CACHE; jj++)
//           {
//             double *p_params = params_ldm + offset + jj * 8 * 4;
//             vec8d_y = simd_set_doublev8(p_params[0 * 4], p_params[1 * 4], p_params[2 * 4], p_params[3 * 4],
//                         p_params[4 * 4], p_params[5 * 4], p_params[6 * 4], p_params[7 * 4]);
//             vec8d_s = simd_set_doublev8(p_params[0 * 4 + 1], p_params[1 * 4 + 1], p_params[2 * 4 + 1], p_params[3 * 4 + 1],
//                         p_params[4 * 4 + 1], p_params[5 * 4 + 1], p_params[6 * 4 + 1], p_params[7 * 4 + 1]);
//             vec8d_c1 = simd_set_doublev8(p_params[0 * 4 + 2], p_params[1 * 4 + 2], p_params[2 * 4 + 2], p_params[3 * 4 + 2],
//                         p_params[4 * 4 + 2], p_params[5 * 4 + 2], p_params[6 * 4 + 2], p_params[7 * 4 + 2]);
//             vec8d_c2 = simd_set_doublev8(p_params[0 * 4 + 3], p_params[1 * 4 + 3], p_params[2 * 4 + 3], p_params[3 * 4 + 3],
//                         p_params[4 * 4 + 3], p_params[5 * 4 + 3], p_params[6 * 4 + 3], p_params[7 * 4 + 3]);

//             // c3 = fabs(c2 * r)
//             vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
//             vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
//             // c4 = c3 + abs(c1 * q)
//             vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
//             vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
//             vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
//             // if (c4 > 0.0) c3 /= c4
//             temp = simd_vdivd(vec8d_c3, vec8d_c4);
//             vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
//             // c4 = c2 + c3 * (c1 - c2)
//             temp = simd_vsubd(vec8d_c1, vec8d_c2);
//             vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
//             // p_f = p_y + q * (s - r * c4)
//             temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
//             temp = simd_vmad(vec8d_q, temp, vec8d_y);
//             // f
//             simd_store(temp, p_f + kk * n_out + j + jj * 8);
//             // p_df = s + dx * (c4 - c2) * (1 - c3)
//             temp = simd_vsubd(vec8d_c4, vec8d_c2);
//             temp = simd_vmuld(vec8d_dx, temp);
//             vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
//             vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
//             // p_df = p_df + (q - r) * c4;
//             temp = simd_vsubd(vec8d_q, vec8d_r);
//             vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
//             //df
//             simd_store(vec8d_p_df, p_df + kk * n_out + j + jj * 8);
//           }
//         }
//       } else {
//         for (int j = 0; j < n_out; j+=8) {
//           simd_store(zero, p_f + kk * n_out + j);
//           simd_store(zero, p_df + kk * n_out + j);
//         }
//       }
//     }
//     CRTS_dma_put(f + k * n_out, p_f, n_out * RESULT_CACHE * sizeof(double));
//     CRTS_dma_put(df + k * n_out, p_df, n_out * RESULT_CACHE * sizeof(double));
//   }

//   int k = k_end - kmod;
//   for (int kk = 0; kk < kmod; kk++) {
//     i = (int) (x[k + kk] * xdx);
//     i = min(size - 2, i);

//     if (i > 0 && i < size - 2) {
//       q = x[k + kk] - ((double) i) * dx;
//       r = dx - q;
//       vec8d_r = r;
//       vec8d_q = q;
//       CRTS_dma_iget(params_ldm, &params[(i * 4) * n_out], 8 * PARAMS_CACHE * 4 * sizeof(double), &reply);
//       for (int j = 0, flag = 0; j < n_out; j += 8 * PARAMS_CACHE) {
//         // load for 64 aligned
//         // uload for unaligned
//         CRTS_dma_wait_value(&reply, 1);
//         reply = 0;
//         int offset = flag * 8 * PARAMS_CACHE * 4;
//         flag++;
//         flag = flag % 2;
//         if (j + 8 * PARAMS_CACHE < n_out) {
//           CRTS_dma_iget(params_ldm + flag * 8 * PARAMS_CACHE * 4, &params[(i * 4) * n_out + (j + 8 * PARAMS_CACHE) * 4], 8 * PARAMS_CACHE * 4 * sizeof(double), &reply);
//         }
        
//         for (int jj = 0; jj < PARAMS_CACHE; jj++)
//         {
//           double *p_params = params_ldm + offset + jj * 8 * 4;
//           vec8d_y = simd_set_doublev8(p_params[0 * 4], p_params[1 * 4], p_params[2 * 4], p_params[3 * 4],
//                       p_params[4 * 4], p_params[5 * 4], p_params[6 * 4], p_params[7 * 4]);
//           vec8d_s = simd_set_doublev8(p_params[0 * 4 + 1], p_params[1 * 4 + 1], p_params[2 * 4 + 1], p_params[3 * 4 + 1],
//                       p_params[4 * 4 + 1], p_params[5 * 4 + 1], p_params[6 * 4 + 1], p_params[7 * 4 + 1]);
//           vec8d_c1 = simd_set_doublev8(p_params[0 * 4 + 2], p_params[1 * 4 + 2], p_params[2 * 4 + 2], p_params[3 * 4 + 2],
//                       p_params[4 * 4 + 2], p_params[5 * 4 + 2], p_params[6 * 4 + 2], p_params[7 * 4 + 2]);
//           vec8d_c2 = simd_set_doublev8(p_params[0 * 4 + 3], p_params[1 * 4 + 3], p_params[2 * 4 + 3], p_params[3 * 4 + 3],
//                       p_params[4 * 4 + 3], p_params[5 * 4 + 3], p_params[6 * 4 + 3], p_params[7 * 4 + 3]);

//           // c3 = fabs(c2 * r)
//           vec8d_c3 = simd_vmuld(vec8d_c2, vec8d_r);
//           vec8d_c3 = simd_smaxd(vec8d_c3, -vec8d_c3);
//           // c4 = c3 + abs(c1 * q)
//           vec8d_c4 = simd_vmuld(vec8d_c1, vec8d_q);
//           vec8d_c4 = simd_smaxd(vec8d_c4, -vec8d_c4);
//           vec8d_c4 = simd_vaddd(vec8d_c3, vec8d_c4);
//           // if (c4 > 0.0) c3 /= c4
//           temp = simd_vdivd(vec8d_c3, vec8d_c4);
//           vec8d_c3 = simd_vfselled(vec8d_c4, vec8d_c3, temp);
//           // c4 = c2 + c3 * (c1 - c2)
//           temp = simd_vsubd(vec8d_c1, vec8d_c2);
//           vec8d_c4 = simd_vmad(vec8d_c3, temp, vec8d_c2);
//           // p_f = p_y + q * (s - r * c4)
//           temp = simd_vnmad(vec8d_r, vec8d_c4, vec8d_s);
//           temp = simd_vmad(vec8d_q, temp, vec8d_y);
//           // f
//           simd_store(temp, p_f + kk * n_out + j + jj * 8);
//           // p_df = s + dx * (c4 - c2) * (1 - c3)
//           temp = simd_vsubd(vec8d_c4, vec8d_c2);
//           temp = simd_vmuld(vec8d_dx, temp);
//           vec8d_p_df = simd_vnmad(temp, vec8d_c3, temp);
//           vec8d_p_df = simd_vaddd(vec8d_p_df, vec8d_s);
//           // p_df = p_df + (q - r) * c4;
//           temp = simd_vsubd(vec8d_q, vec8d_r);
//           vec8d_p_df = simd_vmad(temp, vec8d_c4, vec8d_p_df);
//           //df
//           simd_store(vec8d_p_df, p_df + kk * n_out + j + jj * 8);
//         }
//       }
//     } else {
//       for (int j = 0; j < n_out; j+=8) {
//         simd_store(zero, p_f + kk * n_out + j);
//         simd_store(zero, p_df + kk * n_out + j);
//       }
//     }
//   }
//   CRTS_dma_put(f + k * n_out, p_f, n_out * kmod * sizeof(double));
//   CRTS_dma_put(df + k * n_out, p_df, n_out * kmod * sizeof(double));

//   CRTS_pldm_free(f_ldm, n_out * RESULT_CACHE * sizeof(double));
//   CRTS_pldm_free(df_ldm, n_out * RESULT_CACHE * sizeof(double));
//   CRTS_pldm_free(params_ldm, 8 * 4 * PARAMS_CACHE * 2 * sizeof(double));
// }