#include <crts.h>
#include <slave.h>

#include "common.h"

#define USE_TRANSFORM_A
#define SIMD_SIZE 4

static void transform_A(int transA, int M, int K, const double *A_in,
                        double *A_out)
{
  // from
  // A_in[K][M] or A_in[M][K]
  // to
  // A[M / SIMD_SIZE][K][SIMD_SIZE] concact A[M % SIMD_SIZE][K]
  int m, k;
  // A[M / SIMD_SIZE][K][SIMD_SIZE]
  for (m = 0; m < M; m += SIMD_SIZE) {
    for (k = 0; k < K; k++) {
      for (int i = 0; i < SIMD_SIZE; i++) {
        if (transA)
          // A_in[M][K]
          A_out[i + SIMD_SIZE * k + K * m] = A_in[k + K * (m + i)];
        else
          // A_in[K][M]
          A_out[i + SIMD_SIZE * k + K * m] = A_in[m + i + M * k];
      }
    }
  }
  // A[M % SIMD_SIZE][K]
  for (;m < M; m++) {
    for (k = 0; k < K; k++) {
      if (transA)
        A_out[k + K * m] = A_in[k + K * m];
      else
        A_out[k + K * m] = A_in[m + M * k];
    }
  }
}

static void transpose_mat(int M, int K, const double *A_in, double *A_out)
{
  // A_in[K][M], A_out[M][K]
  // N = M
  // B_in[K][N], B_out[N][K]
  int k, m = 0;
#ifdef SIMD_SIZE
  for (; m <= M - SIMD_SIZE; m += SIMD_SIZE) {
    for (k = 0; k < K; k++) {
      for (int i = 0; i < SIMD_SIZE; i++)
        A_out[k + K * (m + i)] = A_in[m + i + M * k];
    }
  }
#endif
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) A_out[k + K * m] = A_in[m + M * k];
  }
}

static void small_dgemm(int M, int N, int K, double *A, double *B, double *C)
{
  // A[M][K] or A[M / SIMD_SIZE][K][SIMD_SIZE] concact A[M % SIMD_SIZE][K]
  // B[N][K]
  // C[N][M]
  int m, n, k;
  for (n = 0; n < N; n++) {
    m = 0;
#ifdef SIMD_SIZE
    for (; m < M; m += SIMD_SIZE) {
      for (int i = 0; i < SIMD_SIZE; i++) C[m + i + M * n] = 0;
      for (k = 0; k < K; k++) {
        for (int i = 0; i < SIMD_SIZE; i++)
#ifdef USE_TRANSFORM_A
          C[m + i + M * n] += A[i + SIMD_SIZE * k + K * m] * B[k + K * n];
#else
          C[m + i + M * n] += A[k + K * (m + i)] * B[k + K * n];
#endif
      }
    }
#endif
    for (; m < M; m++) {
      C[m + M * n] = 0;
      for (k = 0; k < K; k++) C[m + M * n] += A[k + K * m] * B[k + K * n];
    }
  }
}

void batch_small_dgemm(union athread_args *args)
{
  int global_size = args[0].i;
  int M = args[1].i;
  int N = args[2].i;
  int K = args[3].i;
  int transA = args[4].i;
  int transB = args[5].i;
  double *A = args[6].p;
  double *B = args[7].p;
  double *C = args[8].p;

  int global_id_start, global_id_end;
  athread_get_task_range(global_size, &global_id_start, &global_id_end, 64);
  if (global_id_start >= global_id_end) return;

  double t_A[M * K];
  double t_B[N * K];

#define LDM_SIZE 2

  double *a, *b, *c;
  volatile crts_rply_t rply[LDM_SIZE];
  double ldm_A[LDM_SIZE][M * K];
  double ldm_B[LDM_SIZE][K * N];
  double ldm_C[LDM_SIZE][M * N];

  a = &t_A[0];
  b = &t_B[0];

  int gid = global_id_start;
  for (; gid < global_id_end && gid < global_id_start + LDM_SIZE; gid++) {
    int idx = gid - global_id_start;
    rply[idx] = 0;
    CRTS_dma_iget(&ldm_A[idx][0], &A[M * K * gid], M * K * sizeof(double),
                  &rply[idx]);
    CRTS_dma_iget(&ldm_B[idx][0], &B[K * N * gid], K * N * sizeof(double),
                  &rply[idx]);
  }

  int idx = 0;
  gid = global_id_start;
  for (; gid < global_id_end; gid++) {
    if (gid - LDM_SIZE < global_id_start)
      CRTS_dma_wait_value(&rply[idx], 2);
    else
      CRTS_dma_wait_value(&rply[idx], 3);
    rply[idx] = 0;
#ifdef USE_TRANSFORM_A
    transform_A(transA, M, K, &ldm_A[idx][0], a);
#else
    if (transA)
      a = &ldm_A[idx][0];
    else
      transpose_mat(M, K, &ldm_A[idx][0], a);
#endif
    if (!transB)
      b = &ldm_B[idx][0];
    else
      transpose_mat(N, K, &ldm_B[idx][0], b);
    c = &ldm_C[idx][0];
    small_dgemm(M, N, K, a, b, c);
    CRTS_dma_iput(&C[M * N * gid], c, M * N * sizeof(double), &rply[idx]);
    if (gid + LDM_SIZE < global_id_end) {
      int gid2 = gid + LDM_SIZE;
      CRTS_dma_iget(&ldm_A[idx][0], &A[M * K * gid2], M * K * sizeof(double),
                    &rply[idx]);
      CRTS_dma_iget(&ldm_B[idx][0], &B[K * N * gid2], K * N * sizeof(double),
                    &rply[idx]);
    }
    idx++;
    if (idx >= LDM_SIZE) idx -= LDM_SIZE;
  }
  CRTS_dma_barrier();
}
