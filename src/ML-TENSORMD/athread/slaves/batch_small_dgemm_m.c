#include <crts.h>
#include <slave.h>
#include "common.h"

#define max(x, y)        \
  ({                     \
    typeof(x) _x = (x);  \
    typeof(y) _y = (y);  \
    (void) (&_x == &_y); \
    _x > _y ? _x : _y;   \
  })

static void transpose_mat(int M, int K, double *in)
{
  // out[M/N][K]
  double *out = (double *)CRTS_pldm_malloc(M * K * sizeof(double));
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) out[k + K * m] = in[m + M * k];
  }
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) in[k + K * m] = out[k + K * m];
  }
  CRTS_pldm_free(out, M * K * sizeof(double));
}

static void small_dgemm(int M, int N, int K, double *A, double *B, double *C)
{
  // A[M][K] B[N][K] C[M][N]
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double tmp = 0;
      for (int k = 0; k < K; k++) 
        tmp += A[k + K * m] * B[k + K * n];
      C[m * N + n] = tmp;
    }
  }
}

void batch_small_dgemm_m(union athread_args *args)
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

  double *a, *b, *c;
  volatile crts_rply_t rply = 1;
  double *ldm_A = (double *)CRTS_pldm_malloc(M * K * sizeof(double));
  double *ldm_B = (double *)CRTS_pldm_malloc(N * K * sizeof(double));
  double *ldm_C = (double *)CRTS_pldm_malloc(M * N * sizeof(double));

  for (int gid = global_id_start; gid < global_id_end; gid++) {
    CRTS_dma_get(ldm_A, &A[M * K * gid], M * K * sizeof(double));
    CRTS_dma_get(ldm_B, &B[N * K * gid], N * K * sizeof(double));
    if (transA) {transpose_mat(M, K, ldm_A);}
    a = ldm_A;
    if (transB) {transpose_mat(N, K, ldm_B);}
    b = ldm_B;
    c = ldm_C;
    CRTS_dma_wait_value(&rply, 1);
    rply = 0;    
    small_dgemm(M, N, K, a, b, c);
    CRTS_dma_iput(&C[M * N * gid], c, M * N * sizeof(double), &rply);
  }
  CRTS_dma_barrier();
  CRTS_pldm_free(ldm_A, M * K * sizeof(double));
  CRTS_pldm_free(ldm_B, N * K * sizeof(double));
  CRTS_pldm_free(ldm_C, M * N * sizeof(double));
}
