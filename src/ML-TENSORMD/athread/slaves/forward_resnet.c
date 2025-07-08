#include <math.h>
#include <slave.h>

#include "common.h"

void forward_resnet(union athread_args *args)
{
  const int M = args[0].i;
  const int N = args[1].i;
  const int K = args[2].i;
  const double *input = args[3].p;
  const double *weight = args[4].p;
  const double *bias = args[5].p;
  const int actfn = args[6].i;
  const int resnet_dt = args[7].i;
  double *output = args[8].p;
  double *dC = args[9].p;

  int global_id_start, global_id_end;
  athread_get_task_range(M, &global_id_start, &global_id_end,64);
  if (global_id_start >= global_id_end) return;

  double ldm_weight[K][N];
  double ldm_bias[N];
  double ldm_sum[N];
  double ldm_dC[N];

  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++) ldm_weight[k][n] = weight[n + N * k];
  for (int n = 0; n < N; n++) {
    if (bias)
      ldm_bias[n] = bias[n];
    else
      ldm_bias[n] = 0;
  }

  int gid = global_id_start;
  for (; gid < global_id_end; gid++) {
    for (int n = 0; n < N; n++) ldm_sum[n] = ldm_bias[n];
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++)
        ldm_sum[n] += ldm_weight[k][n] * input[k + K * gid];
    }
    for (int n = 0; n < N; n++) {
      if (actfn == 1) {
        ldm_sum[n] = ldm_sum[n] >= 0 ? ldm_sum[n] : 0;
        ldm_dC[n] = ldm_sum[n] >= 0 ? 1 : 0;
      }
      if (actfn == 2) {
        double exp_sum = exp(ldm_sum[n]);
        ldm_sum[n] = log1p(exp_sum);
        ldm_dC[n] = exp_sum / (1 + exp_sum);
      }
      if (actfn == 3) {
        ldm_sum[n] = tanh(ldm_sum[n]);
        ldm_dC[n] = 1 - ldm_sum[n] * ldm_sum[n];
      }
      if (resnet_dt && K == N) ldm_sum[n] = ldm_sum[n] + input[n + N * gid];
      output[n + N * gid] = ldm_sum[n];
      if (actfn > 0 && actfn < 4) dC[n + N * gid] = ldm_dC[n];
    }
    // for (int n = 0; n < N; n++) {
    //   double sum = ldm_bias[n];
    //   for (int k = 0; k < K; k++) {
    //     sum += ldm_weight[k][n] * input[k + K * gid];
    //   }
    //   // relu
    //   if (actfn == 1) sum = sum > 0 ? sum : 0;
    //   if (actfn == 2) sum = log1p(exp(sum));
    //   if (actfn == 3) sum = tanh(sum);
    //   if (resnet_dt && K == N) sum = sum + input[n + N * gid];
    //   output[n + N * gid] = sum;
    // }
  }
}
