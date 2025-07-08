#include "hip/hip_runtime.h"
template <typename Scalar>
void ration_setup_GPU(Scalar *params, Scalar **params_GPU, int size, int n_out)
{
  hipMalloc(params_GPU, sizeof(Scalar) * size * 4 * n_out);
  hipMemcpy(*params_GPU, params, sizeof(Scalar) * size * 4 * n_out,
             hipMemcpyHostToDevice);
}

template <typename Scalar>
__global__ void ration_computef_device(Scalar *x, int n, Scalar *f, Scalar *df,
                                      Scalar *params, Scalar xdx, int size,
                                      Scalar dx, int n_out,
                                      bool endpoints_force_zero)
{
  int k = blockIdx.x;
  int j = threadIdx.x;
  int i;
  Scalar q, r, s, c1, c2, c3, c4, xk;
  xk = x[k];
  i = (int) (xk * xdx);
  i = min(size - 2, i);
  q = xk - i * dx;
  r = dx - q;
  Scalar *p_y = &params[(i * 4 + 0) * n_out];
  Scalar *p_s = &params[(i * 4 + 1) * n_out];
  Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
  Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
  Scalar *p_f = &f[k * n_out];
  Scalar *p_df = &df[k * n_out];
  if (i == 0) {
    if (endpoints_force_zero) {
      p_f[j] = 0.0;
    } else {
      s = p_s[j];
      c2 = p_c2[j];
      p_f[j] = p_y[j] + q * (s - r * c2);
    }
  } else if (i == size - 2) {
    if (endpoints_force_zero) {
      p_f[j] = 0.0;
    } else {
      s = p_s[j];
      c1 = p_c1[j];
      p_f[j] = p_y[j] + q * (s - r * c1);
    }
  } else {
    s = p_s[j];
    c1 = p_c1[j];
    c2 = p_c2[j];
    c3 = fabs(c2 * r);
    c4 = c3 + fabs(c1 * q);
    if (c4 > 0.0) c3 /= c4;
    c4 = c2 + c3 * (c1 - c2);
    p_f[j] = p_y[j] + q * (s - r * c4);
  }
}

template <typename Scalar>
__global__ void ration_computedf_device(Scalar *x, int n, Scalar *f, Scalar *df,
                                      Scalar *params, Scalar xdx, int size,
                                      Scalar dx, int n_out,
                                      bool endpoints_force_zero)
{
  int k = blockIdx.x;
  int j = threadIdx.x;
  int i;
  Scalar q, r, s, c1, c2, c3, c4, xk;
  xk = x[k];
  i = (int) (xk * xdx);
  i = min(size - 2, i);
  q = xk - i * dx;
  r = dx - q;
  Scalar *p_y = &params[(i * 4 + 0) * n_out];
  Scalar *p_s = &params[(i * 4 + 1) * n_out];
  Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
  Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
  Scalar *p_f = &f[k * n_out];
  Scalar *p_df = &df[k * n_out];
  if (i == 0) {
    if (endpoints_force_zero) {
      p_df[j] = 0.0;
    } else {
      s = p_s[j];
      c2 = p_c2[j];
      p_df[j] = s + (q - r) * c2;
    }
  } else if (i == size - 2) {
    if (endpoints_force_zero) {
      p_df[j] = 0.0;
    } else {
      s = p_s[j];
      c1 = p_c1[j];
      p_df[j] = s + (q - r) * c1;
    }
  } else {
    s = p_s[j];
    c1 = p_c1[j];
    c2 = p_c2[j];
    c3 = fabs(c2 * r);
    c4 = c3 + fabs(c1 * q);
    if (c4 > 0.0) c3 /= c4;
    c4 = c2 + c3 * (c1 - c2);
    p_df[j] = s + (q - r) * c4 + dx * (c4 - c2) * (1 - c3);
  }
}

template <typename Scalar>
void ration_compute_GPU(Scalar *x, int n, Scalar *f, Scalar *df, Scalar *params,
                        Scalar xdx, int size, Scalar dx, int n_out,
                        bool endpoints_force_zero)
{
  hipLaunchKernelGGL(ration_computef_device, dim3(n), dim3(n_out), 0, 0, x, n, f, df, params, xdx, size, dx, n_out,
                                      endpoints_force_zero);
  hipLaunchKernelGGL(ration_computedf_device, dim3(n), dim3(n_out), 0, 0, x, n, f, df, params, xdx, size, dx, n_out,
                                      endpoints_force_zero);
  hipDeviceSynchronize();
}

template void ration_setup_GPU<double>(double *params, double **params_GPU,
                                       int size, int n_out);
template void ration_setup_GPU<float>(float *params, float **params_GPU,
                                      int size, int n_out);
template void ration_compute_GPU<double>(double *x, int n, double *f,
                                         double *df, double *params, double xdx,
                                         int size, double dx, int n_out,
                                         bool endpoints_force_zero);
template void ration_compute_GPU<float>(float *x, int n, float *f, float *df,
                                        float *params, float xdx, int size,
                                        float dx, int n_out,
                                        bool endpoints_force_zero);
