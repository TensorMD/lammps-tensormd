#include "common.cuh"
extern int numneigh_max_global;

template <typename Scalar>
void ration_setup_GPU(Scalar *params, Scalar **params_GPU, int size, int n_out)
{
  CHECK(cudaMalloc(params_GPU, sizeof(Scalar) * size * 4 * n_out));
  CHECK(cudaMemcpy(*params_GPU, params, sizeof(Scalar) * size * 4 * n_out,
             cudaMemcpyHostToDevice));
}

template <typename Scalar>
__global__ void ration_compute_device(Scalar *x, int n, Scalar *f, Scalar *df,
                                      Scalar *params, Scalar xdx, int size,
                                      Scalar dx, int n_out,
                                      bool endpoints_force_zero, int numneigh_max_global)
{
  extern __shared__ double x_shm[];
  int j = threadIdx.x;
  int k = blockIdx.x * numneigh_max_global;
  int i;
  Scalar q, r, s, c1, c2, c3, c4, xk;
  for (int ii = threadIdx.x; ii < numneigh_max_global; ii += blockDim.x) {
    x_shm[ii] = x[k + ii];
  }
  __syncthreads();
  for (int ii = 0; ii < numneigh_max_global; ii++) {
    xk = x_shm[ii];
    i = (int) (xk * xdx);
    i = min(size - 2, i);
    q = xk - i * dx;
    r = dx - q;
    Scalar *p_y = &params[(i * 4 + 0) * n_out];
    Scalar *p_s = &params[(i * 4 + 1) * n_out];
    Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
    Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
    Scalar *p_f = &f[(k + ii) * n_out];
    Scalar *p_df = &df[(k + ii) * n_out];
    if (i == 0) {
      if (endpoints_force_zero) {
        p_f[j] = 0.0;
        p_df[j] = 0.0;
      } else {
        s = p_s[j];
        c2 = p_c2[j];
        p_f[j] = p_y[j] + q * (s - r * c2);
        p_df[j] = s + (q - r) * c2;
      }
    } else if (i == size - 2) {
      if (endpoints_force_zero) {
        p_f[j] = 0.0;
        p_df[j] = 0.0;
      } else {
        s = p_s[j];
        c1 = p_c1[j];
        p_f[j] = p_y[j] + q * (s - r * c1);
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
      p_f[j] = p_y[j] + q * (s - r * c4);
      p_df[j] = s + (q - r) * c4 + dx * (c4 - c2) * (1 - c3);
    }
  }
}

template <typename Scalar>
void ration_compute_GPU(Scalar *x, int n, Scalar *f, Scalar *df, Scalar *params,
                        Scalar xdx, int size, Scalar dx, int n_out,
                        bool endpoints_force_zero)
{
  int shared_size = numneigh_max_global * sizeof(double);
  ration_compute_device<<<n / numneigh_max_global, n_out, shared_size>>>(
      x, n, f, df, params, xdx, size, dx, n_out, endpoints_force_zero,
      numneigh_max_global);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());                                  
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
