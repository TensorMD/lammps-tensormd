#include "common.cuh"
#include "math.cuh"
#include "nnp.cuh"

extern int maxThread;
extern cublasHandle_t handle;
extern void *globalPool;

template <typename Scalar>
__global__ void axpy_squareplus_device(Scalar *bias, Scalar *x, Scalar *dx, int M, int N, bool res, Scalar *res_x)
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  Scalar xi = x[i * N + j];
  xi += bias[j];
  Scalar z = sqrt(xi * xi + 4.0);
  xi = 0.5 * (xi + z);
  if (dx) { dx[i * N + j] = xi / z; } 
  if (res) { xi += res_x[i * N + j]; }
  x[i * N + j] = xi;
}

__global__ void axpy_softplus_device(double *bias, double *x, double *dx, int M, int N, bool res, double *res_x)
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  double xi = x[i * N + j];
  xi += bias[j];
  double exp_ = exp(xi);
  double threshold = log(2.22045e-16) + 2;
  if (xi < -threshold) { xi = log1p(exp_); }
  if (dx) { dx[i * N + j] = 1.0 / (1.0 + 1.0 / exp_); }
  if (res) { xi += res_x[i * N + j]; }
  x[i * N + j] = xi;
}

__global__ void axpy_softplus_device(float *bias, float *x, float *dx, int M, int N, bool res, float *res_x)
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  float xi = x[i * N + j];
  xi += bias[j];
  float exp_ = exp(xi);
  float threshold = log(1.19209e-07) + 2;
  if (xi < -threshold) { xi = log1p(exp_); }
  if (dx) { dx[i * N + j] = 1.0 / (1.0 + 1.0 / exp_); }
  if (res) { xi += res_x[i * N + j]; }
  x[i * N + j] = xi;
}

template <typename Scalar>
__global__ void axpy_device(Scalar *bias, Scalar *x, int M, int N)
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  x[i * N + j] += bias[j];
}

template <typename Scalar>
void forward_fnn_GPU(Scalar *x_in, int n, Scalar *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, Scalar *weights, Scalar *biases)
{
  Scalar *x_GPU, *pool, *dx = nullptr;
  CHECK(cudaMalloc(&x_GPU, sizeof(Scalar) * n * max_size));
  CHECK(cudaMalloc(&pool, sizeof(Scalar) * n * max_size));  
  CHECK(cudaMemcpy(x_GPU, x_in, sizeof(Scalar) * n, cudaMemcpyHostToDevice));
  int i;
  int M, N, K;
  Scalar *x;
  Scalar *C;
  bool C_is_pool;
  Scalar alpha = 1.0, beta = 0.0;

  C_is_pool = true;
  C = pool;
  M = n;
  x = x_GPU;

  for (i = 0; i < nlayers; i++) {
    K = full_sizes[i];
    N = full_sizes[i + 1];
    gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                weights + i * max_size * max_size,
                N, x, K, &beta, C, N);
    if (i < nlayers - 1) {
      bool res = i > 0 && use_resnet_dt && K == N;
      if (actfn == 1) {
        axpy_softplus_device<<<M, N>>>(
            biases + i * max_size, C, dx, M, N, res, x);
      } else if (actfn == 3) {
        axpy_squareplus_device<<<M, N>>>(
            biases + i * max_size, C, dx, M, N, res, x);
      }
    } else {
      if (apply_output_bias) {
        axpy_device<<<M, N>>>(biases + i * max_size, C, M, N);
      }
      break;
    }
    if (C_is_pool) {
      x = pool;
      C = x_GPU;
    } else {
      x = x_GPU;
      C = pool;
    }
    C_is_pool = !C_is_pool;
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(y, C, sizeof(Scalar) * M * n_out, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(x_GPU));
  CHECK(cudaFree(pool));
}

template <typename Scalar>
void free_nnp_GPU(Scalar *weights_GPU, Scalar *biases_GPU, Scalar **dz)
{
  CHECK(cudaFree(weights_GPU));
  CHECK(cudaFree(biases_GPU));
  for (int i = 0; i < 4; i++) { CHECK(cudaFree(*(dz + i))); }
}

template <typename Scalar>
void setup_nnp_GPU(Scalar **weights_GPU, Scalar **biases_GPU, Scalar ***weights,
                   Scalar ***biases, int nlayers, int num_in,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   int nelt, int max_size)
{
  CHECK(cudaMalloc(weights_GPU, sizeof(Scalar) * nlayers * max_size * max_size * nelt));
  CHECK(cudaMalloc(biases_GPU, sizeof(Scalar) * nlayers * max_size * nelt));
  for (int elti = 0; elti < nelt; elti++) {
    for (int i = 0; i < nlayers; i++) {
      CHECK(cudaMemcpy((*weights_GPU) + elti * nlayers * max_size * max_size +
                     i * max_size * max_size,
                 weights[elti][i], sizeof(Scalar) * full_sizes[i + 1] * full_sizes[i],
                 cudaMemcpyHostToDevice));
      if (i < nlayers - 1 || apply_output_bias) {
        CHECK(cudaMemcpy((*biases_GPU) + elti * nlayers * max_size + i * max_size,
                   biases[elti][i], sizeof(Scalar) * full_sizes[i + 1],
                   cudaMemcpyHostToDevice));
      }
    }
  }
}

template <typename Scalar>
void backward_resnet_GPU(int elti, Scalar *grad_in, Scalar *grad_out,
                         int max_size, int n_out, int M, int nlayers,
                         std::map<int, int> &full_sizes, bool use_resnet_dt,
                         Scalar *weights, Scalar *dz)
{
  int i;
  int N, K;
  Scalar *x;
  Scalar alpha = 1.0, beta = 0.0;
  K = n_out;
  N = full_sizes[nlayers - 1];
  if (grad_in == nullptr) {
    x = (Scalar *)globalPool + 2 * M * max_size;
  } else {
    x = grad_in;
  }

  int max_size_dz = 0;
  for (int ii = 1; ii < nlayers; ii++) {
    if (full_sizes[ii] > max_size_dz) {
      max_size_dz = full_sizes[ii];
    }
  }

  gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
              weights + elti * nlayers * max_size * max_size +
                  (nlayers - 1) * max_size * max_size,
              K, x, K, &beta, grad_out, N);

  for (i = nlayers - 1; i > 0; i--) {
    N = full_sizes[i - 1];
    K = full_sizes[i];
    vmul_device<<<(M * K + maxThread - 1) / maxThread, maxThread>>>(dz + (i - 1) * M * max_size_dz, grad_out,
                                                 M * K);
    if (use_resnet_dt && N == K && i > 1) {
      gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                  weights + elti * nlayers * max_size * max_size +
                      (i - 1) * max_size * max_size,
                  K, dz + (i - 1) * M * max_size_dz, K, &alpha, grad_out, N);
    } else {
      gpublasgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                  weights + elti * nlayers * max_size * max_size +
                      (i - 1) * max_size * max_size,
                  K, dz + (i - 1) * M * max_size_dz, K, &beta, grad_out, N);
    }
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
}

template <typename Scalar>
Scalar
forward_resnet_GPU(int elti, Scalar *x_in, int n, Scalar *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, bool sum_output, Scalar *weights,
                   Scalar *biases, Scalar **dz, int nlocal_old, bool DToH)
{
  int i;
  int M, N, K;
  Scalar *x;
  Scalar *C;
  bool C_is_pool_0;
  Scalar alpha = 1.0, beta = 0.0;
  Scalar result = 0;
  Scalar *pool[2], *ones;
  pool[0] = (Scalar *)globalPool;
  pool[1] = pool[0] + n * max_size;
  ones = pool[1] + n * max_size;

  int max_size_dz = 0;
  for (int ii = 1; ii < nlayers; ii++) {
    if (full_sizes[ii] > max_size_dz) {
      max_size_dz = full_sizes[ii];
    }
  }

  if (n > nlocal_old) {
    CHECK(cudaFree(*dz));
    CHECK(cudaMalloc(dz, sizeof(Scalar) * (nlayers - 1) * n * max_size_dz));
  }
  C_is_pool_0 = true;
  C = pool[0];
  M = n;
  x = x_in;
  ones_like_device<<<(M * n_out + maxThread - 1) / maxThread, maxThread>>>(ones, M * n_out);

  for (i = 0; i < nlayers; i++) {
    K = full_sizes[i];
    N = full_sizes[i + 1];
    gpublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                weights + elti * nlayers * max_size * max_size +
                    i * max_size * max_size,
                N, x, K, &beta, C, N);
    if (i < nlayers - 1) {
      bool res = i > 0 && use_resnet_dt && K == N;
      if (actfn == 1) {
        axpy_softplus_device<<<M, N>>>(
            biases + elti * nlayers * max_size + i * max_size, C,
            (*dz) + i * n * max_size_dz, M, N, res, x);
      } else if (actfn == 3) {
        axpy_squareplus_device<<<M, N>>>(
            biases + elti * nlayers * max_size + i * max_size, C,
            (*dz) + i * n * max_size_dz, M, N, res, x);
      }
    } else {
      if (apply_output_bias) {
        axpy_device<<<M, N>>>(biases + elti * nlayers * max_size + i * max_size,
                              C, M, N);
      }
      break;
    }
    if (C_is_pool_0) {
      x = pool[0];
      C = pool[1];
    } else {
      x = pool[1];
      C = pool[0];
    }
    if (i == nlayers - 2 && !DToH) C = y;
    C_is_pool_0 = !C_is_pool_0;
  }
  if (y && DToH) {
    CHECK(cudaMemcpy(y, C, sizeof(Scalar) * M * n_out, cudaMemcpyDeviceToHost));
  }
  if (sum_output) {
    gpublasdot(handle, M * n_out, ones, 1, C, 1, &result);
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  return result;
}

template <typename Scalar>
void forward_setup_tdnp_GPU(int n, Scalar **y, Scalar **dU, Scalar **dS,
                            int ydim)
{
  CHECK(cudaFree(*y));
  CHECK(cudaFree(*dS));
  CHECK(cudaMalloc(y, sizeof(Scalar) * n * ydim));
  *dU = *y;
  CHECK(cudaMalloc(dS, sizeof(Scalar) * n * ydim));
}

template <typename Scalar>
__global__ void set_element_tdnp_device(int n, int ydim, Scalar *x, Scalar init)
{
  int i = (threadIdx.x + blockDim.x * blockIdx.x) * ydim + ydim - 1;
  if (i < n * ydim) { x[i] = init; }
}

template <typename Scalar>
void forward_tdnp_GPU(int nlocal, int ydim, Scalar *y, Scalar etemp)
{
  set_element_tdnp_device<<<(nlocal + maxThread - 1) / maxThread, maxThread>>>(
      nlocal, ydim, y, etemp);
}

template <typename Scalar>
void backward_tdnp_GPU(int nlocal, int ydim, double scale, Scalar *dS, Scalar *dU)
{
  Scalar alpha = -scale;
  Scalar init = 0.0;
  gpublasaxpy(handle, ydim * nlocal, &alpha, dS, 1, dU, 1);
  set_element_tdnp_device<<<(nlocal + maxThread - 1) / maxThread, maxThread>>>(
      nlocal, ydim, dU, init);
}

template double forward_resnet_GPU<double>(
    int elti, double *x_in, int n, double *y, int max_size, int n_out,
    int nlayers, int actfn, std::map<int, int> &full_sizes,
    bool apply_output_bias, bool use_resnet_dt, bool sum_output,
    double *weights, double *biases, double **dz, int nlocal_old, bool DToH);
template float forward_resnet_GPU<float>(
    int elti, float *x_in, int n, float *y, int max_size, int n_out,
    int nlayers, int actfn, std::map<int, int> &full_sizes,
    bool apply_output_bias, bool use_resnet_dt, bool sum_output, float *weights,
    float *biases, float **dz, int nlocal_old, bool DToH);
template void backward_resnet_GPU<double>(
    int elti, double *grad_in, double *grad_out, int max_size, int n_out, int M,
    int nlayers, std::map<int, int> &full_sizes, bool use_resnet_dt,
    double *weights, double *dz);
template void backward_resnet_GPU<float>(
    int elti, float *grad_in, float *grad_out, int max_size, int n_out, int M,
    int nlayers, std::map<int, int> &full_sizes, bool use_resnet_dt,
    float *weights, float *dz);
template void setup_nnp_GPU<double>(double **weights_GPU, double **biases_GPU,
                                    double ***weights, double ***biases,
                                    int nlayers, int num_in,
                                    std::map<int, int> &full_sizes,
                                    bool apply_output_bias, int nelt, int max_size);
template void setup_nnp_GPU<float>(float **weights_GPU, float **biases_GPU,
                                   float ***weights, float ***biases,
                                   int nlayers, int num_in,
                                   std::map<int, int> &full_sizes,
                                   bool apply_output_bias, int nelt, int max_size);
template void free_nnp_GPU<double>(double *weights_GPU, double *biases_GPU, double **dz);
template void free_nnp_GPU<float>(float *weights_GPU, float *biases_GPU, float **dz);
template void forward_fnn_GPU<double>(double *x_in, int n, double *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, double *weights, double *biases);
template void forward_fnn_GPU<float>(float *x_in, int n, float *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, float *weights, float *biases);
template void forward_setup_tdnp_GPU<double>(int n, double **y, double **dU,
                                             double **dS, int ydim);
template void forward_setup_tdnp_GPU<float>(int n, float **y, float **dU,
                                            float **dS, int ydim);
template void forward_tdnp_GPU<double>(int nlocal, int ydim, double *y, double etemp);
template void forward_tdnp_GPU<float>(int nlocal, int ydim, float *y, float etemp);
template void backward_tdnp_GPU<double>(int nlocal, int ydim, double scale,
                                        double *dS, double *dU);
template void backward_tdnp_GPU<float>(int nlocal, int ydim, double scale,
                                       float *dS, float *dU);