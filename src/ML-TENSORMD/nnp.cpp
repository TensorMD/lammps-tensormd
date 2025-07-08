/* ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "nnp.h"
#include "activation.hpp"
#include "cppblas.hpp"
#include "error.h"
#include "memory.h"
#include "tensormd_timer.h"
#include "athread/athread_tensoralloy.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

void calculate_nn_flops_per_one(int nlayers, const int *layer_sizes, int num_in,
                                bool use_resnet_dt, bool apply_output_bias,
                                double &forward, double &backward)
{
  int i;
  int N, K;
  auto full_sizes = new int[nlayers + 1];
  double flops = 0.0;
  for (i = 0; i < nlayers; i++) full_sizes[i + 1] = layer_sizes[i];
  full_sizes[0] = num_in;

  for (i = 0; i < nlayers; i++) {
    K = full_sizes[i];
    N = full_sizes[i + 1];
    flops += K * N * 2 + N + N;
    if (i > 0 && use_resnet_dt && K == N) flops += N;
    if (i == nlayers - 1 && !apply_output_bias) flops -= N;
  }
  flops += 1;
  forward = flops;
  flops += 2 * N * K;

  for (i = nlayers - 1; i > 0; i--) {
    N = full_sizes[i - 1];
    K = full_sizes[i];
    flops += 2 * N * K + K;
    if (use_resnet_dt && N == K && i > 1) flops += N;
    if (i == 1) break;
  }
  backward = flops - forward;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
NeuralNetworkPotential<Scalar>::NeuralNetworkPotential(Memory *mem, Error *err)
{
  memory = mem;
  error = err;

  use_resnet_dt = false;
  apply_output_bias = true;
  nlayers = 0;
  actfn = ReLU;
  max_size = 0;
  nelt = 0;
  n_in = 0;
  n_out = 0;
  n_pool = 0;
  pool = nullptr;
  act = new Activation<Scalar>();
  sum_output = true;

  backward_flops_per_one = 0.0;
  forward_flops_per_one = 0.0;
}

template <typename Scalar>
NeuralNetworkPotential<Scalar>::~NeuralNetworkPotential()
{
  int i, elti;

  for (elti = 0; elti < nelt; elti++) {
    for (i = 0; i < biases.size(); i++) { delete[] biases[elti][i]; }
    for (i = 0; i < weights.size(); i++) { delete[] weights[elti][i]; }
    weights[elti].clear();
    biases[elti].clear();
    memory->destroy(dz[elti]);
  }

  weights.clear();
  biases.clear();
  dz.clear();

  memory->destroy(pool);

  delete act;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::setup_global(cnpy::npz_t &npz)
{
  int i, elti;
  int nelt_;
  int actfn_, nlayers_;
  int num_in_;
  bool use_resnet_dt_, apply_output_bias_;
  int *layer_sizes;
  Scalar ***weights_, ***biases_;

  nelt_ = *npz["nelt"].data<int>();
  nlayers_ = *npz["nlayers"].data<int>();
  actfn_ = *npz["actfn"].data<int>();
  layer_sizes = npz["layer_sizes"].data_copy<int>();
  use_resnet_dt_ = *npz["use_resnet_dt"].data<int>();
  apply_output_bias_ = *npz["apply_output_bias"].data<int>();

  weights_ = new Scalar **[nelt_];
  biases_ = new Scalar **[nelt_];
  for (elti = 0; elti < nelt_; elti++) {
    weights_[elti] = new Scalar *[nlayers_];
    biases_[elti] = new Scalar *[nlayers_];
    for (i = 0; i < nlayers_; i++) {
      weights_[elti][i] =
          npz[fmt::format("weights_{}_{}", elti, i)].data<Scalar>();
      if (i < nlayers_ - 1 || apply_output_bias_) {
        biases_[elti][i] =
            npz[fmt::format("biases_{}_{}", elti, i)].data<Scalar>();
      }
    }
  }
  num_in_ = static_cast<int>(npz["weights_0_0"].shape[0]);

  setup_global(nelt_, num_in_, nlayers_, layer_sizes, actfn_, weights_, biases_,
               use_resnet_dt_, apply_output_bias_);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::setup_global(
    int nelt_, int num_in, int nlayers_, const int *layer_sizes, int actfn_,
    Scalar ***weights_, Scalar ***biases_, bool use_resnet_dt_,
    bool apply_output_bias_)
{
  int i, elti, n_rows, n_cols, n_res;

  this->nelt = nelt_;
  this->nlayers = nlayers_;
  this->use_resnet_dt = use_resnet_dt_;
  this->apply_output_bias = apply_output_bias_;

  if (actfn_ == 0)
    this->actfn = ReLU;
  else if (actfn_ == 1)
    this->actfn = Softplus;
  else if (actfn_ == 2)
    this->actfn = Tanh;
  else
    this->actfn = Squareplus;

  this->full_sizes[0] = num_in;
  this->max_size = num_in;
  this->n_in = num_in;
  this->n_out = layer_sizes[nlayers - 1];
  n_res = 0;
  for (i = 0; i < nlayers; i++) {
    this->full_sizes[i + 1] = layer_sizes[i];
    this->max_size = MAX(max_size, layer_sizes[i]);
    if (i > 0 && full_sizes[i] == full_sizes[i + 1]) n_res++;
  }
  if (n_res == 0) use_resnet_dt = false;

  for (elti = 0; elti < nelt; elti++) {
    nmax[elti] = 0;
    nlocal[elti] = 0;
    for (i = 0; i < nlayers; i++) {
      n_rows = full_sizes[i];
      n_cols = full_sizes[i + 1];
      this->weights[elti][i] = new Scalar[n_rows * n_cols];
      cppblas_copy(n_rows * n_cols, weights_[elti][i], 1,
                   this->weights[elti][i], 1);
      if (i < nlayers - 1 || apply_output_bias) {
        this->biases[elti][i] = new Scalar[n_cols];
        cppblas_copy(n_cols, biases_[elti][i], 1, this->biases[elti][i], 1);
      }
    }
  }

  calculate_nn_flops_per_one(nlayers, layer_sizes, num_in, use_resnet_dt,
                             apply_output_bias, forward_flops_per_one,
                             backward_flops_per_one);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar NeuralNetworkPotential<Scalar>::compute(int elti, Scalar *x_in, int n,
                                               Scalar *y, Scalar *dydx)
{
  double sum;
  sum = forward(elti, x_in, n, y);
  backward(elti, nullptr, dydx);
  return sum;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar NeuralNetworkPotential<Scalar>::forward(int elti, Scalar *x_in, int n,
                                               Scalar *y)
{ 
  if (use_resnet_dt || actfn == Softplus)
    return forward_resnet(elti, x_in, n, y);
  else
    // Computing gradients of softplus(x) will be significantly slower for this
    // implementation.
    return forward_dense(elti, x_in, n, y);
}

template <typename Scalar>
Scalar NeuralNetworkPotential<Scalar>::forward_resnet(int elti, Scalar *x_in,
                                                      int n, Scalar *y)
{
  int i, j;
  int M, N, K;
  Scalar *x;
  Scalar *C;
  bool C_is_pool_0;

  nlocal[elti] = n;
  if (n > nmax[elti]) {
    memory->destroy(dz[elti]);
    nmax[elti] = n;
    memory->create(dz[elti], nlayers - 1, n * max_size, "nnp:dz");
  }
  if (n > n_pool) {
    memory->destroy(pool);
    n_pool = n;
    memory->create(pool, 2, n * max_size, "nnp:pool");
  }

  C_is_pool_0 = true;
  C = pool[0];
  x = x_in;
  M = n;

  for (i = 0; i < this->nlayers; i++) {
    K = this->full_sizes[i];
    N = this->full_sizes[i + 1];

    cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, x, K,
                 this->weights[elti][i], N, 0.0, C, N);
#if defined(BUILD_BASELINE)
    if (i < this->nlayers - 1 || apply_output_bias) {
      for (j = 0; j < M; j++) {
        cppblas_axpy(N, 1.0, this->biases[elti][i], 1, &C[j * N], 1);
      }
    }
    if (i < this->nlayers - 1) {
      athread_tensoralloy_act(actfn, M * N, C, dz[elti][i]);
    } else {
      break;
    }
    if (i > 0 && use_resnet_dt && K == N)
      cppblas_axpy(M * N, 1.0, x, 1.0, C, 1);
#else
    if (i < nlayers - 1) {
      bool res = i > 0 && use_resnet_dt && K == N;
      athread_tensoralloy_axpy_act(actfn, res, M, N, x, C, dz[elti][i], this->biases[elti][i]);
    } else {
      if (apply_output_bias) {
        athread_tensoralloy_axpy(M, N, this->biases[elti][i], C);
      }
      break;
    }
#endif

    if (C_is_pool_0) {
      x = pool[0];
      C = pool[1];
    } else {
      x = pool[1];
      C = pool[0];
    }
    if (i == nlayers - 2 && y) C = y;
    C_is_pool_0 = !C_is_pool_0;
  }

  if (sum_output)
    return cppblas_dot(M * n_out, act->ones_like(M * n_out), 1, C, 1);
  else
    return 0.0;
}

template <typename Scalar>
Scalar NeuralNetworkPotential<Scalar>::forward_dense(int elti, Scalar *x_in,
                                                     int n, Scalar *y)
{
  int i, j;
  int M, N, K;
  Scalar *x;
  Scalar *C;

  nlocal[elti] = n;
  if (n > nmax[elti]) {
    memory->destroy(dz[elti]);
    nmax[elti] = n;
    memory->create(dz[elti], nlayers - 1, n * max_size, "nnp:dz");
  }
  if (n > n_pool) {
    memory->destroy(pool);
    n_pool = n;
    memory->create(pool, 1, n * max_size, "nnp:pool");
  }

  C = dz[elti][0];
  x = x_in;
  M = n;

  for (i = 0; i < this->nlayers; i++) {
    K = this->full_sizes[i];
    N = this->full_sizes[i + 1];
    cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, x, K,
                 this->weights[elti][i], N, 0.0, C, N);
    if (i < this->nlayers - 1 || apply_output_bias) {
#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (j = 0; j < M; j++) {
        cppblas_axpy(N, 1.0, this->biases[elti][i], 1, &C[j * N], 1);
      }
    }
    if (i < this->nlayers - 1) {
      act->activate(actfn, C, M * N);
    } else {
      break;
    }
    x = C;
    if (i < nlayers - 2) {
      C = dz[elti][i + 1];
    } else if (y) {
      C = y;
    } else {
      C = pool[0];
    }
  }

  if (sum_output)
    return cppblas_dot(M * n_out, act->ones_like(M * n_out), 1, C, 1);
  else
    return 0.0;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::backward(int elti, Scalar *grad_in,
                                              Scalar *grad_out)
{
  if (use_resnet_dt || actfn == Softplus)
    backward_resnet(elti, grad_in, grad_out);
  else
    backward_dense(elti, grad_in, grad_out);
}

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::backward_resnet(int elti, Scalar *grad_in,
                                                     Scalar *grad_out)
{
  int i;
  int M, N, K;
  Scalar *x;
  int j;

  M = nlocal[elti];
  K = n_out;
  N = this->full_sizes[nlayers - 1];

  if (grad_in == nullptr) {
    x = act->ones_like(M * K);
  } else {
    x = grad_in;
  }
  cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, x, K,
               this->weights[elti][nlayers - 1], K, 0.0, grad_out, N);

  for (i = nlayers - 1; i > 0; i--) {
    N = this->full_sizes[i - 1];
    K = this->full_sizes[i];
    athread_tensoralloy_vmul(M * K, grad_out, dz[elti][i - 1]);
    if (use_resnet_dt && N == K && i > 1) {
      cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                 dz[elti][i - 1], K, this->weights[elti][i - 1], K, 1.0, grad_out, N);
    } else {
      cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0,
                 dz[elti][i - 1], K, this->weights[elti][i - 1], K, 0.0, grad_out, N);
    }
  }
}

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::backward_dense(int elti, Scalar *grad_in,
                                                    Scalar *grad_out)
{
  int i;
  int M, N, K;
  Scalar *x, *C;
  C = pool[0];
  M = nlocal[elti];
  K = n_out;
  N = this->full_sizes[nlayers - 1];

#if !defined(EIGEN_USE_MKL_ALL)
  int j;
#endif

  if (grad_in == nullptr) {
    x = act->ones_like(M * K);
  } else {
    x = grad_in;
  }
  cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, x, K,
               this->weights[elti][nlayers - 1], K, 0.0, C, N);

  for (i = nlayers - 1; i > 0; i--) {
    N = this->full_sizes[i - 1];
    K = this->full_sizes[i];
    x = dz[elti][i - 1];
    act->grad(actfn, x, M * K);
#if defined(EIGEN_USE_MKL_ALL)
    cppvml_mul(M * K, C, x, x);
#else
    for (j = 0; j < M * K; j++) { x[j] *= C[j]; }
#endif
    if (i == 1) {
      C = grad_out;
    }
    cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, x, K,
                 this->weights[elti][i - 1], K, 0.0, C, N);
    if (i == 1) break;
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double NeuralNetworkPotential<Scalar>::get_memory_usage()
{
  int elti, i;
  int n_rows, n_cols;
  auto bytes = static_cast<double>(n_pool * max_size * 2);
  for (elti = 0; elti < nelt; elti++) {
    bytes += static_cast<double>(nmax[elti] * (nlayers - 1) * max_size);
    for (i = 0; i < nlayers; i++) {
      n_rows = full_sizes[i];
      n_cols = full_sizes[i + 1];
      bytes += static_cast<double>(n_rows * n_cols);
      if (i < nlayers - 1 || apply_output_bias) {
        bytes += static_cast<double>(n_cols);
      }
    }
  }
  bytes *= static_cast<double>(sizeof(Scalar));
  if (act) bytes += act->get_memory_usage();
  return bytes;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double NeuralNetworkPotential<Scalar>::get_flops_per_atom()
{
  return forward_flops_per_one + backward_flops_per_one;
}

template <typename Scalar>
void NeuralNetworkPotential<Scalar>::get_sw_nnp_config(int *use_resnet,
                  int *nlayers, int max_layer, int max_col,
                  int **layer_sizes, int **input_sizes, Scalar **bias,
                  Scalar **last_kernel, Scalar **kernel_matrix_T_global,
                  Scalar **kernel_matrix_bp_global, int config_id)
{
  *use_resnet = this->use_resnet_dt;
  *nlayers = this->nlayers;
  *layer_sizes = new int[this->nlayers];
  *input_sizes = new int[this->nlayers];
  for (int i = 0; i < this->nlayers; i++) {
    (*layer_sizes)[i] = this->full_sizes[i + 1];
    (*input_sizes)[i] = this->full_sizes[i];
  }
  *bias = new Scalar[this->nelt * this->nlayers * max_layer];
  *kernel_matrix_T_global = new Scalar[64 * this->nelt * max_col * max_layer];
  *kernel_matrix_bp_global = new Scalar[64 * this->nelt * max_col * max_layer];
  *last_kernel = new Scalar[this->nelt * max_layer];
  for (int elti = 0; elti < this->nelt; elti++) {
    for (int i = 0; i < this->nlayers; i++) {
      cppblas_copy(this->full_sizes[i + 1], this->biases[elti][i], 1, 
                  *bias + elti * this->nlayers * max_layer + i * max_layer, 1);
    }
  }
  for (int elti = 0; elti < this->nelt; elti++) {
    cppblas_copy(this->full_sizes[this->nlayers - 1], 
                 this->weights[elti][this->nlayers - 1], 1,
                 *last_kernel + elti * max_layer, 1);
  }

  // param is row-major
  for (int elti = 0; elti < this->nelt; elti++) {
    for (int rank = 0; rank < 64; rank++){
      Scalar *kernel_matrix_T = *kernel_matrix_T_global + elti * 64 * 
        max_col * max_layer + rank * max_col * max_layer;
      Scalar *kernel_matrix_bp = *kernel_matrix_bp_global + elti * 64 * 
        max_col * max_layer + rank * max_col * max_layer;
      
      // Predefined config sets
      if (config_id == 1) { // 512 128 128 128 128 1  
        if (rank >= 0 && rank <= 15) {
          for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 512; j++) {
              kernel_matrix_T[i * 512 + j] =
                  weights[elti][0][j * 128 + 8 * rank + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][0] + 32 * rank * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 16 && rank <= 19) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][1][j * 128 + 32 * (rank - 16) + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][1] + 32 * (rank - 16) * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 20 && rank <= 23) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][2][j * 128 + 32 * (rank - 20) + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][2] + 32 * (rank - 20) * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 24 && rank <= 27) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][3][j * 128 + 32 * (rank - 24) + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][3] + 32 * (rank - 24) * 128, 1,
                      kernel_matrix_bp, 1);
        }
      } else if (config_id == 2) { // 128 128 128 128 1
        if (rank >= 0 && rank <= 7) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][0][j * 128 + 16 * rank + i];
            }
          }
          cppblas_copy(16 * 128, weights[elti][0] + 16 * rank * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 8 && rank <= 15) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][1][j * 128 + 16 * (rank - 8) + i];
            }
          }
          cppblas_copy(16 * 128, weights[elti][1] + 16 * (rank - 8) * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 16 && rank <= 23) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][2][j * 128 + 16 * (rank - 16) + i];
            }
          }
          cppblas_copy(16 * 128, weights[elti][2] + 16 * (rank - 16) * 128, 1,
                      kernel_matrix_bp, 1);
        }
      } else if (config_id == 3) { // 128 128 128 128 1
        if (rank >= 0 && rank <= 3) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][0][j * 128 + 32 * rank + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][0] + 32 * rank * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 4 && rank <= 7) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][1][j * 128 + 32 * (rank - 4) + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][1] + 32 * (rank - 4) * 128, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 8 && rank <= 11) {
          for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 128; j++) {
              kernel_matrix_T[i * 128 + j] =
                  weights[elti][2][j * 128 + 32 * (rank - 8) + i];
            }
          }
          cppblas_copy(32 * 128, weights[elti][2] + 32 * (rank - 8) * 128, 1,
                      kernel_matrix_bp, 1);
        }
      } else if (config_id == 4) { // 512 64 64 64 64 1
        if (rank >= 0 && rank <= 15) {
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 512; j++) {
              kernel_matrix_T[i * 512 + j] =
                  weights[elti][0][j * 64 + 4 * rank + i];
            }
          }
          cppblas_copy(32 * 64, weights[elti][0] + 32 * rank * 64, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 16 && rank <= 19) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 64; j++) {
              kernel_matrix_T[i * 64 + j] =
                  weights[elti][1][j * 64 + 16 * (rank - 16) + i];
            }
          }
          cppblas_copy(16 * 64, weights[elti][1] + 16 * (rank - 16) * 64, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 20 && rank <= 23) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 64; j++) {
              kernel_matrix_T[i * 64 + j] =
                  weights[elti][2][j * 64 + 16 * (rank - 20) + i];
            }
          }
          cppblas_copy(16 * 64, weights[elti][2] + 16 * (rank - 20) * 64, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 24 && rank <= 27) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 64; j++) {
              kernel_matrix_T[i * 64 + j] =
                  weights[elti][3][j * 64 + 16 * (rank - 24) + i];
            }
          }
          cppblas_copy(16 * 64, weights[elti][3] + 16 * (rank - 24) * 64, 1,
                      kernel_matrix_bp, 1);
        }
      } else if (config_id == 5) { // 768 64 64 64 1
        if (rank >= 0 && rank <= 15) {
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 768; j++) {
              kernel_matrix_T[i * 768 + j] =
                  weights[elti][0][j * 64 + 4 * rank + i];
            }
          }
          cppblas_copy(48 * 64, weights[elti][0] + 48 * rank * 64, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 16 && rank <= 19) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 64; j++) {
              kernel_matrix_T[i * 64 + j] =
                  weights[elti][1][j * 64 + 16 * (rank - 16) + i];
            }
          }
          cppblas_copy(16 * 64, weights[elti][1] + 16 * (rank - 16) * 64, 1,
                      kernel_matrix_bp, 1);
        } else if (rank >= 20 && rank <= 23) {
          for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 64; j++) {
              kernel_matrix_T[i * 64 + j] =
                  weights[elti][2][j * 64 + 16 * (rank - 20) + i];
            }
          }
          cppblas_copy(16 * 64, weights[elti][2] + 16 * (rank - 20) * 64, 1,
                      kernel_matrix_bp, 1);
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::NeuralNetworkPotential<float>;
template class LAMMPS_NS::NeuralNetworkPotential<double>;
