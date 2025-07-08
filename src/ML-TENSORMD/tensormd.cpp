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

#include "tensormd.h"

#include "cutoff.h"
#include "descriptor.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "error.h"
#include "fnn.h"
#include "memory.h"
#include "nnp.h"
#include "tdnp.h"
#include "tensor_debug.h"
#include "tensor_ops.h"
#include "tensormd_timer.h"
#include "cppblas.hpp"

#include "athread/athread_tensoralloy.h"
#include "athread/sw_sysmem.h"

#include "sw_kernel/include/sw_TensorMD.h"
#include "sw_kernel/include/sw_nnp_main.h"

using namespace LAMMPS_NS;
using Eigen::IndexPair;
using Eigen::Tensor;
using Eigen::TensorMap;

#define MALLOC_BUFF 12

/* ---------------------------------------------------------------------- */

template <typename Scalar>
TensorMD<Scalar>::TensorMD(Memory *mem, Error *err, FILE *scr, FILE *log,
                           int mpid)
{
  memory = mem;
  error = err;
  screen = scr;
  logfile = log;
  mpiid = mpid;

  alocal = 0;
  nmax = 0;
  cmax = 0;
  i2row = nullptr;
  row2i = nullptr;

  neltypes = 0;
  rmax = 0.0;
  num_filters = 0;
  max_moment = 0;

  dims = 0;
  total_dims = 0;
  nv_dims = 0;

  is_T_symmetric = false;

  cutforce = 0;
  cutforcesq = 0;

  eltij_pos = nullptr;

  nnp = nullptr;
  fnn = nullptr;
  tdnp = nullptr;
  descriptor = nullptr;
  cutoff = nullptr;

  disable_timer = false;

  // Initialize the FLOPs timer
  timer = new TensorMDTimer(this->screen, this->logfile);
}

template <typename Scalar> TensorMD<Scalar>::~TensorMD()
{
  if (this->timer && !disable_timer) this->timer->print();

  memory->destroy(this->eltij_pos);

  delete this->nnp;
  delete this->fnn;
  delete this->timer;
  delete this->tdnp;
  delete this->cutoff;
  delete this->descriptor;

  libc_aligned_free(dMdrx);
  libc_aligned_free(V);
  libc_aligned_free(ijlist);
  libc_aligned_free(R);
  libc_aligned_free(mask);
  libc_aligned_free(sij);
  libc_aligned_free(dsij);
  libc_aligned_free(M);
  libc_aligned_free(dEdG);
  libc_aligned_free(P);
  libc_aligned_free(dEdS);
  libc_aligned_free(H);
  libc_aligned_free(dHdr);  
  libc_aligned_free(drdrx);

#if defined(USE_SW_NNP)
  delete[] layer_sizes;
  delete[] input_sizes;
  delete[] bias;
  delete[] kernel_matrix_T;
  delete[] kernel_matrix_bp;
  delete[] last_kernel;
#endif
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::read_npz(cnpy::npz_t &npz, int &nelt,
                                std::vector<int> &numbers,
                                std::vector<double> &masses)
{
  int i, j, k;
  int use_fnn;

  // Global parameters
  rmax = *npz["rmax"].data<Scalar>();
  neltypes = *npz["nelt"].data<int>();
  max_moment = *npz["max_moment"].data<int>();
  for (i = 0; i < npz["masses"].num_vals; i++) {
    masses.push_back(static_cast<double>(npz["masses"].data<Scalar>()[i]));
  }
  for (i = 0; i < npz["numbers"].num_vals; i++) {
    numbers.push_back(npz["numbers"].data<int>()[i]);
  }
  nelt = neltypes;

  // Setup the atomistic neural network potential
  if (npz.find("tdnp") != npz.end() && npz["tdnp"].data<int>()[0] == 1) {
    tdnp = new TDNP<Scalar>(memory, error);
    tdnp->setup_global(npz);
    timer->setup_tdnp(tdnp->get_flops_per_atom());
  } else {
    nnp = new NeuralNetworkPotential<Scalar>(memory, error);
    nnp->setup_global(npz);
    timer->setup_nn(nnp->get_flops_per_atom());
#if defined(USE_SW_NNP)
    nnp->get_sw_nnp_config(&use_resnet, &nlayers, max_layer, max_col,
                           &layer_sizes, &input_sizes, &bias, &last_kernel,
                           &kernel_matrix_T, &kernel_matrix_bp, config_id);
#endif
  }

  // Setup the descriptor
  use_fnn = *npz["use_fnn"].data<int>();
  if (use_fnn) {
    this->fnn = new FeaturePotential<Scalar>(this->memory, this->error, mpiid);
    this->fnn->setup_global(npz, num_filters);
    double forward, backward;
    this->fnn->get_flops_per_atom(forward, backward);
    this->timer->setup_fnn(forward, backward);
  } else {
    this->descriptor = new Descriptor<Scalar>(this->memory, this->error, npz);
    this->num_filters = this->descriptor->get_num_filters();
  }

  // is T_md symmetric?
  if (npz.find("is_T_symmetric") != npz.end() &&
      npz["is_T_symmetric"].data<int>()[0] == 1) {
    is_T_symmetric = true;
  }

  if (*npz["fctype"].data<int>() == 0) {
    cutoff = new Cutoff<Scalar>(rmax);    // Cosine
  } else {
    cutoff = new Cutoff<Scalar>(rmax, 5.0);    // Polynomial
  }

  // Misc
  memory->create(eltij_pos, nelt, nelt, "pair:eltij_pos");
  for (i = 0; i < nelt; i++) {
    k = 1;
    for (j = 0; j < nelt; j++) {
      if (i == j) {
        eltij_pos[i][j] = 0;
      } else {
        eltij_pos[i][j] = k;
        k++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> void TensorMD<Scalar>::setup_global(double *cutmax)
{
  int m;
  const int vmD_nv[6] = {1, 3, 6, 10, 15, 21};

  // Force cutoff
  this->cutforce = this->rmax;
  this->cutforcesq = this->cutforce * this->cutforce;

  // Pass cutoff back to calling program
  *cutmax = static_cast<double>(this->cutforce);

  this->dims = this->num_filters * (this->max_moment + 1);
  this->total_dims = this->dims * this->neltypes;
  this->nv_dims = 0;
  for (m = 0; m < this->max_moment + 1; m++) {
    this->nv_dims += vmD_nv[m];
  }

  this->timer->setup(neltypes, nv_dims, max_moment + 1, num_filters,
                     cutoff->is_cosine());

  // Initialize the multiplicity tensor T_md
  Eigen::array<Eigen::Index, 2> shape{max_moment + 1, nv_dims};
  T.resize(shape);
  T.setZero();
  T(0, 0) = 1;
  if (max_moment > 0) {
    T(1, 1) = 1;    // x
    T(1, 2) = 1;    // y
    T(1, 3) = 1;    // z
    if (max_moment > 1) {
      T(2, 4) = 1;    // xx
      T(2, 5) = 2;    // xy
      T(2, 6) = 2;    // xz
      T(2, 7) = 1;    // yy
      T(2, 8) = 2;    // yz
      T(2, 9) = 1;    // zz
      if (max_moment <= 3 && is_T_symmetric) { T(2, 0) = -1.0 / 3.0; }
      if (max_moment > 2) {
        T(3, 10) = 1;    // xxx
        T(3, 11) = 3;    // xxy
        T(3, 12) = 3;    // xxz
        T(3, 13) = 3;    // xyy
        T(3, 14) = 6;    // xyz
        T(3, 15) = 3;    // xzz
        T(3, 16) = 1;    // yyy
        T(3, 17) = 3;    // yyz
        T(3, 18) = 3;    // yzz
        T(3, 19) = 1;    // zzz
        if (max_moment <= 3 && is_T_symmetric) {
          T(3, 1) = -3.0 / 5.0;
          T(3, 2) = -3.0 / 5.0;
          T(3, 3) = -3.0 / 5.0;
        }
        if (max_moment > 3) {
          T(4, 20) = 1;     // xxxx
          T(4, 21) = 4;     // xxxy
          T(4, 22) = 4;     // xxxz
          T(4, 23) = 6;     // xxyy
          T(4, 24) = 12;    // xxyz
          T(4, 25) = 6;     // xxzz
          T(4, 26) = 4;     // xyyy
          T(4, 27) = 12;    // xyyz
          T(4, 28) = 12;    // xyzz
          T(4, 29) = 4;     // xzzz
          T(4, 30) = 1;     // yyyy
          T(4, 31) = 4;     // yyyz
          T(4, 32) = 6;     // yyzz
          T(4, 33) = 4;     // yzzz
          T(4, 34) = 1;     // zzzz
          if (max_moment > 4) {
            T(5, 35) = 1;     // xxxxx
            T(5, 36) = 5;     // xxxxy
            T(5, 37) = 5;     // xxxxz
            T(5, 38) = 10;    // xxxyy
            T(5, 39) = 20;    // xxxyz
            T(5, 40) = 10;    // xxxzz
            T(5, 41) = 10;    // xxyyy
            T(5, 42) = 30;    // xxyyz
            T(5, 43) = 30;    // xxyzz
            T(5, 44) = 10;    // xxzzz
            T(5, 45) = 5;     // xyyyy
            T(5, 46) = 20;    // xyyyz
            T(5, 47) = 30;    // xyyzz
            T(5, 48) = 20;    // xyzzz
            T(5, 49) = 5;     // xzzzz
            T(5, 50) = 1;     // yyyyy
            T(5, 51) = 5;     // yyyyz
            T(5, 52) = 10;    // yyyzz
            T(5, 53) = 10;    // yyzzz
            T(5, 54) = 5;     // yzzzz
            T(5, 55) = 1;     // zzzzz
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_local(int nlocal, int numneigh_max,
                                   const int *typenums)
{
  int i, elti;
  int a, b, c, d, k, m, x;
  size_t mmax;

  // update the timer
  timer->tic();
  timer->update(nlocal, numneigh_max);

  // grow local arrays if necessary
  if (nlocal > nmax || numneigh_max > cmax) {
    if (neltypes > 1) {
      memory->destroy(i2row);
      memory->destroy(row2i);
    }
    nmax = nlocal + MALLOC_BUFF;
    cmax = numneigh_max + MALLOC_BUFF;
    if (neltypes > 1) {
      memory->create(row2i, nmax, "tensoralloy:row2i");
      memory->create(i2row, nmax, "tensoralloy:i2row");
    }

    a = nmax;
    b = neltypes;
    c = cmax;
    d = nv_dims;
    k = num_filters;
    m = max_moment + 1;
    x = 3;
    
    // The tensor-based algorithm
    libc_aligned_free(ijlist);
    libc_aligned_free(R);
    libc_aligned_free(mask);
    libc_aligned_free(M);
    libc_aligned_free(P);
    libc_aligned_free(dEdS);
    libc_aligned_free(dEdG);
    libc_aligned_free(H);
    libc_aligned_free(dHdr);
    libc_aligned_free(drdrx);
#if defined(BUILD_BASELINE)
    libc_aligned_free(dMdrx);
    dMdrx = (Scalar *) libc_aligned_malloc(a * b * c * x * d * sizeof(Scalar));
    libc_aligned_free(V);
    V = (Scalar *) libc_aligned_malloc(a * b * c * d * sizeof(Scalar));
#endif

    ijlist = (int *) libc_aligned_malloc(a * b * c * 2 * sizeof(int));
    R = (Scalar *) libc_aligned_malloc(a * b * c * sizeof(Scalar));
    mask = (int *) libc_aligned_malloc(a * b * c * sizeof(int));
    if ((fnn && !fnn->is_interpolatable()) ||
        (descriptor && !descriptor->is_interpolatable())) {
      libc_aligned_free(sij);
      libc_aligned_free(dsij);
      sij = (Scalar *) libc_aligned_malloc(a * b * c * sizeof(Scalar));
      dsij = (Scalar *) libc_aligned_malloc(a * b * c * sizeof(Scalar));
    }
    M = (Scalar *) libc_aligned_malloc(a * b * c * d * sizeof(Scalar));

#if defined(USE_SW_NNP)
    a_padded = a + 2 * block * 64 - a % (block * 64);
    mmax = MAX(a_padded * b * k * m, a * b * k * d);
    dEdS = (Scalar *) libc_aligned_malloc(mmax * sizeof(Scalar));
    G = dEdS;
    dEdG = (Scalar *) libc_aligned_malloc(a_padded * b * k * m * sizeof(Scalar));
#else
    dEdS = (Scalar *) libc_aligned_malloc(a * b * k * d * sizeof(Scalar));
    G = dEdS;
    dEdG = (Scalar *) libc_aligned_malloc(a * b * k * m * sizeof(Scalar));
#endif
    H = (Scalar *) libc_aligned_malloc(a * b * c * k * sizeof(Scalar));
    dHdr = (Scalar *) libc_aligned_malloc(a * b * c * k * sizeof(Scalar));
    drdrx = (Scalar *) libc_aligned_malloc(a * b * c * x * sizeof(Scalar));
    mmax = MAX(a * b * k * d, a * b * c * x * 2);
    P = (Scalar *) libc_aligned_malloc(mmax * sizeof(Scalar));
    F1 = P;
    F2 = F1 + a * b * c * x;
#if defined(BUILD_BASELINE)
    S = (Scalar *) libc_aligned_malloc(a * b * k * d * sizeof(Scalar));
#endif
  }

  // Setup imap when `neltypes > 1` because `imap[i] = i` is always
  // true if `neltypes == 1`.

  eltind[0].offset = 0;
  eltind[0].row = 0;
  if (neltypes > 1) {
    for (i = 0; i < nmax; i++) { i2row[i] = row2i[i] = 0; }
    m = typenums[0];
    for (elti = 1; elti < neltypes; elti++) {
      eltind[elti].offset = m;
      eltind[elti].row = m;
      m += typenums[elti];
    }
  }
  alocal = nlocal;
  clocal = numneigh_max;

  timer->record(TIMER::ALLOC);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_tensors(const int *ilist, const int inum,
                                     const int *type, const int *fmap,
                                     double **pos, int numneigh_max,
                                     const int *numneigh, int **firstneigh)
{
  int i, a, elti;

  timer->tic();
  if (neltypes > 1) {
    for (int ii = 0; ii < inum; ii++)
    {
      i = ilist[ii];
      elti = fmap[type[i]];
      a = eltind[elti].row;
      eltind[elti].row++;
      i2row[i] = a;
      row2i[a] = i;
    }
  }
  athread_tensoralloy_setup_tensors(inum, i2row, type, fmap, pos, numneigh, firstneigh, 
                                    neltypes, this->cutforcesq, ilist,
                                    eltij_pos, clocal, nv_dims, 
                                    ijlist, R, mask, M, drdrx, dMdrx);
  timer->record(TIMER::SETUP);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int TensorMD<Scalar>::get_exact_cmax(int *ilist, int inum, const int *type,
                                     const int *fmap, double **pos,
                                     int *numneigh, int **firstneigh)
{
  timer->tic();
  int cmax_ = 0;
  int cmax_array[64] = {0};
  athread_tensoralloy_get_cmax(inum, type, fmap, pos, numneigh, firstneigh, 
                          neltypes, cutforcesq, eltij_pos, ilist, cmax_array);
  for (int i = 0; i < 64; i++) {
    if (cmax_array[i] > cmax_) {
      cmax_ = cmax_array[i];
    }
  }
  timer->record(TIMER::SETUP);
  return cmax_;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::calc_tensor_density(int *ilist, const int inum,
                                           const int *type, const int *fmap,
                                           double **pos, int numneigh_max,
                                           int *numneigh, int **firstneigh)
{
  int a, b, c, d, k, m;

  // Setup the R, M, dMdx, dMdrx tensors
  setup_tensors(ilist, inum, type, fmap, pos, numneigh_max, numneigh,
                firstneigh);

  // Subsets
  a = alocal;
  b = neltypes;
  c = clocal;
  d = nv_dims;
  k = num_filters;
  m = max_moment + 1;
  TensorMap<Tensor<Scalar, 3>> R_{R, {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> sij_{sij, {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> dsij_{dsij, {c, b, a}};
  TensorMap<Tensor<Scalar, 4>> H_{H, {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> dHdr_{dHdr, {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> M_{M, {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> P_{P, {d, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> dEdS_{dEdS, {d, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> G_{G, {m, k, b, a}};
  TensorMap<Tensor<int, 3>> mask_{mask, {c, b, a}};

  // Calculate descriptors: FNN or analytic functions
  if (this->fnn && this->fnn->is_interpolatable()) {
    timer->tic();
    this->fnn->interpolate(R_.data(), a * b * c, H_.data(), dHdr_.data());
    timer->record(TIMER::FNN_INTERP);
  } else if (this->descriptor && this->descriptor->is_interpolatable()) {
    timer->tic();
    this->descriptor->interpolate(R_.data(), a * b * c, H_.data(),
                                  dHdr_.data());
    timer->record(TIMER::DESCRIPTOR);
  } else {
    Tensor<Scalar, 3> mask_f = mask_.cast<Scalar>();
    TensorMap<Tensor<Scalar, 3>> mask_{mask_f.data(), {c, b, a}};

    // Calculate cutoff coefficients
    timer->tic();
    cutoff->compute(R_, mask_, &sij_, &dsij_);
    timer->record(TIMER::CUTOFF);

    timer->tic();
    if (this->fnn) {
      Eigen::array<int, 4> bcast{num_filters, 1, 1, 1};
      Eigen::array<int, 4> shape{1, c, b, a};
      auto tiled_sij = sij_.reshape(shape).broadcast(bcast);
      auto tiled_dsij = dsij_.reshape(shape).broadcast(bcast);
      this->fnn->forward(R_.data(), a * b * c, H_.data());
      timer->record(TIMER::FNN_FORWARD);
      timer->tic();
      dHdr_ = tiled_dsij * H_;
      H_ = tiled_sij * H_;
    } else {
      this->descriptor->compute(R_, sij_, dsij_, &H_, &dHdr_);
    }
    timer->record(TIMER::DESCRIPTOR);
  }

#if defined(BUILD_BASELINE)
  timer->tic();
  kernel_P(M_, H_, &P_);
  TensorMap<Tensor<Scalar, 4>> S_{S, {d, k, b, a}};
  athread_tensoralloy_P_2(a * b * k, m, S_.data(), P_.data());
  cppblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a * b * k, m, d,
                1.0, S_.data(), d, T.data(), m, 0.0, G_.data(), m);
  timer->record(TIMER::Kernel1);
#else
  timer->tic();
  sw_dgemm_kernel1(a * b, k, d, c, m, H, M, T.data(), P, G);
  timer->record(TIMER::Kernel1);
#endif
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_interpolation(Scalar delta, int algo)
{
  if (rmax == 0.0) error->all(FLERR, "Invalid rmax for interpolation");

  if (this->fnn)
    this->fnn->setup_ration(delta, rmax, algo, cutoff);
  else
    this->descriptor->setup_ration(delta, rmax, algo, cutoff);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar TensorMD<Scalar>::run(int *typenums, double etemperature, double *eatom,
                             double **f, double **vatom, double scale,
                             double *satom, double &S, double *aatom, double &A,
                             int atommax)
{
  timer->tic();

  int k, a, b, c, d, x, m;
  int i, j, elti;
  Scalar ytotal[3] = {0.0, 0.0, 0.0};
  Scalar *x_in;
  Scalar *y[3];
  Scalar dy[3];
  double df[3], virial[6];

#ifdef USE_SW_NNP
  y[0] = new Scalar[a_padded];
#else
  y[0] = eatom ? new Scalar[alocal] : nullptr;
#endif
  y[1] = satom ? new Scalar[alocal] : nullptr;
  y[2] = aatom ? new Scalar[alocal] : nullptr;

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < neltypes; elti++) {
    if (typenums[elti] == 0) { continue; }
    x_in = &G[eltind[elti].offset * total_dims];
    if (this->nnp) {
#ifdef USE_SW_NNP
      int typenums_padded = typenums[elti] + block * 64 - typenums[elti] % (block * 64);
      sw_nnp_compute(block, max_col, typenums_padded, total_dims, max_moment, 
                     use_resnet, nlayers, max_layer, kernel_ngroups, 
                     kernel_cpe_id, ncols_on_cpe, kernel_ngroups_bp, 
                     kernel_cpe_id_bp, ncols_on_cpe_bp, layer_sizes, input_sizes, 
                     bias + elti * nlayers * max_layer, 
                     last_kernel + elti * max_layer,
                     kernel_matrix_T + 64 * elti * max_col * max_layer, 
                     kernel_matrix_bp + 64 * elti * max_col * max_layer, 
                     x_in, &y[0][eltind[elti].offset], 
                     &dEdG[eltind[elti].offset * total_dims]);
#else
      ytotal[0] +=
          this->nnp->compute(elti, x_in, typenums[elti],
                             eatom ? &y[0][eltind[elti].offset] : nullptr,
                             &dEdG[eltind[elti].offset * total_dims]);
#endif
    } else {
      ytotal[0] += this->tdnp->compute(
          elti, x_in, typenums[elti],
          &dEdG[eltind[elti].offset * total_dims], etemperature,
          eatom ? &y[0][eltind[elti].offset] : nullptr,
          satom ? &y[1][eltind[elti].offset] : nullptr, dy[1],
          aatom ? &y[2][eltind[elti].offset] : nullptr, dy[2]);
      ytotal[1] += dy[1];
      ytotal[2] += dy[2];
    }
  }

#ifdef USE_SW_NNP
  for (int i = 0; i < alocal; i++) {
    ytotal[0] += y[0][i];
  }
#endif

  S = ytotal[1] * scale;
  A = ytotal[2] * scale;

  if (eatom) {
    for (a = 0; a < alocal; a++) {
      if (neltypes == 1) {
        eatom[a] = static_cast<double>(y[0][a]) * scale;
      } else {
        eatom[row2i[a]] = static_cast<double>(y[0][a]) * scale;
      }
    }
  }
  if (satom) {
    for (a = 0; a < alocal; a++) {
      if (neltypes == 1) {
        satom[a] = static_cast<double>(y[1][a]) * scale;
      } else {
        satom[row2i[a]] = static_cast<double>(y[1][a]) * scale;
      }
    }
  }
  if (aatom) {
    for (a = 0; a < alocal; a++) {
      if (neltypes == 1) {
        aatom[a] = static_cast<double>(y[2][a]) * scale;
      } else {
        aatom[row2i[a]] = static_cast<double>(y[2][a]) * scale;
      }
    }
  }
  timer->record(TIMER::NN_COMPUTE);

  // Declare the tensor dimensions
  a = alocal;
  b = neltypes;
  c = clocal;
  d = nv_dims;
  k = num_filters;
  x = 3;
  m = max_moment + 1;
  TensorMap<Tensor<Scalar, 3>> R_{R, {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> sij_{sij, {c, b, a}};
  TensorMap<Tensor<Scalar, 4>> drdrx_{drdrx, {x, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> F1_{F1, {x, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> F2_{F2, {x, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> H_{H, {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> U_{H, {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> dHdr_{dHdr, {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> M_{M, {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> V_{V, {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> P_{P, {d, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> G_{G, {m, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> dEdG_{dEdG, {m, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> dEdS_{dEdS, {d, k, b, a}};
  TensorMap<Tensor<int, 3>> mask_{mask, {c, b, a}};
  TensorMap<Tensor<int, 4>> ijlist_{ijlist, {2, c, b, a}};

  timer->tic();

  // dEdG_mkba = dEdG_mkba * dGdQ_mkba
  // dGdQ.chip(0, 0) = 0.5 / G.chip(0, 0)
  // dGdQ.chip(m>0, 0) = 1.0
  // dEdP_dkba = 2 * T_md * dGdQ_mkba * P_dkba
  // Eigen::array<IndexPair<int>, 1> contract_dims_1{IndexPair<int>{0, 0}};  
#if defined(BUILD_BASELINE)
  kernel_dEdG(G_, &dEdG_);
  kernel_dEdS(T, dEdG_, P_, &dEdS_);
#else
#ifndef USE_SW_NNP
  athread_tensoralloy_dEdG(a * b * k, m, G_.data(), dEdG_.data());
#endif
  sw_dgemm_transB_vdmul(a * b, k, d, m, 2, dEdG, T.data(), 0, P, dEdS);
#endif
  timer->record(TIMER::DEDP);

  // V_dcba = dEdS_dkba * H_kcba
  // F2_xcba = dMdrx_dxcba * V_dcba
#if defined(BUILD_BASELINE)
  timer->tic();
  kernel_V<Scalar>(dEdS_, H_, &V_);
  timer->record(TIMER::V);
  timer->tic();
  TensorMap<Tensor<Scalar, 5>> dMdrx_{dMdrx, {d, x, c, b, a}};
  kernel_F2<Scalar>(dMdrx_, V_, mask_, &F2_);
  timer->record(TIMER::F2);
#else
  timer->tic();
  sw_dgemm_ddot_sum(a*b, c, k, d, x, m, 1, H, dEdS, 0, drdrx, R, F2);
  timer->record(TIMER::Kernel3);
#endif

  // U_kcba = dEdS_dkba * M_dcba
  // F1_xcba = U_kcba * dHdr_kcba * drdrx_xcba
#if defined(BUILD_BASELINE)
  timer->tic();
  kernel_U<Scalar>(dEdS_, M_, &U_);
  timer->record(TIMER::U);
  timer->tic();
  Eigen::array<int, 1> sum_axis{0};
  Eigen::array<int, 4> shape{1, c, b, a};
  Eigen::array<int, 4> bcast_xcba{x, 1, 1, 1};
  if (this->fnn && !this->fnn->is_interpolatable()) {
    Eigen::array<int, 4> bcast_kcba{num_filters, 1, 1, 1};
    Tensor<Scalar, 4> dHdrp{1, c, b, a};
    Tensor<Scalar, 4> Up = U_ * sij_.reshape(shape).broadcast(bcast_kcba);
    this->fnn->backward(Up.data(), dHdrp.data());
    auto grad = (U_ * dHdr_).sum(sum_axis).reshape(shape) + dHdrp;
    F1_ = grad.broadcast(bcast_xcba) * drdrx_;
    timer->record(TIMER::FNN_BACKWARD);
  } else {
    kernel_F1<Scalar>(U_, dHdr_, drdrx_, mask_, &F1_);
  }
  timer->record(TIMER::F1);    
#else
  timer->tic();
  sw_dgemm_transB_ddot_sum(a*b, c, k, d, x, 1, M, dEdS, 0,
                            dHdr, drdrx, F1, mask);
  timer->record(TIMER::Kernel4);
#endif

  // Scatter update forces
  timer->tic();
  const int num_duplicates = 8;
  const int num_threads = 64;
  int vatom_flag = vatom == nullptr ? 0 : 1;
  farray = (double *)libc_aligned_malloc(sizeof(double) * atommax * num_duplicates * 3);
  athread_initialize_array_1d(atommax * 3 * num_duplicates, farray, 0.0, num_threads);
  if (vatom_flag) {
    varray = (double *)libc_aligned_malloc(sizeof(double) * atommax * num_duplicates * 6);
    athread_initialize_array_1d(atommax * 6 * num_duplicates, varray, 0.0, num_threads);
  }
  athread_tensoralloy_sum_forces(a, b, c, ijlist, F1, F2,
                                  drdrx, R, atommax, vatom_flag, farray, varray, num_duplicates);
  athread_tensoralloy_reduce_forces(atommax, farray, varray, f, vatom, vatom_flag, num_duplicates, num_threads);
  libc_aligned_free(farray);
  if (vatom_flag) libc_aligned_free(varray);
  timer->record(TIMER::FORCES);
  timer->begin();
  delete[] y[0];
  delete[] y[1];
  delete[] y[2];
  return ytotal[0] * scale;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> double TensorMD<Scalar>::memory_usage() const
{
  size_t size = 0;
  size += T.size();
  size += nmax * neltypes * cmax * 3; // R sij dsij
  size += nmax * neltypes * cmax * 2; // ijlist
  size += nmax * neltypes * cmax; // mask
#if defined(BUILD_BASELINE)
  size += nmax * neltypes * cmax * 3 * nv_dims; // dMdrx
  size += nmax * neltypes * cmax * nv_dims; // V
#endif
  size += nmax * neltypes * cmax * nv_dims; // M
  size += nmax * neltypes * cmax * num_filters * 2; // H dHdr
  size += nmax * neltypes * num_filters * (max_moment  + 1); // dEdG
  size += nmax * neltypes * num_filters * nv_dims * 2; // P dEdS
  size += nmax * neltypes * cmax * 3; // drdrx
  auto bytes = static_cast<double>(sizeof(Scalar) * size);
  bytes += static_cast<double>(sizeof(int) * nmax * 2);
  if (fnn) bytes += fnn->get_memory_usage();
  if (nnp) bytes += nnp->get_memory_usage();
  if (tdnp) bytes += tdnp->get_memory_usage();
  return bytes;
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::TensorMD<float>;
template class LAMMPS_NS::TensorMD<double>;
