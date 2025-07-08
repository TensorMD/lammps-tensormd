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
#include "tensor_ops.h"
#include "tensormd_timer.h"
#include "kernel.cuh"

using namespace LAMMPS_NS;
using Eigen::IndexPair;
using Eigen::Tensor;

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

  setup_device_GPU(mpiid, this->screen, this->logfile);
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

#ifdef GPU_BASELINE
  free_tensors_GPU(ilist_GPU, pos_GPU, numneigh_GPU, firstneigh_GPU,
                   ijlist_GPU, i2row_GPU, eltij_pos_GPU, fmap_type_GPU, R_GPU, G_GPU,
                   dEdG_GPU, T_GPU, P_GPU, dEdS_GPU, M_GPU, H_GPU, V_GPU,
                   dHdr_GPU, drdrx_GPU, F_GPU, dMdrx_GPU, mask_GPU, vatom_GPU);
#else
  free_tensors_GPU(ilist_GPU, pos_GPU, numneigh_GPU, firstneigh_GPU, ijlist_GPU,
                   i2row_GPU, eltij_pos_GPU, fmap_type_GPU, R_GPU, G_GPU,
                   dEdG_GPU, T_GPU, P_GPU, dEdS_GPU, M_GPU, H_GPU, V_GPU,
                   dHdr_GPU, drdrx_GPU, mask_GPU, vatom_GPU);
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
  for (m = 0; m < this->max_moment + 1; m++) { this->nv_dims += vmD_nv[m]; }

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

  setup_global_GPU(&T_GPU, T.data(), &eltij_pos_GPU, eltij_pos[0], neltypes,
                   nv_dims, max_moment + 1);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_local(int nlocal, int numneigh_max,
                                   const int *typenums)
{
  int i, elti, m;

  // update the timer
  timer->tic();
  timer->update(nlocal, numneigh_max);

  // grow local arrays if necessary
  if (nlocal > nmax) {
    if (neltypes > 1) {
      memory->destroy(i2row);
      memory->destroy(row2i);
    }
    nmax = nlocal;
    if (neltypes > 1) {
      memory->create(row2i, nmax, "tensoralloy:row2i");
      memory->create(i2row, nmax, "tensoralloy:i2row");
    }
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

  timer->record(TIMER::ALLOC);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int TensorMD<Scalar>::get_exact_cmax(int *ilist, int inum, const int *type,
                                     const int *fmap, double **pos,
                                     int *numneigh, int **firstneigh)
{
  int c0 = 0;

  timer->tic();
#pragma omp parallel for reduction(max : c0)
  for (int ii = 0; ii < inum; ii++) {
    int neigh[neltypes];
    for (int elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    int i = ilist[ii];
    int elti = fmap[type[i]];
    double xtmp = pos[i][0];
    double ytmp = pos[i][1];
    double ztmp = pos[i][2];
    for (int jn = 0; jn < numneigh[i]; jn++) {
      int j = firstneigh[i][jn];
      j &= NEIGHMASK;
      double dij0 = pos[j][0] - xtmp;
      double dij1 = pos[j][1] - ytmp;
      double dij2 = pos[j][2] - ztmp;
      double rij2 = dij0 * dij0 + dij1 * dij1 + dij2 * dij2;
      if (rij2 < this->cutforcesq && rij2 > 1e-10) {
        int eltj = fmap[type[j]];
        int b = eltij_pos[elti][eltj];
        neigh[b]++;
      }
    }
    for (int b = 0; b < neltypes; b++) {
      if (neigh[b] > c0) c0 = neigh[b];
    }
  }
  timer->record(TIMER::SETUP);
  return c0;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::calc_tensor_density(int *ilist, const int inum,
                                           const int *type, const int *fmap,
                                           double **pos, int numneigh_max,
                                           int *numneigh, int **firstneigh,
                                           int nmax_global, bool use_exact_cmax)
{
  int a, b, c, d, k, m;
  timer->tic();
  if (neltypes > 1) {
    int i, elti;
    for (int ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      elti = fmap[type[i]];
      a = eltind[elti].row;
      eltind[elti].row++;
      i2row[i] = a;
      row2i[a] = i;
    }
  }

#ifdef GPU_BASELINE
  numneigh_max_GPU = setup_tensors_GPU(
      ilist, inum, type, fmap, pos[0], numneigh_max, numneigh, &firstneigh[0],
      eltij_pos_GPU, i2row, &R_GPU, &G_GPU, &dEdG_GPU, &P_GPU, &dEdS_GPU,
      &M_GPU, &H_GPU, &V_GPU, &dHdr_GPU, &drdrx_GPU, &F_GPU, &dMdrx_GPU,
      &mask_GPU, &ilist_GPU, &fmap_type_GPU, &pos_GPU, &numneigh_GPU,
      &firstneigh_GPU, &ijlist_GPU, &i2row_GPU, cutforcesq, nmax_global, alocal,
      neltypes, nv_dims, num_filters, max_moment + 1, use_exact_cmax);
#else
  numneigh_max_GPU = setup_tensors_GPU(
      ilist, inum, type, fmap, pos[0], numneigh_max, numneigh, &firstneigh[0],
      eltij_pos_GPU, i2row, &R_GPU, &G_GPU, &dEdG_GPU, &P_GPU, &dEdS_GPU,
      &M_GPU, &H_GPU, &V_GPU, &dHdr_GPU, &drdrx_GPU, &F1_GPU, &F2_GPU,
      &mask_GPU, &ilist_GPU, &fmap_type_GPU, &pos_GPU, &numneigh_GPU,
      &firstneigh_GPU, &ijlist_GPU, &i2row_GPU, cutforcesq, nmax_global, alocal,
      neltypes, nv_dims, num_filters, max_moment + 1, use_exact_cmax);
#endif
  nmax_GPU = nmax_global;
  timer->update(alocal, numneigh_max_GPU);
  timer->record(TIMER::SETUP);

  // Subsets
  a = alocal;
  b = neltypes;
  c = numneigh_max_GPU;
  d = nv_dims;
  k = num_filters;
  m = max_moment + 1;

  // Calculate descriptors: FNN or analytic functions
  if (this->fnn && this->fnn->is_interpolatable()) {
    timer->tic();
    this->fnn->interpolate(R_GPU, a * b * c, H_GPU, dHdr_GPU);
    timer->record(TIMER::FNN_INTERP);
  }

  timer->tic();
#ifdef GPU_BASELINE
  kernel_1_GPU(M_GPU, H_GPU, P_GPU, T_GPU, G_GPU, dEdS_GPU, (int *)F_GPU, a, b, c, d, k, m);
#else
  kernel_1_GPU(M_GPU, H_GPU, P_GPU, T_GPU, G_GPU, dEdS_GPU, (int *) F1_GPU, a,
               b, c, d, k, m);
#endif
  timer->record(TIMER::Kernel1);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_interpolation(Scalar delta)
{
  if (rmax == 0.0) error->all(FLERR, "Invalid rmax for interpolation");

  if (this->fnn)
    this->fnn->setup_ration(delta, rmax, cutoff);
  else
    this->descriptor->setup_ration(delta, rmax, cutoff);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar TensorMD<Scalar>::run(int *typenums, double etemperature, double *eatom,
                             double **f, double **vatom, double scale,
                             double *satom, double &S, double *aatom, double &A)
{
  timer->tic();

  int k, a, b, c, d, x, m;
  int i, j, elti;
  Scalar ytotal[3] = {0.0, 0.0, 0.0};
  Scalar *x_in;
  Scalar *y[3];
  Scalar dy[3];
  double df[3], virial[6];

  y[0] = eatom ? new Scalar[alocal] : nullptr;
  y[1] = satom ? new Scalar[alocal] : nullptr;
  y[2] = aatom ? new Scalar[alocal] : nullptr;

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < neltypes; elti++) {
    if (typenums[elti] == 0) { continue; }
    if (this->nnp) {
      ytotal[0] += this->nnp->compute(
          elti, G_GPU + eltind[elti].offset * total_dims, typenums[elti],
          eatom ? &y[0][eltind[elti].offset] : nullptr,
          dEdG_GPU + eltind[elti].offset * total_dims);
    } else {
      ytotal[0] += this->tdnp->compute(
          elti, G_GPU + eltind[elti].offset * total_dims, typenums[elti],
          dEdG_GPU + eltind[elti].offset * total_dims, etemperature,
          eatom ? &y[0][eltind[elti].offset] : nullptr,
          eatom ? &y[1][eltind[elti].offset] : nullptr, dy[1],
          eatom ? &y[2][eltind[elti].offset] : nullptr, dy[2]);
      ytotal[1] += dy[1];
      ytotal[2] += dy[2];
    }
  }

  // if (isnan(ytotal[0])) {
  //   printf("rank-%d nnp output nan, cmax = %d, alocal = %d\n", this->mpiid,
  //         numneigh_max_GPU, alocal);
  // }

  S = ytotal[1] * scale;
  A = ytotal[2] * scale;

  // int acheck = -1;

  if (eatom) {
    for (a = 0; a < alocal; a++) {
      if (neltypes == 1) {
        // if (isnan(y[0][a])) {
        //    printf("rank-%d eatom nan: %d\n", this->mpiid, a);
        //    if (acheck < 0) acheck = a;
        // }
        eatom[a] = static_cast<double>(y[0][a]) * scale;
      } else {
        eatom[row2i[a]] = static_cast<double>(y[0][a]) * scale;
      }
    }
  }

  // if (isnan(ytotal[0])) {
  //   char hostname[256];
  //   gethostname(hostname, 256);
  //   printf("nan energy on host: %s\n", hostname);
  // }

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
  c = numneigh_max_GPU;
  d = nv_dims;
  k = num_filters;
  x = 3;
  m = max_moment + 1;

  // if (acheck >= 0) {
  //   Scalar *Gp = G_GPU + acheck * b * k * m;
  //   Scalar *Hp = H_GPU + acheck * b * c * k;
  //   Scalar *Mp = M_GPU + acheck * b * c * d;
  //   Scalar *Rp = R_GPU + acheck * b * c;
  //   int *maskp = mask_GPU + acheck * b * c;
  //   int *ijlistp = ijlist_GPU + acheck * b * c * 2;
  //   Scalar *dEdGp = dEdG_GPU + acheck * b * k * m;
  //   char hostname[256];
  //   gethostname(hostname, 256);
  //   printf("dump: a=%d, rank=%d, energy=%lf, hostname=%s\n", acheck, mpiid, ytotal[0], hostname);
  //   debug_array_GPU(numneigh_max_GPU, Gp, Hp, Mp, Rp, maskp, ijlistp, dEdGp);
  // }

  timer->tic();
  kernel2_GPU(G_GPU, T_GPU, dEdG_GPU, P_GPU, dEdS_GPU, m, d, k, b, a);
  timer->record(TIMER::DEDP);

  timer->tic();
  kernel_V_GPU(dEdS_GPU, H_GPU, V_GPU, a, b, d, c, k);
  timer->record(TIMER::V);
  timer->tic();
#ifdef GPU_BASELINE
  kernel_fused_F2_GPU(V_GPU, R_GPU, drdrx_GPU, dMdrx_GPU, F_GPU, a * b * c, d, m);
#else
  kernel_fused_F2_GPU(V_GPU, R_GPU, drdrx_GPU, F2_GPU, a * b * c, d, m);
#endif
  timer->record(TIMER::F2);

#ifdef GPU_BASELINE
  timer->tic();
  kernel_U_GPU(dEdS_GPU, M_GPU, H_GPU, a, b, d, c, k);
  timer->record(TIMER::U);
  timer->tic();
  kernel_F1_GPU(H_GPU, dHdr_GPU, drdrx_GPU, F_GPU, mask_GPU, a, b, c, k);
  timer->record(TIMER::F1);
#else
  timer->tic();
  kernel4_GPU(dEdS_GPU, M_GPU, H_GPU, dHdr_GPU, drdrx_GPU, F1_GPU, mask_GPU, a,
              b, d, c, k);
  timer->record(TIMER::Kernel4);
#endif

  // Scatter update forces
  timer->tic();
#ifdef GPU_BASELINE
  if (vatom)
    sum_forces_GPU(a, F_GPU, R_GPU, M_GPU, ijlist_GPU, mask_GPU,
                   pos_GPU, &vatom_GPU, f[0], vatom[0], scale, nmax_GPU, b, c);
  else {
    sum_forces_GPU(a, F_GPU, R_GPU, M_GPU, ijlist_GPU, mask_GPU,
                   pos_GPU, &vatom_GPU, f[0], nullptr, scale, nmax_GPU, b, c);
  }
#else
  sum_forces_GPU(a, F1_GPU, F2_GPU, R_GPU, drdrx_GPU, ijlist_GPU, mask_GPU,
                 pos_GPU, &vatom_GPU, f[0], vatom, scale, nmax_GPU, b, c);
#endif

  timer->record(TIMER::FORCES);
  timer->begin();

  memory_GPU(mpiid, this->screen, this->logfile);

  delete[] y[0];
  delete[] y[1];
  delete[] y[2];
  return ytotal[0] * scale;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> double TensorMD<Scalar>::memory_usage() const
{
  //TODO
  return 0;
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::TensorMD<float>;
template class LAMMPS_NS::TensorMD<double>;
