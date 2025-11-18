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

using namespace LAMMPS_NS;
using Eigen::IndexPair;
using Eigen::Tensor;
using Eigen::TensorMap;

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

  i2a = nullptr;
  a2i = nullptr;
  rmax = 0.0;
  max_moment = 0;
  is_T_symmetric = false;
  cutforce = 0;
  cutforcesq = 0;
  eltij_pos = nullptr;
  nnp = nullptr;
  fnn = nullptr;
  tdnp = nullptr;
  descriptor = nullptr;
  cutoff = nullptr;

  // Initialize the timer
  disable_timer = false;
  timer = new TensorMDTimer(this->screen, this->logfile);
}

template <typename Scalar> TensorMD<Scalar>::~TensorMD()
{
  if (this->timer && !disable_timer) this->timer->print();

  memory->destroy(this->eltij_pos);
  memory->destroy(a2i);
  memory->destroy(i2a);
  memory->destroy(memory_pool.scalars);
  memory->destroy(memory_pool.integers);

  delete this->nnp;
  delete this->fnn;
  delete this->timer;
  delete this->tdnp;
  delete this->cutoff;
  delete this->descriptor;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::read_npz(cnpy::npz_t &npz, int &neltypes,
                                std::vector<int> &numbers,
                                std::vector<double> &masses)
{
  int i, j, k, num_filters;
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
    fnn = new FeaturePotential<Scalar>(memory, error, mpiid);
    fnn->setup_global(npz, num_filters);
    double forward, backward;
    fnn->get_flops_per_atom(forward, backward);
    timer->setup_fnn(forward, backward);
  } else {
    descriptor = new Descriptor<Scalar>(memory, error, npz);
    num_filters = descriptor->get_num_filters();
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
  memory->create(eltij_pos, neltypes, neltypes, "pair:tensormd:eltij_pos");
  for (i = 0; i < neltypes; i++) {
    k = 1;
    for (j = 0; j < neltypes; j++) {
      if (i == j) {
        eltij_pos[i][j] = 0;
      } else {
        eltij_pos[i][j] = k;
        k++;
      }
    }
  }

  // Setup the global sizes for TensorMD. These are constant values.
  size.b = neltypes;
  size.m = max_moment + 1;
  size.k = num_filters;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> void TensorMD<Scalar>::setup_global(double *cutmax)
{
  int m;
  const int vmD_nv[6] = {1, 3, 6, 10, 15, 21};

  // Force cutoff
  cutforce = rmax;
  cutforcesq = cutforce * cutforce;

  // Pass cutoff back to calling program
  *cutmax = static_cast<double>(cutforce);

  // Accumulate the size `d`, which represents the number of unique moments.
  size.d = 0;
  for (m = 0; m < max_moment + 1; m++) { size.d += vmD_nv[m]; }

  // Setup the timer
  timer->setup(size.b, size.d, size.m, size.k, cutoff->is_cosine());

  // Initialize the multiplicity tensor T_md
  T.resize(Shape2d{size.m, size.d});
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
  int i, m, elti;

  // update the timer
  timer->tic();
  timer->update(nlocal, numneigh_max);

  // Set the local dimension at this step
  if (!size.initialized) {
    size.a = nlocal;
    size.initialized = true;
  }
  size.as = nlocal;
  size.cs = numneigh_max;
  memory_pool.reset();

  // grow local arrays if necessary
  if (size.as > size.a || size.cs > size.c) {
    size.a = size.as;
    size.c = size.cs;

    if (size.b > 1) {
      memory->destroy(i2a);
      memory->destroy(a2i);
      memory->create(a2i, size.a, "pair:tensormd:a2i");
      memory->create(i2a, size.a, "pair:tensormd:i2a");
    }

    // Release the previous pools
    memory->destroy(memory_pool.scalars);
    memory->destroy(memory_pool.integers);

    // Compute the total number of scalars and integers
    bool use_sij = ((fnn && !fnn->is_interpolatable()) ||
                    (descriptor && !descriptor->is_interpolatable()));
    size_t nscalars, nintegers;
    size.get_memory_space(nscalars, nintegers, use_sij);

    // Allocate the new pools
    memory->create(memory_pool.scalars, nscalars, "pair:tensormd:pool:scalars");
    memory->create(memory_pool.integers, nintegers,
                   "pair:tensormd:pool:integers");
  }

  // Rearrange the tensors. `as` and `cs` should be used instead of `a` and `c`.
  // U/H and P/V share the same memory space.
  size_t n = size.as * size.b * size.cs;
  ijlist = memory_pool.iget(n * 2), size.tcba();
  mask = memory_pool.iget(n);
  R = memory_pool.fget(n);
  drdrx = memory_pool.fget(n * 3);
  F1 = memory_pool.fget(n * 3);
  F2 = memory_pool.fget(n * 3);
  H = memory_pool.fget(n * size.k);
  U = H;
  dHdr = memory_pool.fget(n * size.k);
  M = memory_pool.fget(n * size.d);
  if ((fnn && !fnn->is_interpolatable()) ||
      (descriptor && !descriptor->is_interpolatable())) {
    sij = memory_pool.fget(n);
    dsij = memory_pool.fget(n);
  }
  G = memory_pool.fget(size.as * size.b * size.k * size.m);
  dEdG = memory_pool.fget(size.as * size.b * size.k * size.m);
  dEdS = memory_pool.fget(size.as * size.b * size.k * size.d);
  if (size.k > size.cs) {
    P = memory_pool.fget(size.as * size.b * size.d * size.k);
    V = P;
  } else {
    V = memory_pool.fget(size.as * size.b * size.d * size.cs);
    P = V;
  }

  // Setup imap when `neltypes > 1` because `imap[i] = i` is always
  // true if `neltypes == 1`.
  eltind[0].offset = 0;
  eltind[0].row = 0;
  if (size.b > 1) {
    for (i = 0; i < size.as; i++) { i2a[i] = a2i[i] = 0; }
    m = typenums[0];
    for (elti = 1; elti < size.b; elti++) {
      eltind[elti].offset = m;
      eltind[elti].row = m;
      m += typenums[elti];
    }
  }

  timer->record(TIMER::ALLOC);
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
void TensorMD<Scalar>::setup_tensors(const int *ilist, const int *type,
                                     const int *fmap, double **pos,
                                     int *numneigh, int **firstneigh)
{
  const int max_eltypes = 8;
  int i, j, a, b, c, d;
  int ii, jn, elti, eltj;
  int neigh[max_eltypes];
  double xtmp, ytmp, ztmp;
  double rijinv, rij, rij2, dij[3];
  double x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz;
  double yyy, yyz, yzz, zzz, xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz;
  double xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz;
  double xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz;
  double xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz, yyzzz, yzzzz;
  double zzzzz;

  TensorMap<Tensor<Scalar, 3>> R_(R, size.cba());
  TensorMap<Tensor<Scalar, 4>> M_(M, size.dcba());
  TensorMap<Tensor<Scalar, 4>> drdrx_(drdrx, size.xcba());
  TensorMap<Tensor<int, 4>> ijlist_(ijlist, size.tcba());
  TensorMap<Tensor<int, 3>> mask_(mask, size.cba());

  timer->tic();

  for (ii = 0; ii < size.as; ii++) {
    i = ilist[ii];
    elti = fmap[type[i]];
    // Find the row of the density matrix `rho` for atom `i`.
    // If `neltypes` is 1, `row == i` is always true.
    if (size.b == 1) {
      a = i;
    } else {
      a = eltind[elti].row;
      eltind[elti].row++;
    }
    if (size.b > 1) {
      i2a[i] = a;
      a2i[a] = i;
    }
  }

#if defined(_OPENMP)
#pragma omp parallel for private(                                            \
        neigh, ii, jn, elti, eltj, a, b, c, d, i, j, rij, rij2, rijinv, dij, \
            x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz,   \
            yyy, yyz, yzz, zzz, xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy,    \
            xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz, xxxxx, xxxxy,    \
            xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz, xyyyy,   \
            xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz, yyzzz, yzzzz,   \
            zzzzz, xtmp, ytmp, ztmp)
#endif
  for (ii = 0; ii < size.as; ii++) {
    for (elti = 0; elti < size.b; elti++) neigh[elti] = 0;
    i = ilist[ii];
    elti = fmap[type[i]];
    xtmp = pos[i][0];
    ytmp = pos[i][1];
    ztmp = pos[i][2];
    // Find the row of the density matrix `rho` for atom `i`.
    // If `neltypes` is 1, `row == i` is always true.
    if (size.b == 1) {
      a = i;
    } else {
      a = i2a[i];
    }
    for (jn = 0; jn < numneigh[i]; jn++) {
      j = firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = pos[j][0] - xtmp;
      dij[1] = pos[j][1] - ytmp;
      dij[2] = pos[j][2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < this->cutforcesq && rij2 > 1e-10) {
        eltj = fmap[type[j]];
        rij = sqrt(rij2);
        rijinv = 1.0 / rij;
        b = eltij_pos[elti][eltj];
        c = neigh[b];
        if (c < size.cs) {
          x = dij[0] * rijinv;
          y = dij[1] * rijinv;
          z = dij[2] * rijinv;
          R_(c, b, a) = rij;
          mask_(c, b, a) = 1;
          M_(0, c, b, a) = 1.0;
          drdrx_(0, c, b, a) = x;
          drdrx_(1, c, b, a) = y;
          drdrx_(2, c, b, a) = z;
          ijlist_(0, c, b, a) = i;
          ijlist_(1, c, b, a) = j;
          if (max_moment > 0) {
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M_(1, c, b, a) = x;
            M_(2, c, b, a) = y;
            M_(3, c, b, a) = z;
            if (max_moment > 1) {
              xxx = xx * x;
              xxy = xx * y;
              xxz = xx * z;
              xyy = xy * y;
              xyz = xy * z;
              xzz = xz * z;
              yyy = yy * y;
              yyz = yy * z;
              yzz = yz * z;
              zzz = zz * z;
              M_(4, c, b, a) = xx;
              M_(5, c, b, a) = xy;
              M_(6, c, b, a) = xz;
              M_(7, c, b, a) = yy;
              M_(8, c, b, a) = yz;
              M_(9, c, b, a) = zz;
              if (max_moment > 2) {
                xxxx = xxx * x;
                xxxy = xxx * y;
                xxxz = xxx * z;
                xxyy = xxy * y;
                xxyz = xxy * z;
                xxzz = xxz * z;
                xyyy = xyy * y;
                xyyz = xyy * z;
                xyzz = xyz * z;
                xzzz = xzz * z;
                yyyy = yyy * y;
                yyyz = yyy * z;
                yyzz = yyz * z;
                yzzz = yzz * z;
                zzzz = zzz * z;
                M_(10, c, b, a) = xxx;
                M_(11, c, b, a) = xxy;
                M_(12, c, b, a) = xxz;
                M_(13, c, b, a) = xyy;
                M_(14, c, b, a) = xyz;
                M_(15, c, b, a) = xzz;
                M_(16, c, b, a) = yyy;
                M_(17, c, b, a) = yyz;
                M_(18, c, b, a) = yzz;
                M_(19, c, b, a) = zzz;
                if (max_moment > 3) {
                  xxxxx = xxxx * x;
                  xxxxy = xxxx * y;
                  xxxxz = xxxx * z;
                  xxxyy = xxxy * y;
                  xxxyz = xxxy * z;
                  xxxzz = xxxz * z;
                  xxyyy = xxyy * y;
                  xxyyz = xxyy * z;
                  xxyzz = xxyz * z;
                  xxzzz = xxzz * z;
                  xyyyy = xyyy * y;
                  xyyyz = xyyy * z;
                  xyyzz = xyyz * z;
                  xyzzz = xyzz * z;
                  xzzzz = xzzz * z;
                  yyyyy = yyyy * y;
                  yyyyz = yyyy * z;
                  yyyzz = yyyz * z;
                  yyzzz = yyzz * z;
                  yzzzz = yzzz * z;
                  zzzzz = zzzz * z;
                  M_(20, c, b, a) = xxxx;
                  M_(21, c, b, a) = xxxy;
                  M_(22, c, b, a) = xxxz;
                  M_(23, c, b, a) = xxyy;
                  M_(24, c, b, a) = xxyz;
                  M_(25, c, b, a) = xxzz;
                  M_(26, c, b, a) = xyyy;
                  M_(27, c, b, a) = xyyz;
                  M_(28, c, b, a) = xyzz;
                  M_(29, c, b, a) = xzzz;
                  M_(30, c, b, a) = yyyy;
                  M_(31, c, b, a) = yyyz;
                  M_(32, c, b, a) = yyzz;
                  M_(33, c, b, a) = yzzz;
                  M_(34, c, b, a) = zzzz;
                  if (max_moment > 4) {
                    M_(35, c, b, a) = xxxxx;
                    M_(36, c, b, a) = xxxxy;
                    M_(37, c, b, a) = xxxxz;
                    M_(38, c, b, a) = xxxyy;
                    M_(39, c, b, a) = xxxyz;
                    M_(40, c, b, a) = xxxzz;
                    M_(41, c, b, a) = xxyyy;
                    M_(42, c, b, a) = xxyyz;
                    M_(43, c, b, a) = xxyzz;
                    M_(44, c, b, a) = xxzzz;
                    M_(45, c, b, a) = xyyyy;
                    M_(46, c, b, a) = xyyyz;
                    M_(47, c, b, a) = xyyzz;
                    M_(48, c, b, a) = xyzzz;
                    M_(49, c, b, a) = xzzzz;
                    M_(50, c, b, a) = yyyyy;
                    M_(51, c, b, a) = yyyyz;
                    M_(52, c, b, a) = yyyzz;
                    M_(53, c, b, a) = yyzzz;
                    M_(54, c, b, a) = yzzzz;
                    M_(55, c, b, a) = zzzzz;
                  }
                }
              }
            }
          }
        }
        neigh[b]++;
      }
    }

    for (b = 0; b < size.b; b++) {
      for (c = neigh[b]; c < size.cs; c++) {
        ijlist_(0, c, b, a) = -1;
        ijlist_(1, c, b, a) = -1;
        mask_(c, b, a) = 0;
        R_(c, b, a) = 0.0;
        drdrx_(0, c, b, a) = 0.0;
        drdrx_(1, c, b, a) = 0.0;
        drdrx_(2, c, b, a) = 0.0;
        for (d = 0; d < size.d; d++) { M_(d, c, b, a) = 0.0; }
      }
    }
  }
  timer->record(TIMER::SETUP);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
int TensorMD<Scalar>::get_exact_cmax(const int *ilist, const int inum,
                                     const int *type, const int *fmap,
                                     double **pos, int *numneigh,
                                     int **firstneigh)
{
  const int max_eltypes = 8;
  int ii, elti, eltj, i, j, b, jn, neigh[max_eltypes];
  double xtmp, ytmp, ztmp, dij[3], rij2;
  int cmax = 0;
#if defined(_OPENMP)
#pragma omp parallel for private(ii, elti, eltj, i, j, b, jn, neigh, xtmp, \
                                     ytmp, ztmp, dij, rij2)                \
    reduction(max : cmax)
#endif
  for (ii = 0; ii < inum; ii++) {
    for (elti = 0; elti < size.b; elti++) neigh[elti] = 0;
    i = ilist[ii];
    elti = fmap[type[i]];
    xtmp = pos[i][0];
    ytmp = pos[i][1];
    ztmp = pos[i][2];
    for (jn = 0; jn < numneigh[i]; jn++) {
      j = firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = pos[j][0] - xtmp;
      dij[1] = pos[j][1] - ytmp;
      dij[2] = pos[j][2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < this->cutforcesq && rij2 > 1e-10) {
        eltj = fmap[type[j]];
        b = eltij_pos[elti][eltj];
        neigh[b]++;
      }
    }
    for (b = 0; b < size.b; b++) {
      if (neigh[b] > cmax) cmax = neigh[b];
    }
  }
  return cmax;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::compute(int *ilist, const int *type, const int *fmap,
                               double **pos, int *numneigh, int **firstneigh,
                               int *typenums, double *etemp,
                               bool use_atom_etemp, TDPE<double> &pe,
                               double **f, double **vatom, double scale)
{
  // Setup tensors
  setup_tensors(ilist, type, fmap, pos, numneigh, firstneigh);

  // Compute the radial density tensor H
  compute_radial_density();

  // Compute the projected density tensor G
  compute_projected_density();

  // Compute the neural network potential
  if (tdnp) {
    compute_tdnp(typenums, etemp, use_atom_etemp, pe, scale);
  } else {
    compute_nnp(typenums, pe.E, scale);
  }

  // Compute the gradients
  compute_gradients();

  // Compute the forces
  compute_forces(f, vatom, scale);

  // Start the timer after first `compute` call.
  timer->begin();
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> void TensorMD<Scalar>::compute_radial_density()
{
  if (this->fnn && this->fnn->is_interpolatable()) {
    timer->tic();
    this->fnn->interpolate(R, size.as * size.b * size.cs, H, dHdr);
    timer->record(TIMER::FNN_INTERP);
  } else if (this->descriptor && this->descriptor->is_interpolatable()) {
    timer->tic();
    this->descriptor->interpolate(R, size.as * size.b * size.cs, H, dHdr);
    timer->record(TIMER::DESCRIPTOR);
  } else {
    TensorMap<Tensor<Scalar, 3>> R_(R, size.cba());
    TensorMap<Tensor<Scalar, 3>> sij_(sij, size.cba());
    TensorMap<Tensor<Scalar, 3>> dsij_(dsij, size.cba());
    TensorMap<Tensor<Scalar, 4>> H_(H, size.kcba());
    TensorMap<Tensor<Scalar, 4>> dHdr_(dHdr, size.kcba());
    TensorMap<Tensor<int, 3>> mask_(mask, size.cba());

    // Calculate cutoff coefficients
    timer->tic();
    cutoff->compute(R_, mask_, &sij_, &dsij_);
    timer->record(TIMER::CUTOFF);

    timer->tic();
    if (this->fnn) {
      Shape4d bcast{size.k, 1, 1, 1};
      Shape4d shape{1, size.cs, size.b, size.as};
      auto tiled_sij_ = sij_.reshape(shape).broadcast(bcast);
      auto tiled_dsij_ = dsij_.reshape(shape).broadcast(bcast);
      this->fnn->forward(R, size.as * size.b * size.cs, H);
      timer->record(TIMER::FNN_FORWARD);
      timer->tic();
      dHdr_ = tiled_dsij_ * H_;
      H_ = tiled_sij_ * H_;
    } else {
      this->descriptor->compute(R_, sij_, dsij_, &H_, &dHdr_);
    }
    timer->record(TIMER::DESCRIPTOR);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> void TensorMD<Scalar>::compute_projected_density()
{
  TensorMap<Tensor<Scalar, 4>> M_(M, size.dcba());
  TensorMap<Tensor<Scalar, 4>> H_(H, size.kcba());
  TensorMap<Tensor<Scalar, 4>> P_(P, size.dkba());
  TensorMap<Tensor<Scalar, 4>> G_(G, size.mkba());

  // P_dkba = M_dcba * H_kcba
  timer->tic();
  kernel_P(M_, H_, &P_);
  timer->record(TIMER::P);

  // +/- sign for m = 0
  auto sign = P_.chip(0, 0).sign();

  // Q_mkba = T_md * (P_dkba)**2
  timer->tic();
  kernel_Q(P_, T, &G_);
  timer->record(TIMER::Q);

  // G_mkba
  timer->tic();
  G_.chip(0, 0) = G_.chip(0, 0).sqrt() * sign;
  timer->record(TIMER::G);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::compute_nnp(int *typenums, PE<double> &pe, double scale)
{
  timer->tic();

  int a, elti;
  Scalar *x_in;
  Scalar *eatom, *dydx;
  PE<Scalar> y{0.0, nullptr};
  int bkm = get_bkm();

  // Allocate memory for atomic energies
  if (pe.calc_atom()) {
    eatom = new Scalar[size.as];
  } else {
    eatom = nullptr;
  }

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < size.b; elti++) {
    if (typenums[elti] == 0) { continue; }

    // Setup the offset
    x_in = &G[eltind[elti].offset * bkm];
    if (eatom) {
      y.eatom = &eatom[eltind[elti].offset];
    } else {
      y.eatom = nullptr;
    }
    y.toten = 0.0;
    dydx = &dEdG[eltind[elti].offset * bkm];

    // Compute
    nnp->compute(elti, x_in, typenums[elti], y, dydx);

    // Add the scaled energy
    pe.toten += static_cast<double>(y.toten) * scale;
  }

  if (eatom != nullptr) {
    for (a = 0; a < size.as; a++) {
      if (size.b == 1) {
        pe.eatom[a] = static_cast<double>(eatom[a]) * scale;
      } else {
        pe.eatom[a2i[a]] = static_cast<double>(eatom[a]) * scale;
      }
    }
    delete[] eatom;
  }

  timer->record(TIMER::NN_COMPUTE);
}

template <typename Scalar>
void TensorMD<Scalar>::compute_tdnp(int *typenums, double *etemp,
                                    bool use_atom_etemp, TDPE<double> &tdpe,
                                    double scale)
{
  timer->tic();

  int a, elti;
  Scalar *x_in;
  Scalar *eatom[4], *dfdx, *etemp_atom;
  TDPE<Scalar> y{{0.0, nullptr}, {0.0, nullptr}, {0.0, nullptr}};
  int bkm = get_bkm();

  // Allocate memory for atomic free energies (0), energies (1) and electron
  // entropy (2).
  if (tdpe.calc_atom()) {
    eatom[0] = new Scalar[size.as];
    eatom[1] = new Scalar[size.as];
    eatom[2] = new Scalar[size.as];
  } else {
    eatom[0] = nullptr;
    eatom[1] = nullptr;
    eatom[2] = nullptr;
  }

  if (use_atom_etemp) {
    // Allocate memory for atomic electron temperature (3) if needed
    eatom[3] = new Scalar[size.as];

    // Local to global mapping of atomic indices
#if defined(_OPENMP)
#pragma omp parallel for private(a)
#endif
    for (a = 0; a < size.as; a++) {
      if (size.b == 1) {
       eatom[3][a] = static_cast<Scalar>(etemp[a]);
      } else {
        eatom[3][a2i[a]] = static_cast<Scalar>(etemp[a]);
      }
    }
  } else {
    eatom[3] = nullptr;
  }

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < size.b; elti++) {
    if (typenums[elti] == 0) { continue; }

    // Setup the offset
    x_in = &G[eltind[elti].offset * bkm];
    if (tdpe.calc_atom()) {
      y.F.eatom = &eatom[0][eltind[elti].offset];
      y.E.eatom = &eatom[1][eltind[elti].offset];
      y.S.eatom = &eatom[2][eltind[elti].offset];
    } else {
      y.F.eatom = nullptr;
      y.E.eatom = nullptr;
      y.S.eatom = nullptr;
    }
    y.F.toten = 0.0;
    y.E.toten = 0.0;
    y.S.toten = 0.0;
    dfdx = &dEdG[eltind[elti].offset * bkm];

    // Compute
    if (use_atom_etemp) {
      etemp_atom = &eatom[3][eltind[elti].offset];
      tdnp->compute(elti, x_in, typenums[elti], dfdx, etemp_atom, y);
    } else {
      tdnp->compute(elti, x_in, typenums[elti], dfdx, etemp[0], y);
    }

    // Add the scaled values
    tdpe.F.toten += static_cast<double>(y.F.toten) * scale;
    tdpe.E.toten += static_cast<double>(y.E.toten) * scale;
    tdpe.S.toten += static_cast<double>(y.S.toten) * scale;
  }

  if (tdpe.calc_atom()) {
#if defined(_OPENMP)
#pragma omp parallel for private(a)
#endif
    for (a = 0; a < size.as; a++) {
      if (size.b == 1) {
        tdpe.F.eatom[a] = static_cast<double>(eatom[0][a]) * scale;
        tdpe.E.eatom[a] = static_cast<double>(eatom[1][a]) * scale;
        tdpe.S.eatom[a] = static_cast<double>(eatom[2][a]) * scale;
      } else {
        tdpe.F.eatom[a2i[a]] = static_cast<double>(eatom[0][a]) * scale;
        tdpe.E.eatom[a2i[a]] = static_cast<double>(eatom[1][a]) * scale;
        tdpe.S.eatom[a2i[a]] = static_cast<double>(eatom[2][a]) * scale;
      }
    }

    delete[] eatom[0];
    delete[] eatom[1];
    delete[] eatom[2];
  }
  if (use_atom_etemp) { delete[] eatom[3]; }
  timer->record(TIMER::NN_COMPUTE);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> void TensorMD<Scalar>::compute_gradients()
{
  TensorMap<Tensor<Scalar, 4>> G_(G, size.mkba());
  TensorMap<Tensor<Scalar, 4>> dEdG_(dEdG, size.mkba());
  TensorMap<Tensor<Scalar, 4>> P_(P, size.dkba());
  TensorMap<Tensor<Scalar, 4>> dEdS_(dEdS, size.dkba());
  TensorMap<Tensor<Scalar, 4>> M_(M, size.dcba());
  TensorMap<Tensor<Scalar, 4>> H_(H, size.kcba());
  TensorMap<Tensor<Scalar, 4>> dHdr_(dHdr, size.kcba());
  TensorMap<Tensor<Scalar, 4>> U_(U, size.kcba());
  TensorMap<Tensor<Scalar, 4>> V_(V, size.dcba());
  TensorMap<Tensor<Scalar, 4>> F1_(F1, size.xcba());
  TensorMap<Tensor<Scalar, 4>> F2_(F2, size.xcba());
  TensorMap<Tensor<Scalar, 4>> drdrx_(drdrx, size.xcba());
  TensorMap<Tensor<Scalar, 3>> R_(R, size.cba());
  TensorMap<Tensor<int, 3>> mask_(mask, size.cba());

  // dEdG_mkba = dEdG_mkba * dGdQ_mkba
  // dGdQ.chip(0, 0) = 0.5 / G.chip(0, 0)
  // dGdQ.chip(m>0, 0) = 1.0
  // dEdP_dkba = 2 * T_md * dGdQ_mkba * P_dkba
  timer->tic();
  kernel_dEdG(G_, &dEdG_);
  kernel_dEdS(T, dEdG_, P_, &dEdS_);
  timer->record(TIMER::DEDP);

  // V_dcba = dEdS_dkba * H_kcba
  timer->tic();
  kernel_V<Scalar>(dEdS_, H_, &V_);
  timer->record(TIMER::V);

  // F2_xcba = dMdrx_dxcba * V_dcba
  timer->tic();
  kernel_fused_F2<Scalar>(max_moment, V_, R_, drdrx_, mask_, &F2_);
  timer->record(TIMER::F2);

  // U_kcba = dEdS_dkba * M_dcba
  timer->tic();
  kernel_U<Scalar>(dEdS_, M_, &U_);
  timer->record(TIMER::U);

  // F1_xcba = U_kcba * dHdr_kcba * drdrx_xcba
  timer->tic();
  if (this->fnn && !this->fnn->is_interpolatable()) {
    Eigen::array<Eigen::Index, 1> sum_axis{0};
    Shape4d shape{1, size.cs, size.b, size.as};
    Shape4d bcast_xcba{3, 1, 1, 1};
    Eigen::array<Eigen::Index, 4> bcast_kcba{size.k, 1, 1, 1};
    Tensor<Scalar, 4> dHdrp{shape};
    TensorMap<Tensor<Scalar, 3>> sij_(sij, size.cba());
    Tensor<Scalar, 4> Up = U_ * sij_.reshape(shape).broadcast(bcast_kcba);
    this->fnn->backward(Up.data(), dHdrp.data());
    auto grad = (U_ * dHdr_).sum(sum_axis).reshape(shape) + dHdrp;
    F1_ = grad.broadcast(bcast_xcba) * drdrx_;
    timer->record(TIMER::FNN_BACKWARD);
  } else {
    kernel_F1<Scalar>(U_, dHdr_, drdrx_, mask_, &F1_);
    timer->record(TIMER::F1);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::compute_forces(double **f, double **vatom, double scale)
{
  TensorMap<Tensor<int, 3>> mask_(mask, size.cba());
  TensorMap<Tensor<int, 4>> ijlist_(ijlist, size.tcba());
  TensorMap<Tensor<Scalar, 4>> F1_(F1, size.xcba());
  TensorMap<Tensor<Scalar, 4>> F2_(F2, size.xcba());
  TensorMap<Tensor<Scalar, 4>> drdrx_(drdrx, size.xcba());
  TensorMap<Tensor<Scalar, 3>> R_(R, size.cba());

  int a, b, c, x, i, j;
  double df[3], virial[6];

  // Scatter update forces
  timer->tic();
  for (a = 0; a < size.as; a++) {
    for (b = 0; b < size.b; b++) {
      for (c = 0; c < size.cs; c++) {
        if (mask_(c, b, a) == 0) continue;
        i = ijlist_(0, c, b, a);
        j = ijlist_(1, c, b, a);
        if (i >= 0 && j >= 0) {
          for (x = 0; x < size.x; x++) {
            df[x] = (F1_(x, c, b, a) + F2_(x, c, b, a)) * scale;
            f[i][x] += df[x];
            f[j][x] -= df[x];
          }
          if (vatom) {
            virial[0] = -drdrx_(0, c, b, a) * df[0] * R_(c, b, a);
            virial[1] = -drdrx_(1, c, b, a) * df[1] * R_(c, b, a);
            virial[2] = -drdrx_(2, c, b, a) * df[2] * R_(c, b, a);
            virial[3] = -drdrx_(0, c, b, a) * df[1] * R_(c, b, a);
            virial[4] = -drdrx_(0, c, b, a) * df[2] * R_(c, b, a);
            virial[5] = -drdrx_(1, c, b, a) * df[2] * R_(c, b, a);
            vatom[i][0] += 0.5 * virial[0];
            vatom[i][1] += 0.5 * virial[1];
            vatom[i][2] += 0.5 * virial[2];
            vatom[i][3] += 0.5 * virial[3];
            vatom[i][4] += 0.5 * virial[4];
            vatom[i][5] += 0.5 * virial[5];
            vatom[j][0] += 0.5 * virial[0];
            vatom[j][1] += 0.5 * virial[1];
            vatom[j][2] += 0.5 * virial[2];
            vatom[j][3] += 0.5 * virial[3];
            vatom[j][4] += 0.5 * virial[4];
            vatom[j][5] += 0.5 * virial[5];
          }
        }
      }
    }
  }
  timer->record(TIMER::FORCES);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar> double TensorMD<Scalar>::memory_usage() const
{
  size_t nscalars = 0;
  size_t nintegers = 0;
  bool use_sij = ((fnn && !fnn->is_interpolatable()) ||
                  (descriptor && !descriptor->is_interpolatable()));
  size.get_memory_space(nscalars, nintegers, use_sij);

  nscalars += T.size();
  auto bytes =
      static_cast<double>(sizeof(Scalar) * nscalars + sizeof(int) * nintegers);
  bytes += static_cast<double>(sizeof(int) * size.a * 2);
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
