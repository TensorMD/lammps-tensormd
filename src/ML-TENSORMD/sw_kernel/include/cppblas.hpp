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

#ifndef LAMMPS_TENSORALLOY_MATH_H
#define LAMMPS_TENSORALLOY_MATH_H

#if defined(EIGEN_USE_MKL_ALL)
#include "mkl.h"
#elif defined(__APPLE__) && defined(AccelerateFramework)
#include <Accelerate/accelerate.h>
#else
#include "cblas.h"
#endif

static void cppblas_axpy(int n, float alpha, float *x, int incx, float *y,
                         int incy)
{
#ifdef BLAS_SW_XMATH
  saxpy_(&n, &alpha, x, &incx, y, &incy);
#else
  cblas_saxpy(n, alpha, x, incx, y, incy);
#endif
}

static void cppblas_axpy(int n, double alpha, double *x, int incx, double *y,
                         int incy)
{
#ifdef BLAS_SW_XMATH
  daxpy_(&n, &alpha, x, &incx, y, &incy);
#else
  cblas_daxpy(n, alpha, x, incx, y, incy);
#endif
}

static void cppblas_copy(int n, float *x, int incx, float *y, int incy)
{
#ifdef BLAS_SW_XMATH
  scopy_(n, x, incx, y, incy);
#else
  cblas_scopy(n, x, incx, y, incy);
#endif
}

static void cppblas_copy(int n, double *x, int incx, double *y, int incy)
{
#ifdef BLAS_SW_XMATH
  dcopy_(n, x, incx, y, incy);
#else
  cblas_dcopy(n, x, incx, y, incy);
#endif
}

static void cppblas_gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, int M, int N, int K,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb, double beta, double *C,
                         int ldc)
{
#ifdef BLAS_SW_XMATH
  char TA, TB;
  if (order == CblasColMajor)
  {
    if(transA == CblasTrans) TA='T';
    else if ( transA == CblasConjTrans ) TA='C';
    else if ( transA == CblasNoTrans )   TA='N';
    if(transB == CblasTrans) TB='T';
    else if ( transB == CblasConjTrans ) TB='C';
    else if ( transB == CblasNoTrans )   TB='N';
    dgemm_(&TA, &TB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
  else
  {
    if(transA == CblasTrans) TB='T';
    else if ( transA == CblasConjTrans ) TB='C';
    else if ( transA == CblasNoTrans )   TB='N';
    if(transB == CblasTrans) TA='T';
    else if ( transB == CblasConjTrans ) TA='C';
    else if ( transB == CblasNoTrans )   TA='N';
    dgemm_(&TA, &TB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
  }
#else
  cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#endif
}

static void cppblas_gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE transA,
                         CBLAS_TRANSPOSE transB, int M, int N, int K,
                         float alpha, const float *A, int lda, const float *B,
                         int ldb, float beta, float *C, int ldc)
{
#ifdef BLAS_SW_XMATH
  char TA, TB;
  if (order == CblasColMajor)
  {
    if(transA == CblasTrans) TA='T';
    else if ( transA == CblasConjTrans ) TA='C';
    else if ( transA == CblasNoTrans )   TA='N';
    if(transB == CblasTrans) TB='T';
    else if ( transB == CblasConjTrans ) TB='C';
    else if ( transB == CblasNoTrans )   TB='N';
    sgemm_(&TA, &TB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }
  else
  {
    if(transA == CblasTrans) TB='T';
    else if ( transA == CblasConjTrans ) TB='C';
    else if ( transA == CblasNoTrans )   TB='N';
    if(transB == CblasTrans) TA='T';
    else if ( transB == CblasConjTrans ) TA='C';
    else if ( transB == CblasNoTrans )   TA='N';
    sgemm_(&TA, &TB, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
  }
#else
  cblas_sgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#endif
}

static float cppblas_dot(int n, float *x, int incx, float *y, int incy)
{
#ifdef BLAS_SW_XMATH
  return sdot_(n, x, incx, y, incy);
#else
  return cblas_sdot(n, x, incx, y, incy);
#endif
}

static double cppblas_dot(int n, double *x, int incx, double *y, int incy)
{
#ifdef BLAS_SW_XMATH
  return ddot_(n, x, incx, y, incy);
#else
  return cblas_ddot(n, x, incx, y, incy);
#endif
}

#ifdef EIGEN_USE_MKL_ALL

static void cppvml_mul(int n, float *x, float *y, float *out)
{
  vsMul(n, x, y, out);
}

static void cppvml_mul(int n, double *x, double *y, double *out)
{
  vdMul(n, x, y, out);
}

static void cppvml_add(int n, float *x, float *y, float *out)
{
  vsAdd(n, x, y, out);
}

static void cppvml_add(int n, double *x, double *y, double *out)
{
  vdAdd(n, x, y, out);
}

static void cppvml_sub(int n, float *x, float *y, float *out)
{
  vsSub(n, x, y, out);
}

static void cppvml_sub(int n, double *x, double *y, double *out)
{
  vdSub(n, x, y, out);
}

static void cppvml_div(int n, float *x, float *y, float *out)
{
  vsDiv(n, x, y, out);
}

static void cppvml_div(int n, double *x, double *y, double *out)
{
  vdDiv(n, x, y, out);
}

static void cppvml_exp(int n, float *x, float *out)
{
  vsExp(n, x, out);
}

static void cppvml_exp(int n, double *x, double *out)
{
  vdExp(n, x, out);
}

static void cppvml_log1p(int n, float *x, float *out)
{
  vsLog1p(n, x, out);
}

static void cppvml_log1p(int n, double *x, double *out)
{
  vdLog1p(n, x, out);
}

static void cppvml_tanh(int n, float *x, float *out)
{
  vsTanh(n, x, out);
}

static void cppvml_tanh(int n, double *x, double *out)
{
  vdTanh(n, x, out);
}

static void cppvml_sqr(int n, float *x, float *out)
{
  vsSqr(n, x, out);
}

static void cppvml_sqr(int n, double *x, double *out)
{
  vdSqr(n, x, out);
}

static void cppvml_inv(int n, float *x, float *out)
{
  vsInv(n, x, out);
}

static void cppvml_inv(int n, double *x, double *out)
{
  vdInv(n, x, out);
}

#if INTEL_MKL_VERSION >= 20210000

static void cppblas_gemv_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const MKL_INT m,
    const MKL_INT n, const double alpha, const double *a, const MKL_INT lda,
    const MKL_INT stridea, const double *x, const MKL_INT incx,
    const MKL_INT stridex, const double beta, double *y, const MKL_INT incy,
    const MKL_INT stridey, const MKL_INT batch_size)
{
  cblas_dgemv_batch_strided(layout, trans, m, n, alpha, a, lda, stridea, x,
                            incx, stridex, beta, y, incy, stridey, batch_size);
}

static void cppblas_gemv_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE trans, const MKL_INT m,
    const MKL_INT n, const float alpha, const float *a, const MKL_INT lda,
    const MKL_INT stridea, const float *x, const MKL_INT incx,
    const MKL_INT stridex, const float beta, float *y, const MKL_INT incy,
    const MKL_INT stridey, const MKL_INT batch_size)
{
  cblas_sgemv_batch_strided(layout, trans, m, n, alpha, a, lda, stridea, x,
                            incx, stridex, beta, y, incy, stridey, batch_size);
}

static void cppblas_gemm_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n,
    const MKL_INT k, const double alpha, const double *a, const MKL_INT lda,
    const MKL_INT stridea, const double *b, const MKL_INT ldb,
    const MKL_INT strideb, const double beta, double *c, const MKL_INT ldc,
    const MKL_INT stridec, const MKL_INT batch_size)
{
  cblas_dgemm_batch_strided(layout, transa, transb, m, n, k, alpha, a, lda,
                            stridea, b, ldb, strideb, beta, c, ldc, stridec,
                            batch_size);
}

static void cppblas_gemm_batch_strided(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n,
    const MKL_INT k, const float alpha, const float *a, const MKL_INT lda,
    const MKL_INT stridea, const float *b, const MKL_INT ldb,
    const MKL_INT strideb, const float beta, float *c, const MKL_INT ldc,
    const MKL_INT stridec, const MKL_INT batch_size)
{
  cblas_sgemm_batch_strided(layout, transa, transb, m, n, k, alpha, a, lda,
                            stridea, b, ldb, strideb, beta, c, ldc, stridec,
                            batch_size);
}

#endif

#endif

#endif
