#include <athread.h>
#include <mpi.h>
#include <stdio.h>
#include "athread_tensoralloy.h"

extern void SLAVE_FUN(setup_tensors)();

void athread_tensoralloy_setup_tensors(
    int global_size, int *i2row, const int *type, const int *fmap, double **pos,
    const int *numneigh, int **firstneigh, int neltypes, double cutforcesq,
    const int *ilist, int **eltij_pos,
    int size_c, int size_d, int *ijlist, void *R, int *mask, void *M,
    void *drdrx, void *dMdrx)
{
  athread_init();

  void *args[19];
  args[0] = &global_size;
  args[1] = i2row;
  args[2] = type;
  args[3] = fmap;
  args[4] = pos;
  args[5] = numneigh;
  args[6] = firstneigh;
  args[7] = &neltypes;
  args[8] = &cutforcesq;
  args[9] = ilist;
  args[10] = eltij_pos;
  args[11] = &size_c;
  args[12] = &size_d;
  args[13] = ijlist;
  args[14] = R;
  args[15] = mask;
  args[16] = M;
  args[17] = drdrx;
  args[18] = dMdrx;

  athread_spawn(setup_tensors, args);
  athread_join();
}

extern void SLAVE_FUN(get_cmax)();

void athread_tensoralloy_get_cmax(
    int inum, const int *type, const int *fmap, double **pos,
    const int *numneigh, int **firstneigh, int neltypes, double cutforcesq,
    int **eltij_pos, int *ilist, int *cmax_array)
{
  athread_init();

  void *args[11];
  args[0] = &inum;
  args[1] = type;
  args[2] = fmap;
  args[3] = pos;
  args[4] = numneigh;
  args[5] = firstneigh;
  args[6] = &neltypes;
  args[7] = &cutforcesq;
  args[8] = eltij_pos;
  args[9] = ilist;
  args[10] = cmax_array;

  athread_spawn(get_cmax, args);
  athread_join();
}

extern void SLAVE_FUN(batch_ration_compute)();

void athread_tensoralloy_batch_ration_compute(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df)
{
  athread_init();
  void *args[8];
  args[0] = &n;
  args[1] = &n_out;
  args[2] = &size;
  args[3] = &dx;
  args[4] = params;
  args[5] = x;
  args[6] = f;
  args[7] = df;
  athread_spawn(batch_ration_compute, args);
  athread_join();
}

extern void SLAVE_FUN(batch_ration_computef)();

void athread_tensoralloy_batch_ration_computef(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df)
{
  athread_init();
  void *args[8];
  args[0] = &n;
  args[1] = &n_out;
  args[2] = &size;
  args[3] = &dx;
  args[4] = params;
  args[5] = x;
  args[6] = f;
  args[7] = df;
  athread_spawn(batch_ration_computef, args);
  athread_join();
}

extern void SLAVE_FUN(batch_ration_computedf)();

void athread_tensoralloy_batch_ration_computedf(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df)
{
  athread_init();
  void *args[8];
  args[0] = &n;
  args[1] = &n_out;
  args[2] = &size;
  args[3] = &dx;
  args[4] = params;
  args[5] = x;
  args[6] = f;
  args[7] = df;
  athread_spawn(batch_ration_computedf, args);
  athread_join();
}

extern void SLAVE_FUN(batch_dgemv_F2)();

void athread_tensoralloy_batch_dgemv_F2(int batch_size, int x, int d, void *A,
                                        void *B, void *C)
{
  athread_init();
  union athread_args args[6];
  args[0].i = batch_size;
  args[1].i = x;
  args[2].i = d;
  args[3].p = A;
  args[4].p = B;
  args[5].p = C;
  athread_spawn(batch_dgemv_F2, args);
  athread_join();
}

extern void SLAVE_FUN(batch_small_dgemm_m)();

void athread_tensoralloy_batch_small_dgemm(int batch_size, int M, int N, int K,
                                           int transA, int transB, void *A,
                                           void *B, void *C)
{
  athread_init();
  union athread_args args[9];
  args[0].i = batch_size;
  args[1].i = M;
  args[2].i = N;
  args[3].i = K;
  args[4].i = transA;
  args[5].i = transB;
  args[6].p = A;
  args[7].p = B;
  args[8].p = C;
  athread_spawn(batch_small_dgemm_m, args);
  athread_join();
}

extern void SLAVE_FUN(batch_ddot_F1)();

void athread_tensoralloy_batch_ddot_F1(int batch_size, int k, int x, void *A,
                                       void *B, void *C, void *D)
{
  athread_init();
  union athread_args args[7];
  args[0].i = batch_size;
  args[1].i = k;
  args[2].i = x;
  args[3].p = A;
  args[4].p = B;
  args[5].p = C;
  args[6].p = D;
  athread_spawn(batch_ddot_F1, args);
  athread_join();
}

extern void SLAVE_FUN(r_cutoff)();

void athread_tensoralloy_cutoff(int fctype, int size, double cutforce,
                                void *mask, void *R, void *sij, void *dsij)
{
  athread_init();
  union athread_args args[7];
  args[0].i = fctype;
  args[1].i = size;
  args[2].d = cutforce;
  args[3].p = mask;
  args[4].p = R;
  args[5].p = sij;
  args[6].p = dsij;
  athread_spawn(r_cutoff, args);
  athread_join();
}

extern void SLAVE_FUN(descriptor)();

void athread_tensoralloy_descriptor(int interp, int size, int k, void *H,
                                    void *dHdr, void *sij, void *dsij)
{
  athread_init();
  union athread_args args[7];
  args[0].i = interp;
  args[1].i = size;
  args[2].i = k;
  args[3].p = H;
  args[4].p = dHdr;
  args[5].p = sij;
  args[6].p = dsij;
  athread_spawn(descriptor, args);
  athread_join();
}

extern void SLAVE_FUN(contract_dEdP)();

void athread_tensoralloy_contract_dEdP(int size, int m, void *dEdS, void *dEdG,
                                       void *P)
{
  athread_init();
  union athread_args args[5];
  args[0].i = size;
  args[1].i = m;
  args[2].p = dEdS;
  args[3].p = dEdG;
  args[4].p = P;
  athread_spawn(contract_dEdP, args);
  athread_join();
}

extern void SLAVE_FUN(dEdS_P)();

void athread_tensoralloy_dEdS_P(int size, int m, void *dEdS, void *dEdG,
                                       void *P)
{
  athread_init();
  union athread_args args[5];
  args[0].i = size;
  args[1].i = m;
  args[2].p = dEdS;
  args[3].p = dEdG;
  args[4].p = P;
  athread_spawn(dEdS_P, args);
  athread_join();
}

extern void SLAVE_FUN(contract_G)();

void athread_tensoralloy_contract_G(int size, int m, void *G, void *P)
{
  athread_init();
  union athread_args args[4];
  args[0].i = size;
  args[1].i = m;
  args[2].p = G;
  args[3].p = P;
  athread_spawn(contract_G, args);
  athread_join();
}

extern void SLAVE_FUN(P_2)();

void athread_tensoralloy_P_2(int size, int m, void *G, void *P)
{
  athread_init();
  union athread_args args[4];
  args[0].i = size;
  args[1].i = m;
  args[2].p = G;
  args[3].p = P;
  athread_spawn(P_2, args);
  athread_join();
}

extern void SLAVE_FUN(forward_resnet)();

void athread_tensoralloy_forward_resnet(int M, int N, int K, void *input,
                                        void *weight, void *bias, int actfn,
                                        int resnet_dt, void *output, void *dC)
{
  athread_init();
  union athread_args args[10];
  args[0].i = M;
  args[1].i = N;
  args[2].i = K;
  args[3].p = input;
  args[4].p = weight;
  args[5].p = bias;
  args[6].i = actfn;
  args[7].i = resnet_dt;
  args[8].p = output;
  args[9].p = dC;
  athread_spawn(forward_resnet, args);
  athread_join();
}

extern void SLAVE_FUN(sum_forces_virial)();

void athread_tensoralloy_sum_forces(int amax, int bmax, int cmax, int *ijlist,
                                    void *F1, void *F2, void *drdrx, void *R,
                                    int natoms, int vatom_flag, void *farray,
                                    void *varray, int num_threads)
{
  athread_init();

  void *args[13];
  args[0] = (void *) &amax;
  args[1] = (void *) &bmax;
  args[2] = (void *) &cmax;
  args[3] = (void *) ijlist;
  args[4] = (void *) F1;
  args[5] = (void *) F2;
  args[6] = (void *) drdrx;
  args[7] = (void *) R;
  args[8] = (void *) &natoms;
  args[9] = (void *) &vatom_flag;
  args[10] = (void *) farray;
  args[11] = (void *) varray;
  args[12] = (void *) &num_threads;

  athread_spawn(sum_forces_virial, args);
  athread_join();
}

extern void SLAVE_FUN(reduce_forces_virial)();

void athread_tensoralloy_reduce_forces(int natoms, void *farray, void *varray,
                                       double **f, double **vatom, int vatom_flag,
                                       int num_duplicates, int num_threads)
{
  athread_init();
  void *args[8];
  args[0] = (void *) &natoms;
  args[1] = (void *) farray;
  args[2] = (void *) varray;
  args[3] = (void *) f;
  args[4] = (void *) vatom;
  args[5] = (void *) &vatom_flag;
  args[6] = (void *) &num_duplicates;
  args[7] = (void *) &num_threads;

  athread_spawn(reduce_forces_virial, args);
  athread_join();
}

extern void SLAVE_FUN(initialize_array_1d)();

void athread_initialize_array_1d(int size, void *array, double value,
                                 int num_threads)
{
  athread_init();
  void *args[4];
  args[0] = (void *) &size;
  args[1] = (void *) array;
  args[2] = (void *) &value;
  args[3] = (void *) &num_threads;

  athread_spawn(initialize_array_1d, args);
  athread_join();
}

extern void SLAVE_FUN(sigma_dEdG)();

void athread_tensoralloy_dEdG(int size, int m, void *G, void *dEdG)
{
  athread_init();
  void *args[4];
  args[0] = (void *) &size;
  args[1] = (void *) &m;
  args[2] = (void *) G;
  args[3] = (void *) dEdG;

  athread_spawn(sigma_dEdG, args);
  athread_join();
}

extern void SLAVE_FUN(axpy)();

void athread_tensoralloy_axpy(int m, int n, void *x, void *y) {
  athread_init();
  void *args[4];
  args[0] = (void *) &m;
  args[1] = (void *) &n;
  args[2] = (void *) x;
  args[3] = (void *) y;
  athread_spawn(axpy, args);
  athread_join();
}

extern void SLAVE_FUN(vmul)();

void athread_tensoralloy_vmul(int n, void *x, void *y) {
  athread_init();
  void *args[3];
  args[0] = (void *) &n;
  args[1] = (void *) x;
  args[2] = (void *) y;
  athread_spawn(vmul, args);
  athread_join();
}

extern void SLAVE_FUN(softplus)();
extern void SLAVE_FUN(squareplus)();

void athread_tensoralloy_act(int actfn, int n, void *x, void *y) {
  athread_init();
  void *args[3];
  args[0] = (void *) &n;
  args[1] = (void *) x;
  args[2] = (void *) y;
  if (actfn == 1) {
    athread_spawn(softplus, args);
  } else if (actfn == 3) {
    athread_spawn(squareplus, args);
  }
  athread_join();
}

extern void SLAVE_FUN(axpy_softplus)();
extern void SLAVE_FUN(axpy_squareplus)();
extern void SLAVE_FUN(axpy_squareplus_simd)();

void athread_tensoralloy_axpy_act(int actfn, int res, int M, int N, void *old_x,
                                  void *x, void *y, void *bias) {
  athread_init();
  void *args[7];
  args[0] = (void *) &res;
  args[1] = (void *) &M;
  args[2] = (void *) &N;
  args[3] = (void *) old_x;
  args[4] = (void *) x;
  args[5] = (void *) y;
  args[6] = (void *) bias;
  if (actfn == 1) {
    athread_spawn(axpy_softplus, args);
  } else if (actfn == 3) {
    athread_spawn(axpy_squareplus, args);
  }
  athread_join();
}