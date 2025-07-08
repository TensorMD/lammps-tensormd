#ifndef _ATHREAD_TENSORALLOY_H_
#define _ATHREAD_TENSORALLOY_H_

#ifdef __cplusplus
extern "C" {
#endif
union athread_args {
  void *p;
  int i;
  double d;
};
void athread_tensoralloy_setup_tensors(
    int global_size, int *i2row, const int *type, const int *fmap, double **pos,
    const int *numneigh, int **firstneigh, int neltypes, double cutforcesq,
    const int *ilist, int **eltij_pos,
    int size_c, int size_d, int *ijlist, void *R, int *mask, void *M,
    void *drdrx, void *dMdrx);
void athread_tensoralloy_get_cmax(
    int inum, const int *type, const int *fmap, double **pos,
    const int *numneigh, int **firstneigh, int neltypes, double cutforcesq,
    int **eltij_pos, int *ilist, int *cmax_array);
void athread_tensoralloy_batch_ration_compute(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df);
void athread_tensoralloy_batch_ration_computef(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df);
void athread_tensoralloy_batch_ration_computedf(int n, int n_out, int size,
                                              double dx, void *params, void *x,
                                              void *f, void *df);
void athread_tensoralloy_batch_dgemv_F2(int batch_size, int x, int d, void *A,
                                        void *B, void *C);
void athread_tensoralloy_batch_small_dgemm(int batch_size, int M, int N, int K,
                                           int transA, int transB, void *A,
                                           void *B, void *C);
void athread_tensoralloy_batch_ddot_F1(int batch_size, int k, int x, void *A,
                                       void *B, void *C, void *D);
void athread_tensoralloy_cutoff(int fctype, int size, double cutforce,
                                void *mask, void *R, void *sij, void *dsij);
void athread_tensoralloy_descriptor(int interp, int size, int k, void *H,
                                    void *dHdr, void *sij, void *dsij);
void athread_tensoralloy_dEdS_P(int size, int m, void *dEdS, void *dEdG,
                                       void *P);
void athread_tensoralloy_contract_dEdP(int size, int m, void *dEdS, void *dEdG,
                                       void *P);
void athread_tensoralloy_P_2(int size, int m, void *G, void *P);
void athread_tensoralloy_contract_G(int size, int m, void *G, void *P);
void athread_tensoralloy_forward_resnet(int M, int N, int K, void *input,
                                        void *weight, void *bias, int actfn,
                                        int resnet_dt, void *output, void *dC);
void athread_tensoralloy_sum_forces(int amax, int bmax, int cmax, int *ijlist,
                                    void *F1, void *F2, void *drdrx, void *R,
                                    int natoms, int vatom_flag, void *farray,
                                    void *varray, int num_threads);
void athread_tensoralloy_reduce_forces(int natoms, void *farray, void *varray,
                                       double **f, double **vatom, int vatom_flag,
                                       int num_duplicates, int num_threads);
void athread_initialize_array_1d(int size, void *array, double value,
                                 int num_threads);
void athread_tensoralloy_dEdG(int size, int m, void *G, void *dEdG);
void athread_tensoralloy_axpy(int m, int n, void *x, void *y);
void athread_tensoralloy_vmul(int n, void *x, void *y);
void athread_tensoralloy_act(int actfn, int n, void *x, void *y);
void athread_tensoralloy_axpy_act(int actfn, int res, int M, int N, void *old_x,
                                  void *x, void *y, void *bias);
#ifdef __cplusplus
}
#endif

#endif
