template <typename Scalar>
void kernel_F1_GPU(Scalar *U, Scalar *dHdr, Scalar *drdrx, Scalar *F1,
                   int *mask, int a, int b, int c, int k);
template <typename Scalar>
void kernel_fused_F2_GPU(Scalar *V, Scalar *R, Scalar *drdrx, Scalar *dMdrx,
                         Scalar *F2, int n, int d, int m);
template <typename Scalar>
void kernel2_GPU(Scalar *G, Scalar *T, Scalar *dEdG, Scalar *P, Scalar *dEdS,
                 int m, int d, int k, int b, int a);
template <typename Scalar>
void kernel_U_GPU(Scalar *dEdS, Scalar *M, Scalar *U, int a, int b, int d, int c,
                  int k);
template <typename Scalar>
void kernel_V_GPU(Scalar *dEdS, Scalar *H, Scalar *V, int a, int b, int d, int c,
                  int k);
template <typename Scalar>
void kernel_1_GPU(Scalar *M, Scalar *H, Scalar *P, Scalar *T, Scalar *G,
                  Scalar *S, int *sign, int a, int b, int c, int d, int k,
                  int m);
template <typename Scalar>
int setup_tensors_GPU(int *ilist, int inum, const int *type, const int *fmap,
                      double *pos, int old_numneigh_max, int *numneigh,
                      int **firstneigh, int *eltij_pos, int *i2row,
                      Scalar **R_GPU, Scalar **G_GPU, Scalar **dEdG_GPU,
                      Scalar **P_GPU, Scalar **dEdS_GPU, Scalar **M_GPU,
                      Scalar **H_GPU, Scalar **V_GPU, Scalar **dHdr_GPU,
                      Scalar **drdrx_GPU, Scalar **F_GPU, Scalar **dMdrx_GPU,
                      int **mask_GPU, int **ilist_GPU, int **fmap_type_GPU,
                      double **pos_GPU, int **numneigh_GPU,
                      int **firstneigh_GPU, int **ijlist_GPU, int **i2row_GPU,
                      Scalar cutforcesq, int nmax, int alocal, int neltypes,
                      int d, int k, int m, bool use_exact_cmax);
template <typename Scalar>
void free_tensors_GPU(int *ilist_GPU, double *pos_GPU, int *numneigh_GPU,
                      int *firstneigh_GPU, int *ijlist_GPU, int *i2row_GPU,
                      int *eltij_pos_GPU, int *fmap_type_GPU, Scalar *R_GPU,
                      Scalar *G_GPU, Scalar *dEdG_GPU, Scalar *T_GPU,
                      Scalar *P_GPU, Scalar *dEdS_GPU, Scalar *M_GPU,
                      Scalar *H_GPU, Scalar *V_GPU, Scalar *dHdr_GPU,
                      Scalar *drdrx_GPU, Scalar *F_GPU, Scalar *dMdrx_GPU,
                      int *mask_GPU, double *vatom_GPU);
template <typename Scalar>
void setup_global_GPU(Scalar **T_GPU, Scalar *T, int **eltij_pos_GPU,
                      int *eltij_pos, int neltypes, int d, int m);
template <typename Scalar>
void sum_forces_GPU(int n, Scalar *F_GPU, Scalar *R_GPU, Scalar *drdrx_GPU,
                    int *ijlist_GPU, int *mask_GPU, double *f_GPU,
                    double **vatom_GPU, double *f, double *vatom, double scale,
                    int nmax, int nelt, int cmax);
void setup_device_GPU(int mpiid, FILE *screen, FILE *logfile);
void memory_GPU(int mpiid, FILE *screen, FILE *logfile);