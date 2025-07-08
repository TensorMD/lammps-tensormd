#ifdef __cplusplus
extern "C" {
#endif

#ifdef PUT_dS
void sw_dgemm_transB_vdmul(const int AB1,const int K1,const int D1,const int M1, const double alpha1,const void *dQ,const void *T, const double beta1, void *dS, const void *P, void *dP);
#else
void sw_dgemm_transB_vdmul(const int AB1,const int K1,const int D1,const int M1, const double alpha1,const void *dQ,const void *T, const double beta1, const void *P, void *dP);
#endif

#ifdef PUT_V
       void sw_dgemm_ddot_sum(const int AB, const int C, const int K, const int D, const int X, const int M, const double alpha, const void *H, const void *dP, const double beta, void *V_SW, const void *dR, const void *R, void *F2_fused_SW);
#else
       void sw_dgemm_ddot_sum(const int AB, const int C, const int K, const int D, const int X, const int M, const double alpha, const void *H, const void *dP, const double beta, const void *dR, const void *R, void *F2_fused_SW);
#endif

#ifdef PUT_U
void sw_dgemm_transB_ddot_sum(const int AB, const int C, const int K, const int D, const int X, const double alpha, const void *M, const void *dP, const double beta, void *U_SW, const void *dH, const void *dR, void *F1_fused_SW, const void *mask);
#else
void sw_dgemm_transB_ddot_sum(const int AB, const int C, const int K, const int D, const int X, const double alpha, const void *M, const void *dP, const double beta, const void *dH, const void *dR, void *F1_fused_SW, const void *mask);
#endif

void sw_dgemm_kernel1(const int AB1, const int K1, const int D1, const int C1, const int M1, const void *H, const void *M, const void *T, const void *P, const void *G_SW);

#ifdef __cplusplus
}
#endif
