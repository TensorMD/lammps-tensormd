void kernel4_CHK(double *M, double *dP, double *U_CHK, double *dR, double *F1_CHK, double *dH);
void kernel4_masked_CHK(double *M, double *dP, double *U_CHK, double *dR, double *F1_fused_CHK, double *dH, int *mask);
void kernel3_fused_CHK(double *H, double *dP, double *V_CHK, double *drdrx, double *R, double *F2_fused_CHK);

void sw_master_vdMul(long long size, double *dS, double *P, double *dP);
void RESULT_CHK(long long S_size, double *ORI, double *CHK, int CHECK_FLAG);
