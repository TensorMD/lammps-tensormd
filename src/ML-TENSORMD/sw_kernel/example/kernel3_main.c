//#include "mkl.h"
//#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <athread.h>
#include "cblas.h"
#include "math.h"
#include "utils.h"
#include "calculate_dM.h"
#include "bench_chk.h"
#include "sw_TensorMD.h"

// #define STRONG_SCALING

#define CHECK_SW_RESULT

extern char *optarg;
extern int optopt;

long long a = 500, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;
double alpha = 1.0, beta =0.0;

double randfrom(double min, double max)
{
  double range = (max-min);
  double div = RAND_MAX/range;
  return min+(rand()/div);
}
#ifdef CHECK_RESULT
void RESULT_CHK(long long size, double *ORI, double *CHK, int check_flag) {
	int right = 1;
	long long i;
	FILE* fp;
  if((fp = fopen("error","w")) == NULL)
  {
    printf("Error open file error!\n");
    exit(0);
  }

  printf("%d element begin check !\n",size);
#ifndef demo_DEBUG
  for(i=0; i<size; i++)
	{
    if(fabs(CHK[i]-ORI[i])>1.0e-10)
    {
      printf("%ld element result is wrong! Check %d\n",i,check_flag);
			printf("ERR_CHK[%ld]=%20.12lf\n",i,CHK[i]);
			printf("ERR_ORI[%ld]=%20.12lf\n",i,ORI[i]);
			fprintf(fp,"ERR_CHK[%ld]=%20.12lf\n",i,CHK[i]);
			fprintf(fp,"ERR_ORI[%ld]=%20.12lf\n",i,ORI[i]);
      right = 0;
      break;
    }
	}
#else
  for(i=0; i<size; i++)
	{
		if(i==128){
			fprintf(fp,"ERR_CHK[%ld]=%20.12lf\n",i,CHK[i]);
			fprintf(fp,"ERR_ORI[%ld]=%20.12lf\n",i,ORI[i]);
		}
	}
#endif
#ifdef demo_DEBUG
	// code for debug
  for(i=0; i<size; i++){
		if(CHK[i] != 0){
		fprintf(fp,"DEBUG_CHK[%d]=%lf\n",i,CHK[i]);
		fprintf(fp,"DEBUG_ORI[%d]=%lf\n",i,ORI[i]);
		}
  }
#endif
  if(right)
  {
    printf("%d element result is right! Check %d\n",size,check_flag);
		fprintf(fp,"RIGHT_CHK[%ld]=%20.12lf===%d\n",size-1,CHK[size-1],check_flag);
		fprintf(fp,"RIGHT_ORI[%ld]=%20.12lf===%d\n",size-1,ORI[size-1],check_flag);
  }
}
#endif //#ifdef CHECK_RESULT

/* 
 *  dM_size = a * b * c * x * d;
 *  R_size = a * b * c;
 *  D_size = a * b * c * x;
 *
 *  V_size = a * b * c * d;
 *  H_size = a * b * c * k;
 *  P_size = a * b * k * d;
 *  F_size = a * b * c * x;
 *
 * */
#ifdef CHECK_RESULT
void kernel3_fused_CHK(double *H, double *dP, double *V_CHK, double *drdrx, double *R,
                   double *F2_fused_CHK)
{

int abc;
double dM[3*d];
double rij, rijinv;
int t_l,t_c,t_d,t_k,t_x;
double tmp,w;

for(t_d=0; t_d<3*d; t_d++){dM[t_d]=0.0;}

for (t_l = 0; t_l < a*b; t_l++){
for (t_c = 0; t_c < c; t_c++){
  for (t_d = 0; t_d < d; t_d++){
    tmp = 0;
    for (t_k = 0; t_k < k; t_k++){
    tmp += H[ t_l*c*k+t_c * k + t_k ]*dP[ t_l*k*d+t_k * d + t_d ];
    }
    V_CHK[ t_l*c*d + t_c * d + t_d ] = tmp;
  	
  }

	abc = t_l*c + t_c;
  rij = R[abc];
	if( rij == 0 ) {
	   //printf("k3_chk,R[%d]=%lf\n",abc,R[abc]);
	}
	else{
	  //dM = 0;
	  rijinv = 1.0 / rij;
	  calculate_dM(m - 1, rijinv, drdrx + abc * 3, dM);
	  double *V_ic = V_CHK + abc * d;
		for (int t_x = 0; t_x < 3; t_x++) {
	    w = 0.0 ;
			//F2[abc * 3 + ix] = simd_dot(V_ic, &dM[ix * d], d);
			for(t_d = 0; t_d < d; t_d++){
				//w += V_CHK[abc*d+t_d]*dM[t_x*d+t_d];
				w += V_ic[t_d]*dM[t_x*d+t_d];
			}
			F2_fused_CHK[abc * 3 + t_x] = w;
		}
	}
}
}
#if 0
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, d, k,
                            1, H, k, c * k, dP, d, k * d, 0, V, d, d * c,
                            a * b);
  int abc;
  double dM[3 * d];
  double rij, rijinv;
  for (int ia = 0; ia < a; ia++) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        abc = ia * b * c + ib * c + ic;
        rij = R[abc];
        if (rij == 0) break;
        rijinv = 1.0 / rij;
        calculate_dM(m - 1, rijinv, drdrx + abc * 3, dM);
        double *V_ic = V + abc * d;
        for (int ix = 0; ix < 3; ix++) {
          F2[abc * 3 + ix] = simd_dot(V_ic, &dM[ix * d], d);
        }
      }
    }
  }
#endif
}
#endif

#ifdef MKL
void kernel3(double *H, double *dP, double *V, double *dM, double *F2) {
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, d, k,
                            1, H, k, c * k, dP, d, k * d, 0, V, d, d * c,
                            a * b);
  cblas_dgemv_batch_strided(CblasRowMajor, CblasNoTrans, x, d, 1, dM, d, d * x,
                            V, 1, d, 0, F2, 1, x, a * b * c);
}

double simd_dot(double *y, double *kernel, int size) {
  double result = 0;
  for (int i = 0; i < size; i++) {
    result += y[i] * kernel[i];
  }
  return result;
}

void kernel3_fused(double *H, double *dP, double *V, double *drdrx, double *R,
                   double *F2)
{
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, d, k,
                            1, H, k, c * k, dP, d, k * d, 0, V, d, d * c,
                            a * b);
  int abc;
  double dM[3 * d];
  double rij, rijinv;
  for (int ia = 0; ia < a; ia++) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        abc = ia * b * c + ib * c + ic;
        rij = R[abc];
        if (rij == 0) break;
        rijinv = 1.0 / rij;
        calculate_dM(m - 1, rijinv, drdrx + abc * 3, dM);
        double *V_ic = V + abc * d;
        for (int ix = 0; ix < 3; ix++) {
          F2[abc * 3 + ix] = simd_dot(V_ic, &dM[ix * d], d);
        }
      }
    }
  }
}
#endif

int main(int argc, char *argv[]) {
  int world_size, world_rank;

  //MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int arg_temp;

  while (EOF != (arg_temp = getopt(argc, argv, "a::b::c::k::m::"))) {
    switch (arg_temp) {
    case 'a':
      if (optarg == NULL)
        printf("Legal input should be like -a30000, using default "
               "configuration.\n");
      else
        a = atoi(optarg);
      break;
    case 'b':
      if (optarg == NULL)
        printf(
            "Legal input should be like -b1, using default configuration.\n");
      else
        b = atoi(optarg);
      break;
    case 'c':
      if (optarg == NULL)
        printf(
            "Legal input should be like -c200, using default configuration.\n");
      else
        c = atoi(optarg);
      break;
    case 'k':
      if (optarg == NULL)
        printf(
            "Legal input should be like -k64, using default configuration.\n");
      else
        k = atoi(optarg);
      break;
    case 'm':
      if (optarg == NULL) {
        printf(
            "Legal input should be like -m3, using default configuration.\n");
      } else {
        m = atoi(optarg) + 1;
        if (m < 1 || m > 4) {
          printf("Illegal value of m, using default configureation.\n");
          m = 4;
          d = 20;
        } else if (m == 1)
          d = 1;
        else if (m == 2)
          d = 4;
        else if (m == 3)
          d = 10;
      }
      break;
    case '?':
      printf("Unsupported argument %c, using defualt configuration.\n", optopt);
      break;
    }
  }

  // 模拟多进程对张量分块处理

#ifdef STRONG_SCALING
  a = a / world_size;
#endif

  double *D;
  double *R;
  double *drdrx;
  double *dM;
  int *mask;
  long long R_size = a * b * c;
  long long D_size = a * b * c * x;
  long long dM_size = a * b * c * x * d;

  R = (double *) malloc(R_size * sizeof(double));
  D = (double *) malloc(D_size * sizeof(double));
  dM = (double *) malloc(dM_size * sizeof(double));
  drdrx = (double *) malloc(D_size * sizeof(double));
  mask = (int *)malloc(R_size * sizeof(int));

  srand(time(0));
  for (long long ia = 0; ia < a; ia++) {
    for (long long ib = 0; ib < b; ib++) {
      long long rnd = rand() % (c / 2) + c / 2;
      for (long long ic = 0; ic < c; ic++) {
        mask[ia * b * c + ib * c + ic] = 0;
        if (ic < rnd) {
          mask[ia * b * c + ib * c + ic] = 1;
        }
      }
    }
  }
#ifdef demo_DEBUG
  for (long long ia = 0; ia < a; ia++) {
    for (long long ib = 0; ib < b; ib++) {
      for (long long ic = 0; ic < c; ic++) {
       printf("hexiang-mask[%d]=%d\n",ia * b * c + ib * c + ic,mask[ia * b * c + ib * c + ic] );
      }}}
#endif

  for (long long i = 0; i < R_size; i++) {
    if (mask[i] == 0) {
      D[i * 3 + 0] = 0;
      D[i * 3 + 1] = 0;
      D[i * 3 + 2] = 0;
    } else {
      D[i * 3 + 0] = randfrom(1.0, 3.0);
      D[i * 3 + 1] = randfrom(1.0, 3.0);
      D[i * 3 + 2] = randfrom(1.0, 3.0);
    }
  }

  for (long long i = 0; i < a * b * c; i++) {
    if (mask[i] == 0) {
      R[i] = 0;
      drdrx[i * 3 + 0] = 0;
      drdrx[i * 3 + 1] = 0;
      drdrx[i * 3 + 2] = 0;
      for (long long j = 0; j < 3 * d; j++) {
        dM[i * 3 * d + j] = 0;
      }
      continue;
    }
    double dx = D[i * 3 + 0];
    double dy = D[i * 3 + 1];
    double dz = D[i * 3 + 2];
    double rij = sqrt(dx * dx + dy * dy + dz * dz);
    R[i] = rij;
    drdrx[i * 3 + 0] = dx / rij;
    drdrx[i * 3 + 1] = dy / rij;
    drdrx[i * 3 + 2] = dz / rij;
    double rijinv = 1.0 / rij;
    calculate_dM(3, rijinv, drdrx + i * 3, dM + i * 3 * d);
  }

#ifdef demo_DEBUG
  for (long long ia = 0; ia < a; ia++) {
    for (long long ib = 0; ib < b; ib++) {
      for (long long ic = 0; ic < c; ic++) {
       printf("hexiang-R[%d]=%lf\n",ia * b * c + ib * c + ic,R[ia * b * c + ib * c + ic] );
      }}}
#endif

  double *H;
  double *dP;
  double *V;
  double *F2;
  double *F2_fused;
#ifdef CHECK_RESULT
  double *V_CHK, *F2_CHK, *F2_fused_CHK;
#endif
#ifdef CHECK_SW_RESULT
  double *V_SW, *F2_SW, *F2_fused_SW;
#endif

	unsigned long t1,t2;

  long long H_size = a * b * c * k;
  long long P_size = a * b * k * d;
  long long V_size = a * b * c * d;
  long long F_size = a * b * c * x;

  //分配内存空间并初始化张量

  H = (double *)malloc(H_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  V = (double *)malloc(V_size * sizeof(double));
  F2 = (double *)malloc(F_size * sizeof(double));
  F2_fused = (double *)malloc(F_size * sizeof(double));
#ifdef CHECK_RESULT
  V_CHK = (double *)malloc(V_size * sizeof(double));
  //F2_CHK = (double *)malloc(F_size * sizeof(double));
  F2_fused_CHK = (double *)malloc(F_size * sizeof(double));
  for (long long i = 0; i < V_size; i++) {
		V_CHK[i] = 0.0;
  }
  for (long long i = 0; i < F_size; i++) {
    //F2_CHK[i] = 0.0;
    F2_fused_CHK[i] = 0.0;
  }
#endif
#ifdef CHECK_SW_RESULT
  V_SW = (double *)malloc(V_size * sizeof(double));
  //F2_SW = (double *)malloc(F_size * sizeof(double));
  F2_fused_SW = (double *)malloc(F_size * sizeof(double));
  for (long long i = 0; i < V_size; i++) {
		V_SW[i] = 0.0;
  }
  for (long long i = 0; i < F_size; i++) {
    //F2_CHK[i] = 0.0;
    F2_fused_SW[i] = 0.0;
  }
#endif

  for (long long i = 0; i < F_size; i++) {
    F2[i] = 0.0;
    F2_fused[i] = 0.0;
  }
  for (long long i = 0; i < H_size; i++) {
    H[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < P_size; i++) {
    dP[i] = randfrom(0.0, 1.0);
  }

  //kernel3(H, dP, V, dM, F2);
#ifdef MKL
  kernel3_fused(H, dP, V, drdrx, R, F2_fused);
  double diff = 0.0;
  for (long long i = 0; i < F_size; i++) {
    diff += fabs(F2[i] - F2_fused[i]);
  }
#endif

#ifdef CHECK_RESULT
  kernel3_fused_CHK(H, dP, V_CHK, drdrx, R, F2_fused_CHK);
#endif

#ifdef CHECK_SW_RESULT
	athread_init();
  printf("****************************** k3-sw ******************************\n");
  printf("k3-sw-using a=%d, b=%d, c=%d, d=%d, k=%d, m+1=%d, x=%d.\n", a, b, c, d, k, m , x);
	t1 = CRTS_time_cycle();
  //sw_dgemm_ddot_sum(a*b, c, k, d, x, m, alpha, H, dP, beta, V_SW, drdrx, R, F2_fused_SW, mask);
#ifdef PUT_V
  sw_dgemm_ddot_sum(a*b, c, k, d, x, m, alpha, H, dP, beta, V_SW, drdrx, R, F2_fused_SW);
#else
  sw_dgemm_ddot_sum(a*b, c, k, d, x, m, alpha, H, dP, beta, drdrx, R, F2_fused_SW);
#endif
	t2 = CRTS_time_cycle();
	const double cyc = 2.0;
	const double cyc_tm = 0.001 * 0.001 * 0.001;
	double tm1 = cyc_tm*(t2-t1)/cyc;
	printf("k3-sw-cycle = %ld, time = %lf\n",t2-t1,tm1);
#endif

#ifdef CHECK_RESULT
  //RESULT_CHK(V_size, V, V_CHK, 1);
  //RESULT_CHK(F_size, F2_fused, F2_fused_CHK, 2);
	
  //RESULT_CHK(V_size, V_CHK, V_SW, 3);
  RESULT_CHK(F_size, F2_fused_CHK, F2_fused_SW, 4);
#endif

#if 0
  double diff_CHK = 0.0;
  for (long long i = 0; i < F_size; i++) {
    //diff_CHK += fabs(F2_CHK[i] - F2_fused_CHK[i]);
#ifdef demo_DEBUG
    //printf("CHK--F2[%d]=%lf,F2_fused[%d]=%lf: %f\n", i,F2_CHK[i],i,F2_fused_CHK[i]);
#endif
  }
  //if (world_rank == 0) {
    printf("CHK_Diff of kernel 3 and fused kernel 3: %f\n", diff_CHK);
  //}


  if (world_rank == 0) {
    printf("Diff of kernel 3 and fused kernel 3: %f\n", diff);
  }
#endif

  double start, end;
  int iter = 5;
#ifdef MKL
  //MPI_Barrier(MPI_COMM_WORLD);
  //start = MPI_Wtime();

  for (int i = 0; i < iter; i++) {
    kernel3(H, dP, V, dM, F2);
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  //end = MPI_Wtime();

  //if (world_rank == 0) {
  if (1 == 0) {
    printf("Kernel3 with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }
#endif

  free(dM);
  free(H);
  free(dP);
  free(V);
  free(F2);
  free(R);
  free(D);
  free(drdrx);
  free(F2_fused);
  free(mask);
#ifdef CHECK_RESULT
  free(V_CHK);
  free(F2_fused_CHK);
#endif
#ifdef CHECK_RESULT
  free(V_SW);
  free(F2_fused_SW);
#endif

  //MPI_Finalize();
  return 0;
}
