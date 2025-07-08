#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "math.h"
//#include <crts.h>
#include <athread.h>
#include "cblas.h"
#include "utils.h"
#include "bench_chk.h"
#include "sw_TensorMD.h"

// #define STRONG_SCALING

//#define demo_debug
#define CHECK_SW_RESULT

extern char *optarg;
extern int optopt;

long long a = 5000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;
//long long a = 40, b = 1, c = 10, d = 20, k = 6, m = 4, x = 3;
double alpha = 1.0, beta =0.0;

#if 0

double randfrom(double min, double max)
{
  double range = (max-min);
  double div = RAND_MAX/range;
  return min+(rand()/div);
}
#endif

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

#ifdef MKL
void kernel4(double *M, double *dP, double *U, double *dR, double *F1,
             double *dH) {
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, c, k, d, 1,
                            M, d, c * d, dP, d, k * d, 0, U, k, k * c, a * b);

  for (int i = 0; i < a * b * c; i++) {
    double temp = cblas_ddot(k, U + i * k, 1, dH + i * k, 1);
    int pos = i * 3;
    F1[pos + 0] = temp * dR[pos];
    F1[pos + 1] = temp * dR[pos + 1];
    F1[pos + 2] = temp * dR[pos + 2];
  }
}

void kernel4_masked(double *M, double *dP, double *U, double *dR, double *F1_fused,
                    double *dH, int *mask)
{
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, c, k, d, 1,
                            M, d, c * d, dP, d, k * d, 0, U, k, k * c, a * b);
  int abc;
  double dM[3 * d];
  double rij, rijinv;
  for (int ia = 0; ia < a; ia++) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        abc = ia * b * c + ib * c + ic;
        if (mask[abc] == 0) {
          break;
        }
        double temp = cblas_ddot(k, U + abc * k, 1, dH + abc * k, 1);
        int pos = abc * 3;

        F1_fused[pos + 0] = temp * dR[pos + 0];
        F1_fused[pos + 1] = temp * dR[pos + 1];
        F1_fused[pos + 2] = temp * dR[pos + 2];
        //F1[pos + 0] = temp * dR[pos + 0];
        //F1[pos + 1] = temp * dR[pos + 1];
        //F1[pos + 2] = temp * dR[pos + 2];
      }
    }
  }
}

#endif //ifdef MKL

/* 
 *  M_size = a * b * c * d;
 *  H_size = a * b * c * k;
 *  P_size = a * b * k * d;
 *  U_size = a * b * c * k;
 *  R_size = a * b * c * x;
 *  F_size = a * b * c * x;
 *
 * */
void kernel4_CHK(double *M, double *dP, double *U_CHK, double *dR, double *F1_CHK, double *dH) {
		int li,ki,di,mi,ci,xi;

		double tmp,w;
		for(li=0; li<a*b; li++){
    for(ci=0; ci<c; ci++){
			w = 0.0;
      for(ki=0; ki<k; ki++)
		  {
		  	tmp=0.0;
        for(di=0; di<d; di++)
		  	{
#ifdef demo_DEBUG
		  	  if(li==64){
      	    printf("-----M[%d]=%lf,dP[%d]=%lf\n",li*c*k+ci*d+di,M[li*c*d+ci*d+di],li*k*d+di*k+ki,,dP[li*k*d+di*k+ki]);
		  	  }
#endif
          //tmp += M[li*c*d+ci*d+di] * dP1[li*k*d+di*k+ki];
          tmp += M[li*c*d+ci*d+di] * dP[li*k*d+ki*d+di];
#ifdef demo_DEBUG
		  	  if(li==64){
		      printf("-----tmp=%lf\n",tmp);
		  	  }
#endif
		  	}
		  	U_CHK[li*c*k+ci*k+ki] = tmp ;
#ifdef demo_DEBUG
		  	if(li==64){
		    printf("before dP U_CHK[%d]=%lf,tmp=%lf\n",li*c*k+ci*k+ki,U_CHK[li*c*k+ci*k+ki],tmp);
		  	}
#endif
        //U_CHK[li*c*k+ci*k+ki]=U_CHK[li*c*k+ci*k+ki]*dH[li*c*k+ci*k+ki];
        w = w + U_CHK[li*c*k+ci*k+ki]*dH[li*c*k+ci*k+ki];
		  }
#ifdef demo_DEBUG
                printf("%d check_w=%lf\n",li*c*x+ci,w);
#endif
      for(xi=0; xi<x; xi++){
				F1_CHK[li*c*x+ci*x+xi] = w * dR[li*c*x+ci*x+xi];
#ifdef demo_DEBUG
                printf("%d check_w=%lf,dR[%d]=%lf\n",li*c*x+ci*x+xi,w,li*c*x+ci*x+xi,dR[li*c*x+ci*x+xi]);
#endif
			}
		}
		}

		//printf("in kernel4------=%lf,%lf\n",U_CHK[0],F_CHK[0]);

		return;
}

void kernel4_masked_CHK(double *M, double *dP, double *U_CHK, double *dR, double *F1_fused_CHK, double *dH, int *mask) {
		int li,ki,di,mi,ci,xi;

		double tmp,w;
		for(li=0; li<a*b; li++){
    for(ci=0; ci<c; ci++){
			w = 0.0;
      for(ki=0; ki<k; ki++)
		  {
		  	tmp=0.0;
        for(di=0; di<d; di++)
		  	{
#ifdef demo_DEBUG
		  	  if(li==64){
      	    printf("---M[%d]=%lf,dP[%d]=%lf\n",li*c*k+ci*d+di,M[li*c*d+ci*d+di],li*k*d+di*k+ki,,dP[li*k*d+di*k+ki]);
		  	  }
#endif
          //tmp += M[li*c*d+ci*d+di] * dP1[li*k*d+di*k+ki];
          tmp += M[li*c*d+ci*d+di] * dP[li*k*d+ki*d+di];
#ifdef demo_DEBUG
				  printf("--li=%d,ci=%d,M[%d]=%lf,dP[%d]=%lf\n",li,ci,li*k*d+ki*d+di,M[li*k*d+ki*d+di],li*k*d+ki*d+di,dP[li*k*d+ki*d+di]);
		  	  if(li==64){
		      printf("-----tmp=%lf\n",tmp);
		  	  }
#endif
		  	}
		  	U_CHK[li*c*k+ci*k+ki] = tmp ;
#ifdef demo_DEBUG
				printf("tmp-U_CHK[%d]=%20.12lf\n",li*c*k+ci*k+ki,U_CHK[li*c*k+ci*k+ki]);
		  	if(li==64){
		    printf("before dP U_CHK[%d]=%lf,tmp=%lf\n",li*c*k+ci*k+ki,U_CHK[li*c*k+ci*k+ki],tmp);
		  	}
#endif
		  }
#ifdef demo_DEBUG
      printf("mask-check-indx mask[%d]=%d,c=%d\n",li*c+ci,mask[li*c+ci],c);
#endif
      if (mask[li*c+ci] == 0) {
#ifdef demo_DEBUG
				printf("mask[%d]=%d\n",li*c+ci,mask[li*c+ci]);
#endif
      }
      else{
               for(ki=0; ki<k; ki++)
		           {
#ifdef demo_DEBUG
				 printf("mask[%d]=%d,U[%d]=%lf,dH[%d]=%lf\n",li*c+ci,mask[li*c+ci],li*c*k+ci*k+ki,U_CHK[li*c*k+ci*k+ki],li*c*k+ci*k+ki,dH[li*c*k+ci*k+ki]);
#endif
                     w = w + U_CHK[li*c*k+ci*k+ki]*dH[li*c*k+ci*k+ki];
		           }
#ifdef demo_DEBUG
               printf("%d check_w=%lf\n",li*c*x+ci,w);
#endif
               for(xi=0; xi<x; xi++){
		                 F1_fused_CHK[li*c*x+ci*x+xi] = w * dR[li*c*x+ci*x+xi];
#ifdef demo_DEBUG
				 printf("F1_fused_CHK[%d]=%lf,dR[%d]=%lf\n",li*c*x+ci*x+xi,F1_fused_CHK[li*c*x+ci*x+xi],li*c*x+ci*x+xi,dR[li*c*x+ci*x+xi]);
#endif
#ifdef demo_DEBUG
                 printf("%d check_w=%lf,dR[%d]=%lf\n",li*c*x+ci*x+xi,w,li*c*x+ci*x+xi,dR[li*c*x+ci*x+xi]);
#endif
		           }
       };
		}//for(ci=0; ci<c; ci++)
		}

		//printf("in kernel4------=%lf,%lf\n",U_CHK[0],F_CHK[0]);

		return;
}

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

  R = (double *) malloc(R_size * sizeof(double));
  D = (double *) malloc(D_size * sizeof(double));
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
  }

  double *dH;
  double *M;
  double *dP;
  double *U;
  double *F1;
  double *F1_fused;
#ifdef CHECK_RESULT
  double *U_CHK, *F1_CHK, *F1_fused_CHK;
#endif
#ifdef CHECK_SW_RESULT
  double *U_SW, *F1_SW, *F1_fused_SW;
#endif
	unsigned long t1, t2, t3, t4;

  long long M_size = a * b * c * d;
  long long H_size = a * b * c * k;
  long long P_size = a * b * k * d;
  long long U_size = a * b * c * k;
  long long F_size = a * b * c * x;

  //分配内存空间并初始化张量

  dH = (double *)malloc(H_size * sizeof(double));
  M = (double *)malloc(M_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  U = (double *)malloc(U_size * sizeof(double));
  F1 = (double *)malloc(F_size * sizeof(double));
  F1_fused = (double *)malloc(F_size * sizeof(double));
#ifdef CHECK_RESULT
  U_CHK = (double *)malloc(U_size * sizeof(double));
  F1_CHK = (double *)malloc(F_size * sizeof(double));
  F1_fused_CHK = (double *)malloc(F_size * sizeof(double));
  for (long long i = 0; i < U_size; i++) {
		U_CHK[i] = 0.0;
  }
  for (long long i = 0; i < F_size; i++) {
    F1_CHK[i] = 0.0;
    F1_fused_CHK[i] = 0.0;
  }
#endif
#ifdef CHECK_SW_RESULT
  U_SW = (double *)malloc(U_size * sizeof(double));
  F1_SW = (double *)malloc(F_size * sizeof(double));
  F1_fused_SW = (double *)malloc(F_size * sizeof(double));
  for (long long i = 0; i < U_size; i++) {
		U_SW[i] = 0.0;
  }
  for (long long i = 0; i < F_size; i++) {
    F1_SW[i] = 0.0;
    F1_fused_SW[i] = 0.0;
  }
#endif

  for (long long i = 0; i < F_size; i++) {
    F1[i] = 0.0;
    F1_fused[i] = 0.0;
  }
  for (long long i = 0; i < H_size; i++) {
    dH[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < M_size; i++) {
    M[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < P_size; i++) {
    dP[i] = randfrom(0.0, 1.0);
  }
#ifdef MKL
  kernel4(M, dP, U, drdrx, F1, dH);
  kernel4_masked(M, dP, U, drdrx, F1_fused, dH, mask);
#endif

#ifdef CHECK_RESULT
  kernel4_CHK(M, dP, U_CHK, drdrx, F1_CHK, dH);
  kernel4_masked_CHK(M, dP, U_CHK, drdrx, F1_fused_CHK, dH, mask);
#endif
	athread_init();
#ifdef CHECK_SW_RESULT
  printf("****************************** k4-sw ******************************\n");
  printf("k4-sw-using a=%d, b=%d, c=%d, d=%d, k=%d, m+1=%d, x=%d.\n", a, b, c, d, k, m , x);
	t1 = CRTS_time_cycle();
#ifdef PUT_U
  sw_dgemm_transB_ddot_sum(a*b, c, k, d, x, alpha, M, dP, beta, U_SW, dH, drdrx, F1_fused_SW, mask);
#else
  sw_dgemm_transB_ddot_sum(a*b, c, k, d, x, alpha, M, dP, beta, dH, drdrx, F1_fused_SW, mask);
#endif
	t2 = CRTS_time_cycle();
	const double cyc = 2.0;
	const double cyc_tm = 0.001 * 0.001 * 0.001;
	double tm1 = cyc_tm*(t2-t1)/cyc;
	printf("k4-sw-cycle =%ld, time = %lf\n",t2-t1,tm1);
#endif

#ifdef CHECK_RESULT
#ifdef MKL
  RESULT_CHK(U_size, U, U_CHK, 1);
  RESULT_CHK(F_size, F1, F1_CHK, 2);
  RESULT_CHK(F_size, F1_fused, F1_fused_CHK, 3);
#endif
#ifdef CHECK_SW_RESULT
  //RESULT_CHK(U_size, U_CHK, U_SW, 4);
  RESULT_CHK(F_size, F1_fused_CHK, F1_fused_SW, 5);
#endif
#endif

  double diff = 0.0;
  for (long long i = 0; i < F_size; i++) {
    diff += fabs(F1[i] - F1_fused[i]);
#ifdef demo_DEBUG
    printf("ORI--F1[%d]=%lf,F1_fused[%d]=%lf: %f\n", i,F1[i],i,F1_fused[i]);
#endif

  }

  //if (world_rank == 0) {
    printf("Diff of kernel 4 and fused kernel 4: %f\n", diff);
  //}

#ifdef CHECK_RESULT
  double diff_CHK = 0.0;
  for (long long i = 0; i < F_size; i++) {
    diff_CHK += fabs(F1_CHK[i] - F1_fused_CHK[i]);
#ifdef demo_DEBUG
    printf("CHK--F1[%d]=%lf,F1_fused[%d]=%lf: %f\n", i,F1_CHK[i],i,F1_fused_CHK[i]);
#endif
  }
  //if (world_rank == 0) {
    printf("CHK_Diff of kernel 4 and fused kernel 4: %f\n", diff_CHK);
  //}
#endif

  double start, end;
  int iter = 5;

  //MPI_Barrier(MPI_COMM_WORLD);
  //start = MPI_Wtime();
#ifdef MKL 
  for (int i = 0; i < iter; i++) {
    kernel4(M, dP, U, drdrx, F1, dH);
  }
#endif

  //MPI_Barrier(MPI_COMM_WORLD);
  //end = MPI_Wtime();

  //if (world_rank == 0) {
  if (1 == 0) {
    printf("Kernel4 with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }

  free(M);
  free(dH);
  free(dP);
  free(U);
  free(drdrx);
  free(F1);
  free(F1_fused);
  free(D);
  free(mask);
  free(R);
#ifdef CHECK_RESULT
  free(U_CHK);
  free(F1_CHK);
  free(F1_fused_CHK);
#endif
#ifdef CHECK_SW_RESULT
  free(U_SW);
  free(F1_SW);
  free(F1_fused_SW);
#endif

  //MPI_Finalize();
  return 0;
}
