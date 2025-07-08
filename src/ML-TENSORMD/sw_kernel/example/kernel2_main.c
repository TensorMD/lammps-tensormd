//#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <crts.h>
#include <athread.h>
#include "cblas.h"
#include "bench_chk.h"
#include "sw_TensorMD.h"


#define TEST
#define CHECK_SW_VDMUL
#define CHECK_SW_RESULT

//#define demo_DEBUG

extern char *optarg;
extern int optopt;

long long a = 5000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;
double alpha = 2.0, beta = 0.0;

void kernel2(double *dQ, double *T, double *dS, double *P, double *dP) {
#ifdef TEST
//  dS=2*Q[a,b,k,m]*Trans(T[d][m]);
//	cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
//	          const int M, const int N, const int K,
	            a * b * k, d, m, 
//  const double alpha, const double *A, const int lda, const double *B, const int ldb,
							2, dQ, m, T, m,
//  const double beta, double *C, const int ldc)
              0, dS, d);
#else
//  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, k, d, m, 2,
//                            dQ, m, m * k, T, m, 0, 0, dS, d, d * k, a * b);
#endif
//  dP[a,b,k,d]=dS[a,b,k,d]*P[a,b,k,d];
#ifdef CHECK_SW_VDMUL
  sw_master_vdMul(a * b * k * d, dS, P, dP);
#else
  vdMul(a * b * k * d, dS, P, dP);
#endif
}
#ifdef CHECK_SW_VDMUL
void sw_master_vdMul(long long  size, double *dS, double *P, double *dP){
		for(long long li=0; li<size; li++){
			dP[li] = dS[li]*P[li];
		}
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
		fprintf(fp,"RIGHT_CHK[%ld]=%20.12lf\n",size-1,CHK[size-1]);
		fprintf(fp,"RIGHT_ORI[%ld]=%20.12lf\n",size-1,ORI[size-1]);
  }
}

void kernel2_CHK(double *dQ, double *T, double *dS_CHK, double *P, double *dP_CHK) {
		int li,ki,di,mi;
		double tmp;
		for(li=0; li<a*b; li++)
    for(ki=0; ki<k;  ki++)
    for(di=0; di<d;  di++)
		{
			tmp=0.0;
      for(mi=0; mi<m; mi++)
			{
#ifdef demo_DEBUG
			  if(li==64){
		    printf("-----dQ[%d]=%lf,T[%d]=%lf\n",li*k*m+ki*m+mi,dQ[li*k*m+ki*m+mi],di*m+mi,T[di*m+mi]);
			  }
#endif
        tmp += dQ[li*k*m+ki*m+mi] * T[di*m+mi];
#ifdef demo_DEBUG
			  if(li==64){
		    printf("-----tmp=%lf\n",tmp);
			  }
#endif
			}
			dS_CHK[li*k*d+ki*d+di] = alpha * tmp ;//+ dS_CHK[li*k*d+ki*d+di] * beta;
#ifdef demo_DEBUG
			if(li==64){
		  printf("before dP dS_CHK[%d]=%lf,tmp=%lf\n",li*k*d+ki*d+di,dS_CHK[li*k*d+ki*d+di],tmp);
		  //printf("before0 dP %ld li=%ld,ki=%ld,di=%ld,kernel2_CHK------=%lf,%lf,%lf\n",li*k*d+ki*d+di,li,ki,di,dS_CHK[0],P[0],dP_CHK[0]);
			}
#endif
			dP_CHK[li*k*d+ki*d+di] = dS_CHK[li*k*d+ki*d+di]*P[li*k*d+ki*d+di];
#ifdef demo_DEBUG
			if(li==64){
		  printf("dP dP_CHK[%d]=%lf,dS_CHK[%d]=%lf,P[%d]=%lf\n",li*k*d+ki*d+di,dP_CHK[li*k*d+ki*d+di],li*k*d+ki*d+di,dS_CHK[li*k*d+ki*d+di],li*k*d+ki*d+di,P[li*k*d+ki*d+di]);
		  //printf("in %ld li=%ld,ki=%ld,di=%ld,kernel2_CHK------=%lf,%lf,%lf\n",li*k*d+ki*d+di,li,ki,di,dS_CHK[li*k*d+ki*d+di],P[li*k*d+ki*d+di],dP_CHK[li*k*d+ki*d+di]);
		  //printf("in0 %ld li=%ld,ki=%ld,di=%ld,kernel2_CHK------=%lf,%lf,%lf\n",li*k*d+ki*d+di,li,ki,di,dS_CHK[0],P[0],dP_CHK[0]);
			}
#endif

		}

		//printf("in kernel2_CHK------=%lf,%lf,%lf\n",dS_CHK[0],P[0],dP_CHK[0]);

		return;
}
#endif //CHECK_RESULT

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
				//printf("hexiang=====m=%d\n",m);
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

  // a = a / world_size;

  double *P;
  double *dP;
  double *dS;
  double *T;
  double *dQ;
#ifdef CHECK_RESULT
  double *dP_CHK, *dS_CHK;
#endif
#ifdef CHECK_SW_RESULT
  double *dP_SW, *dS_SW;
#endif

  long long P_size = a * b * k * d;
  long long S_size = a * b * k * d;
  long long T_size = d * m;
  long long Q_size = a * b * k * m;

  // 分配内存空间并初始化张量

  P = (double *)malloc(P_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  dS = (double *)malloc(S_size * sizeof(double));
  T = (double *)malloc(T_size * sizeof(double));
  dQ = (double *)malloc(Q_size * sizeof(double));
#ifdef CHECK_RESULT
  dS_CHK = (double *)malloc(S_size * sizeof(double));
  dP_CHK = (double *)malloc(P_size * sizeof(double));
  for (long long i = 0; i < P_size; i++) {
		dP_CHK[i] = 0.0;
  }
  for (long long i = 0; i < S_size; i++) {
    dS_CHK[i] = 0.0;
  }
#endif
#ifdef CHECK_SW_RESULT
  dS_SW = (double *)malloc(S_size * sizeof(double));
  dP_SW = (double *)malloc(P_size * sizeof(double));
  for (long long i = 0; i < P_size; i++) {
		dP_SW[i] = 0.0;
  }
  for (long long i = 0; i < S_size; i++) {
    dS_SW[i] = 0.0;
  }
#endif
  long long total_size = (P_size+P_size+S_size+T_size+Q_size+Q_size+P_size);
	total_size = total_size * sizeof(double);
	//total_size = total_size /(1024*1024*1024);
	//printf("Total size of mem is %ld\n",total_size);

  srand(time(0));
  for (long long i = 0; i < P_size; i++) {
    //P[i] = rand() / RAND_MAX;
    P[i] = 1.0 * rand() / RAND_MAX;
		dP[i] = 0.0;
  }
  for (long long i = 0; i < Q_size; i++) {
    //dQ[i] = rand() / RAND_MAX;
    dQ[i] = 1.0 * rand() / RAND_MAX;
  }
  for (long long i = 0; i < S_size; i++) {
    dS[i] = 0.0;
  }
  for (long long i = 0; i < T_size; i++) {
    //T[i] = 0.0;
    T[i] = 1.0*rand() / RAND_MAX;
  }
  T[0] = 1;
  if (m > 1) {
    T[5] = 1;
    T[9] = 1;
    T[13] = 1;
  }
  if (m > 2) {
    T[2] = a;
    T[18] = 1;
    T[22] = 2;
    T[26] = 2;
    T[30] = 1;
    T[34] = 2;
    T[38] = 1;
  }
  if (m > 3) {
    T[7] = b;
    T[11] = b;
    T[15] = b;
    T[43] = 1;
    T[47] = 3;
    T[51] = 3;
    T[55] = 3;
    T[59] = 6;
    T[63] = 3;
    T[67] = 1;
    T[71] = 3;
    T[75] = 3;
    T[79] = 1;
  }

  //for (long long i = 0; i < T_size; i++) {
  //  T[i] = 1.0 * i;
  //  //T[i] = rand() / RAND_MAX;
  //}

  double start, end;
	unsigned long t1, t2, t3, t4;
  int iter = 1;
  double tm1 = 0.0;
  double tm2 = 0.0;
  double tm3 = 0.0;

  //printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k, m - 1, x);
  //MPI_Barrier(MPI_COMM_WORLD);
  //start = MPI_Wtime();

  for (int i = 0; i < iter; i++) {
	  t1 = CRTS_time_cycle();
    kernel2(dQ, T, dS, P, dP);
	  t2 = CRTS_time_cycle();
#ifdef CHECK_RESULT
    kernel2_CHK(dQ, T, dS_CHK, P, dP_CHK);
#endif
	  t3 = CRTS_time_cycle();

#ifdef CHECK_SW_RESULT
    athread_init();
    printf("****************************** k2-sw ******************************\n");
    printf("k2-sw-using a=%d, b=%d, c=%d, d=%d, k=%d, m+1=%d, x=%d.\n", a, b, c, d, k, m , x);
#ifdef PUT_dS
    sw_dgemm_transB_vdmul(a*b, k, d, m, alpha, dQ, T, beta, dS_SW, P, dP_SW);
#else
    sw_dgemm_transB_vdmul(a*b, k, d, m, alpha, dQ, T, beta, P, dP_SW);
#endif
#endif
	  t4 = CRTS_time_cycle();
		const double cyc = 2.0;
		const double cyc_tm = 0.001 * 0.001 * 0.001;
		tm1 = tm1 + cyc_tm*(t2-t1)/cyc;
		tm2 = tm2 + cyc_tm*(t3-t2)/cyc;
		tm3 = tm3 + cyc_tm*(t4-t3)/cyc;
		//printf("%d kernel2(cblas_dgemm)time     = %lf\n",iter, tm1);
		//printf("%d kernel2_CHK(master) time    = %lf\n",iter, tm2);
		printf("k2-sw-time  = %lf\n", tm3/iter);
  }

  //MPI_Barrier(MPI_COMM_WORLD);
  //end = MPI_Wtime();

#ifdef CHECK_RESULT
//    RESULT_CHK(S_size, dS, dS_CHK);
//    RESULT_CHK(P_size, dP, dP_CHK);
//    RESULT_CHK(S_size, dS, dS_SW, 1);
    RESULT_CHK(P_size, dP, dP_SW, 2);
#endif

  //MPI_Barrier(MPI_COMM_WORLD);
	
  //if (world_rank == 0) {
  if (1==0) {
    printf("Kernel2+ with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }

  free(P);
  free(dP);
  free(dS);
  free(T);
  free(dQ);

#ifdef CHECK_RESULT
  free(dS_CHK);
  free(dP_CHK);
#endif
#ifdef CHECK_SW_RESULT
  free(dS_SW);
  free(dP_SW);
#endif

  //MPI_Finalize();
  return 0;
}
