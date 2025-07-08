//#include "mkl.h"
//#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <crts.h>
#include "cblas.h"
#include "sw_TensorMD.h"

#define TEST
//#define demo_DEBUG

extern char *optarg;
extern int optopt;

//long long a = 30000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;
long long a = 30000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;
double alpha = 1.0, beta = 0.0;

void kernel1(double *H, double *M, double *P, double *S, double *T, double *Q,
             double *G, double *sign) {
  //cblas_dgemm_batch_strided(CblasRowMajor, CblasTrans, CblasNoTrans, k, d, c, 1,
  //                          H, k, c * k, M, d, c * d, 0, P, d, d * k, a * b);
	for (int i=0;i<a*b;i++){
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, d, c, 1,
                            &H[i*k*c], k, &M[i*c*d], d, 0, &P[i*k*d], d);
	}
//	printf("after gemm1, H==%lf,M==%lf,P==%lf\n",H[0],M[0],P[0]);
  for (int i = 0; i < a * b * k; i++) {
    sign[i] = P[i * m] < 0 ? -1 : 1;
  }
  //vdSqr(a * b * k * d, P, S);
    sw_vdSqr(a * b * k * d, P, S);
//	printf("after sqr,P==%lf,S==%lf\n",P[0],S[0]);
	 
#ifdef TEST
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a * b * k, m, d, 1, S,
              d, T, m, 0, Q, m);
//	printf("after gemm2,S==%lf,T==%lf,Q==%lf\n",S[0],T[0],Q[0]);
#else
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, k, m, d,
                            1, S, d, k * d, T, m, 0, 0, Q, m, k * m, a * b);
#endif
  //vdSqrtI(a * b * k, Q, m, G, m);
  //sw_vdSqrtI(a * b * k, Q, m, G, m);
  //sw_vdSqrtI1(a * b , Q, k, G, m);
		for(long long li=0; li<a*b; li++){
			for(long long ki=0; ki<k;ki++)
			Q[li*k*m+ki*m+0] =sqrt(Q[li*k*m+ki*m+0])*sign[ki];
		}
//	printf("after sqrt,Q==%lf,G==%lf\n",Q[1],G[1]);
}
//验证串行代码正确
void kernel1_bak(double *H, double *M, double *P_CHK, double *S, double *T, double *Q_CHK,  double *G, double *sign) {
	int li,ki,di,ci,mi;
	double tmp,tmp1;
	//for (int i=0;i<a*b;i++){
  //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, d, c, 1,
  //                          &H[i*k*c], k, &M[i*c*d], d, 0, &P[i*k*d], d);
	//}
	for(li=0; li<a*b;li++){
  for(ki=0; ki<k;  ki++){
  for(di=0; di<d;  di++)
	{
		tmp=0.0;
      for(ci=0; ci<c;  ci++)
			{
        tmp += H[li*k*c+ci*k+ki] * M[li*c*d+ci*d+di];
//		if(li==0 && ki==0 && di==0 && ci<10) printf("in mas, h=%lf,m=%lf,tmp=%lf,loc_M=%lx\n",H[li*k*c+ci*k+ki],M[li*c*d+ci*d+di],tmp,M);
      }
			//P_CHK[li*k*d+ki*d+di] = alpha * tmp *alpha *tmp ;//+ dS_CHK[li*k*d+ki*d+di] * beta;
			P_CHK[li*k*d+ki*d+di] = alpha *tmp ;//+ dS_CHK[li*k*d+ki*d+di] * beta;
	}
    sign[li*k+ki] = P_CHK[li*k*d+ki*d] < 0 ? -1 : 1;//P可以变为局部变量,降维度
	}

  for(ki=0; ki<k;  ki++)
  for(di=0; di<d;  di++)
	  	P_CHK[li*k*d+ki*d+di] = P_CHK[li*k*d+ki*d+di] *	P_CHK[li*k*d+ki*d+di];

 // for (int i = 0; i < k; i++) {
 //   sign[li*k+i] = P_CHK[li*k*d+i*d] < 0 ? -1 : 1;//P可以变为局部变量,降维度
 // }

  for(ki=0; ki<k;  ki++)
  for(mi=0; mi<m;  mi++)
	{
		tmp1=0.0;
    for(di=0; di<d;  di++){
      tmp1 += P_CHK[li*k*d+ki*d+di] * T[di*m+mi];
		}
		Q_CHK[li*k*m+ki*m+mi] = alpha*tmp1 ;//+ dS_CHK[li*k*d+ki*d+di] * beta;   
	}
//sqrt
//	if(li==0) printf("111 sqrt,Q_chk==%lf\n",Q_CHK[1]);
  for(ki=0; ki<k;  ki++)
		Q_CHK[li*k*m+ki*m+0] = sqrt(Q_CHK[li*k*m+ki*m+0])*sign[ki] ;//+ dS_CHK[li*k*d+ki*d+di] * beta;   
  
  }   
//	printf("222 sqrt,Q_chk==%lf\n",Q_CHK[1]);
	
}

void sw_vdSqr(long long  size, double *P, double *S){
		for(long long li=0; li<size; li++){
			S[li] = P[li]*P[li];
		}
}
void sw_vdSqrtI(long long  size, double *Q,long long size1, double *G,long long size2){
		for(long long li=0; li<size; li++){
			G[li*m] = sqrt(Q[li*m]);
		}
}
void sw_vdSqrtI1(long long  size1, double *Q,long long size2, double *G,long long size3){
		for(long long li=0; li<size1; li++){
			for(long long ki=0; ki<size2;ki++)
			G[li*size2*size3+ki*size3+0] =sqrt(Q[li*size2*size3+ki*size3+0]);
		}
}

void RESULT_CHK(long long size, double *ORI, double *CHK) {
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
    if(fabs(CHK[i]-ORI[i])>1.0e-5)
    //if(CHK[i] != ORI[i])
    {
      printf("%ld element result is wrong!\n",i);
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
    printf("%d element result is right!\n",size);
		fprintf(fp,"RIGHT_CHK[%ld]=%20.12lf\n",size-1,CHK[size-1]);
		fprintf(fp,"RIGHT_ORI[%ld]=%20.12lf\n",size-1,ORI[size-1]);
  }
}


int main(int argc, char *argv[]) {
  int world_size, world_rank;

//  MPI_Init(&argc, &argv);
//  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

  // a = a / world_size;

  double *M;
  double *H;
  double *P;
  double *P_CHK;
  double *S;
  double *T;
  double *Q;
  double *Q_CHK;
  double *G;
  double *sign;
  double *G_SW;
  double *P_SW;

  long long M_size = a * b * c * d;
  long long H_size = a * b * c * k;
  long long P_size = a * b * k * d;
  long long S_size = a * b * k * d;
  long long T_size = d * m;
  long long Q_size = a * b * k * m;
  long long G_size = a * b * k * m;

  //分配内存空间并初始化张量

  M = (double *)malloc(M_size * sizeof(double));
  H = (double *)malloc(H_size * sizeof(double));
  P = (double *)malloc(P_size * sizeof(double));
  P_CHK = (double *)malloc(P_size * sizeof(double));
  S = (double *)malloc(S_size * sizeof(double));
  T = (double *)malloc(T_size * sizeof(double));
  Q = (double *)malloc(Q_size * sizeof(double));
  Q_CHK = (double *)malloc(Q_size * sizeof(double));
  G = (double *)malloc(G_size * sizeof(double));
  sign = (double *)malloc(a * b * k * sizeof(double));
  G_SW = (double *)malloc(G_size * sizeof(double));
  //P_SW = (double *)malloc(1 * sizeof(double));
  P_SW = (double *)malloc(P_size * sizeof(double));

  srand(time(0));
  for (long long i = 0; i < M_size; i++) {
    M[i] = 1.0*rand() / RAND_MAX;
  }
  for (long long i = 0; i < H_size; i++) {
    H[i] = 1.0*rand() / RAND_MAX;
  }
  for (long long i = 0; i < T_size; i++) {
    T[i] = 0.0;
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

  double start, end;
	unsigned long t1, t2, t3, t4,t5;
  int iter = 5;

  //MPI_Barrier(MPI_COMM_WORLD);
 // start = MPI_Wtime();
    athread_init();
    printf("****************************** k1-sw ******************************\n");
    printf("k1-sw-using a=%d, b=%d, c=%d, d=%d, k=%d, m+1=%d, x=%d.\n", a, b, c, d, k, m , x);

  //for (int i = 0; i < iter; i++) {
	  t1 = CRTS_time_cycle();
    kernel1(H, M, P, S, T, Q, G, sign);
	  t2 = CRTS_time_cycle();
  //}
    kernel1_bak(H, M, P_CHK, S, T, Q_CHK, G, sign);
	  t3 = CRTS_time_cycle();
//printf("m====%d\n",m);
    sw_dgemm_kernel1(a*b, k, d, c, m, H, M, T, P_SW, G_SW);
    //sw_dgemm_kernel1(a*b, k, d, c, m, H, M, T, G_SW);
	  t4 = CRTS_time_cycle();

		printf("============================check---P,S==========================\n");
    RESULT_CHK(P_size, S, P_CHK);
		printf("============================check---G==========================\n");
    RESULT_CHK(Q_size, Q, Q_CHK);
		printf("============================check-slave--P==========================\n");
    //RESULT_CHK(P_size, S, P_SW);
    RESULT_CHK(P_size, P, P_SW);
		printf("============================check-slave--G==========================\n");
    RESULT_CHK(Q_size, Q, G_SW);

 // MPI_Barrier(MPI_COMM_WORLD);
 // end = MPI_Wtime();

		const double cyc = 2.0;
		const double cyc_tm = 0.001 * 0.001 * 0.001;


printf("All time of kernel1, xmath=%lf,master=%lf,slave=%lf\n",(t2-t1)*cyc_tm/cyc,(t3-t2)*cyc_tm/cyc,(t4-t3)*cyc_tm/cyc);
  free(M);
  free(H);
	free(P);
  free(P_CHK);
  free(S);
  free(T);
  free(Q);
  free(Q_CHK);
  free(G);
  free(sign);
  free(G_SW);
  free(P_SW);

//  if (world_rank == 0) {
//    printf("Kernel1 with %d processes computing time %f sec, ", world_size,
//           (end - start) / iter);
//    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
//           m - 1, x);
//  }

//  MPI_Finalize();
  return 0;
}
