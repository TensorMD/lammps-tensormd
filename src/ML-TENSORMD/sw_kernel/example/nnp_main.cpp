#include "cnpy.h"
//#include "mpi.h"
#include "nnp.h"
//#include "nnp_pseudo_athread.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <crts.h>
#include "sw_nnp_main.h"

// a应为4 * 64的倍数
// 为便于编程，此处的kernel_cpe_id与ncols_on_cpe均padding为(nlayers - 1) * 16，若需修改需要同时修改main.cpp nnp_pseudo_athread.c nnp.cpp中的相关代码

static size_t a = 2560, b = 1, k = 128, m = 4;
static int block = 4;
static int kernel_ngroups[4] = {16, 4, 4, 4};
static int kernel_cpe_id[4][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19},
    {20, 21, 22, 23},
    {24, 25, 26, 27}};
static int ncols_on_cpe[4][16] = {
    {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
    {32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32}};
static int kernel_ngroups_bp[4] = {16, 4, 4, 4};
static int kernel_cpe_id_bp[4][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19},
    {20, 21, 22, 23},
    {24, 25, 26, 27}};
static int ncols_on_cpe_bp[4][16] = {
    {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32}};
static int max_col = 32;

int main(int argc, char *argv[])
{
  int world_size, world_rank;




  // 主核代码
    auto model = new NeuralNetworkPotential<double>();
    auto npz = cnpy::npz_load("Mo_fp64_large.npz");
    model->setup_global(npz);

    size_t G_size = a * b * k * m;
    double *G = new double[G_size];
    double *y = new double[a];
    double *dydx = new double[G_size];
    unsigned long t1,t2,t3,t4;
    srand(time(0));
    //srand(1);
for (size_t i = 0; i < G_size; i++) { G[i] = 1.0* rand() / RAND_MAX; }
		//	printf("before compute!\n");
			t1=CRTS_time_cycle();
    model->compute(0, G, a, y, dydx);
		 t2=CRTS_time_cycle();
		//	printf("after compute!,world_size===%d\n",world_size);

      int M = a;
      int N = b * k * m;
      int max_moment = m - 1;
      int beta;
      int nlayers;
      int max_layer;
      model->get_config(&beta, &nlayers, &max_layer);
      int *layer_sizes = new int[nlayers];
      int *input_sizes = new int[nlayers];
      model->get_layer_sizes(layer_sizes, input_sizes);
    printf("****************************** nnp-sw ******************************\n");
			 printf("nnp-sw-using, block=%d,max_col=%d,M=%d,N=%d,max_moment=%d,beta=%d,nlayers=%d,max_layer=%d\n",block,max_col,M,N,max_moment,beta,nlayers,max_layer);
//for(int i=0;i<nlayers;i++)
//	printf("in master, layer_sizes[%d]===%d, input_sizes[%d]===%d\n",i,layer_sizes[i],i,input_sizes[i]);
      double *bias = new double[nlayers * max_layer];
      double *kernel_matrix_on_this_cpe_T = new double[64*max_col * max_layer];
      double *kernel_matrix_on_this_cpe_bp = new double[64*max_col * max_layer];
      double *last_kernel = new double[max_layer];

	    CRTS_init();
      model->get_bias(bias);
      model->get_last_kernel(last_kernel);
      for (int cid = 0; cid < 64; cid++) {
        model->get_parameter(kernel_matrix_on_this_cpe_T+cid*max_col * max_layer,
                             kernel_matrix_on_this_cpe_bp+cid*max_col * max_layer, cid);
      }

      double *y_pseudo_athread = new double[a];
      double *dydx_pseudo_athread = new double[G_size];
      double *y_sw = new double[a];
      double *dydx_sw = new double[G_size];
  // 从核代码
t3=CRTS_time_cycle();
			sw_nnp_compute(block,max_col,M,N,max_moment,beta,nlayers,max_layer,kernel_ngroups,kernel_cpe_id,ncols_on_cpe,kernel_ngroups_bp,kernel_cpe_id_bp,ncols_on_cpe_bp,layer_sizes,input_sizes,bias,last_kernel,kernel_matrix_on_this_cpe_T,kernel_matrix_on_this_cpe_bp,G,y_sw,dydx_sw);
	t4=CRTS_time_cycle();	

	printf("Begin check!!,a===%d,G_size===%d\n",a,G_size);
      bool y_flag = true;
      bool dydx_flag = true;
      for (int i = 0; i < a; i++) {
				if(i<5) printf("y_sw[%d]===%f,y[%d]===%f\n",i,y_sw[i],i,y[i]);
        if (abs(y_sw[i] - y[i]) > 10e-6) {
          y_flag = false;
				printf("y-test-fail!,y_sw[%d]===%f,y[%d]===%f\n",i,y_sw[i],i,y[i]);
          break;
        }
      }
      for (int i = 0; i < G_size; i++) {
				if(i<5) printf("dydx_sw[%d]===%f,dydx[%d]===%f\n",i,dydx_sw[i],i,dydx[i]);
        if (abs(dydx_sw[i] - dydx[i]) > 10e-6) {
          dydx_flag = false;
          break;
        }
      }
      if (y_flag) {
        printf("y test passed.\n");
      } else {
        printf("y test failed.\n");
      }
      if (dydx_flag) {
        printf("dydx test passed.\n");
      } else {
        printf("dydx test failed.\n");
      }
			const double cyc=2.0;
			const double cyc_tm=0.001*0.001*0.001;
printf("Final,master_time(xmath)===%lf,slave_time===%lf\n",cyc_tm*(t2-t1)/cyc,cyc_tm*(t4-t3)/cyc);
      delete[] y_pseudo_athread;
      delete[] dydx_pseudo_athread;
      delete[] y_sw;
      delete[] dydx_sw;
      delete[] bias;
      delete[] kernel_matrix_on_this_cpe_T;
      delete[] kernel_matrix_on_this_cpe_bp;
      delete[] last_kernel;
      delete[] layer_sizes;
      delete[] input_sizes;




    delete[] G;
    delete[] y;
    delete[] dydx;
    delete model;
  return 0;
}
