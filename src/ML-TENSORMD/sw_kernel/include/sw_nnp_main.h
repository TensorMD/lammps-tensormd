extern "C"{
void sw_nnp_compute(const int block, const int max_col, const int M, const int N, const int max_moment, const int beta, const int nlayers, const int max_layer, const void *kernel_ngroups, const void *kernel_cpe_id, const void *ncols_on_cpe, const void *kernel_ngroups_bp, const void *kernel_cpe_id_bp, const void *ncols_on_cpe_bp,  const void *layer_sizes, const void *input_sizes, const void *bias, const void *last_kernel, const void *kernel_matrix_on_this_cpe_T, const void *kernel_matrix_on_this_cpe_bp, const void *G, const void *y_sw, const void *dydx_sw);
}
