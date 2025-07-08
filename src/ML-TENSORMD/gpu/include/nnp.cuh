#include <map>

template <typename Scalar>
Scalar
forward_resnet_GPU(int elti, Scalar *x_in, int n, Scalar *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, bool sum_output, Scalar *weights,
                   Scalar *biases, Scalar **dz, int nlocal_old, bool DToH);
template <typename Scalar>
void backward_resnet_GPU(int elti, Scalar *grad_in, Scalar *grad_out,
                         int max_size, int n_out, int M, int nlayers,
                         std::map<int, int> &full_sizes, bool use_resnet_dt,
                         Scalar *weights, Scalar *dz);
template <typename Scalar>
void setup_nnp_GPU(Scalar **weights_GPU, Scalar **biases_GPU, Scalar ***weights,
                   Scalar ***biases, int nlayers, int num_in,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   int nelt, int max_size);
template <typename Scalar>
void free_nnp_GPU(Scalar *weights_GPU, Scalar *biases_GPU, Scalar **dz);
template <typename Scalar>
void forward_fnn_GPU(Scalar *x_in, int n, Scalar *y, int max_size,
                   int n_out, int nlayers, int actfn,
                   std::map<int, int> &full_sizes, bool apply_output_bias,
                   bool use_resnet_dt, Scalar *weights, Scalar *biases);
template <typename Scalar>
void forward_setup_tdnp_GPU(int n, Scalar **y, Scalar **dU, Scalar **dS,
                            int ydim);
template <typename Scalar>
void forward_tdnp_GPU(int nlocal, int ydim, Scalar *y, Scalar etemp);
template <typename Scalar>
void backward_tdnp_GPU(int nlocal, int ydim, double scale, Scalar *dS, Scalar *dU);