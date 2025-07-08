template <typename Scalar>
void ration_setup_GPU(Scalar *params, Scalar **params_GPU, int size, int n_out);
template <typename Scalar>
void ration_compute_GPU(Scalar *x, int n, Scalar *f, Scalar *df, Scalar *params,
                        Scalar xdx, int size, Scalar dx, int n_out,
                        bool endpoints_force_zero);
