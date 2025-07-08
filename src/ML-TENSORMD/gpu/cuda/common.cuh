#ifndef LAMMPS_TENSORMD_GPU_COMMON
#define LAMMPS_TENSORMD_GPU_COMMON

void error_handle_GPU(cudaError_t err, const char *file, int line);

#define CHECK(err) (error_handle_GPU(err, __FILE__, __LINE__))

#endif