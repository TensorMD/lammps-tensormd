#include <stdio.h>

void error_handle_GPU(cudaError_t err, const char *file, int line)
{
#if defined(GPU_DEBUG)
  if (err != cudaSuccess) {
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
  }
#endif
}
