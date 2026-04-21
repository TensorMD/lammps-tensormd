#include <stdio.h>
#include "hip/hip_runtime.h"

void error_handle_GPU(hipError_t err, const char *file, int line)
{
#if defined(GPU_DEBUG)
  if (err != hipSuccess) {
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), hipGetErrorString(err), file, line);
		exit(int(err));
  }
#endif
}
