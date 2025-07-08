#include <stdlib.h>

int64_t get_max(void) // Modified by WD in 2019-05-10
{
  int64_t tmp;
  int64_t cur = (1L << 34);
  void *ptr = NULL;
  tmp = cur;
  while (1) {
    ptr = malloc(tmp);
    if (ptr != NULL) {
      free(ptr);
    } else {
      tmp -= cur;
    }

    if (cur == 2048)
      break;

    cur >>= 1;
    tmp += cur;
  }
  return tmp;
}

int64_t tell_avail_memory() // Modified by WD in 2019-05-10
{
  return get_max();
}
