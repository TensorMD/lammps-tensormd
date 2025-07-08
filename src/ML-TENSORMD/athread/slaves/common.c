#include <slave.h>
#include "common.h"

void athread_get_task_range(int global_size, int *start, int *end, int threads)
{
    int athread_rows, remainder, athread_start, athread_end;
    athread_rows = global_size / threads;
    remainder = global_size % threads;
    athread_start = athread_rows * _MYID;
    athread_end = athread_rows * (_MYID + 1);
    if (_MYID < remainder) {
      athread_start += _MYID;
      athread_end += _MYID + 1;
    } else {
      athread_start += remainder;
      athread_end += remainder;
    }
    if (_MYID < threads) {
      *start = athread_start;
      *end = athread_end;
    } else {
      *start = 0;
      *end = 0;
    }
}
