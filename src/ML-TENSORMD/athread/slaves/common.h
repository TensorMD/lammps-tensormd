union athread_args
{
    void *p;
    int i;
    double d;
};

void athread_get_task_range(int global_size, int *start, int *end, int threads);
