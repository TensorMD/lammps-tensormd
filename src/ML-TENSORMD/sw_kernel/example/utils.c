#include "utils.h"
#include <stdlib.h>


double randfrom(double min, double max) 
{
  double range = (max - min); 
  double div = RAND_MAX / range;
  return min + (rand() / div);
}
