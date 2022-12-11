/**********************************************************************
  ML.c:

     MLDFT.c is a subroutine to perform ML prediction of atomic force

  Log of ML.c:

     10/Sep/2021  Added by Hengyu Li

     27/April/2022  ver 1.0.2 Parallel generate matrix and train

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define Pi acos(-1)