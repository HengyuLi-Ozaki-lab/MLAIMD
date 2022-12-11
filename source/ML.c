/**********************************************************************
  ML.c:

     MLDFT.c is a subroutine to perform ML prediction of atomic force

  Log of ML.c:

     21/Aug/2022  Added by Hengyu Li

     ver 1.0.9 Radial part is replaced by (x-1)^n serise polynomial

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define Pi acos(-1.0)

/*******************************************************
            Arry shared among functions
*******************************************************/

/* Atom information */

static int ***nei_info;
static int ***nei_info_cache;

static int ***nei_info3;
static int ***nei_info3_cache;

static int *ang_num;
static int *ang_num_cache;
static double **dis_nei;
static double **dis_nei3;
static double **ang_nei;

/* Parameter matrix */

static double ***matrix_a;
static double ***matrix_b;
static double ***matrix_a_prime;
static double ***matrix_b_prime;
static double **matrix_c;
static double **matrix_c_prime;

/* Parameter matrix cache */

static double ***matrix_a_cache;
static double ***matrix_b_cache;
static double ***matrix_a_prime_cache;
static double ***matrix_b_prime_cache;
static double **matrix_c_cache;
static double **matrix_c_prime_cache;

/* matrix for solver */

static double **constant_matrix;
static double **parameter_matrix;
static double **parameter_matrix_fake;

/* Model energy & force*/

static double *fitted_energy;
static double **current_model;

/* Computing time analyze */

double matrix_time;
double solver_time;
double energy_time;
double force_time;

/*******************************************************
                Subfunction of ML
*******************************************************/

/* Allocate the atoms in unit cell to processors and fixed along MD*/

void ML_allocate_atom2cpu()
{

  int i;

  for (i=1;i<=Matomnum;i++){

    ML_M2G[i] = M2G[i]; // Assume the gnum is fixed along MD

  }

  ML_Matomnum = Matomnum;

  new_train = 0; /*Initial the new train*/

}

/* Compute three-body combination number */

int factorial(int m, int n)
{

  int i,j;
	int ans = 1;
  
	if(m < n-m) m = n-m;
	for(i = m+1; i <= n; i++) ans *= i;
	for(j = 1; j <= n - m; j++) ans /= j;

	return ans;

}

/* Weight for L2 regularization term to ensure farer atom has smaller parameter and contribution */

double ML_L2_weight(double distance, double r_cut)
{

  double weight;

  //weight = 1 + 2*distance/r_cut + pow(2*distance/r_cut,3) + pow(2*distance/r_cut,5) + pow(2*distance/r_cut,7); // too large weight will cause overflow

  weight = 1000000/(1+exp(-15*(distance-0.9*r_cut))) + 1;

  return weight;
  
}

double ML_orth_poly(double distance, double r_cut, int order)
{

  double x;
  
  x = distance/r_cut;

  if (order==0){

    return x;

  }

  else if (order==1){

    return sqrt(7)*pow(x-1,3);

  }

  else if (order==2){

    return 3*pow(x-1,3)*(8*x-1);

  }

  else if (order==3){

    return sqrt(11)*pow(x-1,3)*(45*x*x-18*x+1);

  }

  else if (order==4){

    return sqrt(13)*pow(x-1,3)*(220*pow(x,3)-165*x*x+30*x-1);

  }

  else if (order==5){

    return sqrt(15)*pow(x-1,3)*(1001*pow(x,4)-1144*pow(x,3)+396*x*x-44*x+1);

  }

  else if (order==6){

    return sqrt(17)*pow(x-1,3)*(4368*pow(x,5)-6825*pow(x,4)+3640*pow(x,3)-780*x*x+60*x-1);

  }

  else if (order==7){

    return sqrt(19)*pow(x-1,3)*(1-78*x+1365*x*x-9100*pow(x,3)+27300*pow(x,4)-37128*pow(x,5)+18564*pow(x,6));

  }

  else{

    printf("Order of polynomial exceeds the maximum\n");

  }

}

double ML_orth_poly_deriv(double distance, double r_cut, int order)
{

  double x;
  
  x = distance/r_cut;

  if (order==0){

    return 1/r_cut;

  }

  else if (order==1){

    return 3*sqrt(7)*pow(x-1,2)/r_cut;

  }

  else if (order==2){

    return 3*pow(x-1,2)*(32*x-11)/r_cut;

  }

  else if (order==3){

    return 3*sqrt(11)*pow(x-1,2)*(75*x*x-54*x+7)/r_cut;

  }

  else if (order==4){

    return 3*sqrt(13)*pow(x-1,2)*(440*pow(x,3)-495*x*x+150*x-11)/r_cut;

  }

  else if (order==5){

    return sqrt(15)*pow(x-1,2)*(7007*pow(x,4)-10868*pow(x,3)+5412*x*x-968*x+47)/r_cut;

  }

  else if (order==6){

    return 3*sqrt(17)*pow(x-1,2)*(11648*pow(x,5)-23205*pow(x,4)+16380*pow(x,3)-4940*x*x+600*x-21)/r_cut;

  }

  else if (order==7){

    return 3*sqrt(19)*pow(x-1,2)*(55692*pow(x,6)-136136*pow(x,5)+125580*pow(x,4)-54600*pow(x,3)+11375*x*x-1014*x+27)/r_cut;

  }

  else{

    printf("Order of polynomial exceeds the maximum (Derivative)\n");

  }

}

double ML_orth_ang(double distance, double r_cut, int order)
{

  double x;
  
  x = distance/r_cut;

  if (order==0){

    return x;

  }

  else if (order==1){

    return sqrt(7)*pow(x-1,3);

  }

  else if (order==2){

    return 3*pow(x-1,3)*(8*x-1);

  }

  else if (order==3){

    return sqrt(11)*pow(x-1,3)*(45*x*x-18*x+1);

  }

  else if (order==4){

    return sqrt(13)*pow(x-1,3)*(220*pow(x,3)-165*x*x+30*x-1);

  }

  else if (order==5){

    return sqrt(15)*pow(x-1,3)*(1001*pow(x,4)-1144*pow(x,3)+396*x*x-44*x+1);

  }

  else if (order==6){

    return sqrt(17)*pow(x-1,3)*(4368*pow(x,5)-6825*pow(x,4)+3640*pow(x,3)-780*x*x+60*x-1);

  }

  else if (order==7){

    return sqrt(19)*pow(x-1,3)*(1-78*x+1365*x*x-9100*pow(x,3)+27300*pow(x,4)-37128*pow(x,5)+18564*pow(x,6));

  }

  else{

    printf("Order of polynomial exceeds the maximum\n");

  }

}

double ML_orth_ang_deriv(double distance, double r_cut, int order)
{

  double x;
  
  x = distance/r_cut;

  if (order==0){

    return 1/r_cut;

  }

  else if (order==1){

    return 3*sqrt(7)*pow(x-1,2)/r_cut;

  }

  else if (order==2){

    return 3*pow(x-1,2)*(32*x-11)/r_cut;

  }

  else if (order==3){

    return 3*sqrt(11)*pow(x-1,2)*(75*x*x-54*x+7)/r_cut;

  }

  else if (order==4){

    return 3*sqrt(13)*pow(x-1,2)*(440*pow(x,3)-495*x*x+150*x-11)/r_cut;

  }

  else if (order==5){

    return sqrt(15)*pow(x-1,2)*(7007*pow(x,4)-10868*pow(x,3)+5412*x*x-968*x+47)/r_cut;

  }

  else if (order==6){

    return 3*sqrt(17)*pow(x-1,2)*(11648*pow(x,5)-23205*pow(x,4)+16380*pow(x,3)-4940*x*x+600*x-21)/r_cut;

  }

  else if (order==7){

    return 3*sqrt(19)*pow(x-1,2)*(55692*pow(x,6)-136136*pow(x,5)+125580*pow(x,4)-54600*pow(x,3)+11375*x*x-1014*x+27)/r_cut;

  }

  else{

    printf("Order of polynomial exceeds the maximum (Derivative)\n");

  }

}

/* Chebyshev function for angular part */

double ML_Chebyshev(double angular, double order)
{

  if (order==0){

    return 1;

  }

  else if (order==1){

    return angular;

  }

  else{

    return 2*angular*ML_Chebyshev(angular,order-1)-ML_Chebyshev(angular,order-2);

  }

}

double ML_Chebyshev_deriv(double angular, double order)
{

  if (order==0){

    return 0;

  }

  else if (order==1){

    return 1;

  }

  else{

    return 2*ML_Chebyshev(angular,order-1)+2*angular*ML_Chebyshev_deriv(angular,order-1)-ML_Chebyshev_deriv(angular,order-2);

  }

}

/* Compute distance for each two-body combination */

void cal_dis(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  
  int i,j,k,nei_num,nei_gnum,nei_cnum,dis_count2,dis_count3,species,centra_gnum,myid,ID,global_num;
  double dx,dy,dz,distance;

  /* MPI */

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Allocate neighbor array respect to processor*/

  dis_nei = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));
  dis_nei3 = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum =ML_M2G[i];
    nei_num = FNAN[centra_gnum];

    dis_nei[i] = (double*)malloc(sizeof(double)*(nei_num+1));
    memset(dis_nei[i],0,(nei_num+1)*sizeof(double));

    dis_nei3[i] = (double*)malloc(sizeof(double)*(nei_num+1));
    memset(dis_nei3[i],0,(nei_num+1)*sizeof(double));     
  
  }

  nei_info = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));
  nei_info3 = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num = FNAN[centra_gnum];

    nei_info[i] = (int**)malloc(sizeof(int*)*(nei_num+1));
    nei_info3[i] = (int**)malloc(sizeof(int*)*(nei_num+1));

    for (j=0;j<=nei_num;j++){

      nei_info[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info[i][j],0,3*sizeof(int));

      nei_info3[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info3[i][j],0,3*sizeof(int));

    }

  }

  /* Allocate global neighbor array */

  nei_info_global = (int***)malloc(sizeof(int**)*(atomnum+1));
  nei_info3_global = (int***)malloc(sizeof(int**)*(atomnum+1));
  dis_nei_global = (double**)malloc(sizeof(double*)*(atomnum+1));
  dis_nei3_global = (double**)malloc(sizeof(double*)*(atomnum+1));
  
  for (i=1;i<=atomnum;i++){

    nei_num = FNAN[i];

    dis_nei_global[i] = (double*)malloc(sizeof(double)*(nei_num+1));
    memset(dis_nei_global[i],0,(nei_num+1)*sizeof(double));

    dis_nei3_global[i] = (double*)malloc(sizeof(double)*(nei_num+1));
    memset(dis_nei3_global[i],0,(nei_num+1)*sizeof(double));

    nei_info_global[i] = (int**)malloc(sizeof(int*)*(nei_num+1));
    nei_info3_global[i] = (int**)malloc(sizeof(int*)*(nei_num+1));

    for (j=0;j<=nei_num;j++){

      nei_info_global[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info_global[i][j],0,3*sizeof(int));

      nei_info3_global[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info3_global[i][j],0,3*sizeof(int));

    }

  }
 
  /* Select neighbor list need to be modified */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];

    nei_num = FNAN[centra_gnum];
    dis_count2 = 1;
    dis_count3 = 1;

    for (j=1;j<=nei_num;j++){

      nei_gnum = natn[centra_gnum][j];
      nei_cnum = ncn[centra_gnum][j];

      dx = fabs(Gxyz[nei_gnum][1]-Gxyz[centra_gnum][1]+atv[nei_cnum][1]);
      dy = fabs(Gxyz[nei_gnum][2]-Gxyz[centra_gnum][2]+atv[nei_cnum][2]);
      dz = fabs(Gxyz[nei_gnum][3]-Gxyz[centra_gnum][3]+atv[nei_cnum][3]);

      distance = sqrt(dx*dx+dy*dy+dz*dz);
      
      if (distance<=r_cut2){

        nei_info[i][0][0] += 1;
        nei_info_global[centra_gnum][0][0] += 1;

        nei_info[i][dis_count2][1] = nei_gnum;  // record global number
        nei_info[i][dis_count2][2] = nei_cnum;  // record cell number

        nei_info_global[centra_gnum][dis_count2][1] = nei_gnum;
        nei_info_global[centra_gnum][dis_count2][2] = nei_cnum;

        dis_nei[i][dis_count2] = distance;

        dis_nei_global[centra_gnum][dis_count2] = distance;

        dis_count2 += 1;

      }

      if (distance<=r_cut3){

        nei_info3[i][0][0] += 1;
        nei_info3_global[centra_gnum][0][0] += 1;

        nei_info3[i][dis_count3][1] = nei_gnum;  // record global number
        nei_info3[i][dis_count3][2] = nei_cnum;  // record cell number

        nei_info3_global[centra_gnum][dis_count3][1] = nei_gnum;
        nei_info3_global[centra_gnum][dis_count3][2] = nei_cnum;

        dis_nei3[i][dis_count3] = distance;

        dis_nei3_global[centra_gnum][dis_count3] = distance;

        dis_count3 += 1;

      }

    }
  
  }

  for (i=1;i<=atomnum;i++){

    nei_num = FNAN[i];

    MPI_Allreduce(MPI_IN_PLACE, &dis_nei_global[i][0], nei_num+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    MPI_Allreduce(MPI_IN_PLACE, &dis_nei3_global[i][0], nei_num+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  }

  for (i=1;i<=atomnum;i++){

    nei_num = FNAN[i];

    for (j=0;j<=nei_num;j++){

      MPI_Allreduce(MPI_IN_PLACE, &nei_info_global[i][j][0], 3, MPI_INT, MPI_SUM, mpi_comm_level1);

      MPI_Allreduce(MPI_IN_PLACE, &nei_info3_global[i][j][0], 3, MPI_INT, MPI_SUM, mpi_comm_level1);

    }

  }

}

/* Compute angular for each three-body combination */

void cal_ang(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".cal_ang";
  int i,j,k,nei_gnum1,nei_gnum2,nei_cnum1,nei_cnum2,nei_num,ang_count,centra_gnum,myid;

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Allocate angular number & angular array respect to processor */

  ang_nei = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){

    ang_num[i] = factorial(2,nei_info3[i][0][0]);

    ang_nei[i] = (double*)malloc(sizeof(double)*(ang_num[i]+1));
    memset(ang_nei[i],0,(ang_num[i]+1)*sizeof(double));

  }

  /* Allocate angular number & angular array global */

  ang_nei_global = (double**)malloc(sizeof(double*)*(atomnum+1));
  ang_num_global = (int*)malloc(sizeof(int)*(atomnum+1));

  for (i=1;i<=atomnum;i++){

    ang_num_global[i] = factorial(2,nei_info3_global[i][0][0]);

    ang_nei_global[i] = (double*)malloc(sizeof(double)*(ang_num_global[i]+1));
    memset(ang_nei_global[i],0,(ang_num_global[i]+1)*sizeof(double));

  }

  /* Calculate angle of all three-body combination */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num = nei_info3[i][0][0];
    ang_count = 1;

    if (nei_num>1){
      for (j=1;j<=nei_num-1;j++){

        nei_gnum1 = nei_info3[i][j][1];
        nei_cnum1 = nei_info3[i][j][2];

        for(k=j+1;k<=nei_num;k++){

          nei_gnum2 = nei_info3[i][k][1];
          nei_cnum2 = nei_info3[i][k][2];

          ang_nei[i][ang_count] = ((atv[nei_cnum1][1]+Gxyz[nei_gnum1][1]-Gxyz[centra_gnum][1])*(atv[nei_cnum2][1]+Gxyz[nei_gnum2][1]-Gxyz[centra_gnum][1])+\
                                  (atv[nei_cnum1][2]+Gxyz[nei_gnum1][2]-Gxyz[centra_gnum][2])*(atv[nei_cnum2][2]+Gxyz[nei_gnum2][2]-Gxyz[centra_gnum][2])+\
                                  (atv[nei_cnum1][3]+Gxyz[nei_gnum1][3]-Gxyz[centra_gnum][3])*(atv[nei_cnum2][3]+Gxyz[nei_gnum2][3]-Gxyz[centra_gnum][3]))/(dis_nei3[i][j]*dis_nei3[i][k]);

          ang_nei_global[centra_gnum][ang_count] = ang_nei[i][ang_count];

          ang_count ++;

        }
      }

    }

    else{

      continue;

    }

  }

  for (i=1;i<=atomnum;i++){

    MPI_Allreduce(MPI_IN_PLACE, &ang_nei_global[i][0], ang_num_global[i]+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  }
 
}

/* Compute cutoff coefficient */

double cut_off(double distance,double r_cut,int grad)
{
  double coefficient;

  /* Output normal cutoff function */

  if (grad == 0){
    if (distance <= r_cut){

      coefficient = 0.5*(cos(Pi*distance/r_cut)+1);

      return coefficient;

    }

    else{

      return 0; // 0

    }

  }

  /* Output cutoff function derivative */
  
  else if (grad == 1){
    if (distance <= r_cut){

      coefficient = 0.5*Pi*sin(Pi*distance/r_cut)/r_cut; // (rjx-rix)/r_ij is not included

      return coefficient;

    }

    else{

      return 0;

    }

  }

  /* Input error */

  else{

    printf("Check grad control in cut_off function\n");
    return 0;

  }
}

/* Get decomposed energy repect to atom on each processor */

void Get_decomposed_ene(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int i, j, spin, local_num, global_num, ID, species, myid;
  double energy;

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (i=1;i<=atomnum;i++){

    Dec_tot_global[i] = 0;

  }

  for (i=1;i<=Matomnum;i++){

    global_num = M2G[i];

    species = WhatSpecies[global_num];
    energy = 0;

    if (SpinP_switch==0){

      for (j=0;j<Spe_Total_CNO[species];j++){

        energy += 2*DecEkin[0][i][j]; 
        energy += 2*DecEv[0][i][j];
        energy += 2*DecEcon[0][i][j];
        energy += 2*DecEscc[0][i][j];
        energy += 2*DecEvdw[0][i][j];

      }

    }

    if (SpinP_switch==1 || SpinP_switch==3){
      
      for (j=0;j<Spe_Total_CNO[species];j++){
        for (spin=0;spin<=1;spin++){

          energy += DecEkin[spin][i][j];
          energy += DecEv[spin][i][j];
          energy += DecEcon[spin][i][j];
          energy += DecEscc[spin][i][j];
          energy += DecEvdw[spin][i][j];

        }

      }

    }

    Dec_tot_global[global_num] = energy;

  }

  MPI_Allreduce(MPI_IN_PLACE, &Dec_tot_global[0], atomnum+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

}

/* Transform the atom information of last iter to cache */

void ML_Tran_cache(int iter)
{

  int i,j,k,nei_num;
  int nei_num_current,nei_num_last;

  /* Allocate nei_info_cache */

  nei_info_cache = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    ang_num_cache[i] = ang_num[i]; // transform angular number to cache
    nei_info_cache[i] = (int**)malloc(sizeof(int*)*(nei_num+1));

    for (j=0;j<=nei_num;j++){

      nei_info_cache[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info_cache[i][j],0,3*sizeof(int));

    }

  }

  /* Allocate nei_info3_cache */

  nei_info3_cache = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info3[i][0][0];

    nei_info3_cache[i] = (int**)malloc(sizeof(int*)*(nei_num+1));

    for (j=0;j<=nei_num;j++){

      nei_info3_cache[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info3_cache[i][j],0,3*sizeof(int));

    }

  }

  /* Transform neighbor information to cache */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    nei_info_cache[i][0][0] = nei_num;

    for (j=1;j<=nei_num;j++){

      nei_info_cache[i][j][1] = nei_info[i][j][1];
      nei_info_cache[i][j][2] = nei_info[i][j][2];

    }

  }

  /* Transform 3neighbor information to cache */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info3[i][0][0];
    nei_info3_cache[i][0][0] = nei_num;

    for (j=1;j<=nei_num;j++){

      nei_info3_cache[i][j][1] = nei_info3[i][j][1];
      nei_info3_cache[i][j][2] = nei_info3[i][j][2];

    }

  }

  /* Free nei_list_ana to store next iter information */

  if (iter>1){

    for (i=1;i<=ML_Matomnum;i++){

      nei_num_current = nei_info[i][0][0];

      for (j=1;j<=nei_num_current;j++){

        free(nei_list_ana[i][j]);
        
      }

    }

    free(nei_list_ana);
    nei_list_ana = NULL;

    for (i=1;i<=ML_Matomnum;i++){

      nei_num_current = nei_info3[i][0][0];

      for (j=1;j<=nei_num_current;j++){

        free(nei_list3_ana[i][j]);
        
      }

    }

    free(nei_list3_ana);
    nei_list3_ana = NULL;

  }

  /* Free nei_info to store next iter information */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    for (j=0;j<=nei_num;j++){

      free(nei_info[i][j]);

    }

    free(nei_info[i]);

  }

  free(nei_info);
  nei_info = NULL;

  /* Free nei_info3 to store next iter information */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info3[i][0][0];

    for (j=0;j<=nei_num;j++){

      free(nei_info3[i][j]);

    }

    free(nei_info3[i]);

  }

  free(nei_info3);
  nei_info3 = NULL;


  /* Free nei_info_global to store next iter information */

  for (i=1;i<=atomnum;i++){

    nei_num = nei_info_global[i][0][0];

    for (j=0;j<=nei_num;j++){

      free(nei_info_global[i][j]);

    }

    free(nei_info_global[i]);

  }

  free(nei_info_global);
  nei_info_global = NULL;

  /* Free nei_info_global to store next iter information */

  for (i=1;i<=atomnum;i++){

    nei_num = nei_info3_global[i][0][0];

    for (j=0;j<=nei_num;j++){

      free(nei_info3_global[i][j]);

    }

    free(nei_info3_global[i]);

  }

  free(nei_info3_global);
  nei_info3_global = NULL;

  /* nei_dis */

  for (i=1;i<=ML_Matomnum;i++){

    free(dis_nei[i]);

  }

  free(dis_nei);
  dis_nei = NULL;

  /* nei_dis3 */

  for (i=1;i<=ML_Matomnum;i++){

    free(dis_nei3[i]);

  }

  free(dis_nei3);
  dis_nei3 = NULL;

  /* nei_dis global */

  for (i=1;i<=atomnum;i++){

    free(dis_nei_global[i]);

  }

  free(dis_nei_global);
  dis_nei_global = NULL;

  /* nei_dis3 global */

  for (i=1;i<=atomnum;i++){

    free(dis_nei3_global[i]);

  }

  free(dis_nei3_global);
  dis_nei3_global = NULL;  

  /* nei_ang */

  for (i=1;i<=ML_Matomnum;i++){

    free(ang_nei[i]);

  }

  free(ang_nei);
  ang_nei = NULL;  

  /* nei_ang global*/

  for (i=1;i<=ML_Matomnum;i++){

    free(ang_nei_global[i]);

  }

  free(ang_nei_global);
  ang_nei_global = NULL; 

  if (iter==MD_IterNumber){

    /* Free ML_M2G */

    free(ML_M2G);

    ML_M2G = NULL;
    
  }

}

/* Search the original position in nei_info */

int ML_search_nei(int centra_gnum, int tar_pos)
{

  int i,left1,right1,left2,right2,mid1,mid2,result,nei_num,target1,target2;
  int floor, ceil;

  nei_num = nei_info_cache[centra_gnum][0][0];

  left1 = 1;
  left2 = 1;
  right1 = nei_num;
  right2 = nei_num;
  target1 = nei_info[centra_gnum][tar_pos][1];
  target2 = nei_info[centra_gnum][tar_pos][2];

  result = -1;

  while (left1<=right1){

    mid1 = left1 + (right1-left1)/2;

    if (nei_info_cache[centra_gnum][mid1][1]>=target1){

      right1 = mid1 - 1;

    }

    else{

      left1 = mid1+1;

    }

  }

  if (left1 <= nei_num && target1==nei_info_cache[centra_gnum][left1][1]){
    
    floor = left1;
    result = 0;

  }

  while (left2<=right2){

    mid2 = left2 + (right2-left2)/2;

    if (nei_info_cache[centra_gnum][mid2][1]<=target1){

      left2 = mid2+1;

    }

    else{

      right2 = mid2-1;

    }

  }

  if (left2 >= 1 && target1==nei_info_cache[centra_gnum][right2][1]){

    ceil = right2;
    result = 0;

  }

  if (result!=-1){
    for (i=floor;i<=ceil;i++){
      if (target2==nei_info_cache[centra_gnum][i][2]){

        result = i;
        break;

      }
      
      result = -1;

    }
  }

  return result;

}

int ML_search_nei3(int centra_gnum, int tar_pos)
{

  int i,left1,right1,left2,right2,mid1,mid2,result,nei_num,target1,target2;
  int floor, ceil;

  nei_num = nei_info3_cache[centra_gnum][0][0];

  left1 = 1;
  left2 = 1;
  right1 = nei_num;
  right2 = nei_num;
  target1 = nei_info3[centra_gnum][tar_pos][1];
  target2 = nei_info3[centra_gnum][tar_pos][2];

  result = -1;

  while (left1<=right1){

    mid1 = left1 + (right1-left1)/2;

    if (nei_info3_cache[centra_gnum][mid1][1]>=target1){

      right1 = mid1 - 1;

    }

    else{

      left1 = mid1+1;

    }

  }

  if (left1 <= nei_num && target1==nei_info3_cache[centra_gnum][left1][1]){
    
    floor = left1;
    result = 0;

  }

  while (left2<=right2){

    mid2 = left2 + (right2-left2)/2;

    if (nei_info3_cache[centra_gnum][mid2][1]<=target1){

      left2 = mid2+1;

    }

    else{

      right2 = mid2-1;

    }

  }

  if (left2 >= 1 && target1==nei_info3_cache[centra_gnum][right2][1]){

    ceil = right2;
    result = 0;

  }

  if (result!=-1){
    for (i=floor;i<=ceil;i++){
      if (target2==nei_info3_cache[centra_gnum][i][2]){

        result = i;
        break;

      }
      
      result = -1;

    }
  }

  return result;

}


/* Check the neighbor information and record in nei_list_ana */

void ML_check_nei()
{

  int i,j,k;
  int nei_gnum_current,nei_gnum_last,nei_cnum_current,nei_cnum_last,nei_num_current,nei_num_last,myid,centra_gnum;

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (i=1;i<=atomnum;i++){

    signal[i] = 0;

  }

  /* Allocate nei_list_ana */

  nei_list_ana = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){
    
    nei_num_current = nei_info[i][0][0];

    nei_list_ana[i] = (int**)malloc(sizeof(int*)*(nei_num_current+1));
    
    for (j=1;j<=nei_num_current;j++){

      nei_list_ana[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_list_ana[i][j],0,3*sizeof(int));
      
    }

  }

  /* Allocate nei_list_ana */

  nei_list3_ana = (int***)malloc(sizeof(int**)*(ML_Matomnum+1));

  for (i=1;i<=ML_Matomnum;i++){
    
    nei_num_current = nei_info3[i][0][0];

    nei_list3_ana[i] = (int**)malloc(sizeof(int*)*(nei_num_current+1));
    
    for (j=1;j<=nei_num_current;j++){

      nei_list3_ana[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_list3_ana[i][j],0,3*sizeof(int));
      
    }

  }

  /* Analyse the diff of neighbor information bewteem current iter and last iter */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num_current = nei_info[i][0][0];
    nei_num_last = nei_info_cache[i][0][0];

    if (nei_num_current>nei_num_last){

      for (j=1;j<=nei_num_last;j++){

        nei_gnum_current = nei_info[i][j][1];
        nei_cnum_current = nei_info[i][j][2];
        nei_gnum_last = nei_info_cache[i][j][1];
        nei_cnum_last = nei_info_cache[i][j][2];

        if (nei_gnum_current!=nei_gnum_last || nei_cnum_current!=nei_cnum_last){

          nei_list_ana[i][j][1] = 0;

          nei_list_ana[i][j][2] = ML_search_nei(i,j);

          if (nei_list_ana[i][j][2]==-1){

            signal[centra_gnum] += 1;

          }

        }

        else{

          nei_list_ana[i][j][1] = 1;

          nei_list_ana[i][j][2] = j;

        }

      }

      for (j=nei_num_last+1;j<=nei_num_current;j++){

        nei_gnum_current = nei_info[i][j][1];
        nei_cnum_current = nei_info[i][j][2];

        signal[centra_gnum] += 1;

        nei_list_ana[i][j][1] = 0;

        nei_list_ana[i][j][2] = ML_search_nei(i,j);

      }

    }

    if (nei_num_current<=nei_num_last){

      for (j=1;j<=nei_num_current;j++){

        nei_gnum_current = nei_info[i][j][1];
        nei_cnum_current = nei_info[i][j][2];
        nei_gnum_last = nei_info_cache[i][j][1];
        nei_cnum_last = nei_info_cache[i][j][2];

        if (nei_gnum_current!=nei_gnum_last || nei_cnum_current!=nei_cnum_last){

          nei_list_ana[i][j][1] = 0;

          nei_list_ana[i][j][2] = ML_search_nei(i,j);

          if (nei_list_ana[i][j][2]==-1){

            signal[centra_gnum] += 1;

          }

        }

        else{

          nei_list_ana[i][j][1] = 1;

          nei_list_ana[i][j][2] = j;

        }

      }

    }
    
  }

  /* Analyse 3body list diff */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num_current = nei_info3[i][0][0];
    nei_num_last = nei_info3_cache[i][0][0];

    if (nei_num_current>nei_num_last){

      for (j=1;j<=nei_num_last;j++){

        nei_gnum_current = nei_info3[i][j][1];
        nei_cnum_current = nei_info3[i][j][2];
        nei_gnum_last = nei_info3_cache[i][j][1];
        nei_cnum_last = nei_info3_cache[i][j][2];

        if (nei_gnum_current!=nei_gnum_last || nei_cnum_current!=nei_cnum_last){

          nei_list3_ana[i][j][1] = 0;

          nei_list3_ana[i][j][2] = ML_search_nei3(i,j);

          if (nei_list3_ana[i][j][2]==-1){

            signal[centra_gnum] += 1;

          }

        }

        else{

          nei_list3_ana[i][j][1] = 1;

          nei_list3_ana[i][j][2] = j;

        }

      }

      for (j=nei_num_last+1;j<=nei_num_current;j++){

        nei_gnum_current = nei_info3[i][j][1];
        nei_cnum_current = nei_info3[i][j][2];

        signal[centra_gnum] += 1;

        nei_list3_ana[i][j][1] = 0;

        nei_list3_ana[i][j][2] = ML_search_nei3(i,j);

      }

    }

    if (nei_num_current<=nei_num_last){

      for (j=1;j<=nei_num_current;j++){

        nei_gnum_current = nei_info3[i][j][1];
        nei_cnum_current = nei_info3[i][j][2];
        nei_gnum_last = nei_info3_cache[i][j][1];
        nei_cnum_last = nei_info3_cache[i][j][2];

        if (nei_gnum_current!=nei_gnum_last || nei_cnum_current!=nei_cnum_last){

          nei_list3_ana[i][j][1] = 0;

          nei_list3_ana[i][j][2] = ML_search_nei3(i,j);

          if (nei_list3_ana[i][j][2]==-1){

            signal[centra_gnum] += 1;

          }

        }

        else{

          nei_list3_ana[i][j][1] = 1;

          nei_list3_ana[i][j][2] = j;

        }

      }

    }
    
  }
 
  MPI_Allreduce(MPI_IN_PLACE, &signal[0], atomnum+1, MPI_INT, MPI_SUM, mpi_comm_level1);
  
}

/* Allocate working array which will not change with MD */

void ML_allocate_static()
{

  int i,j,k,nei_num;

  ang_num = (int*)malloc(sizeof(int)*(Matomnum+1));
  memset(ang_num,0,(Matomnum+1)*sizeof(int));

  ang_num_cache = (int*)malloc(sizeof(int)*(Matomnum+1));
  memset(ang_num_cache,0,(Matomnum+1)*sizeof(int));

  signal = (int*)malloc(sizeof(int)*(atomnum+1));
  memset(signal,0,(atomnum+1)*sizeof(int));

  ML_M2G = (int*)malloc(sizeof(int)*(Matomnum+1));
  memset(ML_M2G,0,(Matomnum+1)*sizeof(int));

  //printf("Angular number array allocate Pass\n");

  Dec_tot_global = (double*)malloc(sizeof(double)*(atomnum+1));
  memset(Dec_tot_global,0,(atomnum+1)*sizeof(double));

  //printf("Decomposed energy array allocate Pass\n");

  fitted_energy = (double*)malloc(sizeof(double)*(Matomnum+1));
  memset(fitted_energy,0,(Matomnum+1)*sizeof(double));

  fitted_energy_global = (double*)malloc(sizeof(double)*(atomnum+1));
  memset(fitted_energy_global,0,(atomnum+1)*sizeof(double));

  //printf("Model fitting energy array allocate Pass\n");

  energy_error = (double*)malloc(sizeof(double)*(Matomnum+1));
  memset(energy_error,0,(Matomnum+1)*sizeof(double));

  //printf("Fitting energy error array allocate Pass\n");

  model_force = (double**)malloc(sizeof(double*)*(atomnum+1));

  for (i=1;i<=atomnum;i++){

    model_force[i] = (double*)malloc(sizeof(double)*4);
    memset(model_force[i],0,4*sizeof(double));

  }

  //printf("Model force array allocate Pass\n");

  numerical_force = (double**)malloc(sizeof(double*)*(atomnum+1));

  for (i=1;i<=atomnum;i++){

    numerical_force[i] = (double*)malloc(sizeof(double)*4);
    memset(numerical_force[i],0,4*sizeof(double));

  }

  //printf("Numerical force array allocate Pass\n");

  total_force = (double*)malloc(sizeof(double)*4);
  memset(total_force,0,4*sizeof(double));

  //printf("Total force array allocate Pass\n");

}

/* Allocate working array and matrice for ML */

void ML_allocate_matrix()
{
  int i,j,k,nei_num;

  //printf("Angular calculation Pass\n");

  matrix_a = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_a[i] = (double**)malloc(sizeof(double*)*(nei_num*(Max_order-Min_order+1)+1));

    for (j=1; j<=(nei_num*(Max_order-Min_order+1)); j++){
      
      matrix_a[i][j] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1)); 
      memset(matrix_a[i][j],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

    }
  }

  //printf("Matrix A allocate Pass\n");

  matrix_b = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_b[i] = (double**)malloc(sizeof(double*)*(nei_num*(Max_order-Min_order+1)+1)); 
    
    for (j=1; j<=(nei_num*(Max_order-Min_order+1)); j++){

      matrix_b[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
      memset(matrix_b[i][j],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double));

    }
  }

  //printf("Matrix B allocate Pass\n");

  matrix_c = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    matrix_c[i] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1));
    memset(matrix_c[i],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

  }

  //printf("Matrix C allocate Pass\n");

  matrix_a_prime = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_a_prime[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*ang_num[i]+1));

    for (j=1; j<=((Max_order-Min_order+1)*ang_num[i]); j++){

      matrix_a_prime[i][j] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1));
      memset(matrix_a_prime[i][j],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

    }
  }

  //printf("Matrix A' allocate Pass\n");

  matrix_b_prime = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    matrix_b_prime[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*ang_num[i]+1)); 
    
    for (j=1; j<=((Max_order-Min_order+1)*ang_num[i]); j++){

      matrix_b_prime[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
      memset(matrix_b_prime[i][j],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double)); 

    }
  }

  //printf("Matrix B' allocate Pass\n");

  matrix_c_prime = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    matrix_c_prime[i] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
    memset(matrix_c_prime[i],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double));

  }

  //printf("Matrix C' allocate Pass\n");

  //printf("All array allocate Pass\n");

}

void ML_allocate_matrix_cache(){

  int i,j,k,nei_num;

  //printf("Angular calculation Pass\n");

  matrix_a_cache = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_a_cache[i] = (double**)malloc(sizeof(double*)*(nei_num*(Max_order-Min_order+1)+1));

    for (j=1; j<=(nei_num*(Max_order-Min_order+1)); j++){
      
      matrix_a_cache[i][j] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1)); 
      memset(matrix_a_cache[i][j],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

    }
  }

  //printf("Matrix A allocate Pass\n");

  matrix_b_cache = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_b_cache[i] = (double**)malloc(sizeof(double*)*(nei_num*(Max_order-Min_order+1)+1)); 
    
    for (j=1; j<=(nei_num*(Max_order-Min_order+1)); j++){

      matrix_b_cache[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
      memset(matrix_b_cache[i][j],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double));

    }
  }

  //printf("Matrix B allocate Pass\n");

  matrix_c_cache = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    matrix_c_cache[i] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1));
    memset(matrix_c_cache[i],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

  }

  //printf("Matrix C allocate Pass\n");

  matrix_a_prime_cache = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    nei_num = nei_info[i][0][0];

    matrix_a_prime_cache[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*ang_num[i]+1));

    for (j=1; j<=((Max_order-Min_order+1)*ang_num[i]); j++){

      matrix_a_prime_cache[i][j] = (double*)malloc(sizeof(double)*(nei_num*(Max_order-Min_order+1)+1));
      memset(matrix_a_prime_cache[i][j],0,(nei_num*(Max_order-Min_order+1)+1)*sizeof(double));

    }
  }

  //printf("Matrix A' allocate Pass\n");

  matrix_b_prime_cache = (double***)malloc(sizeof(double**)*(ML_Matomnum+1));

  for (i=1; i<=ML_Matomnum; i++){

    matrix_b_prime_cache[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*ang_num[i]+1)); 
    
    for (j=1; j<=((Max_order-Min_order+1)*ang_num[i]); j++){

      matrix_b_prime_cache[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
      memset(matrix_b_prime_cache[i][j],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double)); 

    }
  }

  //printf("Matrix B' allocate Pass\n");

  matrix_c_prime_cache = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    matrix_c_prime_cache[i] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*ang_num[i]+1));
    memset(matrix_c_prime_cache[i],0,((Max_order-Min_order+1)*ang_num[i]+1)*sizeof(double));

  }

}

void ML_allocate_dyna(){

  int i,j,k,nei_num;

  parameter_matrix = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    parameter_matrix[i] = (double*)malloc(sizeof(double)*(pow((nei_num+ang_num[i])*(Max_order-Min_order+1),2)+1));
    memset(parameter_matrix[i],0,(pow((nei_num+ang_num[i])*(Max_order-Min_order+1),2)+1)*sizeof(double));

  }

  parameter_matrix_fake = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    parameter_matrix_fake[i] = (double*)malloc(sizeof(double)*(pow((nei_num+ang_num[i])*(Max_order-Min_order+1),2)+1));
    memset(parameter_matrix_fake[i],0,(pow((nei_num+ang_num[i])*(Max_order-Min_order+1),2)+1)*sizeof(double));

  }

  //printf("Parameter array allocate Pass\n");

  constant_matrix = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    constant_matrix[i] = (double*)malloc(sizeof(double)*((nei_num+ang_num[i])*(Max_order-Min_order+1)+1));
    memset(constant_matrix[i],0,((nei_num+ang_num[i])*(Max_order-Min_order+1)+1)*sizeof(double));

  }

  //printf("Constant array allocate Pass\n");

  //printf("All array allocate Pass\n");

}

void ML_allocate_model()
{

  int i,nei_num;

  current_model = (double**)malloc(sizeof(double*)*(ML_Matomnum+1));

  for(i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    current_model[i] = (double*)malloc(sizeof(double)*((nei_num+ang_num[i])*(Max_order-Min_order+1)+1));
    memset(current_model[i],0,((nei_num+ang_num[i])*(Max_order-Min_order+1)+1)*sizeof(double));

  }

  current_model_global = (double**)malloc(sizeof(double*)*(atomnum+1));

  for(i=1;i<=atomnum;i++){

    nei_num = nei_info_global[i][0][0];
    current_model_global[i] = (double*)malloc(sizeof(double)*((nei_num+ang_num_global[i])*(Max_order-Min_order+1)+1));
    memset(current_model_global[i],0,((nei_num+ang_num_global[i])*(Max_order-Min_order+1)+1)*sizeof(double));

  }

}

/* Free working array and matrice */

void ML_free_static(){

  int i,j;

  /* Free ang_num */

  free(ang_num);

  ang_num = NULL;

  //printf("ang_num free Pass\n");

  /* Free ang_num_cache */

  free(ang_num_cache);

  ang_num_cache = NULL;

  //printf("ang_num_cache free Pass\n");

  /* Free signal */

  free(signal);

  signal = NULL;

  //printf("signal free Pass\n");

  /* Free Dec_tot global */

  free(Dec_tot_global);

  Dec_tot_global = NULL;

  //printf("Dec_tot_global free Pass\n");

  //printf("Free dect pass\n");

  /* Free fitted_energy */

  free(fitted_energy);

  fitted_energy = NULL;

  //printf("fitted_energy free Pass\n");

  /* Free fitted_energy global */

  free(fitted_energy_global);

  fitted_energy_global = NULL;

  //printf("fitted_energy_global free Pass\n");

  //printf("Free fitted ene pass\n");

  /* Free model_force */

  for (i=1;i<=atomnum;i++){

    free(model_force[i]);

  }

  free(model_force);

  model_force = NULL;

  //printf("Free model force pass\n");

  /* Free energy_error */

  free(energy_error);

  energy_error = NULL;

  //printf("Free ene error pass\n");

  /* Free numerical_force */

  for (i=1;i<=atomnum;i++){

    free(numerical_force[i]);

  }

  free(numerical_force);

  numerical_force = NULL;

  //printf("Free num force pass\n");

  /* Free total_force */

  free(total_force);

  total_force = NULL;

  //printf("Free tot force pass\n");

}

void ML_free_matrix()
{
  int i,j,nei_num;

  /* Free Matrix A */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){
      
      free(matrix_a[i][j]);

    }

    free(matrix_a[i]);

  }

  free(matrix_a);
  matrix_a = NULL;

  //printf("Free a pass\n");

  /* Free Matrix B */
  
  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){

      free(matrix_b[i][j]);

    }

    free(matrix_b[i]);

  }

  free(matrix_b);
  matrix_b = NULL;

  //printf("Free b pass\n");

  /* Free Matrix C */
  
  for (i=1;i<=ML_Matomnum;i++){

    free(matrix_c[i]);

  }

  free(matrix_c);
  matrix_c = NULL;

  //printf("Free c pass\n");

  /* Free Matrix A' */
  
  for (i=1;i<=ML_Matomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

      free(matrix_a_prime[i][j]);

    }

    free(matrix_a_prime[i]);

  }

  free(matrix_a_prime);
  matrix_a_prime = NULL;

  //printf("Free a' pass\n");

  /* Free Matrix B' */
  
  for (i=1;i<=ML_Matomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

      free(matrix_b_prime[i][j]);

    }

    free(matrix_b_prime[i]);

  }

  free(matrix_b_prime);
  matrix_b_prime = NULL;

  //printf("Free b' pass\n");

  /* Free Matrix C' */
  
  for (i=1;i<=ML_Matomnum;i++){

    free(matrix_c_prime[i]);
    
  }

  free(matrix_c_prime);
  matrix_c_prime = NULL;

  //printf("Free c' pass\n");

}

void ML_free_matrix_cache()
{
  int i,j,nei_num,nei_num_last;

  /* Free Matrix A */
  
  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info_cache[i][0][0];

    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){
      
      free(matrix_a_cache[i][j]);

    }

    free(matrix_a_cache[i]);

  }

  free(matrix_a_cache);
  matrix_a_cache = NULL;

  //printf("free a pass\n");
  
  /* Free Matrix B */
  
  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info_cache[i][0][0];

    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){

      free(matrix_b_cache[i][j]);

    }

    free(matrix_b_cache[i]);

  }

  free(matrix_b_cache);
  matrix_b_cache = NULL;

  //printf("free b pass\n");
  
  /* Free Matrix C */
  
  for (i=1;i<=ML_Matomnum;i++){

    free(matrix_c_cache[i]);

  }

  free(matrix_c_cache);
  matrix_c_cache = NULL;
  
  /* Free Matrix A' */
  
  for (i=1;i<=ML_Matomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*ang_num_cache[i];j++){

      free(matrix_a_prime_cache[i][j]);

    }

    free(matrix_a_prime_cache[i]);

  }

  free(matrix_a_prime_cache);
  matrix_a_prime_cache = NULL;
  
  /* Free Matrix B' */

  for (i=1;i<=ML_Matomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*ang_num_cache[i];j++){

      free(matrix_b_prime_cache[i][j]);

    }

    free(matrix_b_prime_cache[i]);

  }

  free(matrix_b_prime_cache);
  matrix_b_prime_cache = NULL;

  /* Free Matrix C' */
  
  for (i=1;i<=ML_Matomnum;i++){

    free(matrix_c_prime_cache[i]);
    
  }

  free(matrix_c_prime_cache);
  matrix_c_prime_cache = NULL;
  
}

void ML_free_dyna(){

  int i,j,k,nei_num;

  /* Free Para matrix */

  for (i=1;i<=ML_Matomnum;i++){

    free(parameter_matrix[i]);

  }
  
  free(parameter_matrix);
  parameter_matrix = NULL;

  //printf("Free para pass\n");

  /* Free fake Para matrix */

  for (i=1;i<=ML_Matomnum;i++){

    free(parameter_matrix_fake[i]);

  }
  
  free(parameter_matrix_fake);
  parameter_matrix_fake = NULL;

  //printf("Free para fake pass\n");

  /* Free Const matrix */

  for (i=1;i<=ML_Matomnum;i++){

    free(constant_matrix[i]);

  }

  free(constant_matrix);
  constant_matrix = NULL;

  //printf("Free const pass\n");

  /* Free Decomposed energy,fitted energy,energy error,current model array */

  //printf("Free array pass\n");

}

void ML_free_model()
{

  int i;

  for (i=1;i<=ML_Matomnum;i++){

    free(current_model[i]);

  }

  free(current_model);
  current_model = NULL;
  
  for (i=1;i<=atomnum;i++){

    free(current_model_global[i]);

  }

  free(current_model_global);
  current_model_global = NULL;

}

void ML_free_nei_cache()
{

  int i,j;

  for (i=1;i<=ML_Matomnum;i++){

    for (j=0;j<=nei_info_cache[i][0][0];j++){

      free(nei_info_cache[i][j]);

    }

    free(nei_info_cache[i]);

  }

  free(nei_info_cache);
  nei_info_cache = NULL;

  for (i=1;i<=ML_Matomnum;i++){

    for (j=0;j<=nei_info3_cache[i][0][0];j++){

      free(nei_info3_cache[i][j]);

    }

    free(nei_info3_cache[i]);

  }

  free(nei_info3_cache);
  nei_info3_cache = NULL;

}

int ML_trace_ang(int i, int j, int nei_num){

  int first,second; // first is the 1st 

  if (i>j){

    first = j;
    second = i;

  }

  else{

    first = i;
    second = j;

  }

  return (-first*first+2*nei_num*first+first-2*nei_num)/2 + second-first; // Trace the combination num by 1st & 2nd in nei_list

}

void ML_matrix_pre(int iter, char filepath[YOUSO10], char filename[YOUSO10]){

  int i,j,k,p,j_1,j_2,k_1,p_1;
  int row_current,row_last,column_current,column_last;
  int trace1,trace2,trace3,trace4;
  int centra_num,nei_num_current,nei_num_last,nei_num3_current,nei_num3_last,myid,centra_gnum;
  char filelast[YOUSO10] = ".matrix_pre";
  FILE *fp;

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  ML_allocate_matrix();

  //printf("ML_allocate_matrix pass\n");

  /* Preprocessing of Matrix_A */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num_current = nei_info[i][0][0];

    /* Preprocessing of Matrix_A */

    for (j=1;j<=nei_num_current;j++){   // row loop for current iter

      trace1 = nei_list_ana[i][j][2];

      if (trace1!=-1){

        for (j_1=1;j_1<=nei_num_current;j_1++){   // column loop for current iter

          trace2 = nei_list_ana[i][j_1][2];

          if (trace2!=-1){

            for (p=Min_order;p<=Max_order;p++){  // row loop for subblock by order

              for (p_1=Min_order;p_1<=Max_order;p_1++){   // column loop for subblock by order 

                row_current = (Max_order-Min_order+1)*(j-1)+p-Min_order+1;
                column_current = (Max_order-Min_order+1)*(j_1-1)+p_1-Min_order+1;

                row_last = (Max_order-Min_order+1)*(trace1-1)+p-Min_order+1;
                column_last = (Max_order-Min_order+1)*(trace2-1)+p_1-Min_order+1;

                matrix_a[i][row_current][column_current] = matrix_a_cache[i][row_last][column_last];

              }
            }

          }

          else{

            continue;

          }

        }

      }

      else{

        continue;

      }

    }

  }

  /* Preprocessing of Matrix_B */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num_current = nei_info[i][0][0];

    nei_num3_current = nei_info3[i][0][0];
    nei_num3_last = nei_info3_cache[i][0][0];

    for (j=1;j<=nei_num_current;j++){

      trace1 = nei_list_ana[i][j][2];

      if (trace1!=-1){

        for (j_1=1;j_1<=nei_num3_current-1;j_1++){

          trace2 = nei_list3_ana[i][j_1][2];

          if (trace2!=-1){

            for (k=j_1+1;k<=nei_num3_current;k++){

              trace3 = nei_list3_ana[i][k][2];

              if (trace3!=-1){

                for (p=Min_order;p<=Max_order;p++){

                  for (p_1=Min_order;p_1<=Max_order;p_1++){

                    row_current = (Max_order-Min_order+1)*(j-1)+p-Min_order+1;
                    column_current = (Max_order-Min_order+1)*(ML_trace_ang(j_1,k,nei_num3_current)-1)+p_1-Min_order+1;
                    
                    row_last = (Max_order-Min_order+1)*(trace1-1)+p-Min_order+1;
                    column_last = (Max_order-Min_order+1)*(ML_trace_ang(trace2,trace3,nei_num3_last)-1)+p_1-Min_order+1;

                    matrix_b[i][row_current][column_current] = matrix_b_cache[i][row_last][column_last];

                  }

                }

              }

              else{

                continue;

              }

            }

          }

          else{

            continue;

          }

        }

      }

      else{

        continue;

      }

    }

    //printf("Matrix B pass on %d\n",ML_M2G[i]);

  }

  /* Preprocessing of Matrix_A' */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num_current = nei_info[i][0][0];

    nei_num3_current = nei_info3[i][0][0];
    nei_num3_last = nei_info3_cache[i][0][0];

    for (j=1;j<=nei_num3_current-1;j++){

      trace1 = nei_list3_ana[i][j][2];

      if (trace1!=-1){

        for (k=j+1;k<=nei_num3_current;k++){

          trace2 = nei_list3_ana[i][k][2];

          if (trace2!=-1){

            for (j_1=1;j_1<=nei_num_current;j_1++){

              trace3 = nei_list_ana[i][j_1][2];

              if (trace3!=-1){

                for (p=Min_order;p<=Max_order;p++){

                  for (p_1=Min_order;p_1<=Max_order;p_1++){

                    row_current = (Max_order-Min_order+1)*(ML_trace_ang(j,k,nei_num3_current)-1)+p-Min_order+1;
                    column_current = (Max_order-Min_order+1)*(j_1-1)+p_1-Min_order+1;

                    row_last = (Max_order-Min_order+1)*(ML_trace_ang(trace1,trace2,nei_num3_last)-1)+p-Min_order+1;
                    column_last = (Max_order-Min_order+1)*(trace3-1)+p_1-Min_order+1;

                    matrix_a_prime[i][row_current][column_current] = matrix_a_prime_cache[i][row_last][column_last];

                  }

                }

              }

              else{

                continue;

              }

            }

          }

          else{

            continue;

          }

        }

      }

      else{

        continue;

      }

    }

    //printf("Matrix A' pass on %d\n",ML_M2G[i]);

  }

  /* Preprocessing of Matrix_B' */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num3_current = nei_info3[i][0][0];
    nei_num3_last = nei_info3_cache[i][0][0];

    for (j=1;j<=nei_num3_current-1;j++){

      trace1 = nei_list3_ana[i][j][2];

      if (trace1!=-1){

        for (k=j+1;k<=nei_num3_current;k++){

          trace2 = nei_list3_ana[i][k][2];

          if (trace2!=-1){

            for (j_1=1;j_1<=nei_num3_current-1;j_1++){

              trace3 = nei_list3_ana[i][j_1][2];

              if (trace3!=-1){

                for (k_1=j_1+1;k_1<=nei_num3_current;k_1++){

                  trace4 = nei_list3_ana[i][k_1][2];

                  if (trace4!=-1){

                    for (p=Min_order;p<=Max_order;p++){

                      for (p_1=Min_order;p_1<=Max_order;p_1++){

                        row_current = (Max_order-Min_order+1)*(ML_trace_ang(j,k,nei_num3_current)-1)+p-Min_order+1;
                        column_current = (Max_order-Min_order+1)*(ML_trace_ang(j_1,k_1,nei_num3_current)-1)+p_1-Min_order+1;

                        row_last = (Max_order-Min_order+1)*(ML_trace_ang(trace1,trace2,nei_num3_last)-1)+p-Min_order+1;
                        column_last = (Max_order-Min_order+1)*(ML_trace_ang(trace3,trace4,nei_num3_last)-1)+p_1-Min_order+1;

                        matrix_b_prime[i][row_current][column_current] = matrix_b_prime_cache[i][row_last][column_last];

                      }
                    }

                  }

                  else{

                    continue;

                  }

                }

              }

              else{

                continue;

              }

            }

          }

          else{

            continue;

          }

        }

      }

      else{

        continue;

      }

    }

    //printf("Matrix B' pass on %d\n",ML_M2G[i]);

  }

  /* Preprocessing of Matrix_C */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num_current = nei_info[i][0][0];

    for (j=1;j<=nei_num_current;j++){

      trace1 = nei_list_ana[i][j][2];

      if (trace1!=-1){

        for (p=Min_order;p<=Max_order;p++){

          row_current = (Max_order-Min_order+1)*(j-1)+p-Min_order+1;

          row_last = (Max_order-Min_order+1)*(trace1-1)+p-Min_order+1;

          matrix_c[i][row_current] = matrix_c_cache[i][row_last];

        }

      }

      else{

        continue;

      }

    }

  }

  /* Preprocessing of Matrix_C' */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num3_current = nei_info3[i][0][0];
    nei_num3_last = nei_info3_cache[i][0][0];

    for (j=1;j<=nei_num3_current-1;j++){

      trace1 = nei_list3_ana[i][j][2];

      if (trace1!=-1){

        for (k=j+1;k<=nei_num3_current;k++){

          trace2 = nei_list3_ana[i][k][2];

          if (trace2!=-1){

            for (p=Min_order;p<=Max_order;p++){

              row_current = (Max_order-Min_order+1)*(ML_trace_ang(j,k,nei_num3_current)-1)+p-Min_order+1;

              row_last = (Max_order-Min_order+1)*(ML_trace_ang(trace1,trace2,nei_num3_last)-1)+p-Min_order+1;

              matrix_c_prime[i][row_current] = matrix_c_prime_cache[i][row_last];

            }

          }

          else{
  
            continue;
  
          }

        }

      }

      else{

        continue;

      }

    }

    //printf("Matrix C' pass on %d\n",ML_M2G[i]);   

  }

  if (myid==2){

    fprintf(fp,"Matrix at iter = %d\n",iter);

    for (i=1;i<=ML_Matomnum;i++){

      fprintf(fp,"Matrix C \n");

      for (j=1;j<=(Max_order-Min_order+1)*nei_info[i][0][0];j++){

        fprintf(fp,"%16.14f ",matrix_c_cache[i][j]);

      }

      fprintf(fp,"\n");

      fprintf(fp,"Matrix C' \n");

      for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

        fprintf(fp,"%16.14f ",matrix_c_prime_cache[i][j]);

      }

      fprintf(fp,"\n");
      
    }

    fclose(fp);

  }

  ML_free_matrix_cache();

  //printf("ML_free_matrix_cache pass\n");

  ML_free_nei_cache();

}

/* Generate the matrice for linear solver */

void ML_matrix_gen(int iter, char filepath[YOUSO10], char filename[YOUSO10])
{
  int i,j,k,p,j_1,j_2,k_1,p_1,p_2;
  int count_ang,count_ang1,count_ang2,row,column1,column2,parameter_count,centra_gnum;
  int species,nei_num,nei_num3,myid;
  int n,lda,info,lwork;
  int *ipiv,*iwork;
  double *work;
  double start_time,end_time,norm,condition_num;
  char filelast[YOUSO10] = ".matrix";
  char filelast1[YOUSO10] = ".matrix2";
  char filelast2[YOUSO10] = ".matrix3";
  FILE *fp;
  FILE *fp1;
  FILE *fp2;

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  fnjoint(filepath,filename,filelast1);
  fp1 = fopen(filelast1,"a");

  fnjoint(filepath,filename,filelast2);
  fp2 = fopen(filelast2,"a");

  dtime(&start_time);

  if (myid==2){

    fprintf(fp2,"Matrix at iter = %d\n",iter);

    for (i=1;i<=ML_Matomnum;i++){

      fprintf(fp2,"Matrix C \n");

      for (j=1;j<=(Max_order-Min_order+1)*nei_info[i][0][0];j++){

        fprintf(fp2,"%16.14f ",matrix_c[i][j]);

      }

      fprintf(fp2,"\n");

      fprintf(fp2,"Matrix C' \n");

      for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

        fprintf(fp2,"%16.14f ",matrix_c_prime[i][j]);

      }

      fprintf(fp2,"\n");
      
    }

    fclose(fp2);

  }

  for (i=1;i<=ML_Matomnum;i++){

    row = 1;
    centra_gnum = ML_M2G[i];
    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];

    for (j=1;j<=nei_num;j++){
      for (p=Min_order;p<=Max_order;p++){

        column1 = 1;

        /* Generation of A */

        for (j_1=1;j_1<=nei_num;j_1++){
          for (p_1=Min_order;p_1<=Max_order;p_1++){

            matrix_a[i][row][column1] += 2*lammda1*ML_orth_poly(dis_nei[i][j],r_cut2,p)*ML_orth_poly(dis_nei[i][j_1],r_cut2,p_1)*\
                                          cut_off(dis_nei[i][j],r_cut2,0)*cut_off(dis_nei[i][j_1],r_cut2,0);

            if (iter==Train_iter || iter==new_train){

              if (j==j_1 && p==p_1){

                matrix_a[i][row][column1] += 2*lammda2*ML_L2_weight(dis_nei[i][j],r_cut2);

              }

            }

            column1 += 1;

          }
        }

        /* Generation of B */

        if (nei_num3>1){

          column2 = 1;
          count_ang = 1;

          for (j_2=1;j_2<=nei_num3-1;j_2++){
            for (k=j_2+1;k<=nei_num3;k++){
              for (p_2=Min_order;p_2<=Max_order;p_2++){

                matrix_b[i][row][column2] += 2*lammda1*pow(ang_nei[i][count_ang],p_2)*ML_orth_poly(dis_nei[i][j],r_cut2,p)*\
                                             cut_off(dis_nei3[i][j_2],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)*cut_off(dis_nei[i][j],r_cut2,0);
                                            
                column2 ++;

              }

              count_ang ++;

            }
          }

        }

        /* Generation of C */

        matrix_c[i][row] += 2*lammda1*ML_orth_poly(dis_nei[i][j],r_cut2,p)*cut_off(dis_nei[i][j],r_cut2,0)*Dec_tot_global[centra_gnum];    
        
        row += 1;

      }
    }

  }

  //printf("Generate A B C Pass\n");

  /* Generation of matrix A' B' C' */

  for (i=1;i<=ML_Matomnum;i++){
    
    centra_gnum = ML_M2G[i];
    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];
    row = 1;
    count_ang1 = 1;

    if (nei_num3>1){

      /* Generation of A' */

      for (j=1;j<=nei_num3-1;j++){
        for (k=j+1;k<=nei_num3;k++){
          for (p=Min_order;p<=Max_order;p++){

            column1 = 1;

            for (j_1=1;j_1<=nei_num;j_1++){
              for (p_1=Min_order;p_1<=Max_order;p_1++){

                matrix_a_prime[i][row][column1] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*ML_orth_poly(dis_nei[i][j_1],r_cut2,p_1)*\
                                                   cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)*cut_off(dis_nei[i][j_1],r_cut2,0);
                
                column1 += 1;

              }

            }

            /* Generation of B' */

            column2 = 1;
            count_ang2 = 1;

            for (j_2=1;j_2<=nei_num3-1;j_2++){
              for (k_1=j_2+1;k_1<=nei_num3;k_1++){
                for (p_2=Min_order;p_2<=Max_order;p_2++){
                  
                  matrix_b_prime[i][row][column2] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*pow(ang_nei[i][count_ang2],p_2)*\
                                                     cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)*cut_off(dis_nei3[i][j_2],r_cut3,0)*cut_off(dis_nei3[i][k_1],r_cut3,0);
                  
                  if (iter==Train_iter || iter==new_train){

                    if (j==j_2 && k==k_1 && p==p_2){

                      matrix_b_prime[i][row][column2] += 2*lammda2;//*(ML_L2_weight(dis_nei3[i][j],r_cut3)+ML_L2_weight(dis_nei3[i][k],r_cut3));

                    }     

                  }

                  column2 ++;

                }

                count_ang2 ++;

              }
            }            

            /* Generation of C' */ 

            matrix_c_prime[i][row] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)*Dec_tot_global[centra_gnum];

            row ++;

          }

          count_ang1 ++;

        }
      }

    }

  }

  /* Transform matrice to array for solver */

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];
    parameter_count = 1;

    for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){
      for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){

        parameter_matrix[i][parameter_count] = matrix_a[i][j][k];
        parameter_matrix_fake[i][parameter_count] = matrix_a[i][j][k];
        parameter_count ++;

      }

      if (nei_num3>1){
        for (p=1;p<=(Max_order-Min_order+1)*ang_num[i];p++){

          parameter_matrix[i][parameter_count] = matrix_b[i][j][p];
          parameter_matrix_fake[i][parameter_count] = matrix_b[i][j][p];
          parameter_count ++;

        }

      }

    }

    if (nei_num3>1){
      for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){
        for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){

          parameter_matrix[i][parameter_count] = matrix_a_prime[i][j][k];
          parameter_matrix_fake[i][parameter_count] = matrix_a_prime[i][j][k];
          parameter_count ++;

        }

        for (p=1;p<=(Max_order-Min_order+1)*ang_num[i];p++){

          parameter_matrix[i][parameter_count] = matrix_b_prime[i][j][p];
          parameter_matrix_fake[i][parameter_count] = matrix_b_prime[i][j][p];
          parameter_count ++;

        }

      }

    }

  }

  //printf("Transform para array Pass\n");

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];
    row = 1;

    for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){

      constant_matrix[i][row] = matrix_c[i][j];
      row ++;

    }

    if (nei_num3>1){
      for (k=1;k<=(Max_order-Min_order+1)*ang_num[i];k++){

        constant_matrix[i][row] = matrix_c_prime[i][k];
        row ++;

      }

    }

  }
  
  /* Calculate the condition number of parameter matrix */

  for (i=1;i<=Matomnum;i++){

    nei_num = nei_info[i][0][0];

    n = (nei_num+ang_num[i])*(Max_order-Min_order+1);
    lwork = 64*n; // check
    lda = (nei_num+ang_num[i])*(Max_order-Min_order+1);

    ipiv = (int*)malloc(sizeof(int)*n);
    memset(ipiv,0,n*sizeof(int));

    iwork = (int*)malloc(sizeof(int)*n);
    memset(iwork,0,n*sizeof(int));

    work = (double*)malloc(sizeof(double)*lwork);
    memset(work,0,lwork*sizeof(double));

    norm = F77_NAME(dlansy,DLANSY)("1","L",&n,&parameter_matrix_fake[i][1],&lda,work);

    F77_NAME(dsytrf,DSYTRF)("L",&n,&parameter_matrix_fake[i][1],&lda,ipiv,work,&lwork,&info);

    F77_NAME(dsycon,DSYCON)("L",&n,&parameter_matrix_fake[i][1],&lda,ipiv,&norm,&condition_num,work,iwork,&info);

    free(ipiv);
    ipiv = NULL;

    free(iwork);
    iwork = NULL;

    free(work);
    work = NULL;

  }

  dtime(&end_time);
  matrix_time += end_time-start_time;

  if (myid==2){

    fprintf(fp,"Matrix at iter = %d\n",iter);

    for (i=1;i<=ML_Matomnum;i++){

      fprintf(fp,"Matrix C \n");

      for (j=1;j<=(Max_order-Min_order+1)*nei_info[i][0][0];j++){

        fprintf(fp,"%16.14f ",matrix_c[i][j]);

      }

      fprintf(fp,"\n");

      fprintf(fp,"Matrix C' \n");

      for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

        fprintf(fp,"%16.14f ",matrix_c_prime[i][j]);

      }

      fprintf(fp,"\n");
      
    }

    fclose(fp);

  }

  if (myid==1){

    fprintf(fp1,"Matrix at iter = %d\n",iter);

    for (i=1;i<=ML_Matomnum;i++){

      fprintf(fp1,"Matrix C \n");

      for (j=1;j<=(Max_order-Min_order+1)*nei_info[i][0][0];j++){

        fprintf(fp1,"%16.14f ",matrix_c[i][j]);

      }

      fprintf(fp1,"\n");

      fprintf(fp1,"Matrix C' \n");

      for (j=1;j<=(Max_order-Min_order+1)*ang_num[i];j++){

        fprintf(fp1,"%16.14f ",matrix_c_prime[i][j]);

      }

      fprintf(fp1,"\n");
      
    }

    fclose(fp1);

  }

}

void ML_tran_matrix(){

  int i,j,k,k_1;
  int nei_num,count_ang;

  ML_allocate_matrix_cache();

  /* Transform matrix to cache */
  
  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];
    count_ang = ang_num[i];

    for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){

      for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){

        matrix_a_cache[i][j][k] = matrix_a[i][j][k];

      }

      for (k_1=1;k_1<=count_ang*(Max_order-Min_order+1);k_1++){

        matrix_b_cache[i][j][k_1] = matrix_b[i][j][k_1];

      }

      matrix_c_cache[i][j] = matrix_c[i][j];

    }

    for (j=1;j<=count_ang*(Max_order-Min_order+1);j++){

      for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){

        matrix_a_prime_cache[i][j][k] = matrix_a_prime[i][j][k];

      }

      for (k_1=1;k_1<=count_ang*(Max_order-Min_order+1);k_1++){

        matrix_b_prime_cache[i][j][k_1] = matrix_b_prime[i][j][k_1];

      }

      matrix_c_prime_cache[i][j] = matrix_c_prime[i][j];

    }

  }

  ML_free_matrix(); // Free matrix for next iteration

}

/* Linear solver for fitting */

void ML_DSYSV_solver(int iter, char filepath[YOUSO10], char filename[YOUSO10])
{
  int i,j,nei_num,nei_num3,centra_gnum;
  int n,nrhs,lda,ldb,info,lwork;
  double *work,start_time,end_time;
  int *ipiv;
  char filelast[YOUSO10] = ".solver_info";
  FILE *fp;

  fnjoint(filepath,filename,filelast);

  fp = fopen(filelast,"a");

  /* DSYSV Solver */

  //printf("Start Solver\n");

  dtime(&start_time);

  for (i=1;i<=ML_Matomnum;i++){

    nei_num = nei_info[i][0][0];

    nei_num3 = nei_info3[i][0][0];

    if (nei_num3<=1){

      n = nei_num*(Max_order-Min_order+1);
      nrhs = 1;
      lda = nei_num*(Max_order-Min_order+1);
      ldb = nei_num*(Max_order-Min_order+1);
      lwork = 64*nei_num*(Max_order-Min_order+1);

    }

    else{

      n = (nei_num+ang_num[i])*(Max_order-Min_order+1);
      nrhs = 1;
      lda = (nei_num+ang_num[i])*(Max_order-Min_order+1); 
      ldb = (nei_num+ang_num[i])*(Max_order-Min_order+1);
      lwork = (nei_num+ang_num[i])*(Max_order-Min_order+1);

    }

    /* Allocate the work array */

    work= (double*)malloc(sizeof(double)*lwork);
    memset(work,0,lwork*sizeof(double));

    ipiv = (int*)malloc(sizeof(int)*n);
    memset(ipiv,0,n*sizeof(int));

    /* Call LAPACK solver DSYCV */

    F77_NAME(dsysv,DSYSV)("L", &n, &nrhs, &parameter_matrix[i][1], &lda, ipiv, &constant_matrix[i][1], &ldb, work, &lwork, &info);
 
    /* Free work array */

    free(ipiv);
    ipiv = NULL;

    free(work);
    work = NULL;
    
    
    if (info!=0) {

      fprintf(fp,"info=%d for atom %d at MD iter %d\n",info,ML_M2G[i],iter);

    }

  }

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];

    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];

    if (nei_num3<=1){

      for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){

        current_model[i][j] = constant_matrix[i][j];

        current_model_global[centra_gnum][j] = constant_matrix[i][j];

      }

    }

    else{

      for (j=1;j<=(nei_num+ang_num[i])*(Max_order-Min_order+1);j++){

        current_model[i][j] = constant_matrix[i][j];

        current_model_global[centra_gnum][j] = constant_matrix[i][j];

      }

    }

  }

  for (i=1;i<=atomnum;i++){

    nei_num = nei_info_global[i][0][0];

    MPI_Allreduce(MPI_IN_PLACE, &current_model_global[i][0], (nei_num+ang_num_global[i])*(Max_order-Min_order+1)+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  }

  dtime(&end_time);
  solver_time += end_time-start_time;

}

/* Calculate the model energy and energy error respect to current model */

double ML_model_energy(int iter)
{
  int i,j,j_1,k,p,p_1,count_para,count_ang,nei_num,nei_num3,species,centra_gnum;
  double test,pre_ene,start_time,end_time;

  dtime(&start_time);

  for (i=1;i<=atomnum;i++){

    fitted_energy_global[i] = 0;

  }

  for (i=1;i<=ML_Matomnum;i++){

    fitted_energy[i] = 0;

  }

  /* Rebuild model energy */ 
  
  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];

    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];
    count_para = 1;

    for (j=1;j<=nei_num;j++){
      for (p=Min_order;p<=Max_order;p++){
        
        pre_ene = current_model[i][count_para]*cut_off(dis_nei[i][j],r_cut2,0)*ML_orth_poly(dis_nei[i][j],r_cut2,p);
        fitted_energy[i] += pre_ene;
        fitted_energy_global[centra_gnum] += pre_ene;

        count_para ++;
        
      }

    }
    
    if (nei_num3>1){

      count_ang = 1;

      for (j_1=1;j_1<=nei_num3-1;j_1++){
        for (k=j_1+1;k<=nei_num3;k++){
          for (p_1=Min_order;p_1<=Max_order;p_1++){

            pre_ene = current_model[i][count_para]*cut_off(dis_nei3[i][j_1],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)*pow(ang_nei[i][count_ang],p_1);
            fitted_energy[i] += pre_ene;
            fitted_energy_global[centra_gnum] += pre_ene;

            count_para ++;

          }

          count_ang ++;

        }

      }   

    }

  }

  /* Compute error */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    energy_error[i] = fitted_energy[i]-Dec_tot_global[centra_gnum];

  }

  MPI_Allreduce(MPI_IN_PLACE, &fitted_energy_global[0], atomnum+1, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

  dtime(&end_time);
  energy_time += end_time-start_time;

}

/* Calculate force */

void ML_force(int iter, int new_train, char filepath[YOUSO10], char filename[YOUSO10])
{

  int i,i_,j,j_,k,k_,p,axis,species,myid;
  int nei_num,nei_num3,nei_num1,nei_num2,nei_gnum1,nei_gnum2,count_ang,nei_num_,centra_gnum;
  int nei_cnum1,nei_cnum2,count_para,count_para1,count_para2;
  double force_pre,force_pre1,ref_energy,energy_plus,energy_minus,numerical_precision,dis_shift1,dis_shift2,ang_shift,start_time,end_time,d1,d2,d3,d4,d5,d6;

  numerical_precision = 0.0000001;

  /* MPI */

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  dtime(&start_time);

  /* Initial force array */

  for (i=1;i<=atomnum;i++){

    total_force[i] = 0;

    for (j=1;j<=3;j++){

      model_force[i][j] = 0;

    }

  }

  /* Force */
  
  /* Self gradient */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num = nei_info[i][0][0];
    nei_num3 = nei_info3[i][0][0];
    count_para = 1;

    /* Two-body contribution */

    for (j=1;j<=nei_num;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (p=Min_order;p<=Max_order;p++){
        for (axis=1;axis<=3;axis++){

          d1 = (Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-atv[nei_cnum1][axis])/dis_nei[i][j];
          d2 = (-Gxyz[centra_gnum][axis]+Gxyz[nei_gnum1][axis]+atv[nei_cnum1][axis])/dis_nei[i][j];
          
          model_force[centra_gnum][axis] += current_model[i][count_para]*d1*ML_orth_poly_deriv(dis_nei[i][j],r_cut2,p)*cut_off(dis_nei[i][j],r_cut2,0)\
                                           +current_model[i][count_para]*d2*ML_orth_poly(dis_nei[i][j],r_cut2,p)*cut_off(dis_nei[i][j],r_cut2,1);

        }

        count_para++;

      }

    }

    /* Three-body contribution */
    
    count_ang = 1;

    for (j=1;j<=nei_num3-1;j++){

      nei_gnum1 = nei_info3[i][j][1];
      nei_cnum1 = nei_info3[i][j][2];

      for (k=j+1;k<=nei_num3;k++){

        nei_gnum2 = nei_info3[i][k][1];
        nei_cnum2 = nei_info3[i][k][2];

        for (p=Min_order;p<=Max_order;p++){ /*Need to be modified*/
          for (axis=1;axis<=3;axis++){

            d1 = ang_nei[i][count_ang]*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/pow(dis_nei3[i][j],2);
            d2 = ang_nei[i][count_ang]*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/pow(dis_nei3[i][k],2);
            d3 = (2*Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-Gxyz[nei_gnum2][axis]-atv[nei_cnum1][axis]-atv[nei_cnum2][axis])/(dis_nei3[i][j]*dis_nei3[i][k]);
            d4 = (atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/dis_nei3[i][j];
            d5 = (atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/dis_nei3[i][k];

            if (p==0){

              force_pre = current_model[i][count_para]*d4*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,1)*cut_off(dis_nei3[i][k],r_cut3,0)\
                          +current_model[i][count_para]*d5*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,1);

            }

            else{

              force_pre = current_model[i][count_para]*(d1+d2+d3)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)\
                        +current_model[i][count_para]*d4*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,1)*cut_off(dis_nei3[i][k],r_cut3,0)\
                        +current_model[i][count_para]*d5*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,1);

            }


            model_force[centra_gnum][axis] += force_pre;


          }

          count_para ++;

        }

        count_ang ++;

      }

    }

  }

  printf("Self gradient done\n");

  /* Gradient respect to neighbor atom */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num = nei_info[i][0][0];
    count_para = 1;

    /* Two body contribution */

    for (j=1;j<=nei_num;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (p=Min_order;p<=Max_order;p++){
        for (axis=1;axis<=3;axis++){

          d1 = (-Gxyz[centra_gnum][axis]+Gxyz[nei_gnum1][axis]+atv[nei_cnum1][axis])/dis_nei[i][j];
          d2 = (Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-atv[nei_cnum1][axis])/dis_nei[i][j];
          
          model_force[nei_gnum1][axis] += current_model[i][count_para]*d1*ML_orth_poly_deriv(dis_nei[i][j],r_cut2,p)*cut_off(dis_nei[i][j],r_cut2,0)\
                                        +current_model[i][count_para]*d2*ML_orth_poly(dis_nei[i][j],r_cut2,p)*cut_off(dis_nei[i][j],r_cut2,1);

        }

        count_para++;

      }

    }
  
  }

  /* Three body contribution */

  for (i=1;i<=ML_Matomnum;i++){

    centra_gnum = ML_M2G[i];
    nei_num3 = nei_info3[i][0][0];
    count_para = nei_num*(Max_order-Min_order+1)+1;
    count_ang = 1;

    for (j=1;j<=nei_num3-1;j++){

      nei_gnum1 = nei_info3[i][j][1];
      nei_cnum1 = nei_info3[i][j][2];

      for (k=j+1;k<=nei_num3;k++){

        nei_gnum2 = nei_info3[i][k][1];
        nei_cnum2 = nei_info3[i][k][2];

        for(p=Min_order;p<=Max_order;p++){
          for (axis=1;axis<=3;axis++){

            d1 = (atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/(dis_nei3[i][j]*dis_nei3[i][k]);
            d2 = ang_nei[i][count_ang]*(Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/pow(dis_nei3[i][j],2);
            d3 = (Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/dis_nei3[i][j];

            d4 = (atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/(dis_nei3[i][j]*dis_nei3[i][k]);
            d5 = ang_nei[i][count_ang]*(Gxyz[centra_gnum][axis]-atv[nei_cnum2][axis]-Gxyz[nei_gnum2][axis])/pow(dis_nei3[i][k],2);
            d6 = (Gxyz[centra_gnum][axis]-atv[nei_cnum2][axis]-Gxyz[nei_gnum2][axis])/dis_nei3[i][k];

            if (p==0){

              force_pre = current_model[i][count_para]*d3*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,1)*cut_off(dis_nei3[i][k],r_cut3,0);

              force_pre1 = current_model[i][count_para]*d6*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,1);

            }

            else{

              force_pre = current_model[i][count_para]*(d1+d2)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)\
                         +current_model[i][count_para]*d3*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,1)*cut_off(dis_nei3[i][k],r_cut3,0);

              force_pre1 = current_model[i][count_para]*(d4+d5)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,0)\
                          +current_model[i][count_para]*d6*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei3[i][j],r_cut3,0)*cut_off(dis_nei3[i][k],r_cut3,1);

            }

            model_force[nei_gnum1][axis] += force_pre;

            model_force[nei_gnum2][axis] += force_pre1;
          
          }

          count_para += 1;

        }

        count_ang += 1;

      }

    }

  }

  printf("Nei gradient done\n");

  /* Sum the separate force by MPI reduce */

  //MPI_Barrier(mpi_comm_level1);

  for (i=1;i<=atomnum;i++){

    MPI_Allreduce(MPI_IN_PLACE, &model_force[i][0], 4, MPI_DOUBLE, MPI_SUM, mpi_comm_level1); // for loop

  }


  /* Calculate total force */

  if (myid==Host_ID){
    for (axis=1;axis<=3;axis++){
      for (i=1;i<=atomnum;i++){

        total_force[axis] += model_force[i][axis];

      }
    }
  }

  //printf("Total force done\n");

  /* Calculate numerical force */
  
  if (myid==Host_ID){

    for (i=1;i<=atomnum;i++){
      for (axis=1;axis<=3;axis++){

        energy_plus = 0;
        ref_energy = 0;
        Gxyz[i][axis] += numerical_precision;

        for (i_=1;i_<=atomnum;i_++){

          count_para = 1;
          nei_num = nei_info_global[i_][0][0];
          nei_num3 = nei_info3_global[i_][0][0];

          for (j=1;j<=nei_num;j++){

            nei_gnum1 = nei_info_global[i_][j][1];
            nei_cnum1 = nei_info_global[i_][j][2];

            d1 = atv[nei_cnum1][1]+Gxyz[nei_gnum1][1]-Gxyz[i_][1];
            d2 = atv[nei_cnum1][2]+Gxyz[nei_gnum1][2]-Gxyz[i_][2];
            d3 = atv[nei_cnum1][3]+Gxyz[nei_gnum1][3]-Gxyz[i_][3];

            dis_shift1 = sqrt(d1*d1+d2*d2+d3*d3);

            for (p=Min_order;p<=Max_order;p++){

              energy_plus += current_model_global[i_][count_para]*cut_off(dis_shift1,r_cut2,0)*ML_orth_poly(dis_shift1,r_cut2,p);
              count_para += 1;

            }
          }

          for (j=1;j<=nei_num3-1;j++){

            nei_gnum1 = nei_info3_global[i_][j][1];
            nei_cnum1 = nei_info3_global[i_][j][2];

            d1 = atv[nei_cnum1][1]+Gxyz[nei_gnum1][1]-Gxyz[i_][1];
            d2 = atv[nei_cnum1][2]+Gxyz[nei_gnum1][2]-Gxyz[i_][2];
            d3 = atv[nei_cnum1][3]+Gxyz[nei_gnum1][3]-Gxyz[i_][3];

            dis_shift1 = sqrt(d1*d1+d2*d2+d3*d3);

            for (k=j+1;k<=nei_num3;k++){

              nei_gnum2 = nei_info3_global[i_][k][1];
              nei_cnum2 = nei_info3_global[i_][k][2];

              d4 = atv[nei_cnum2][1]+Gxyz[nei_gnum2][1]-Gxyz[i_][1];
              d5 = atv[nei_cnum2][2]+Gxyz[nei_gnum2][2]-Gxyz[i_][2];
              d6 = atv[nei_cnum2][3]+Gxyz[nei_gnum2][3]-Gxyz[i_][3];
              
              dis_shift2 = sqrt(d4*d4+d5*d5+d6*d6);

              ang_shift = (d1*d4+d2*d5+d3*d6)/(dis_shift1*dis_shift2);

              for (p=Min_order;p<=Max_order;p++){

                energy_plus += current_model_global[i_][count_para]*cut_off(dis_shift1,r_cut3,0)*cut_off(dis_shift2,r_cut3,0)*pow(ang_shift,p);
                count_para += 1;
                
              }

            }

          }

          ref_energy += fitted_energy_global[i_];

        }

        Gxyz[i][axis] -= numerical_precision;
        numerical_force[i][axis] = -(ref_energy-energy_plus)/numerical_precision;

      }

    }

  }

  printf("Num force done\n");

  /*

  dtime(&end_time);
  force_time += end_time-start_time;

  */

  /* Replace the DFT force */

  if (ML_force_status==1){

    if (iter>Train_iter && iter>new_train){

      for (i=1;i<=atomnum;i++){
        for (axis=17;axis<=19;axis++){

          Gxyz[i][axis] = model_force[i][axis-16];

        }
      }

      printf("Atomic force is provided by model\n");
      
    }

  }

}