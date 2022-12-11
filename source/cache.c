  if (myid==Host_ID){

    /* Calculate force error */

    for (i=1;i<=atomnum;i++){

      force_error[i][iter] = pow((pow(model_force[i][1],2)+pow(model_force[i][2],2)+pow(model_force[i][3],2)),0.5)\
                            -pow((pow(Gxyz[i][17],2)+pow(Gxyz[i][18],2)+pow(Gxyz[i][19],2)),0.5);

    }

    printf("error force pass\n");

    /* Calculate total force */

    for (axis=1;axis<=3;axis++){
      for (i=1;i<=atomnum;i++){

        total_force[axis][iter] += model_force[i][axis];

      }
    }

    printf("tot force pass\n");

  }


  /* Copy the neighbor information to global */

  nei_info_global = allocarr_int_dyna3d(atomnum,FNAN,3,1);
  dis_nei_global = allocarr_double_dyna2d(atomnum,FNAN,1);

  for (i=1;i<=Matomnum;i++){

    global_num = M2G[i];
    nei_num = nei_info[i][0][0];
    nei_info_global[global_num-1][0][0] = nei_info[i][0][0];

    for (j=1;j<=nei_num;j++){
      
      nei_info_global[global_num-1][j-1][1] = nei_info[i][j][1];
      nei_info_global[global_num-1][j-1][2] = nei_info[i][j][2];
      dis_nei_global[global_num-1][j-1] = dis_nei[i][j];

    }

  }
  
  for (i=0;i<atomnum;i++){

    nei_num = FNAN[i];

    MPI_Allreduce(MPI_IN_PLACE, &dis_nei_global[i][0], nei_num, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    for (j=0;j<nei_num;j++){

      MPI_Allreduce(MPI_IN_PLACE, &nei_info_global[i][j][0], 3, MPI_DOUBLE, MPI_SUM, mpi_comm_level1);

    }

  }

  if (myid==Host_ID){

    for (i=0;i<atomnum;i++){

      printf("nei for %d %d\n",i+1,nei_info_global[i][0][0]);

      for (j=0;j<nei_info_global[i][0][0];j++){

        printf("gnei %d cnei %d dis %8.6f\n",nei_info_global[i][j][1],nei_info_global[i][j][2],dis_nei_global[i][j]);

      }
    }

  }

  /* Allocate nei_info_cache */

  nei_info_cache = (int***)malloc(sizeof(int**)*(Matomnum+1));

  for (i=1;i<=Matomnum;i++){

    nei_num = nei_info[i][0][0];
    nei_info_cache[i] = (int**)malloc(sizeof(int*)*(nei_num+1));

    for (j=0;j<=nei_num;j++){

      nei_info_cache[i][j] = (int*)malloc(sizeof(int)*3);
      memset(nei_info_cache[i][j],0,3*sizeof(int));

    }

  }

  /* Transform neighbor information to cache */

  for (i=1;i<=Matomnum;i++){

    nei_num = nei_info[i][0][0];
    nei_info_cache[i][0][0] = nei_num;

    for (j=1;j<=nei_num;j++){

      nei_info_cache[i][j][1] = nei_info[i][j][1];
      nei_info_cache[i][j][2] = nei_info[i][j][2];

    }

  }


static double total_time;

void ML_main(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int myid;
  double start_time;
  double end_time;

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);


  /* Allocate array */

  if (iter==1){

    ML_allocate_static();

    printf("Allocate static pass\n");

  }

  /* Get decomposed energy */

  Get_decomposed_ene(iter,filepath,filename);

  //dtime(&start_time);

  //ML_output(iter,filepath,filename,".coord");

  if (iter<=Train_iter){

    /* Calculate distance and angular for central atom */

    cal_dis(iter,filepath,filename);
    printf("cal_dis pass\n");
    cal_ang(iter,filepath,filename);
    printf("cal_ang pass\n");

    if (iter==1){

      ML_allocate();
      printf("ML_allocate pass\n");

    }

    if (iter>1){

      ML_check_nei();
      printf("ML_check_nei pass\n");
      //printf("Signal = %d\n",signal);

    }

    /* Generate parameter matrix and constant matrix */

    ML_matrix_gen(iter,filepath,filename);
    printf("ML_matrix_gen pass\n");

    /* Run linear solver to fit the polynomial */

    ML_DSYSV_solver(iter,filepath,filename);
    printf("ML_DSYSV_solver pass\n");

    /* Compute model energy and error */

    ML_model_energy(iter);
    printf("ML_model_energy pass\n");

    ML_force(iter,filepath,filename);
    printf("ML_force pass\n");

    /* Free array */

    if (iter==MD_IterNumber){
      /*
      ML_output(iter,filepath,filename,".fitted_energy");
      printf("out fitted_energy pass\n");
      ML_output(iter,filepath,filename,".total_force");
      printf("out total_force pass\n");
      ML_output(iter,filepath,filename,".threebody_energy");
      printf("out threebody_energy pass\n");
      ML_output(iter,filepath,filename,".twobody_energy");
      printf("out twobody_energy pass\n");
      ML_output(iter,filepath,filename,".ref_energy");
      printf("out ref_energy pass\n");
      ML_output(iter,filepath,filename,".energy_error");
      printf("out energy_error pass\n");
      ML_output(iter,filepath,filename,".force_error");
      printf("out force_error pass\n");
      ML_output(iter,filepath,filename,".time_ana");
      printf("out time_ana pass\n");
      ML_output(iter,filepath,filename,".matrix_ratio");
      printf("out matrix_ratio pass\n");
      */
      ML_free_array();
      ML_free_static();

    }

    ML_Tran_cache();
    printf("ML_Tran_cache pass\n");
      
  }

  else if (iter>Train_iter && (iter-Train_iter)%Correction_iter==0){

    /* Calculate distance and angular for central atom */

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    /* Generate parameter matrix and constant matrix */

    ML_matrix_gen(iter,filepath,filename);

    /* Run linear solver to fit the polynomial */

    ML_DSYSV_solver(iter,filepath,filename);

    /* Compute model energy and error */

    ML_model_energy(iter);

    ML_force(iter,filepath,filename);

    /* Free array */

    if (iter==MD_IterNumber){
      /*
      ML_output(iter,filepath,filename,".fitted_energy");
      printf("out fitted_energy pass\n");
      ML_output(iter,filepath,filename,".total_force");
      printf("out total_force pass\n");
      ML_output(iter,filepath,filename,".threebody_energy");
      printf("out threebody_energy pass\n");
      ML_output(iter,filepath,filename,".twobody_energy");
      printf("out twobody_energy pass\n");
      ML_output(iter,filepath,filename,".ref_energy");
      printf("out ref_energy pass\n");
      ML_output(iter,filepath,filename,".energy_error");
      printf("out energy_error pass\n");
      ML_output(iter,filepath,filename,".force_error");
      printf("out force_error pass\n");
      ML_output(iter,filepath,filename,".time_ana");
      printf("out time_ana pass\n");
      ML_output(iter,filepath,filename,".matrix_ratio");
      printf("out matrix_ratio pass\n");
      */
      ML_free_array();
      ML_free_static();

    }

    ML_Tran_cache();

  }

  else if (iter>Train_iter && (iter-Train_iter)%Correction_iter!=0){

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    ML_model_energy(iter);

    ML_force(iter,filepath,filename);

    /* Free array */

    if (iter==MD_IterNumber){
      /*
      ML_output(iter,filepath,filename,".fitted_energy");
      printf("out fitted_energy pass\n");
      ML_output(iter,filepath,filename,".total_force");
      printf("out total_force pass\n");
      ML_output(iter,filepath,filename,".threebody_energy");
      printf("out threebody_energy pass\n");
      ML_output(iter,filepath,filename,".twobody_energy");
      printf("out twobody_energy pass\n");
      ML_output(iter,filepath,filename,".ref_energy");
      printf("out ref_energy pass\n");
      ML_output(iter,filepath,filename,".energy_error");
      printf("out energy_error pass\n");
      ML_output(iter,filepath,filename,".force_error");
      printf("out force_error pass\n");
      ML_output(iter,filepath,filename,".time_ana");
      printf("out time_ana pass\n");
      ML_output(iter,filepath,filename,".matrix_ratio");
      printf("out matrix_ratio pass\n");
      */
      ML_free_array();
      ML_free_static();

    }

    ML_Tran_cache();

  }

  //dtime(&end_time);
  //total_time += end_time-start_time;

}

  twobody_ene = (double***)malloc(sizeof(double**)*(Matomnum+1));

  for(i=1;i<=Matomnum;i++){

    nei_num = nei_info[i][0][0];
    twobody_ene[i] = (double**)malloc(sizeof(double*)*(MD_IterNumber+1));

    for (j=1;j<=MD_IterNumber;j++){

      twobody_ene[i][j] = (double*)malloc(sizeof(double)*(nei_num+1));
      memset(twobody_ene[i][j],0,(nei_num+1)*sizeof(double));

    }
  }

  //printf("Twobody energy array allocate Pass\n");

  threebody_ene = (double***)malloc(sizeof(double**)*(Matomnum+1));

  for(i=1;i<=Matomnum;i++){

    threebody_ene[i] = (double**)malloc(sizeof(double*)*(MD_IterNumber+1));

    for (j=1;j<=MD_IterNumber;j++){

      threebody_ene[i][j] = (double*)malloc(sizeof(double)*(ang_num[i]+1));
      memset(threebody_ene[i][j],0,(ang_num[i]+1)*sizeof(double));

    }
  }

  //printf("Threebody energy array allocate Pass\n");


  else if (iter>Train_iter && (iter-Train_iter)%Correction_iter==0){

    /* Calculate distance and angular for central atom */

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    /* Generate parameter matrix and constant matrix */

    ML_matrix_gen(iter,filepath,filename);

    /* Run linear solver to fit the polynomial */

    ML_DSYSV_solver(iter,filepath,filename);

    /* Compute model energy and error */

    ML_model_energy(iter);

    ML_force(iter,filepath,filename);

    /* Free array */

    if (iter==MD_IterNumber){
      /*
      ML_output(iter,filepath,filename,".fitted_energy");
      printf("out fitted_energy pass\n");
      ML_output(iter,filepath,filename,".total_force");
      printf("out total_force pass\n");
      ML_output(iter,filepath,filename,".threebody_energy");
      printf("out threebody_energy pass\n");
      ML_output(iter,filepath,filename,".twobody_energy");
      printf("out twobody_energy pass\n");
      ML_output(iter,filepath,filename,".ref_energy");
      printf("out ref_energy pass\n");
      ML_output(iter,filepath,filename,".energy_error");
      printf("out energy_error pass\n");
      ML_output(iter,filepath,filename,".force_error");
      printf("out force_error pass\n");
      ML_output(iter,filepath,filename,".time_ana");
      printf("out time_ana pass\n");
      ML_output(iter,filepath,filename,".matrix_ratio");
      printf("out matrix_ratio pass\n");
      */
      ML_free_array();
      ML_free_static();

    }

    ML_Tran_cache();

  }

  else if (iter>Train_iter && (iter-Train_iter)%Correction_iter!=0){

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    ML_model_energy(iter);

    ML_force(iter,filepath,filename);

    /* Free array */

    if (iter==MD_IterNumber){
      /*
      ML_output(iter,filepath,filename,".fitted_energy");
      printf("out fitted_energy pass\n");
      ML_output(iter,filepath,filename,".total_force");
      printf("out total_force pass\n");
      ML_output(iter,filepath,filename,".threebody_energy");
      printf("out threebody_energy pass\n");
      ML_output(iter,filepath,filename,".twobody_energy");
      printf("out twobody_energy pass\n");
      ML_output(iter,filepath,filename,".ref_energy");
      printf("out ref_energy pass\n");
      ML_output(iter,filepath,filename,".energy_error");
      printf("out energy_error pass\n");
      ML_output(iter,filepath,filename,".force_error");
      printf("out force_error pass\n");
      ML_output(iter,filepath,filename,".time_ana");
      printf("out time_ana pass\n");
      ML_output(iter,filepath,filename,".matrix_ratio");
      printf("out matrix_ratio pass\n");
      */
      ML_free_array();
      ML_free_static();

    }

    ML_Tran_cache();

  }

  //dtime(&end_time);
  //total_time += end_time-start_time;


  /* Gradient respect to neighbor atom */

  for (i=1;i<=Matomnum;i++){

    centra_gnum = M2G[i];

    species = WhatSpecies[centra_gnum];
    r_cut = Spe_Atom_Cut1[species];

    nei_num = nei_info[i][0][0];
    count_para = 1;

    /* Two body contribution */

    for (j=1;j<=nei_num;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      if (nei_cnum1==0){

        for (p=Min_order;p<=Max_order;p++){
          for (axis=1;axis<=3;axis++){

            d1 = (-Gxyz[centra_gnum][axis]+Gxyz[nei_gnum1][axis]+atv[nei_cnum1][axis])/dis_nei[i][j];
            d2 = (Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-atv[nei_cnum1][axis])/dis_nei[i][j];
            
            model_force[nei_gnum1][axis] += current_model[i][count_para]*d1*ML_orth_poly_deriv(dis_nei[i][j],r_cut,p)*cut_off(dis_nei[i][j],r_cut,0)\
                                          +current_model[i][count_para]*d2*ML_orth_poly(dis_nei[i][j],r_cut,p)*cut_off(dis_nei[i][j],r_cut,1);

          }

          count_para++;

        }

      }

      else{

        count_para += Max_order-Min_order+1;

      }

    }
  
  }

  /* Three body contribution */

  for (i=1;i<=Matomnum;i++){

    centra_gnum = M2G[i];

    species = WhatSpecies[centra_gnum];
    r_cut = Spe_Atom_Cut1[species];

    nei_num = nei_info[i][0][0];
    count_para = nei_num*(Max_order-Min_order+1)+1;
    count_para1 = nei_num*(Max_order-Min_order+1)+1;
    count_para2 = nei_num*(Max_order-Min_order+1)+1;
    count_ang = 1;

    for (j=1;j<=nei_num-1;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (k=j+1;k<=nei_num;k++){

        nei_gnum2 = nei_info[i][k][1];
        nei_cnum2 = nei_info[i][k][2];

        if (nei_cnum1==0){

          for(p=Min_order;p<=Max_order;p++){
            for (axis=1;axis<=3;axis++){

              d1 = (atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/(dis_nei[i][j]*dis_nei[i][k]);
              d2 = ang_nei[i][count_ang]*(Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/pow(dis_nei[i][j],2);
              d3 = (Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/dis_nei[i][j];

              model_force[nei_gnum1][axis] += current_model[i][count_para1]*(d1+d2)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)\
                                            +current_model[i][count_para1]*d3*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,1)*cut_off(dis_nei[i][k],r_cut,0);

            }

            count_para1 += 1;

          }

        }

        else{

          count_para1 += Max_order-Min_order+1;

        }
        
        if (nei_cnum2==0){

          for(p=Min_order;p<=Max_order;p++){
            for (axis=1;axis<=3;axis++){

              d1 = (atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/(dis_nei[i][j]*dis_nei[i][k]);
              d2 = ang_nei[i][count_ang]*(Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/pow(dis_nei[i][j],2);
              d3 = (Gxyz[centra_gnum][axis]-atv[nei_cnum1][axis]-Gxyz[nei_gnum1][axis])/dis_nei[i][j];

              model_force[nei_gnum1][axis] += current_model[i][count_para2]*(d1+d2)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)\
                                            +current_model[i][count_para2]*d3*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,1)*cut_off(dis_nei[i][k],r_cut,0);

              d4 = (atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/(dis_nei[i][j]*dis_nei[i][k]);
              d5 = ang_nei[i][count_ang]*(Gxyz[centra_gnum][axis]-atv[nei_cnum2][axis]-Gxyz[nei_gnum2][axis])/pow(dis_nei[i][k],2);
              d6 = (Gxyz[centra_gnum][axis]-atv[nei_cnum2][axis]-Gxyz[nei_gnum2][axis])/dis_nei[i][k];

              model_force[nei_gnum2][axis] += current_model[i][count_para2]*(d4+d5)*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)\
                                            +current_model[i][count_para2]*d6*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,1);

            }

            count_para2 += 1;

          }
          
        }

        count_ang += 1;

      }
    }

  }

/**********************************************************************
  ML_output.c:

     ML_output.c is a subroutine to out put necessary information and
     result of ML.c

  Log of ML_output.c:

     19/Sep/2022  Added by Hengyu Li

     ver 0.0.1 First edition of parallel IO

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"

#define Pi 3.141592654

/* Output information */

void ML_output(int iter, char filepath[YOUSO10], char filename[YOUSO10], char keyword[YOUSO10])
{
  int i,j,k,nei_num;
  int species;
  char target_file[YOUSO10];
  FILE *fp;

  strcpy(target_file, keyword);

  fnjoint(filepath,filename,target_file);

  fp = fopen(target_file,"a");
  
  if (keyword==".matrix"){

    fprintf(fp,"Matrices at MD iter =%d\n",iter);

    /* Output matrix A */

    fprintf(fp,"Matrix A\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*nei_info[i][0][0]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*nei_info[i][0][0]; k++){
          fprintf(fp,"%16.14f ",matrix_a[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C */

    fprintf(fp,"Matrix C\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*nei_info[i][0][0]; j++){
        fprintf(fp,"%16.14f ",matrix_c[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B */

    fprintf(fp,"Matrix B\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*nei_info[i][0][0]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*angular_num[i]; k++){
          fprintf(fp,"%16.14f ",matrix_b[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix A' */

    fprintf(fp,"Matrix A'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*nei_info[i][0][0]; k++){
          fprintf(fp,"%16.14f ",matrix_a_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C' */

    fprintf(fp,"Matrix C'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        fprintf(fp,"%16.14f ",matrix_c_[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B' */

    fprintf(fp,"Matrix B'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*angular_num[i]; k++){
          fprintf(fp,"%16.14f ",matrix_b_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }

    fclose(fp);

    //printf("Out matrix Pass\n");

  }

  else if (keyword==".solver_input"){

    fprintf(fp,"Parameter and constant array at MD iter =%d\n",iter);

    for (i=1;i<=atomnum;i++){
      nei_num = nei_info[i][0][0];
      fprintf(fp,"Parameter matrix for atom %d\n",i);
      for (j=1;j<=pow((nei_num+angular_num[i])*(Max_order-Min_order+1),2);j++){
        fprintf(fp,"%16.14f ",parameter_matrix[i][j]); 
      }
      fprintf(fp,"\n");
      fprintf(fp,"Constant matrix for atom %d\n",i);
      for (k=1;k<=(nei_num+angular_num[i])*(Max_order-Min_order+1);k++){ 
        fprintf(fp,"%16.14f ",constant_matrix[i][k]);
      }
      fprintf(fp,"\n");
    }

    fclose(fp);

    //printf("Out solver input Pass\n");

  }

  else if (keyword==".fitted_parameter"){

    fprintf(fp,"Fitted parameters for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){
      nei_num = nei_info_global[i][0][0];
      fprintf(fp,"atom %d\n",i);
      for (j=1;j<=(nei_num+angular_num_global[i])*(Max_order-Min_order+1);j++){
        fprintf(fp,"%16.14f ",current_model_global[i][j]); //constant_matrix[i][j]
      }
      fprintf(fp,"\n");
    }

    fclose(fp);

    //printf("Out fitted parameter Pass\n");

  }
  
  else if (keyword==".energy_error"){

    fprintf(fp,"Fitting error for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %8.8f",i,fitted_energy_global[i]-Dec_tot_global[i]);

    }

    fclose(fp);

    //printf("Out fitting energy error Pass\n");
    
  }

  else if (keyword==".force_error"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Fitting error for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          fprintf(fp,"%16.14f ",force_error[i][j]);
        }
        fprintf(fp,"\n");
      }
      
      fprintf(fp,"\n");
      fclose(fp);

      //printf("Out fitting force error Pass\n");
    }
  }
  
  else if (keyword==".fitted_energy"){

    fprintf(fp,"Fitted energy for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %8.10f\n",i,fitted_energy_global[i]);

    }

    fclose(fp);      

    //printf("Out fitted energy Pass\n");

  }

  else if (keyword==".twobody_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Twobody energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        nei_num = nei_info[i][0][0];
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          for (k=1;k<=nei_num;k++){
            fprintf(fp,"%16.14f ",twobody_ene[i][j][k]);
          }
          fprintf(fp,",");
        }
        fprintf(fp,"\n");
      }

      fclose(fp);      
    }

    //printf("Out Twobody energy Pass\n");

  }

  else if (keyword==".threebody_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Twobody energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          for (k=1;k<=angular_num[i];k++){
            fprintf(fp,"%16.14f ",threebody_ene[i][j][k]);
          }
          fprintf(fp,",");
        }
        fprintf(fp,"\n");
      }

      fclose(fp);      
    }

    //printf("Out Threebody energy Pass\n");

  }

  else if (keyword==".ref_energy"){
    
    fprintf(fp,"Reference energy for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %8.10f\n",i,Dec_tot_global[i]);

    }

    fclose(fp);

    //printf("Out reference decomposed energy Pass\n");

  }

  else if (keyword==".fitted_force"){

    fprintf(fp,"Fitted force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=1;j<=3;j++){
        fprintf(fp,"%16.14f ",model_force[i][j]);
      }

      fprintf(fp,"\n");

    }

    fclose(fp);

    //printf("Out fitted force Pass\n");

  }

  else if (keyword==".ref_force"){

    fprintf(fp,"Reference force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=17;j<=19;j++){
        fprintf(fp,"%16.14f ",Gxyz[i][j]);
      }

      fprintf(fp,"\n");
    }

    fclose(fp);

    //printf("Out reference force Pass\n");

  }

  else if (keyword==".total_force"){

    fprintf(fp,"Total force for x,y,z\n");

    fprintf(fp,"x: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%16.14f ", total_force[1][i]);
    }
    fprintf(fp,"\n");
    fprintf(fp,"y: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%16.14f ", total_force[2][i]);
    }
    fprintf(fp,"\n");
    fprintf(fp,"z: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%16.14f ", total_force[3][i]);
    }

  }

  else if (keyword==".numerical_force"){

    fprintf(fp,"Numerical force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=1;j<=3;j++){
        fprintf(fp,"%16.14f ",numerical_force[i][j]);
      }

      fprintf(fp,"\n");
    }

    fclose(fp);

    //printf("Out Numerical force Pass\n");

  }

  else if (keyword==".coord"){

    if (iter==1){

      fprintf(fp,"%d\n",atomnum);
      fprintf(fp,"\n");

    }

    for (i=1;i<=atomnum;i++){

      species = WhatSpecies[i];

      fprintf(fp,"%s ",SpeName[species]);

      for (j=1;j<=3;j++){
        fprintf(fp,"%16.14f ",Gxyz[i][j]);
      }

      fprintf(fp,"\n");
      
    }

    fclose(fp);

    //printf("Out Coordinate Pass\n");

  }

  else if (keyword==".time_ana"){

    fprintf(fp,"Computing time analyze\n\n");

    fprintf(fp,"Total computing time for ML.c = %8.4f\n\n",total_time);
    
    fprintf(fp,"Matrix generation = %8.4f\n\n",matrix_time);

    fprintf(fp,"DSYSV Solver = %8.4f\n\n",solver_time);

    fprintf(fp,"Model energy = %8.4f\n\n",energy_time);

    fprintf(fp,"Model force = %8.4f\n\n",force_time);

    fclose(fp);

    //printf("Out time analyze Pass\n");

  }

  else if (keyword==".matrix_ratio"){

    if (iter==MD_IterNumber){

      fprintf(fp,"Matrix element ratio for each atom\n");

      for (i=1;i<=MD_IterNumber;i++){

        for (j=1;j<=atomnum;j++){

          fprintf(fp,"Ratio for %d A %16.14f\n",j,ratio_a[j][i]);

          fprintf(fp,"Ratio for %d C %16.14f\n",j,ratio_c[j][i]);

        }

        fprintf(fp,"\n");

      }

      fclose(fp);

    }
  }

  else{
    printf("Check output keyword \n");
  }

}