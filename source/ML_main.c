/**********************************************************************
  ML_main.c:

     ML_main.c is the main function for on-the-fly MD

  Log of ML_main.c:

     19/Sep/2022  Added by Hengyu Li

     ver 0.0.1 First separate from ML.c One can design the on-the-fly
     algorithm here.

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"

#define Pi 3.141592654

/*******************************************************
                Main function of ML
*******************************************************/

static double total_time;
static int dft_time = 0;
static int pattern = 0; // 0: training; 1: Model unchanged; 2: Trace differance; 3: Re-train

void ML_main(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int i,j,centra_gnum,myid,count;
  double start_time;
  double end_time;
  char filelast[YOUSO10] = ".type_ana";
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  count = 0; // count the number of neighbor changed atoms in unit cell

  /* Allocate array */

  if (iter==1){

    ML_allocate_static();
    printf("Allocate static pass\n");

    ML_allocate_atom2cpu();
    printf("Allocate cpu pass\n");

  }
  
  //dtime(&start_time);

  //ML_output(iter,filepath,filename,".coord");

  if (iter<=Train_iter){

    /* Calculate distance and angular for central atom */     

    cal_dis(iter,filepath,filename);
    printf("cal_dis pass\n");

    cal_ang(iter,filepath,filename);
    printf("cal_ang pass\n");

    if (iter==1){

      ML_allocate_matrix();
      printf("Allocate matrix pass\n");

    }

    /* Get decomposed energy */

    Get_decomposed_ene(iter,filepath,filename);
    printf("Get_decomposed_ene pass\n");

    ML_allocate_dyna();
    printf("ML_allocate_dyna pass\n");

    ML_allocate_model();
    printf("ML_allocate_model pass\n");

    if (iter>1){

      ML_check_nei();
      printf("ML_check_nei pass\n");

      ML_matrix_pre(iter,filepath,filename);
      printf("ML_matrix_pre pass\n");
      
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

    ML_force(iter,new_train,filepath,filename);
    printf("ML_force pass\n");

    if (myid==Host_ID){

      ML_output(iter,filepath,filename,".fitted_energy");

      ML_output(iter,filepath,filename,".fitted_force");

      ML_output(iter,filepath,filename,".ref_force");

      ML_output(iter,filepath,filename,".ref_energy");

      ML_output(iter,filepath,filename,".energy_error");

      ML_output(iter,filepath,filename,".fitted_parameter");

      ML_output(iter,filepath,filename,".dis");

      ML_output(iter,filepath,filename,".dis3");

      ML_output(iter,filepath,filename,".ang");

      ML_output(iter,filepath,filename,".total_force");

      ML_output(iter,filepath,filename,".numerical_force");

      ML_output(iter,filepath,filename,".signal");

    }

    ML_output(iter,filepath,filename,".2cpu");

    if (iter>1){

      ML_output(iter,filepath,filename,".nei_ana");

      ML_output(iter,filepath,filename,".nei_ana3");

    }    

    /* Free array */

    ML_free_dyna();
    printf("ML_free_dyna pass\n");

    if (iter!=Train_iter){

      ML_free_model();
      printf("ML_free_model pass\n");

    }

    ML_tran_matrix();
    printf("ML_tran_matrix pass\n");

    ML_Tran_cache(iter);
    printf("ML_Tran_cache pass\n");

    if (iter==MD_IterNumber){

      ML_free_static();
      printf("ML_free_static pass\n");

    }    
      
  }

  if (iter>Train_iter){

    cal_dis(iter,filepath,filename);
    printf("cal_dis pass\n");

    cal_ang(iter,filepath,filename);
    printf("cal_ang pass\n");

    Get_decomposed_ene(iter,filepath,filename);
    printf("Get_decomposed_ene pass\n");

    ML_check_nei();
    printf("ML_check_nei pass\n");

    /* Recognize whether retrain or just keep the model */

    if (iter<=new_train){

      pattern = 3;

      printf("pattern = %d\n",pattern);

    }

    else{

      for (i=1;i<=atomnum;i++){

        count += signal[i];
        
      }

      if (count==0){

        pattern = 1;

      }

      else{

        pattern = 2;

      }

      printf("pattern = %d\n",pattern);

    }

    if (pattern==1){

      if (myid==Host_ID){

        printf("Model unchanged\n");

        fprintf(fp,"Iter %d Model unchanged\n",iter);

      }

      ML_free_nei_cache();
      printf("ML_free_nei_cache pass\n");

      ML_model_energy(iter);
      printf("ML_model_energy pass\n");

      ML_force(iter,new_train,filepath,filename);
      printf("ML_force pass\n");

      if (myid==Host_ID){

        ML_output(iter,filepath,filename,".fitted_energy");

        ML_output(iter,filepath,filename,".fitted_force");

        ML_output(iter,filepath,filename,".ref_force");

        ML_output(iter,filepath,filename,".ref_energy");

        ML_output(iter,filepath,filename,".energy_error");

        ML_output(iter,filepath,filename,".dis");

        ML_output(iter,filepath,filename,".dis3");

        ML_output(iter,filepath,filename,".ang");

        ML_output(iter,filepath,filename,".numerical_force");

        ML_output(iter,filepath,filename,".total_force");

        ML_output(iter,filepath,filename,".signal");

      }

      ML_output(iter,filepath,filename,".nei_ana");

      ML_output(iter,filepath,filename,".nei_ana3");

      ML_Tran_cache(iter);
      printf("ML_Tran_cache pass\n");

      if (iter==MD_IterNumber){

        ML_free_static();
        printf("ML_free_static pass\n");

      }

    }

    if (pattern==2){

      new_train = iter + Correction_iter-1;

      dft_time += 1;

      if (myid==Host_ID){

        printf("Trace diff in nei_info\n");

        fprintf(fp,"Iter %d Trace diff in nei_info\n",iter);

      }

      ML_free_model();
      printf("ML_free_model pass\n");

      ML_allocate_dyna();
      printf("ML_allocate_dyna pass\n");

      ML_allocate_model();
      printf("ML_allocate_model pass\n");

      ML_matrix_pre(iter,filepath,filename);
      printf("ML_matrix_pre pass\n");

      ML_matrix_gen(iter,filepath,filename);
      printf("ML_matrix_gen pass\n");

      ML_DSYSV_solver(iter,filepath,filename);
      printf("ML_DSYSV_solver pass\n");

      ML_model_energy(iter);
      printf("ML_model_energy pass\n");

      ML_force(iter,new_train,filepath,filename);
      printf("ML_force pass\n");

      if (myid==Host_ID){

        ML_output(iter,filepath,filename,".fitted_energy");

        ML_output(iter,filepath,filename,".fitted_force");

        ML_output(iter,filepath,filename,".ref_force");

        ML_output(iter,filepath,filename,".ref_energy");

        ML_output(iter,filepath,filename,".energy_error");

        ML_output(iter,filepath,filename,".fitted_parameter");

        ML_output(iter,filepath,filename,".dis");

        ML_output(iter,filepath,filename,".dis3");

        ML_output(iter,filepath,filename,".ang");

        ML_output(iter,filepath,filename,".numerical_force");

        ML_output(iter,filepath,filename,".total_force");

        ML_output(iter,filepath,filename,".signal");

      }

      ML_output(iter,filepath,filename,".2cpu");

      ML_output(iter,filepath,filename,".nei_ana");

      ML_output(iter,filepath,filename,".nei_ana3");

      /* Free array */

      ML_free_dyna();
      printf("ML_free_dyna pass\n");

      ML_free_model();
      printf("ML_free_model pass\n");

      ML_tran_matrix();
      printf("ML_tran_matrix pass\n");

      ML_Tran_cache(iter);
      printf("ML_Tran_cache pass\n");

      if (iter==MD_IterNumber){

        ML_free_static();
        printf("ML_free_static pass\n");

      }

    }

    if (pattern==3){

      if (myid==Host_ID){

        printf("Retrain model %d iter\n",Correction_iter+iter-new_train);

        fprintf(fp,"Iter %d Re-train %d iter\n",iter,Correction_iter+iter-new_train);

      }

      dft_time += 1;

      ML_allocate_dyna();
      printf("ML_allocate_dyna pass\n");

      ML_allocate_model();

      ML_matrix_pre(iter,filepath,filename);
      printf("ML_matrix_pre pass\n");

      ML_matrix_gen(iter,filepath,filename);
      printf("ML_matrix_gen pass\n");

      ML_DSYSV_solver(iter,filepath,filename);
      printf("ML_DSYSV_solver pass\n");

      ML_model_energy(iter);
      printf("ML_model_energy pass\n");

      ML_force(iter,new_train,filepath,filename);
      printf("ML_force pass\n");

      if (myid==Host_ID){

        ML_output(iter,filepath,filename,".fitted_energy");

        ML_output(iter,filepath,filename,".fitted_force");

        ML_output(iter,filepath,filename,".ref_force");

        ML_output(iter,filepath,filename,".ref_energy");

        ML_output(iter,filepath,filename,".energy_error");

        ML_output(iter,filepath,filename,".fitted_parameter");

        ML_output(iter,filepath,filename,".dis");

        ML_output(iter,filepath,filename,".dis3");

        ML_output(iter,filepath,filename,".ang");

        ML_output(iter,filepath,filename,".numerical_force");

        ML_output(iter,filepath,filename,".total_force");

        ML_output(iter,filepath,filename,".signal");

      }

      ML_output(iter,filepath,filename,".2cpu");

      ML_output(iter,filepath,filename,".nei_ana");

      ML_output(iter,filepath,filename,".nei_ana3");

      /* Free array */

      ML_free_dyna();
      printf("ML_free_dyna pass\n");

      if (iter!=new_train){

        ML_free_model();
        printf("ML_free_model pass\n");

      }

      ML_tran_matrix();
      printf("ML_tran_matrix pass\n");

      ML_Tran_cache(iter);
      printf("ML_Tran_cache pass\n");

      if (iter==MD_IterNumber){

        ML_free_static();
        printf("ML_free_static pass\n");

      }

    }

  }

  if (iter==MD_IterNumber){

    if (myid==Host_ID){

      printf("Extra DFT %d\n",dft_time);

    }

  }

  fclose(fp);

}