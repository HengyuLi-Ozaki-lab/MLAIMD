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
  int i,j,k,nei_num,myid,centra_gnum;
  int species;
  char target_file[YOUSO10];
  FILE *fp;

  MPI_Comm_rank(mpi_comm_level1,&myid);

  strcpy(target_file, keyword);

  fnjoint(filepath,filename,target_file);

  fp = fopen(target_file,"a");
  

  if (keyword==".fitted_parameter"){

    fprintf(fp,"Fitted parameters for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      nei_num = nei_info_global[i][0][0];

      fprintf(fp,"atom %d\n",i);

      for (j=1;j<=(nei_num+ang_num_global[i])*(Max_order-Min_order+1);j++){

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

      fprintf(fp,"Atom %d %16.14f\n",i,fitted_energy_global[i]-Dec_tot_global[i]);

    }

    fclose(fp);

    //printf("Out fitted energy error Pass\n");
    
  }
  
  else if (keyword==".fitted_energy"){

    fprintf(fp,"Fitted energy for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %16.14f\n",i,fitted_energy_global[i]);

    }

    fclose(fp);      

    //printf("Out fitted energy Pass\n");

  }

  else if (keyword==".ref_energy"){
    
    fprintf(fp,"Reference energy for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %16.14f\n",i,Dec_tot_global[i]);

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

  else if (keyword==".total_force"){

    fprintf(fp,"Total force at iter = %d\n",iter);

    fprintf(fp,"x: %16.14f y: %16.14f z: %16.14f\n",total_force[1],total_force[2],total_force[3]);

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

  else if (keyword==".dis"){

    fprintf(fp,"Neighbor info and distance at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d\n",i);

      for (j=1;j<=nei_info_global[i][0][0];j++){

        fprintf(fp,"gnum %d cnum %d dis %16.14f\n",nei_info_global[i][j][1],nei_info_global[i][j][2],dis_nei_global[i][j]);

      }

      fprintf(fp,"\n");

    }

    fclose(fp);

    //printf("Out Numerical force Pass\n");

  }

  else if (keyword==".dis3"){

    fprintf(fp,"3 Neighbor info and distance at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d\n",i);

      for (j=1;j<=nei_info3_global[i][0][0];j++){

        fprintf(fp,"gnum %d cnum %d dis %16.14f\n",nei_info3_global[i][j][1],nei_info3_global[i][j][2],dis_nei3_global[i][j]);

      }

      fprintf(fp,"\n");

    }

    fclose(fp);

  }

  else if (keyword==".ang"){

    fprintf(fp,"Angular at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d\n",i);

      for (j=1;j<=ang_num_global[i];j++){

        fprintf(fp,"%16.14f ",ang_nei_global[i][j]);

      }

      fprintf(fp,"\n");
      
    }

    fclose(fp);

    //printf("Out Numerical force Pass\n");

  }

  else if (keyword==".signal"){

    fprintf(fp,"Signal at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d %d\n",i,signal[i]);
      
    }

    fclose(fp);

    //printf("Out Numerical force Pass\n");

  }

  else if (keyword==".2cpu"){

    for (i=0;i<Num_Procs;i++){

      if (myid==i){

        fprintf(fp,"Iter %d Proc %d Matom %d Ref %d\n",iter,i,ML_Matomnum,Matomnum);

        for (j=1;j<=ML_Matomnum;j++){

          fprintf(fp,"gnum %d ",ML_M2G[j]);

        }

        for (j=1;j<=Matomnum;j++){

          fprintf(fp,"ref gnum %d ",M2G[j]);

        }        

        fprintf(fp,"\n");

      }
    
    }

    fclose(fp);

    //printf("Out Numerical force Pass\n");

  }

  else if (keyword==".nei_ana"){

    for (i=0;i<Num_Procs;i++){

      if (myid==i){

        for (j=1;j<=ML_Matomnum;j++){

          centra_gnum = ML_M2G[j];

          fprintf(fp,"Iter %d Atom %d\n",iter,centra_gnum);

          fprintf(fp,"pattern %d\n",pattern);

          for (k=1;k<=nei_info_global[centra_gnum][0][0];k++){

            fprintf(fp,"nei %d sign %d trace %d\n",k,nei_list_ana[j][k][1],nei_list_ana[j][k][2]);

          }

          fprintf(fp,"\n");

        }

        fprintf(fp,"\n");

      }
    
    }

    fclose(fp);

  }

  else if (keyword==".nei_ana3"){

    for (i=0;i<Num_Procs;i++){

      if (myid==i){

        for (j=1;j<=ML_Matomnum;j++){

          centra_gnum = ML_M2G[j];

          fprintf(fp,"Iter %d Atom %d\n",iter,centra_gnum);

          fprintf(fp,"pattern %d\n",pattern);

          for (k=1;k<=nei_info3_global[centra_gnum][0][0];k++){

            fprintf(fp,"nei %d sign %d trace %d\n",k,nei_list3_ana[j][k][1],nei_list3_ana[j][k][2]);

          }

          fprintf(fp,"\n");

        }

        fprintf(fp,"\n");

      }
    
    }

    fclose(fp);

  }

  else{

    printf("Check output keyword \n");

  }

}