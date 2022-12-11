  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){

          printf("Matrix A for %d\n",i);

          for (row=1;row<=nei_list[j][0]*(Max_order-Min_order+1);row++){
            for (column1=1;column1<=nei_list[j][0]*(Max_order-Min_order+1);column1++){
              printf("%8.6f ",matrix_a[j][row][column1]);
            }
            printf("\n");
          }
          printf("\n");            
          
        }
      }
    }
  }
  */
  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){
          if (nei_list[j][0]>1){
            printf("Matrix B for %d\n",i);
            for (row=1;row<=nei_list[j][0]*(Max_order-Min_order+1);row++){
              for (column1=1;column1<=angular_num[j]*(Max_order-Min_order+1);column1++){
                printf("%8.6f ",matrix_b[j][row][column1]);
              }
              printf("\n");
            }
            printf("\n");            
          }
        }
      }
    }
  }
  */
  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){

          printf("Matrix C for %d\n",i);
          for (row=1;row<=nei_list[j][0]*(Max_order-Min_order+1);row++){

            printf("%8.6f ",matrix_c[j][row]);

          }
          printf("\n");            

        }
      }
    }
  }
  */
  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){
          if (nei_list[j][0]>1){
            printf("Matrix A' for %d\n",i);
            for (row=1;row<=angular_num[j]*(Max_order-Min_order+1);row++){
              for (column1=1;column1<=nei_list[j][0]*(Max_order-Min_order+1);column1++){
                printf("%8.6f ",matrix_a_[j][row][column1]);
              }
              printf("\n");
            }
            printf("\n");            
          }
        }
      }
    }
  }
  */
  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){
          if (nei_list[j][0]>1){
            printf("Matrix B' for %d\n",i);
            for (row=1;row<=angular_num[j]*(Max_order-Min_order+1);row++){
              for (column1=1;column1<=angular_num[j]*(Max_order-Min_order+1);column1++){
                printf("%8.6f ",matrix_b_[j][row][column1]);
              }
              printf("\n");
            }
            printf("\n");            
          }
        }
      }
    }
  }
  */
  /*
  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){
          if (nei_list[j][0]>1){
            printf("Matrix C' for %d\n",i);
            for (row=1;row<=nei_list[j][0]*(Max_order-Min_order+1);row++){

              printf("%8.6f ",matrix_c_[j][row]);

            }
            printf("\n");            
          }
        }
      }
    }
  }
  */

  for (i=1;i<=atomnum;i++){

    ID = G2ID[i];

    if (myid==ID){
      for (j=1;j<=Matomnum;j++){
        if (M2G[j]==i){

          if (nei_list[j][0]>1){

            printf("Para for %d\n",i);

            for (row=1;row<=pow((nei_list[j][0]+angular_num[i])*(Max_order-Min_order+1),2);row++){
              printf("%8.6f ",parameter_matrix[j][row]);
            }
            printf("\n");              
          }

          if (nei_list[j][0]==1){

            printf("Para for %d\n",i);

            for (row=1;row<=pow(nei_list[j][0]*(Max_order-Min_order+1),2);row++){
              printf("%8.6f ",parameter_matrix[j][row]);
            }
            printf("\n");              
          }
          
        }
      }
    }
  }


  cal_dis_parallel();
  printf("Dis pass\n");
  cal_ang_parallel();
  printf("Ang pass\n");
  
  if (iter>1){

    ML_Tran_cache();
    printf("Tran nei pass\n");

  }
  
  if (iter==1){

    ML_allocate_dynamic();
    printf("Allocate dynamic pass\n");

  }

  if (iter<=Train_iter){

    get_dec_ene_parallel();
    printf("Dect pass\n");
    ML_matrix_gen_parallel(iter);
    printf("Matrix gen pass\n");
    ML_DSYSV_solver_parallel();
    printf("Solver pass\n");

  }

  ML_model_energy_parallel(iter);
  printf("Energy pass\n");
  ML_force_parallel();
  printf("Force pass\n");

  if (iter>1){

    ML_check_nei();

    printf("Check nei pass\n");

    printf("Signal = %d\n",signal);

  }
  
  if (iter==MD_IterNumber){

    ML_free_stastic();
    ML_free_dynamic();
    printf("Free pass\n");

  }

  dtime(&end_time);

  tot_time += end_time-start_time;

  if (iter==MD_IterNumber){
    printf("Tot time on %d = %8.6f\n",myid,tot_time);
  }


void ML_force_parallel()
{

  int i,i_,j,j_,k,k_,p,axis,species,centra_gnum,myid;
  int nei_num,nei_num1,nei_num2,nei_gnum1,nei_gnum2,nei_cnum1,nei_cnum2,count_para,count_ang,r_cut,nei_num_;
  double test,ref_energy,energy_plus,dis_shift1,dis_shift2,ang_shift,force_pre,start_time,end_time;

  /* MPI */

  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Initial force array */

  for (i=1;i<=atomnum;i++){
    for (j=1;j<=3;j++){
      model_force_sep[i][j] = 0;
    }
  }

  for (i=1;i<=atomnum;i++){
    for (j=1;j<=3;j++){
      model_force[i][j] = 0;
    }
  }

  /* Gradient respec to all neighbor atom */

  for (i=1;i<=Matomnum;i++){

    centra_gnum = M2G[i];
    nei_num = nei_info[i][0][0];
    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];
    count_para = 1;

    /* Self gradient */

    /* Two-body contribution */

    for (j=1;j<=nei_num;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (p=Min_order;p<=Max_order;p++){
        for (axis=1;axis<=3;axis++){

          model_force_sep[centra_gnum][axis] += current_model[i][count_para]*pow(nei_dis[i][j],p-2)*(Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-atv[nei_cnum1][axis])*\
                                                (p*cut_off_parallel(nei_dis[i][j],r_cut,0)-nei_dis[i][j]*cut_off_parallel(nei_dis[i][j],r_cut,1));

        }

        count_para += 1;

      }
    }

    /* Three-body contribution */
    
    count_ang = 1;
    
    for (j=1;j<=nei_num-1;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (k=j+1;k<=nei_num;k++){

        nei_gnum2 = nei_info[i][k][1];
        nei_cnum2 = nei_info[i][k][2];

        for (p=Min_order;p<=Max_order;p++){
          for (axis=1;axis<=3;axis++){

            force_pre = +current_model[i][count_para]*p*pow(nei_ang[i][count_ang],p)*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/pow(nei_dis[i][k],2)\
                        +current_model[i][count_para]*p*pow(nei_ang[i][count_ang],p)*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/pow(nei_dis[i][j],2)\
                        +current_model[i][count_para]*p*pow(nei_ang[i][count_ang],p-1)*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)*(2*Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-Gxyz[nei_gnum2][axis]-atv[nei_cnum1][axis]-atv[nei_cnum2][axis])/(nei_dis[i][j]*nei_dis[i][k])\
                        +current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*cut_off_parallel(nei_dis[i][j],r_cut,1)*cut_off_parallel(nei_dis[i][k],r_cut,0)*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])/nei_dis[i][j]\
                        +current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,1)*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])/nei_dis[i][k];

            model_force_sep[centra_gnum][axis] += force_pre;
                      
          }

          count_para ++;

        }

        count_ang ++;

      }
      
    }

    /* Gradient respect to neighbor atom */

    count_para = 1;

    /* Two body contribution */

    for (j=1;j<=nei_num;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (p=Min_order;p<=Max_order;p++){
        for (axis=1;axis<=3;axis++){

          model_force_sep[nei_gnum1][axis] += current_model[i][count_para]*pow(nei_dis[i][j],p-2)*(Gxyz[centra_gnum][axis]-Gxyz[nei_gnum1][axis]-atv[nei_cnum1][axis])*\
                                           (-p*cut_off_parallel(nei_dis[i][j],r_cut,0)+nei_dis[i][j]*cut_off_parallel(nei_dis[i][j],r_cut,1));

        }

        count_para += 1;

      }
    }

    /* Three body contribution */

    count_ang = 1;

    for (j=1;j<=nei_num-1;j++){

      nei_gnum1 = nei_info[i][j][1];
      nei_cnum1 = nei_info[i][j][2];

      for (k=j+1;k<=nei_num;k++){

        nei_gnum2 = nei_info[i][k][1];
        nei_cnum2 = nei_info[i][k][2];

        for(p=Min_order;p<=Max_order;p++){
          for (axis=1;axis<=3;axis++){

            model_force_sep[nei_gnum1][axis] += current_model[i][count_para]*pow(nei_ang[i][count_ang],p-1)*p*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)/(nei_dis[i][j]*nei_dis[i][k])\
                                         -current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*p*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)/pow(nei_dis[i][j],2)\
                                         -current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,1)*cut_off_parallel(nei_dis[i][k],r_cut,0)/nei_dis[i][j];

            model_force_sep[nei_gnum2][axis] += current_model[i][count_para]*pow(nei_ang[i][count_ang],p-1)*p*(atv[nei_cnum1][axis]+Gxyz[nei_gnum1][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)/(nei_dis[i][j]*nei_dis[i][k])\
                                         -current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*p*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,0)/pow(nei_dis[i][k],2)\
                                         -current_model[i][count_para]*pow(nei_ang[i][count_ang],p)*(atv[nei_cnum2][axis]+Gxyz[nei_gnum2][axis]-Gxyz[centra_gnum][axis])*cut_off_parallel(nei_dis[i][j],r_cut,0)*cut_off_parallel(nei_dis[i][k],r_cut,1)/nei_dis[i][k];

          }

          count_para += 1;

        }

        count_ang += 1;

      }
    }

  }

  for (i=1;i<=atomnum;i++){

    MPI_Allreduce(&model_force_sep[i][0], &model_force[i][0], 4, MPI_DOUBLE, MPI_SUM, mpi_comm_level1); // for loop

  }
  /*
  if (myid == Host_ID){
    for (i=1;i<=atomnum;i++){
      printf("Fit Force of %d\n",i);
      for (axis=1;axis<=3;axis++){
        printf("%8.6f ",model_force[i][axis]);
      }
      printf("\n");

      printf("Ref Force of %d\n",i);
      for (axis=17;axis<=19;axis++){
        printf("%8.6f ",Gxyz[i][axis]);
      }
      printf("\n");
    }
  }
  */
}