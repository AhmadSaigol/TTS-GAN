#!/usr/bin/bash
# just tell computer to use bash -- essential part of bash script!
# run 'which bash' in terminal to find the above directory.
# the bash script can be executed in terminal by './multiple_runs.sh'
# If you run into the issue of 'permission denied', run 'chmod u+x multiple_runs.sh' before executing the script.

# get path and name of the current script (optional, I use it to automatically restart script if python exits with exit code != 0, i.e., an error of some sort)
#SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
#SCRIPTNAME=${0##*/}
#THIS_SCRIPT="$SCRIPTPATH/$SCRIPTNAME"

# define a variable for the python script you want to run
#SCRIPT_DIR="$HOME/path/to/script/"
#SCRIPT_DIR="$HOME/DeployedProjects/ML_tool/"

# activate python environment, here using pyenv. Python calls will now use you environment for the runtime of the script
#source /home/cmw3307/.pyenv/versions/base/bin/activate

# set TF environment variables for runtime to reduce clutter in console output
#export TF_CPP_MIN_LOG_LEVEL=2  # 0: log all, 1: all but infos, 2: only errors, 3: log neither infos, warnings nor errors
#export TF_ENABLE_ONEDNN_OPTS=1
#export TF_GPU_THREAD_MODE=gpu_private
#export CUDA_VISIBLE_DEVICES=0 # which gpu to use

# go to script directory. If it does not exist, bash script will exit
#cd $SCRIPT_DIR || exit


#############################################################################
#### Case 1: run a single script. Rerun script if errors occur in Python ####
#############################################################################

# run python script and get exit status.
#python multiple_trainings_run.py
#exit_status=$?
#echo "exit ${exit_status}"

# restart script if exit status != 0. Again, I don't know how this works when you use the loop above!
#if [ "${exit_status}" -ne 0 ];  # check if exit status is != 0
#then
#    echo "exit ${exit_status}"
#    exec "$THIS_SCRIPT"  # rerun this script in new process
#fi
#echo "execution successful (EXIT 0)"

######################################################
#### Case 2: run a single python script in a loop ####
######################################################
# you can also do it in a loop, however I do not know how to handle automatic restart in case of errors then.
# you can also loop over a set of arguments for the python script. Take a look into python argumentparser for this. Below, I pass i as an argument to the python script

#NUMBER_OF_EXPERIMENTS=43
#NUMBER_OF_RETRIES=5 # number of times to try to re-run the script if error occurs

#count=0
#for (( i=0 ; i<$NUMBER_OF_EXPERIMENTS ; i++ ))
#do
#  python multiple_trainings_run.py --run_no $i
#  exit_status=$?#

#  if [ "${exit_status}" -ne 0 ];  # check if exit status is != 0
#  then
#    echo "encountered error in run no: ${i} exit code: ${exit_status}"
#    echo "Trying to perform the run again . . ."#

    # Rerun the script until exit status becomes zero (no error) or counter reaches number of retries
#    while [ "${exit_status}" -ne 0  -a  "${count}" -ne "${NUMBER_OF_RETRIES}" ]; do
#      echo "Retry Number: ${count}"
#      python multiple_trainings_run.py --run_no $i #run the script
#      exit_status=$? # get exit status
#      (( count += 1 )) # increment counter
#    done

    # if the script is returning an error even after re-running it multiple times
#    if [ "${exit_status}" -ne 0  -a  "${count}" -eq "${NUMBER_OF_RETRIES}" ];
#    then
#      echo "Exceeded number of tries (${NUMBER_OF_RETRIES}). Unable to complete run no: ${i} exit code: ${exit_status}. Skipping this run . . ."
#    fi

#    count=0 #reset counter
#  fi
#done

module load cuda/11.8

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-versions/11.8

export CUDA_VISIBLE_DEVICES=0

python train_GAN.py \
-gen_bs 32 \
-dis_bs 32 \
--world-size 1 \
--rank 0 \
--dataset UniMiB \
--bottom_width 8 \
--max_iter 500000 \
--img_size 32 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--g_heads 4 \
--d_depth 3 \
--g_depth 3 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--beta1 0.9 \
--beta2 0.999 \
--batch_size 32 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--class_name Running \
--channels 3 \
--seq_len 150 \
--exp_name Running_FULL

python train_GAN.py \
-gen_bs 32 \
-dis_bs 32 \
--world-size 1 \
--rank 0 \
--dataset particle_data \
--data_path "/home/ahmad/TimeGAN_tf2.16/timeGAN/data/PM2.5_bins_split_type_kfold_files_REFs_QTIs" \
--which_folder_data complete \
--fold_no 1 \
--upsample_Y \
--bottom_width 8 \
--max_iter 500000 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--g_heads 4 \
--d_depth 3 \
--g_depth 3 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--beta1 0.9 \
--beta2 0.999 \
--batch_size 32 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--class_name particle_data \
--channels 8 \
--seq_len 300 \
--exp_name particle_data_run_with_default_settings

python train_GAN.py \
-gen_bs 32 \
-dis_bs 32 \
--world-size 1 \
--rank 0 \
--dataset particle_data \
--data_path "/home/ahmad/TimeGAN_tf2.16/timeGAN/data/PM2.5_bins_split_type_kfold_files_REFs_QTIs" \
--which_folder_data complete \
--fold_no 1 \
--upsample_Y \
--bottom_width 8 \
--max_iter 500000 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--g_heads 4 \
--d_depth 2 \
--g_depth 2 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--beta1 0.9 \
--beta2 0.999 \
--batch_size 32 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--class_name particle_data \
--channels 8 \
--seq_len 300 \
--exp_name test_particle_data_run_with_2depth

python train_GAN.py \
-gen_bs 32 \
-dis_bs 32 \
--world-size 1 \
--rank 0 \
--dataset particle_data \
--data_path "/home/ahmad/TimeGAN_tf2.16/timeGAN/data/PM2.5_bins_split_type_kfold_files_REFs_QTIs" \
--which_folder_data complete \
--fold_no 1 \
--upsample_Y \
--bottom_width 8 \
--max_iter 500000 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--g_heads 4 \
--d_depth 4 \
--g_depth 4 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--optimizer adam \
--loss lsgan \
--beta1 0.9 \
--beta2 0.999 \
--batch_size 32 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--class_name particle_data \
--channels 8 \
--seq_len 300 \
--exp_name test_particle_data_run_with_4depth
