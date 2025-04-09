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

export CUDA_VISIBLE_DEVICES=1

python generate_synthetic_data.py
