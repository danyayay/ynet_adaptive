#!/usr/bin/bash
#SBATCH --chdir /home/dli/master-project/ynet_adaptive_work/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 90G
#SBATCH --gres gpu:1
#SBATCH --time 11:00:00
#SBATCH --job-name DC_bias_i_lr_train
#SBATCH --output DC_bias_i_lr_train.out

source ~/.bashrc
conda activate ynetv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dli/anaconda3/lib/

bash run_tune.sh