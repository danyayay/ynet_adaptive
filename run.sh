#!/usr/bin/bash
#SBATCH --chdir /home/dli/master-project/ynet_adaptive_work/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 90G
#SBATCH --gres gpu:1

# module load gcc python 
# source /home/dli/venvs/ynet/bin/activate

source ~/.bashrc
conda activate ynetv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dli/anaconda3/lib/

bash run_train.sh