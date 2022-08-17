#!/usr/bin/bash
#SBATCH --chdir /home/dli/master-project/ynet_adaptive_work/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 90G
#SBATCH --gres gpu:1
#SBATCH --time 15:00:00
#SBATCH --job-name ped2ped_scene134to2__pretrain
#SBATCH --output ped2ped_scene134to2__pretrain.out

source ~/.bashrc
conda activate ynetv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dli/anaconda3/lib/

bash scripts/inD/ped2ped_scene134to2/ynetmod/pretrain.sh