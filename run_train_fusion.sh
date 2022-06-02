list_seed=(1) 
batch_size=10
n_epoch=150

val_ratio=0.1
train_net=train 

declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100)
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990) 
declare -A B_=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.7_1.3.pkl" ["n_test"]=650) # 6670
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000)
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350)
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500)
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500)
declare -A AB=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl 0.5_1.5.pkl" ["n_test"]="100 990")

dataset_path=${E["dataset_path"]}
train_files=${E["filename"]}
val_files=${E["filename"]}
n_leftouts=${E["n_test"]}
ckpt_path=ckpts_fusion2

for seed in ${list_seed[@]}; do
    python train.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --ckpt_path $ckpt_path --n_fusion 2
done