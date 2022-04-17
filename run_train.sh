list_seed=(1) # Train the model on different seeds
batch_size=8
n_epoch=100

dataset_name=sdd
out_csv_dir=csv # /path/to/csv where the output results are written to
val_ratio=0.1 # Split train dataset into a train and val split in case the domains are the same
train_net=all # Train either all parameters, only the encoder or the modulator: (all encoder modulator)

declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100)
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990) 
declare -A B_=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.7_1.3.pkl" ["n_test"]=650) # 6670
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000)
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350)
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500)
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500)
declare -A AB=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl 0.5_1.5.pkl" ["n_test"]="100 990")

dataset_path=${B_["dataset_path"]}
train_files=${B_["filename"]}
val_files=${B_["filename"]}
n_leftouts=${B_["n_test"]}

for seed in ${list_seed[@]}; do
    python train.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_name $dataset_name --dataset_path $dataset_path --train_files $train_files --val_files $val_files --out_csv_dir $out_csv_dir --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net
done