list_seed=(1) 
n_epoch=100
batch_size=10
list_n_train_batch=(1 2 4 20 40 200) 

dataset_name=sdd
out_csv_dir=csv 
val_ratio=0.1
train_net=encoder

declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100 \
    ["ckpt"]="ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt")
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990 \
    ["ckpt"]="ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt")
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000 \
    ["ckpt"]="ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt")
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350 \
    ["ckpt"]="ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt")
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500 \
    ["ckpt"]="ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt")
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500 \
    ["ckpt"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt")

# ## pretrained_on: 
dataset_path=${E["dataset_path"]}
ckpt=${E["ckpt"]}
# ## finetune_encoder_of
train_files=${F["filename"]}
val_files=${F["filename"]}
n_leftouts=${F["n_test"]}

for seed in ${list_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        python train.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_name $dataset_name --dataset_path $dataset_path --out_csv_dir $out_csv_dir --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --ckpt $ckpt --lr 0.00005 --n_train_batch $n_train_batch
    done
done