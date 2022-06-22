# config 
list_train_seed=(1) 
batch_size=10
n_epoch=1
n_round=1
config_filename=sdd_train.yaml
ckpt_path=ckpts

# model 
network=fusion
n_fusion=2
train_net=train 

# data path 
dataset_path=filter/avg_vel/dc_013/Biker/0.5_3.5
load_data=predefined


for train_seed in ${list_train_seed[@]}; do
    python train.py --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --train_net $train_net --n_fusion $n_fusion --ckpt_path $ckpt_path --augment --n_round $n_round --n_train_batch 1 
done 
