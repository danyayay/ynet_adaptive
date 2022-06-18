list_train_seed=(1) 
batch_size=10
n_epoch=300
n_round=3
config_filename=inD_longterm_train.yaml
ckpt_path=ckpts/inD/ped_to_ped/ynetmod

# model 
network=fusion
n_fusion=2
train_net=train 

# data path 
dataset_path=filter/agent_type/scene234/pedestrian
load_data=predefined


for train_seed in ${list_train_seed[@]}; do
    python train.py --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --train_net $train_net --ckpt_path $ckpt_path --augment 
done 
