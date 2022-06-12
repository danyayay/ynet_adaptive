list_train_seed=(1) 
batch_size=10
n_epoch=300
n_round=3
ckpt_path=inD_ckpt
network=original
train_net=train 

########## if use predefined data
dataset_path=filter/agent_type/scene1/car_8_12
load_data=predefined


for train_seed in ${list_train_seed[@]}; do
    python inD_train.py --config_filename 'inD_shortterm_train.yaml' --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --train_net $train_net --ckpt_path $ckpt_path --augment --n_round $n_round 
done
