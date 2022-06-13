list_train_seed=(1) 
batch_size=10
n_epoch=300
n_round=3
ckpt_path=inD_ckpts
network=fusion
n_fusion=2
train_net=train 

########## if use predefined data
dataset_path=filter/agent_type/scene1/car_8_12
load_data=predefined


for train_seed in ${list_train_seed[@]}; do
    CUDA_VISIBLE_SCENE=0 python inD_train.py --config_filename 'inD_shortterm_train.yaml' --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --train_net $train_net  --n_fusion $n_fusion --ckpt_path $ckpt_path --augment --n_round $n_round 
done
