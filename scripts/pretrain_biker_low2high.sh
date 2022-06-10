list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3
ckpt_path=ckpts_fusion
network=fusion
n_fusion=2
train_net=train 


########## if use predefined data
dataset_path=filter/avg_vel/dc_013/Biker/0.5_3.5
load_data=predefined

for seed in ${list_train_seed[@]}; do
    python train.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --load_data $load_data --train_net $train_net --n_fusion $n_fusion --ckpt_path $ckpt_path --augment --n_round $n_round 
done 
