list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3
ckpt_path=ckpts
network=original
train_net=train 

########## if use sequentially split data 
dataset_path=filter/agent_type
train_files=Pedestrian.pkl
val_files=Pedestrian.pkl
val_split=0.1
test_splits=1500
load_data=sequential

for train_seed in ${list_train_seed[@]}; do
    python train.py --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --train_files $train_files --val_files $val_files --val_split $val_split --test_splits $test_splits --train_net $train_net --ckpt_path $ckpt_path --augment --n_round $n_round 
done 


# ########## if use predefined data
# dataset_path=filter/agent_type/Pedestrian
# load_data=predefined

# for seed in ${list_train_seed[@]}; do
#     python train.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --load_data $load_data --train_net $train_net --ckpt_path $ckpt_path --augment --n_round $n_round 
# done 