# config 
list_train_seed=(1) 
batch_size=10
n_epoch=1
n_round=1
config_filename=sdd_train.yaml

# model 
network=original

# pretrained model 
pretrained_ckpt=ckpts/sdd__ynet__ped_to_biker.pt
ckpt_path=ckpts/sdd/ped_to_biker

# data path 
dataset_path=filter/agent_type/deathCircle_0
train_files=Biker.pkl
val_files=Biker.pkl
val_split=80
test_splits=500
load_data=sequential

# fine-tune setting 
train_net=all
list_n_train_batch=(1) 
list_lr=(0.00005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --train_files $train_files --val_files $val_files --val_split $val_split --test_splits $test_splits --pretrained $pretrained_ckpt --train_net $train_net --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr
        done 
    done 
done 