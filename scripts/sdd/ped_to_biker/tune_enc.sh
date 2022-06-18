# config 
list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3
config_filename=sdd_train.yaml

# model 
network=original

# pretrained_ckpt
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
train_net=encoder
list_position=("0 1 2 3 4")  
list_n_train_batch=(0.5 1 2 4 8) 
list_lr=(0.0005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for position in "${list_position[@]}"; do
                python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr
            done 
        done 
    done 
done