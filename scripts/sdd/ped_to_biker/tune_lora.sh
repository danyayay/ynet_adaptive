# config 
list_train_seed=(1) 
batch_size=10
n_epoch=120
n_round=3
config_filename=sdd_train.yaml

# model 
network=original

# pretrained model 
pretrained_ckpt=ckpts/sdd_trajnet_weights.pt
ckpt_path=ckpts/sdd/ped_to_biker

# data path 
dataset_path=filter/agent_type/deathCircle_0/Biker
load_data=predefined

# fine-tune setting 
list_train_net=(lora_1 lora_2 lora_4)
list_position=("0 1 2 3 4") 
list_n_train_batch=(0.5 1 2 4 8) 
list_lr=(0.005 0.003 0.001 0.0005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do
                    python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr 
                done 
            done 
        done 
    done 
done