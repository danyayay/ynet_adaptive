# config 
list_train_seed=(1) 
batch_size=10
n_epoch=1
n_round=1
config_filename=sdd_train.yaml
steps=20

# model 
network=fusion
n_fusion=2

# pretrained model
pretrained_ckpt=ckpts/sdd__ynetmod__biker_low_to_high.pt
ckpt_path=ckpts/sdd/biker_low_to_high

# data path 
dataset_path=filter/avg_vel/dc_013/Biker/4_8
load_data=predefined

# fine-tune setting  
list_train_net=(lora_1)
list_position=("scene")  
list_n_train_batch=(1) 
list_lr=(0.0005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do 
                    python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --n_fusion $n_fusion --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val --steps $steps 
                done 
            done 
        done 
    done 
done 
