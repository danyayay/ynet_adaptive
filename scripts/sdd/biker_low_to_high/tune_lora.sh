# config 
list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3
config_filename=sdd_train.yaml

# model 
network=fusion
n_fusion=2

# pretrained model
pretrained_ckpt=ckpts_fusion/Seed_1__Tr_0.5_3.5__Val_0.5_3.5__ValRatio_0.1__filter_avg_vel_dc_013_Biker__train__fusion_2.pt
ckpt_path=ckpts/sdd/biker_low_to_high

# data path 
dataset_path=filter/avg_vel/dc_013/Biker/4_8
load_data=predefined

# fine-tune setting  
list_n_train_batch=(0.1 0.5 1 2 4) 
list_lr=(0.0005)
list_train_net=(lora_1 lora_2 lora_4)
list_position=("scene" "motion" "scene fusion" "motion fusion" "scene motion fusion")  


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do 
                    python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --n_fusion $n_fusion --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val --steps 20 
                done 
            done 
        done 
    done 
done 
