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
pretrained_ckpt=ckpts/Seed_1__filter_avg_vel_dc_013_Biker_2.75_7.5__train__fusion_2.pt
ckpt_path=ckpts/sdd/biker_high_to_low

# data path 
dataset_path=filter/avg_vel/dc_013/Biker/0.5_2.25
load_data=predefined

# fine-tune setting 
list_train_net=("scene" "motion" "scene_fusion" "motion_fusion" "scene_motion_fusion")
list_n_train_batch=(1) 
list_lr=(0.005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --n_fusion $n_fusion --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr 
                done 
            done 
        done 
    done 
done 

