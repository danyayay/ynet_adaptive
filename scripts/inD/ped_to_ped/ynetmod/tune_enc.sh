# config 
list_train_seed=(1 2 3 4 5) 
batch_size=10
n_epoch=100
n_round=3
config_filename=inD_longterm_train.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
pretrained_ckpt=ckpts/inD__ynetmod__ped_to_ped.pt
ckpt_path=ckpts/inD/ped_to_ped/ynetmod

# data path 
dataset_path=filter/agent_type/scene1/pedestrian_40
load_data=predefined

# fine-tune setting 
list_train_net=("scene" "motion" "scene_fusion" "motion_fusion" "scene_motion_fusion") 
list_n_train_batch=(3 4) 
list_lr=(0.0005)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in "${list_train_net[@]}"; do
                python train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --pretrained $pretrained_ckpt --train_net $train_net --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val
            done 
        done 
    done 
done 
