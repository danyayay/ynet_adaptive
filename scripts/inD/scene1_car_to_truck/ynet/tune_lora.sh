# config 
list_train_seed=(10) 
batch_size=10
n_epoch=300
n_round=3
config_filename=inD_shortterm_train.yaml

# model 
network=original

# pretrained model 
pretrained_ckpt=ckpts/Seed_1__filter_agent_type_scene1_car_filter__train__fusion_2_weights.pt
ckpt_path=ckpts/inD/scene1_car_to_truck/ynet

# data path 
dataset_path=filter/agent_type/scene1/truck_bus_filter
load_data=predefined

# fine-tune setting 
list_train_net=(lora_1)
list_position=("scene" "motion" "scene fusion" "motion fusion" "scene motion fusion")  
list_n_train_batch=(2) 
list_lr=(0.001)


for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do
                    python inD_train.py --fine_tune --config_filename $config_filename --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_round $n_round --n_train_batch $n_train_batch --lr $lr --smooth_val 
                done 
            done 
        done 
    done 
done