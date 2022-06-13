list_train_seed=(1 2 3 4 5) 
batch_size=10
n_epoch=120
n_round=3
ckpt_path=inD_ckpts_fusion
network=fusion
n_fusion=2
train_net=train 

list_n_train_batch=(2) 
list_train_net=(lora_1)
# list_position=("0 1 2 3 4")  # non-baseline models need position 
list_position=("scene" "motion" "scene fusion" "motion fusion" "scene motion fusion")  # non-baseline models need position 
# list_position=("scene" "motion" "scene motion fusion")  # non-baseline models need position 
# list_position=("scene motion fusion")  # non-baseline models need position 
list_lr=(0.005)
# list_lr=(0.00005)

# pretrained_ckpt=${ckpt_path}/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
# pretrained_ckpt=${ckpt_path}/inD_longterm_weights.pt
# pretrained_ckpt=${ckpt_path}/Seed_1__filter_agent_type_scene234_pedestrian___train__original_weights.pt
pretrained_ckpt=${ckpt_path}/Seed_1__filter_agent_type_scene234_pedestrian___train__fusion_2_weights.pt


########## if use sequentially split data 
dataset_path=filter/agent_type/scene1/pedestrian
train_files=pedestrian.pkl
val_files=pedestrian.pkl
val_split=40
test_splits=50
# load_data=sequential
load_data=predefined

for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do
                    python inD_train.py --fine_tune --position $position --n_fusion $n_fusion --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --train_files $train_files --val_files $val_files --val_split $val_split --test_splits $test_splits --pretrained $pretrained_ckpt --train_net $train_net --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --smooth_val
                done 
            done 
        done 
    done 
done 

# --position $position 