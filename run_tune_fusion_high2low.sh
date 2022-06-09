list_train_seed=(1) 
n_epoch=100
batch_size=10
val_split=50   # validation split 
n_fusion=2
n_round=3
train_files=0.5_2.25.pkl
val_files=0.5_2.25.pkl

#### pretrained model 
pretrained_ckpt=ckpts_fusion/Seed_1__Tr_2.75_7.5__Val_2.75_7.5__ValRatio_0.1__filter_avg_vel_dc_013_Biker__train__fusion_${n_fusion}.pt

dataset_path=filter/avg_vel/dc_013/Biker/
n_leftouts=250  # number of test
ckpt_path=ckpts_fusion  # path to save tuned files

list_n_train_batch=(2 4 8)
list_lr=(0.0005)

#### finetuning (encoder) with different position
list_train_net=("scene" "motion" "scene_fusion" "motion_fusion" "scene_motion_fusion")  
for seed in ${list_train_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            for train_net in ${list_train_net[@]}; do
                CUDA_VISIBLE_DEVICES=1 python train_fusion.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --n_fusion $n_fusion --steps 20 --shuffle_data --n_round $n_round
            done 
        done 
    done 
done


# ##### finetune lora with different position
# list_train_net=(lora_1)
# list_position=("scene" "motion" "scene motion" "fusion" "scene motion fusion")  
# for seed in ${list_train_seed[@]}; do
#     for n_train_batch in ${list_n_train_batch[@]}; do
#         for lr in ${list_lr[@]}; do
#             for train_net in ${list_train_net[@]}; do
#                 for position in "${list_position[@]}"; do 
#                     python train_fusion.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --n_fusion $n_fusion --steps 20
#                 done 
#             done 
#         done 
#     done 
# done




