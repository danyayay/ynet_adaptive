list_seed=(1 2 3)
batch_size=10
val_split=0


dataset_path="filter/avg_vel/Pedestrian"
val_files="0.25_0.5.pkl"
n_leftouts=128
n_fusion=2
pretrained_ckpt=ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion_${n_fusion}.pt

python test_fusion.py --seed 1 --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt jjj --n_leftouts $n_leftouts --n_fusion $n_fusion


# list_n_train=(20 40 80 160)
# list_lr=(0.005)
# list_train_net=(lora_1)
# list_position=("scene" "scene_fusion" "motion" "motion_fusion" "scene_motion_fusion")
# list_tuned_ckpt=()
# for n_train in ${list_n_train[@]}; do
#     for lr in ${list_lr[@]}; do
#         for train_net in ${list_train_net[@]}; do 
#             for position in ${list_position[@]}; do
#                 list_tuned_ckpt+=("ckpts_fusion/lora/position_seed1_fusion2/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__fusion_${n_fusion}.pt")
#             done 
#         done 
#     done
# done

for seed in ${list_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_fusion $n_fusion
    done 
done 

