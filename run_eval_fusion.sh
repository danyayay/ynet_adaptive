list_seed=(1 2 3)
batch_size=10
val_ratio=0

declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100 \
    ["ckpt"]="ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt")
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990 \
    ["ckpt"]="ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt")
declare -A AB=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl 0.5_1.5.pkl" ["n_test"]="100 990" \
    ["ckpt"]="ckpts/Seed_1_Train__0.1_0.3__0.5_1.5__Val__0.1_0.3__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt")
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000 \
    ["ckpt"]="ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt")
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350 \
    ["ckpt"]="ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt")
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500 \
    ["ckpt"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt")
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500 \
    ["ckpt"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt")

# ## pretrained_on: 
# dataset_path=${AB["dataset_path"]}
# ckpt=${AB["ckpt"]}
# # ## evaluate_on
# dataset_path=${F["dataset_path"]}
# val_files=${F["filename"]}
# n_leftouts=${F["n_test"]}

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")





dataset_path=filter/agent_type/deathCircle_0/

# ############ fusion model (n_fusion=3)
ckpts=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion3.pt
ckpts_name=OODG
# evaluate on ped
val_files=Pedestrian.pkl
n_leftouts=1500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 3
done 
# evaluate on biker 
val_files=Biker.pkl
n_leftouts=500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 3
done 


# ############ fusion model (n_fusion=4)
ckpts=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion4.pt
ckpts_name=OODG
# evaluate on ped
val_files=Pedestrian.pkl
n_leftouts=1500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 4
done 
# evaluate on biker 
val_files=Biker.pkl
n_leftouts=500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 4
done 


# ############ fusion model (n_fusion=4)
ckpts=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion2.pt
ckpts_name=OODG
# evaluate on ped
val_files=Pedestrian.pkl
n_leftouts=1500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 2
done 
# evaluate on biker 
val_files=Biker.pkl
n_leftouts=500
for seed in ${list_seed[@]}; do
    python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name $ckpts_name --n_leftouts $n_leftouts --n_fusion 2
done 




# list_n_train=(20 40 80 160)
# list_lr=(0.0005 0.00005)
# list_train_net=(lora_1)
# position="0_1_2_3_4"
# list_tuned_ckpt=()
# for n_train in ${list_n_train[@]}; do
#     for lr in ${list_lr[@]}; do
#         for train_net in ${list_train_net[@]}; do 
#             list_tuned_ckpt+=("ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}.pt")
#         done 
#     done
# done

# for seed in ${list_seed[@]}; do
#     for tuned_ckpt in ${list_tuned_ckpt[@]}; do
#         python test_copy.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --swap_semantic
#     done 
# done 

# python test.py --seed 1 --batch_size 8 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpt "ckpts/DC0__lora/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.0005.pt" --n_leftouts 10  --study_semantic pavement_terrain
