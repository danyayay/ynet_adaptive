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


# for seed in ${list_seed[@]}
# do
#     python test.py --seed $seed --batch_size $batch_size --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --out_csv_dir $out_csv_dir --val_ratio $val_ratio --ckpt $ckpt --n_leftouts $n_leftouts
# done


# dataset_path="filter/agent_type/"
# val_files="Biker.pkl"
# n_leftouts=500
# list_n_train=(10 20 40 200 400 2000)
# list_train_net=("all_FT" "encoder")
# list_ckpt=()
# for n_train in ${list_n_train[@]}; do
#     for train_net in ${list_train_net[@]}; do
#         list_ckpt+=("ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_${train_net}_${n_train}_weights.pt")
#     done
# done

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_tuned_ckpt=()
# for n_train in ${list_n_train[@]}; do
#     for train_net in ${list_train_net[@]}; do
#         list_tuned_ckpt+=("ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__TrN_${n_train}.pt")
#     done
# done

# dataset_path=filter/agent_type/deathCircle_0
# val_files=Biker.pkl
# n_leftouts=500
# pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt
# list_train_net=(adapter_serial)
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")
# list_n_train=(10 20 30 40)
# list_tuned_ckpt=()
# for train_net in ${list_train_net[@]}; do
#     for adapter_position in ${list_adapter_position[@]}; do
#         for n_train in ${list_n_train[@]}; do
#             list_tuned_ckpt+=("ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__${adapter_position}__TrN_${n_train}.pt")
#         done
#     done 
# done

dataset_path=filter/agent_type/deathCircle_0/
val_files=Biker.pkl
n_leftouts=500
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt

list_train_net=(lora_1)
list_n_train=(20 40) 
list_lr=(0.05 0.001 0.005)
list_tuned_ckpt=()
for train_net in ${list_train_net[@]}; do
    for n_train in ${list_n_train[@]}; do
        for lr in ${list_lr[@]}; do
            list_tuned_ckpt+=("ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_0_1_2_3_4__TrN_${n_train}__lr_${lr}__bias.pt")
        done 
    done 
done

for seed in ${list_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do 
        python test.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts
    done 
done 
