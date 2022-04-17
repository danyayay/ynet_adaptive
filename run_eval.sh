list_seed=(1 2 3)
batch_size=10

dataset_name=sdd
out_csv_dir=csv 
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
    ["ckpt"]="ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt")
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500 \
    ["ckpt"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt")

# ## pretrained_on: 
# dataset_path=${AB["dataset_path"]}
# ckpt=${AB["ckpt"]}
# # ## evaluate_on
# val_files=${B["filename"]}
# n_leftouts=${B["n_test"]}


# for seed in ${list_seed[@]}
# do
#     python test.py --seed $seed --batch_size $batch_size --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --out_csv_dir $out_csv_dir --val_ratio $val_ratio --ckpt $ckpt --n_leftouts $n_leftouts
# done


dataset_path="filter/agent_type/"
val_files="Biker.pkl"
n_leftouts=500
list_n_train=(10 20 40 200 400 2000)
list_train_net=("all_FT" "encoder")
list_ckpt=()
for n_train in ${list_n_train[@]}; do
    for train_net in ${list_train_net[@]}; do
        list_ckpt+=("ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_${train_net}_${n_train}_weights.pt")
    done
done


for seed in ${list_seed[@]}; do
    for ckpt in ${list_ckpt[@]}; do
        python test.py --seed $seed --batch_size $batch_size --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --out_csv_dir $out_csv_dir --val_ratio $val_ratio --ckpt $ckpt --n_leftouts $n_leftouts
    done 
done



# ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model
