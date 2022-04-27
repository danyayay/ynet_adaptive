list_seed=(1)
dataset_name=sdd
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["pretrained"]="ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt" ["tuned"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["tuned_name"]="FT ET")

# python -m importance_analysis --dataset_path filter/agent_type/ --files Biker.pkl --n_leftouts 20 --pretrained ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --tuned_name FT ET

dataset_path=${F["dataset_path"]}
pretrained=${F["pretrained"]}
tuned=${F["tuned"]}
tuned_name=${F["tuned_name"]}
files=${F["filename"]}
n_leftouts=${F["n_test"]}

for seed in ${list_seed[@]}; do
    python -m importance_analysis --seed $seed --dataset_name $dataset_name --dataset_path $dataset_path --files $files --n_leftouts $n_leftouts --pretrained $pretrained --tuned $tuned --tuned_name $tuned_name
done 