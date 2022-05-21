list_seed=(1)
dataset_name=sdd

declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["pretrained"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt" ["tuned"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["tuned_name"]="FT ET")

declare -A F_FT=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["pretrained"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt" ["tuned"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt" 
    ["tuned_name"]="FT")

declare -A F_ET=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["pretrained"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt" ["tuned"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["tuned_name"]="ET")

# python -m pdb importance_analysis.py --dataset_path filter/agent_type/ --files Biker.pkl --n_leftouts 20 --pretrained ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --tuned_name FT ET --depth 1 --replace_pretrained

# python -m pdb importance_analysis.py --dataset_path filter/agent_type/ --files Biker.pkl --n_leftouts 10 --pretrained ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --tuned_name ET --depth 1 --replace_pretrained --generate_csv

dataset_path=${F["dataset_path"]}
pretrained=${F["pretrained"]}
# tuned=${F["tuned"]}
# tuned_name=${F["tuned_name"]}
files=${F["filename"]}
n_leftouts=${F["n_test"]}

list_depth=(1 2 -1)

for seed in ${list_seed[@]}; do
    # ET
    python -m importance_analysis --seed $seed --dataset_name $dataset_name --dataset_path $dataset_path --files $files --n_leftouts $n_leftouts --pretrained $pretrained --tuned ${F_ET["tuned"]} --tuned_name ${F_ET["tuned_name"]} --depth -1 --replace_pretrained --generate_csv
    # FT
    for depth in ${list_depth[@]}; do
        python -m importance_analysis --seed $seed --dataset_name $dataset_name --dataset_path $dataset_path --files $files --n_leftouts $n_leftouts --pretrained $pretrained --tuned ${F_FT["tuned"]} --tuned_name ${F_FT["tuned_name"]} --depth $depth --replace_pretrained --generate_csv
    done 
done 