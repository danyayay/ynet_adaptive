list_seed=(1)
dataset_name=sdd
declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100 
    ["ckpts"]="ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt"
    ["ckpts_name"]="INDG OODG FT ET")
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990
    ["ckpts"]="ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000
    ["ckpts"]="ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350
    ["ckpts"]="ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0_1.3__Val__0_1.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__1.7_4.3__Val__1.7_4.3__Val_Ratio_0.1_filter_avg_den100_Pedestrian__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500
    ["ckpts"]="ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
# declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
#     ["ckpts"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
#     ["ckpts_name"]="INDG OODG FT ET")
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["ckpts"]="ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["ckpts_name"]="OODG FT ET")

# python -m pdb test_ckpts.py --dataset_path filter/avg_vel/Pedestrian/ --ckpts ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt --ckpts_name OODG FT ET --val_files 0.1_0.3.pkl --n_leftouts 10 --viz --depth 2

# python -m pdb test_ckpts.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt --ckpts_name OODG FT --val_files Biker.pkl --n_leftouts 500 --viz --depth 2

# python -m pdb test_ckpts.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --ckpts_name OODG FT ET --val_files Biker.pkl --n_leftouts 10 --viz --depth 1

depth=1

dataset_path=${F["dataset_path"]}
ckpts=${F["ckpts"]}
ckpts_name=${F["ckpts_name"]}
val_files=${F["filename"]}
n_leftouts=${F["n_test"]}

for seed in ${list_seed[@]}; do
    python -m test_ckpts --seed $seed --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --n_leftouts $n_leftouts --ckpts $ckpts --ckpts_name $ckpts_name --depth $depth --viz --limit_by n_viz --n_viz 150
done 
