dataset_name=sdd
declare -A A=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl" ["n_test"]=100 
    ["ckpts"]="ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt"
    ["ckpts_name"]="INDG OODG FT ET")
declare -A B=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.5_1.5.pkl" ["n_test"]=990
    ["ckpts"]="ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
declare -A C=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="0_1.3.pkl" ["n_test"]=1000)
declare -A D=( ["dataset_path"]="filter/avg_den100/Pedestrian/" ["filename"]="1.7_4.3.pkl" ["n_test"]=350)
declare -A E=( ["dataset_path"]="filter/agent_type/" ["filename"]="Pedestrian.pkl" ["n_test"]=1500)
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500)

# python -m viz_same_data_diff_model --dataset_path filter/avg_vel/Pedestrian/ --ckpts ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.5_1.5__Val__0.5_1.5__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_all_FT_weights.pt ckpts/Seed_1_Train__0.1_0.3__Val__0.1_0.3__Val_Ratio_0.1_filter_avg_vel_Pedestrian__train_encoder_weights.pt --ckpts_name INDG OODG FT ET --val_files 0.5_1.5.pkl --n_leftouts 990

dataset_path=${A["dataset_path"]}
ckpts=${A["ckpts"]}
ckpts_name=${A["ckpts_name"]}
val_files=${A["filename"]}
n_leftouts=${A["n_test"]}

depth=0
  
python -m viz_same_data_diff_model --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --n_leftouts $n_leftouts --ckpts $ckpts --ckpts_name $ckpts_name --depth $depth
