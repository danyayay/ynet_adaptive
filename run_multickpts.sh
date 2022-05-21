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
    ["ckpts"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["ckpts_name"]="INDG OODG FT ET")
# declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
#     ["ckpts"]="ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
#     ["ckpts_name"]="INDG OODG FT ET")
declare -A F=( ["dataset_path"]="filter/agent_type/" ["filename"]="Biker.pkl" ["n_test"]=500
    ["ckpts"]="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt" 
    ["ckpts_name"]="OODG FT ET")

# list_seed=(1)
# dataset_name=sdd

# dataset_path=filter/agent_type/deathCircle_0
# pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt
# tuned_ckpts="ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-1__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-2__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-3__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-4__TrN_20.pt"
# val_files=Biker.pkl
# n_leftouts=500

# for seed in ${list_seed[@]}; do
#     python -m evaluator.evaluate_multickpts --dataset_path $dataset_path --val_files $val_files --n_leftouts $n_leftouts --pretrained_ckpt $pretrained_ckpt --tuned_ckpts $tuned_ckpts
# done 


# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt --n_round 10 




# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --ckpts ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0__TN_160_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0-4__TN_160_weights.pt --ckpts_name OODG "encoder_0(160)" "encoder_0-4(160)" --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --n_round 10 

# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --ckpts ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0__TN_160_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0-4__TN_160_weights.pt --ckpts_name OODG "encoder_0(160)" "encoder_0-4(160)" --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --n_round 10 




# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0_1_2_3_4__TrN_20.pt  --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --n_round 10 

# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0_1_2_3_4__TrN_20.pt  --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --n_round 10 


# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0_1_2_3_4__TrN_20__AUG.pt  --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --n_round 10 



# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-4__TrN_20.pt --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --n_round 10 

# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-4__TrN_20.pt  --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --n_round 10 


python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.0005.pt --n_round 10 --given_meta_ids 5711 6060 5670 5334 6063 6269 5885 6310 5767 6322 5445 --viz 
