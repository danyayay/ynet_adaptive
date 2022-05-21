# list_seed=(1)
# dataset_name=sdd
# dataset_path=filter/agent_type/deathCircle_0
# pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt
# tuned_ckpts=ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt
# val_files=Biker.pkl
# n_leftouts=500
# # result_path='./csv/comparison/1__filter_agent_type_deathCircle_0__Biker/OODG_encoder_0(20)_encoder_0-1(20).csv'
# # result_name='ade_OODG__ade_encoder_0(20)__diff'
# # result_limited=20

# for seed in ${list_seed[@]}; do
#     python -m evaluator.visualize_activation --seed $seed --dataset_path $dataset_path --pretrained_ckpt $pretrained_ckpt --tuned_ckpts $tuned_ckpts --val_files $val_files --n_leftouts $n_leftouts --compare_diff --compare_overlay # --result_path $result_path --result_name $result_name --result_limited $result_limited
# done 


# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --ckpts ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0__TN_160_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0-4__TN_160_weights.pt --ckpts_name OODG "encoder_0(160)" "encoder_0-4(160)" --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --compare_relative

# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --ckpts ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0__TN_160_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_encoder_0-4__TN_160_weights.pt --ckpts_name OODG "encoder_0(160)" "encoder_0-4(160)" --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --compare_relative



# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0_1_2_3_4__TrN_20.pt --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --compare_raw --compare_diff --compare_overlay --compare_relative

# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_parallel__0_1_2_3_4__TrN_20.pt  --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --compare_raw --compare_diff --compare_overlay --compare_relative



# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-4__TrN_20.pt --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5334 5445 5466 5607 5635 5711 5726 5767 5775 5885 --compare_raw --compare_diff --compare_overlay --compare_relative

# python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-4__TrN_20.pt  --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5890 5972 5982 6060 6063 6098 6228 6252 6269 6310 --compare_raw --compare_diff --compare_overlay --compare_relative


python -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.0005.pt --val_files Biker.pkl --n_leftouts 500 --given_meta_ids 5358 5883 5982
