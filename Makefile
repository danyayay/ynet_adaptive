@ inD_make_data_35:
	# Construct
	# python -m utils.inD_dataset

	# Filter Agents and Scenes
	python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene1'
	python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene2'
	python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene3'
	python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene4'

	python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene1'
	python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene2'
	python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene3'
	python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene4'

	python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene1'
	python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene2'
	python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene3'
	python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene4'

	python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene1'
	python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene2'
	python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene3'
	python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene4'


@ inD_make_data_20:
	# Construct
	# python -m utils.inD_dataset --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8

	# Filter Agents and Scenes
	# python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene1' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene2' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene3' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'pedestrian' --selected_scenes 'scene4' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8

	# python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene1' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene2' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene3' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'car' --selected_scenes 'scene4' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8

	# python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene1' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene2' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene3' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'bicycle' --selected_scenes 'scene4' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8

	# python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene1' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene2' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene3' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8
	# python -m utils.inD_dataset --reload --labels 'truck_bus' --selected_scenes 'scene4' --raw_data_filename 'data_8_12_2_5fps.pkl' --step 10 --window_size 20 --stride 20 --obs_len 8

@ ynet_1:
	# bash run_tune_embed.sh
	# bash run_all_tune.sh
	# bash run_enc_tune.sh
	# bash inD_scripts/inD_ped_ped/lora_1.sh > inD_lora_30_first_second_third_train.out
	# bash inD_scripts/inD_ped_ped/lora_2.sh > inD_lora_10_20.out
	# bash inD_scripts/inD_ped_ped/enc_tune.sh > inD_enc_30_third_train.out

	# sdd
	# bash scripts/sdd_ped_to_biker/encoder_tune.sh > sdd_enc_30_train.out
	# bash scripts/sdd_ped_to_biker/finetune.sh > sdd_ft_30_train.out
	# bash scripts/sdd_ped_to_biker/pa_layer_tune.sh > sdd_pa_30_train.out
	# bash scripts/sdd_ped_to_biker_changing/pa.sh > sdd_pa_1_2_3_train.out
	# bash scripts/sdd_ped_to_biker_changing/finetune.sh > sdd_ft_1_2_3_train.out
	bash scripts/sdd_ped_to_biker/finetune.sh > sdd_old_bs_ft_1_2_3_train.out

@ ynet_2:
	# bash run_enc_tune.sh
	# bash run_pa_layer_tune.sh
	# bash inD_scripts/inD_ped_ped/finetune.sh > inD_ft_30_first_second_third_train.out
	# bash inD_scripts/inD_ped_ped/enc_tune.sh > inD_enc_30_40.out
	# bash inD_scripts/inD_ped_ped/pa_layer_tune.sh > inD_pa_30_40.out
	# bash inD_scripts/inD_ped_ped/pa_layer_tune.sh > inD_pa_30_third_train.out
	# bash scripts/sdd_ped_to_biker/pa_layer_tune.sh > sdd_pa_30_train.out
	# bash scripts/sdd_ped_to_biker/lora_4.sh > sdd_lora_4_10_20_train.out
	# bash scripts/sdd_ped_to_biker_changing/encoder.sh > sdd_enc_1_2_3_train.out
	bash scripts/sdd_ped_to_biker/pa_layer_tune.sh > sdd_old_bs_pa_1_2_3_train.out
