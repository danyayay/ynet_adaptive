# Construct short-term data 
python -m utils.sdd_dataset 

# Experiment: sdd ped to biker expt
python -m utils.sdd_dataset --reload --varf agent_type --labels Pedestrian Biker

python -m utils.sdd_dataset --reload --varf agent_type --labels Biker --selected_scenes deathCircle_0

python -m utils.split_dataset --data_dir data/sdd/filter/shortterm/agent_type/deathCircle_0 --data_filename Biker.pkl --val_split 80 --test_split 500 --seed 1

# Experiment: sdd biker low to high 
python -m utils.sdd_dataset --reload --varf agent_type --labels Biker --selected_scenes deathCircle_0 deathCircle_1 deathCircle_3 

mv data/sdd/filter/shortterm/agent_type/deathCircle_0__deathCircle_1__deathCircle_3 data/sdd/filter/shortterm/agent_type/dc_013
rm -r data/sdd/filter/shortterm/agent_type/deathCircle_1 data/sdd/filter/shortterm/agent_type/deathCircle_3 

python -m utils.sdd_dataset --reload --raw_data_dir data/sdd/filter/shortterm/agent_type/dc_013 --raw_data_filename Biker.pkl --varf avg_vel --labels Biker

mkdir data/sdd/filter/shortterm/avg_vel/dc_013
mv data/sdd/filter/shortterm/avg_vel/Biker data/sdd/filter/shortterm/avg_vel/dc_013/

python -m utils.split_dataset --data_dir data/sdd/filter/shortterm/avg_vel/dc_013/Biker --data_filename 0.5_3.5.pkl --val_split 0.1 --test_split 250 --seed 1 

python -m utils.split_dataset --data_dir data/sdd/filter/shortterm/avg_vel/dc_013/Biker --data_filename 4_8.pkl --val_split 50 --test_split 250 --seed 1 




# Long-term split 
python -m utils.sdd_dataset --additional_data_dir /data/dli-data/sdd/raw --raw_data_dir /data/dli-data/sdd/raw --raw_data_filename data_5_30_1fps.pkl --step 30 --window_size 35 --stride 35 --obs_len 5 --varf agent_type --labels Pedestrian Biker --filter_data_dir /data/dli-data/sdd/filter/longterm 

python -m utils.split_dataset --data_dir /data/dli-data/sdd/filter/longterm/agent_type --data_filename Pedestrian.pkl --val_split 0.1 --test_split 0.2 --seed 1 

python -m utils.filter_dataset --data_path /data/dli-data/sdd/filter/longterm/agent_type/Pedestrian.pkl --varf_path /data/dli-data/sdd/raw/varf_8_12_2_5fps.pkl --lower_bound 0.2

python -m utils.split_dataset --data_dir /data/dli-data/sdd/filter/longterm/agent_type --data_filename Pedestrian_filter.pkl --val_split 0.1 --test_split 0.2 --seed 1 
