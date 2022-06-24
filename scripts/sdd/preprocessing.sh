# Construct short-term data 
python -m utils.sdd_dataset 

# Experiment: sdd ped to biker expt
python -m utils.sdd_dataset --reload --varf agent_type --labels Pedestrian Biker

python -m utils.sdd_dataset --reload --varf agent_type --labels Biker --selected_scenes deathCircle_0

python -m utils.split_dataset --data_dir data/sdd/filter/agent_type/deathCircle_0 --data_filename Biker.pkl --val_split 80 --test_split 500 --seed 1

# Experiment: sdd biker low to high 
python -m utils.sdd_dataset --reload --varf agent_type --labels Biker --selected_scenes deathCircle_0 deathCircle_1 deathCircle_3 

mv data/sdd/filter/agent_type/deathCircle_0__deathCircle_1__deathCircle_3 data/sdd/filter/agent_type/dc_013
rm -r data/sdd/filter/agent_type/deathCircle_1 data/sdd/filter/agent_type/deathCircle_3 

python -m utils.sdd_dataset --reload --raw_data_dir data/sdd/filter/agent_type/dc_013 --raw_data_filename Biker.pkl --varf avg_vel --labels Biker

mkdir data/sdd/filter/avg_vel/dc_013
mv data/sdd/filter/avg_vel/Biker data/sdd/filter/avg_vel/dc_013/

python -m utils.split_dataset --data_dir data/sdd/filter/avg_vel/dc_013/Biker --data_filename 0.5_3.5.pkl --val_split 0.1 --test_split 250 --seed 1 

python -m utils.split_dataset --data_dir data/sdd/filter/avg_vel/dc_013/Biker --data_filename 4_8.pkl --val_split 50 --test_split 250 --seed 1 