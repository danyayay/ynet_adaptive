raw_data_dir=data/sdd
raw_data_name=filter/agent_type/hyang_0145/Pedestrian.pkl 
filter_data_dir=data/sdd/filter 
step=12
window_size=20
stride=20
labels=(Pedestrian) # Choose a subset from: Biker, Bus, Car, Cart, Pedestrian, Skater
python utils/dataset.py --raw_data_dir $raw_data_dir --raw_data_name $raw_data_name --filter_data_dir $filter_data_dir --step $step --window_size $window_size --stride $stride --labels $labels --reload
