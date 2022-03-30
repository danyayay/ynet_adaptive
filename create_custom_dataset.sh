raw_data_dir=data/sdd/raw # See README.md on how to download the raw dataset
filter_data_dir=data/sdd/filter # Path to new dataset in sdd_ynet directory
step=12
window_size=20
stride=20
labels=(Pedestrian) # Choose a subset from: Biker, Bus, Car, Cart, Pedestrian, Skater
python utils/dataset.py --raw_data_dir $raw_data_dir --filter_data_dir $filter_data_dir --step $step --window_size $window_size --stride $stride --labels $labels
