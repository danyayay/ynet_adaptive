declare -A AB=( ["dataset_path"]="filter/avg_vel/Pedestrian/" ["filename"]="0.1_0.3.pkl 0.5_1.5.pkl" ["n_test"]="100 990")

dataset_path=${AB["dataset_path"]}
train_files=${AB["filename"]}
val_files=${AB["filename"]}
n_leftout=${AB["n_test"]}

echo $dataset_path
echo $train_files
echo $val_files
echo $n_leftout