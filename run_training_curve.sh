list_window=(9)

list_file_path=()
list_trainnet=(baseline)
list_n_train=(40 80)
list_position=("scene-motion" "fusion" "scene-motion-fusion")
for trainnet in ${list_trainnet[@]}; do 
    for n_train in ${list_n_train[@]}; do
        for position in ${list_position[@]}; do 
            list_file_path+=(${trainnet}_n${n_train}_${position}_seed1-2-3)
        done 
    done 
done 

for window in ${list_window[@]}; do 
    for file_path in ${list_file_path[@]}; do 
        python -m utils.extract_training_curve --file_path ./logs_plot/${file_path}.out --out_dir ./figures/training_curve
    done 
done