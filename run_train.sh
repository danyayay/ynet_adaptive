seeds=(1) # Train the model on different seeds
batch_size=8
num_epochs=50

dataset_name=sdd y
dataset_path=sherwin/dataset_ped_biker/gap/ 
out_csv_dir=csv # /path/to/csv where the output results are written to

train_files='0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl' # Position the dataset files in /path/to/dataset_path/{dataset}
val_files='0.25_0.75.pkl 1.25_1.75.pkl 2.25_2.75.pkl' # Position the dataset files in /path/to/dataset_path/{dataset}
val_ratio=0.3 # Split train dataset into a train and val split in case the domains are the same

train_net=all # Train either all parameters, only the encoder or the modulator: (all encoder modulator)

for seed in ${seeds[@]}
do    
    python train_SDD.py --seed $seed --batch_size $batch_size --num_epochs $num_epochs --dataset_name $dataset_name --dataset_path $dataset_path --train_files $train_files --val_files $val_files --out_csv_dir $out_csv_dir --val_ratio $val_ratio --train_net $train_net 
done