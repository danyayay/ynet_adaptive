seeds=(1 2 3) # Fine-tune the model on different seeds
num_epochs=30
list_batch_size=(1)
list_num_batches=(1 2 3 4 5) # Fine-tune the model with a given number of batches

dataset_name=sdd
dataset=sherwin/dataset_ped_biker/gap/
out_csv_dir=csv # /path/to/csv where the output results are written to

train_net=encoder # Train either all or part of the parameters

train_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
val_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
val_ratio=0.5 # Split train dataset into a train and val split in case the domains are the same

ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model

for seed in ${seeds[@]}
do
    for nb in ${list_num_batches[@]}
    do
    	for bs in ${list_batch_size[@]}
    	do
        python train_SDD.py --fine_tune --seed $seed --batch_size $bs --num_epochs $num_epochs --dataset_name $dataset_name --dataset_path $dataset_path --out_csv_dir $out_csv_dir --train_files $train_files --val_files $val_files --val_ratio $val_ratio --num_train_batches $nb --train_net $train_net --ckpt $ckpt --learning_rate 0.00005
        done
    done
done