seeds=(1 2 3)
batch_size=8

dataset_name=sdd
dataset_path=sherwin/dataset_ped_biker/gap/
out_csv_dir=csv # /path/to/csv where the output results are written to

val_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
val_ratio=0.5 # Only use a subset of the dataset for evaluation to make it comparable to fine-tuning

ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model

for seed in ${seeds[@]}
do
    python evaluate_SDD.py --seed $seed --batch_size $batch_size --dataset_name $dataset_name --dataset_path $dataset_path --val_files $val_files --out_csv_dir $out_csv_dir --ckpt $ckpt
done