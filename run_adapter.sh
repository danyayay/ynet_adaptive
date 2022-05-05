list_seed=(1) 
n_epoch=100
batch_size=10
n_train_batch=16

dataset_name=sdd
out_csv_dir=csv 
val_ratio=0.1

dataset_path=filter/agent_type/deathCircle_0/
ckpt=ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt
train_files=Biker.pkl
val_files=Biker.pkl
n_leftouts=500
train_net=adapter
adapter_types=(series parallel)

for seed in ${list_seed[@]}; do
    for adapter_type in ${adapter_types[@]}; do
        python train_adapter.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_name $dataset_name --dataset_path $dataset_path --out_csv_dir $out_csv_dir --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --ckpt $ckpt --lr 0.00005 --adapter_type $adapter_type --adapter_position 0 1 2 3 4
    done 
done

# python -m pdb train.py --fine_tune --seed 1 --batch_size 8 --n_epoch 10 --dataset_name sdd --dataset_path filter/agent_type/deathCircle_0/ --out_csv_dir csv --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 500 --ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --lr 0.00005 --n_train_batch 1 --train_net encoder_0