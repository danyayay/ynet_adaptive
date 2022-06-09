list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3

val_split=0.1
train_net=train 

dataset_path="filter/agent_type/hyang_0145/"
train_files="Pedestrian.pkl"
val_files="Pedestrian.pkl"
n_leftouts=500

ckpt_path=ckpts_hyang0145_original

for seed in ${list_train_seed[@]}; do
    python train_embed.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --ckpt_path $ckpt_path --is_augment_data --shuffle_data --n_round $n_round --config_filename sdd_raw_pretrain.yaml
done 
