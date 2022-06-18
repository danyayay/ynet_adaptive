# config 
list_eval_seed=(1) 
batch_size=10
n_round=1
config_filename=sdd_eval.yaml

# model 
network=original

# pretrained model 
ckpts=ckpts/sdd__ynet__ped_to_biker.pt
ckpts_name=OODG

# data path 
dataset_path=filter/agent_type/deathCircle_0
val_files=Biker.pkl
val_split=80
test_splits=500
load_data=sequential


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --val_files $val_files --val_split $val_split --test_splits $test_splits --ckpts $ckpts --ckpts_name $ckpts_name
done 

