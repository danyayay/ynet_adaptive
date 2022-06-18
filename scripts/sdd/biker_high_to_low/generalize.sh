# config 
list_eval_seed=(1) 
batch_size=10
n_round=3
config_filename=sdd_eval.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
ckpts=ckpts/Seed_1__filter_avg_vel_multi_easy_Biker_2.75_7.5__train__fusion_2.pt
ckpts_name=OODG

# data path 
dataset_path=filter/avg_vel/dc_013/Biker/0.5_2.25
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --ckpts $ckpts --ckpts_name $ckpts_name --n_fusion $n_fusion
done 

