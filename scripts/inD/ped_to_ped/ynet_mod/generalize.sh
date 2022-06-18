# config 
list_eval_seed=(1) 
batch_size=10
n_round=3
config_filename=inD_longterm_eval.yaml

# model 
network=fusion
n_fusion=2

# pretrained model 
pretrained_ckpt=ckpts/Seed_1__filter_agent_type_scene234_pedestrian__train__original.pt

# data path 
dataset_path=filter/agent_type/scene1/pedestrian
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python inD_test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --dataset_path $dataset_path --network $network --n_fusion $n_fusion --load_data $load_data --n_round $n_round --pretrained_ckpt $pretrained_ckpt
done
