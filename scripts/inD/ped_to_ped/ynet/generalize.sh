# config 
list_eval_seed=(1) 
batch_size=10
n_round=1
config_filename=inD_longterm_eval.yaml

# model 
network=original

# pretrained model 
ckpts=ckpts/inD__ynet__ped_to_ped.pt
ckpts_name=OODG

# data 
dataset_path=filter/agent_type/scene1/pedestrian
load_data=predefined


for eval_seed in ${list_eval_seed[@]}; do
    python test.py --config_filename $config_filename --seed $eval_seed --batch_size $batch_size --dataset_path $dataset_path --network $network --load_data $load_data --n_round $n_round --ckpts $ckpts --ckpts_name $ckpts_name
done
