list_eval_seed=(1) 
batch_size=10
n_round=3
network=fusion
n_fusion=2

load_data=predefined
pretrained_ckpt=inD_ckpts/Seed_1__filter_agent_type_scene1_car_filter__train__fusion_2_weights.pt


dataset_path=filter/agent_type/scene1/truck_bus_filter

for eval_seed in ${list_eval_seed[@]}; do
    python inD_test.py --config_filename 'inD_shortterm_eval.yaml' --seed $eval_seed --batch_size $batch_size --dataset_path $dataset_path --network $network --load_data $load_data --n_fusion $n_fusion --n_round $n_round --pretrained_ckpt $pretrained_ckpt
done
