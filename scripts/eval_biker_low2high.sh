list_eval_seed=(1) 
batch_size=10
n_round=3
network=fusion
n_fusion=2


########## if use predefined data
dataset_path=filter/avg_vel/dc_013/Biker/4_8
load_data=predefined


ckpt_path=multi_hard_early40
pretrained_ckpt=${ckpt_path}/Seed_1__filter_avg_vel_multi_hard_Biker_0.5_3.5__train__fusion_2.pt

for eval_seed in ${list_eval_seed[@]}; do
    python test.py --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --n_fusion $n_fusion
done 


ckpt_path=multi_hard_earlyno
pretrained_ckpt=${ckpt_path}/Seed_1__filter_avg_vel_multi_hard_Biker_0.5_3.5__train__fusion_2.pt

for eval_seed in ${list_eval_seed[@]}; do
    python test.py --seed $eval_seed --batch_size $batch_size --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --n_fusion $n_fusion
done 