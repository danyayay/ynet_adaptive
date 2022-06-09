list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3

val_split=0.1
train_net=train 

dataset_path="filter/avg_vel/hyang_0145/Pedestrian"
train_files="0.25_1.25.pkl"
val_files="0.25_1.25.pkl"
n_leftouts=250
n_fusion=2

ckpt_path=ckpts_fusion/vel_ped

for seed in ${list_train_seed[@]}; do
    python train_fusion.py --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --ckpt_path $ckpt_path --n_fusion $n_fusion --is_augment_data --shuffle_data --n_round $n_round --config_filename sdd_raw_pretrain.yaml
done





# # evaluate on test data 
# list_eval_seed=(1 2 3)
# n_round=10
# dataset_path=filter/agent_type/

# # ## fusion=2
# list_ckpts=()
# for seed in ${list_train_seed[@]}; do
#     list_ckpts+=(${ckpt_path}/Seed_${seed}__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt)
# done 
# # evaluate on ped
# val_files=Pedestrian.pkl
# n_leftouts=1500
# for seed in ${list_seed[@]}; do
#     for ckpts in ${list_ckpts[@]}; do
#         python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round
#     done 
# done 
# # evaluate on biker 
# val_files=Biker.pkl
# n_leftouts=500
# for seed in ${list_seed[@]}; do
#     for ckpts in ${list_ckpts[@]}; do
#         python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round
#     done 
# done 


# # ### evaluate on dc0
# dataset_path=filter/agent_type/deathCircle_0/

# list_ckpts=()
# for seed in ${list_train_seed[@]}; do
#     list_ckpts+=(${ckpt_path}/Seed_${seed}__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt)
# done 

# # evaluate on ped
# val_files=Pedestrian.pkl
# n_leftouts=1500
# for seed in ${list_seed[@]}; do
#     for ckpts in ${list_ckpts[@]}; do
#         python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round
#     done 
# done 

# # evaluate on biker 
# val_files=Biker.pkl
# n_leftouts=500
# for seed in ${list_seed[@]}; do
#     python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round
# done 


