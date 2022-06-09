list_eval_seed=(1 2 3)
batch_size=10
val_split=0
n_round=10

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")


list_ckpts=(ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__deathCirle_013__train__fusion_2.pt)

# evaluate on biker 
val_files=Biker.pkl

dataset_path=filter/agent_type/deathCirle_013/
n_leftouts=1983
for seed in ${list_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion 2 --n_round $n_round
    done 
done 


dataset_path=filter/agent_type/deathCirle_0/
n_leftouts=500
for seed in ${list_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion 2 --n_round $n_round
    done 
done 


dataset_path=filter/agent_type/deathCirle_1/
n_leftouts=500
for seed in ${list_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion 2 --n_round $n_round
    done 
done 

dataset_path=filter/agent_type/deathCirle_3/
n_leftouts=500
for seed in ${list_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_fusion 2 --n_round $n_round
    done 
done 