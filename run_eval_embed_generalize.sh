list_eval_seed=(1 2 3)
batch_size=10
val_split=0
n_round=3
val_files=Biker.pkl

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")

list_ckpts=(ckpts_dc013_original/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.07__filter_agent_type_deathCircle_013__train__TrN_1376__AUG.pt)

dataset_path=filter/agent_type/deathCircle_013/
n_leftouts=1983
for seed in ${list_eval_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_round $n_round
    done 
done 


dataset_path=filter/agent_type/deathCircle_0/
n_leftouts=500
for seed in ${list_eval_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_round $n_round
    done 
done 


dataset_path=filter/agent_type/deathCircle_1/
n_leftouts=500
for seed in ${list_eval_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_round $n_round
    done 
done 


dataset_path=filter/agent_type/deathCircle_3/
n_leftouts=500
for seed in ${list_eval_seed[@]}; do
    for ckpts in ${list_ckpts[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --ckpts $ckpts --ckpts_name OODG --n_leftouts $n_leftouts --n_round $n_round
    done 
done 
