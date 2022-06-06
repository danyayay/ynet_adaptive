list_train_seed=(1 2 3)
batch_size=10
val_split=0

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")


dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
val_files=Biker.pkl
n_leftouts=500

list_n_train=(20 40 80 160)
list_lr=(0.005)
train_net='lora_1'
position="0_1_2_3_4"
list_tuned_ckpt=()
for seed in ${list_train_seed[@]}; do
    for n_train in ${list_n_train[@]}; do
        for lr in ${list_lr[@]}; do
            list_tuned_ckpt+=("ckpts/Seed_${seed}__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}.pt")
        done 
    done
done

list_eval_seed=(1 2 3)
for seed in ${list_eval_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts
    done 
done 


