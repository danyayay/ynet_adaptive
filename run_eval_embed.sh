list_train_seed=(3)
batch_size=10
val_split=0

# list_train_net=("encoder_0" "encoder_0-1" "encoder_0-2" "encoder_0-3" "encoder_0-4" "encoder_1" "encoder_1-2" "encoder_1-3" "encoder_1-4" "encoder_2" "encoder_2-3" "encoder_2-4" "encoder_3" "encoder_3-4" "encoder_4")
# list_adapter_position=("0" "0_1" "0_1_2" "0_1_2_3" "0_1_2_3_4" "1" "1_2" "1_2_3" "1_2_3_4" "2" "2_3" "2_3_4" "3" "3_4" "4")


dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
val_files=Biker.pkl
n_leftouts=500
n_round=3

list_n_train=(20)

list_lr=(0.005)
train_net='lora_1'
position="0_1_2_3_4"
# ckpt_path=ckpts/lora/val_test_noearly
ckpt_path=noearly_shuffle
list_eval_seed=(1)
list_tuned_ckpt=()
for train_seed in ${list_train_seed[@]}; do
    for n_train in ${list_n_train[@]}; do
        for lr in ${list_lr[@]}; do
            for e in $(seq 0 1 99); do
                list_tuned_ckpt+=("${ckpt_path}/Seed_${train_seed}__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__epoch_${e}.pt")
            done 
        done 
    done
done


for seed in ${list_eval_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_round $n_round --given_test_file testset/Seed_3__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0___lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.005.csv
    done 
done 
