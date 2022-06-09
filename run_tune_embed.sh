list_train_seed=(1 2 3) 
n_epoch=100
n_round=3
batch_size=10
val_split=0.1

dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
train_files=Biker.pkl
val_files=Biker.pkl
n_leftouts=500

# list_position=("0" "0 1" "0 1 2" "0 1 2 3" "0 1 2 3 4" "1" "1 2" "1 2 3" "1 2 3 4" "2" "2 3" "2 3 4" "3" "3 4" "4")

ckpt_path=noearly_shuffle/
list_n_train_batch=(2)
list_lr=(0.005)
train_net='lora_1'
position="0 1 2 3 4"
for seed in ${list_train_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            python train_embed.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --position $position --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --config_filename sdd_raw_noearly.yaml --n_round $n_round --shuffle_data 
        done 
    done 
done


# python train_embed.py --fine_tune --seed 1 --batch_size 8 --n_epoch 3 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_split 0.1 --n_leftouts 500 --train_net lora_1 --position 0 1 2 3 4 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt --lr 0.005 --n_train_batch 2 --ckpt_path ckpts/noearly_shuffle/ --config_filename sdd_raw_noearly.yaml --n_round 3 --shuffle_data 

# evaluate 
# position="0_1_2_3_4"
# list_n_train=(20)
# list_tuned_ckpt=()
# for seed in ${list_train_seed[@]}; do
#     for n_train in ${list_n_train[@]}; do
#         for lr in ${list_lr[@]}; do
#             for e in $(seq 0 1 99); do
#                 list_tuned_ckpt+=("${ckpt_path}/Seed_${seed}__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__epoch_${e}.pt")
#             done 
#         done 
#     done
# done

# list_eval_seed=(1)
# for seed in ${list_eval_seed[@]}; do
#     for tuned_ckpt in ${list_tuned_ckpt[@]}; do
#         python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_round $n_round
#     done 
# done 

# python train_embed.py --fine_tune --seed 1 --batch_size 8 --n_epoch 3 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_split 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt --lr 0.00005 --train_net encoder --position 0 1 2 3 4 --n_train_batch 1