list_train_seed=(1)
batch_size=10
val_split=0
n_fusion=2
n_round=3
val_files=Biker.pkl

#### pretrained model 
pretrained_ckpt=ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type_deathCircle_013__train__fusion_${n_fusion}.pt  # pretrained filename to deatchcircle model
# pretrained_ckpt=ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type_hyang_0145__train__fusion_2.pt    # pretrained filename to hyang model

dataset_path=filter/agent_type/deathCircle_0/
n_leftouts=500

list_n_train=(20 40 80 160)
list_lr=(0.005)
list_train_net=(lora_1)
list_position=("scene" "scene_fusion" "motion" "motion_fusion" "scene_motion_fusion")
list_tuned_ckpt=()
for n_train in ${list_n_train[@]}; do
    for lr in ${list_lr[@]}; do
        for train_net in ${list_train_net[@]}; do 
            for position in ${list_position[@]}; do
                list_tuned_ckpt+=("ckpts_fusion/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__fusion_${n_fusion}.pt")  # change path possibly 
            done 
        done 
    done
done

for seed in ${list_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round
    done 
done 





list_lr=(0.005)
train_net='lora_1'
position="0_1_2_3_4"
list_tuned_ckpt=()
for seed in ${list_train_seed[@]}; do
    for n_train in ${list_n_train[@]}; do
        for lr in ${list_lr[@]}; do
            for e in $(seq 100); do
                list_tuned_ckpt+=("ckpts/Seed_${seed}__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__epoch_${e}.pt")
            done 
        done 
    done
done

list_eval_seed=(1)
for seed in ${list_eval_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_embed.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_round $n_round
    done 
done 