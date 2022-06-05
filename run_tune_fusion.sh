list_seed=(1) 
n_epoch=100
batch_size=10
val_ratio=0.1

n_fusion=2
dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts_fusion/Seed_2__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion_${n_fusion}.pt
train_files=Biker.pkl
val_files=Biker.pkl
n_leftouts=500
ckpt_path=ckpts_fusion/lora/rank1

# list_position=("0" "0 1" "0 1 2" "0 1 2 3" "0 1 2 3 4" "1" "1 2" "1 2 3" "1 2 3 4" "2" "2 3" "2 3 4" "3" "3 4" "4")

list_n_train_batch=(2 4 8 16)
list_lr=(0.005)
list_train_net=(lora_1)
list_position=("scene" "scene fusion" "motion" "motion fusion" "scene motion fusion" "scene motion" "fusion")
for seed in ${list_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            for train_net in ${list_train_net[@]}; do
                for position in "${list_position[@]}"; do 
                    python train_fusion.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --position $position --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --n_fusion $n_fusion --steps 20
                done 
            done 
        done 
    done 
done



# test 
list_seed=(1 2 3)
list_n_train=(20 40 80 160)
list_position=("scene" "scene_fusion" "motion" "motion_fusion" "scene_motion_fusion" "scene_motion" "fusion")
list_tuned_ckpt=()
for n_train in ${list_n_train[@]}; do
    for lr in ${list_lr[@]}; do
        for train_net in ${list_train_net[@]}; do 
            for position in ${list_position[@]}; do
                list_tuned_ckpt+=("${ckpt_path}/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__${train_net}__Pos_${position}__TrN_${n_train}__lr_${lr}__fusion_${n_fusion}.pt")
            done 
        done 
    done
done

for seed in ${list_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_ratio $val_ratio --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_fusion $n_fusion
    done 
done 

# python -m pdb train_fusion.py --fine_tune --seed 1 --batch_size 8 --n_epoch 2 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_2__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__fusion_2.pt --lr 0.00005 --n_train_batch 1 --train_net scene_fusion
