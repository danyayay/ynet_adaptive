list_train_seed=(1) 
batch_size=10
n_epoch=100
n_round=3
ckpt_path=ckpts_fusion
network=fusion
n_fusion=2

list_n_train_batch=(2) 
list_train_net=(lora_1)
list_position=("scene" "motion" "scene motion fusion")  # non-baseline models need position 
list_lr=(0.0005)

pretrained_ckpt=${ckpt_path}/Seed_1__0.5_3.5__Val_0.5_3.5__ValRatio_0.1__filter_avg_vel_dc_013_Biker__train__fusion_2.pt


########## if use predefined data
dataset_path=filter/avg_vel/dc_013/Biker/4_8
load_data=predefined

for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do 
                    python train.py --fine_tune --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net --position $position --n_fusion $n_fusion --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr 
                done 
            done 
        done 
    done 
done 
