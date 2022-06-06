list_train_seed=(2 3) 
n_epoch=100
batch_size=10
val_split=0.1

dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
train_files=Biker.pkl
val_files=Biker.pkl
n_leftouts=500

# list_position=("0" "0 1" "0 1 2" "0 1 2 3" "0 1 2 3 4" "1" "1 2" "1 2 3" "1 2 3 4" "2" "2 3" "2 3 4" "3" "3 4" "4")

ckpt_path=ckpts/
list_n_train_batch=(2 4 8 16)
list_lr=(0.005)
train_net='encoder'
position="0 1 2 3 4"
for seed in ${list_train_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            python train_embed.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --position $position --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path
        done 
    done 
done


# python train_embed.py --fine_tune --seed 1 --batch_size 8 --n_epoch 3 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_split 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt --lr 0.00005 --train_net encoder --position 0 1 2 3 4 --n_train_batch 1