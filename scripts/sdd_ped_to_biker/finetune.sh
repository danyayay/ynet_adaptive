list_train_seed=(2 3 4 5) 
batch_size=10
n_epoch=100
n_round=3
ckpt_path=ckpts
network=original
train_net=train 

list_n_train_batch=(0.5 1 2 4 8) 
list_train_net=(all)
list_position=("0 1 2 3 4")  # non-baseline models need position 
list_lr=(0.00005 0.00001)

# pretrained_ckpt=${ckpt_path}/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt
pretrained_ckpt=${ckpt_path}/sdd_trajnet_weights.pt


########## if use sequentially split data 
dataset_path=filter/agent_type/deathCircle_0/Biker
# train_files=Biker.pkl
# val_files=Biker.pkl
# val_split=80
# test_splits=500
load_data=predefined

for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                for position in "${list_position[@]}"; do
                    # python train.py --fine_tune --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --train_files $train_files --val_files $val_files --val_split $val_split --test_splits $test_splits --pretrained $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr --init_check
                    python train.py --fine_tune --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --n_round $n_round --dataset_path $dataset_path --network $network --load_data $load_data --pretrained $pretrained_ckpt --train_net $train_net --position $position --ckpt_path $ckpt_path --n_train_batch $n_train_batch --lr $lr
                done 
            done 
        done 
    done 
done 