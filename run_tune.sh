list_seed=(1) 
n_epoch=100
batch_size=10
val_ratio=0.1

dataset_path=filter/agent_type/deathCircle_0/
pretrained_ckpt=ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt
train_files=Biker.pkl
val_files=Biker.pkl
n_leftouts=500

# list_position=("0" "0 1" "0 1 2" "0 1 2 3" "0 1 2 3 4" "1" "1 2" "1 2 3" "1 2 3 4" "2" "2 3" "2 3 4" "3" "3 4" "4")

list_n_train_batch=(2 4)
list_lr=(0.00005 0.00001 0.0005 0.0001 0.005 0.001)
list_train_net=(biasGoal biasTraj)
for seed in ${list_seed[@]}; do
    for train_net in ${list_train_net[@]}; do
        for lr in ${list_lr[@]}; do
            for n_train_batch in ${list_n_train_batch[@]}; do 
                python train.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch
            done 
        done 
    done 
done



list_n_train_batch=(2 4)
list_lr=(0.0001 0.0005 0.00005)
list_train_net=(bias)
for seed in ${list_seed[@]}; do
    for train_net in ${list_train_net[@]}; do
        for lr in ${list_lr[@]}; do
            for n_train_batch in ${list_n_train_batch[@]}; do 
                python train.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch
            done 
        done 
    done 
done


list_n_train_batch=(2 4)
list_lr=(0.00001 0.0001 0.005 0.001)
list_train_net=(biasEncoder)
for seed in ${list_seed[@]}; do
    for train_net in ${list_train_net[@]}; do
        for lr in ${list_lr[@]}; do
            for n_train_batch in ${list_n_train_batch[@]}; do 
                python train.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_ratio $val_ratio --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch
            done 
        done 
    done 
done


# python -m pdb train.py --fine_tune --seed 1 --batch_size 8 --n_epoch 10 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --lr 0.00005 --n_train_batch 1 --train_net encoder_0

# python train_copy.py --fine_tune --seed 1 --batch_size 8 --n_epoch 3 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 500 --train_net adapter --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt --lr 0.00005 --adapter_type parallel_3x3 --adapter_position 0 --n_train_batch 1