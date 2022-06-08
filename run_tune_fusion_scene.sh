list_train_seed=(1) 
n_epoch=100
batch_size=10
val_split=0.1   # validation split 
n_fusion=2
n_round=3
train_files=Biker.pkl
val_files=Biker.pkl

#### pretrained model 
pretrained_ckpt=ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type_deathCircle_013__train__fusion_${n_fusion}.pt  # pretrained filename to deatchcircle model
# pretrained_ckpt=ckpts_fusion/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type_hyang_0145__train__fusion_2.pt    # pretrained filename to hyang model

dataset_path=filter/agent_type/deathCircle_0/ # folder to Biker.pkl
n_leftouts=500  # number of test
ckpt_path=ckpts_fusion  # path to save tuned files

list_n_train_batch=(2)
list_lr=(0.005)

#### finetuning (baseline)
list_train_net=(all encoder)
for seed in ${list_train_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            for train_net in ${list_train_net[@]}; do
                python train_fusion.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --n_fusion $n_fusion --steps 20
            done 
        done 
    done 
done


##### finetune lora 
list_train_net=(lora_1)
list_position=("scene" "motion" "scene motion" "fusion" "scene motion fusion")  
for seed in ${list_train_seed[@]}; do
    for n_train_batch in ${list_n_train_batch[@]}; do
        for lr in ${list_lr[@]}; do
            for train_net in ${list_train_net[@]}; do
                for position in "${list_position[@]}"; do 
                    python train_fusion.py --fine_tune --seed $seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --train_files $train_files --val_files $val_files --val_split $val_split --n_leftouts $n_leftouts --train_net $train_net --pretrained_ckpt $pretrained_ckpt --lr $lr --n_train_batch $n_train_batch --ckpt_path $ckpt_path --n_fusion $n_fusion --steps 20
                done 
            done 
        done 
    done 
done




