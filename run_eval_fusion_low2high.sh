list_eval_seed=(1)
batch_size=10
val_split=0
n_fusion=2
n_round=3
val_files=4_8.pkl

pretrained_ckpt=ckpts_fusion/Seed_1__Tr_0.5_3.5__Val_0.5_3.5__ValRatio_0.1__filter_avg_vel_dc_013_Biker__train__fusion_${n_fusion}.pt  

dataset_path=filter/avg_vel/dc_013/Biker/
n_leftouts=250

list_n_train=(20)
list_lr=(0.0005)
list_train_net=("scene")
list_tuned_ckpt=()
for n_train in ${list_n_train[@]}; do
    for lr in ${list_lr[@]}; do
        for train_net in ${list_train_net[@]}; do 
            list_tuned_ckpt+=("ckpts_fusion/Seed_1__Tr_4_8__Val_4_8__ValRatio_50__filter_avg_vel_dc_013_Biker__${train_net}__TrN_${n_train}__lr_${lr}__fusion_${n_fusion}.pt")
        done 
    done
done

for seed in ${list_eval_seed[@]}; do
    for tuned_ckpt in ${list_tuned_ckpt[@]}; do
        CUDA_VISIBLE_DEVICES=1 python test_fusion.py --seed $seed --batch_size $batch_size --dataset_path $dataset_path --val_files $val_files --val_split $val_split --pretrained_ckpt $pretrained_ckpt --tuned_ckpt $tuned_ckpt --n_leftouts $n_leftouts --n_fusion $n_fusion --n_round $n_round --given_test_file testset/Seed_1__Tr_4_8__Val_4_8__ValRatio_50__filter_avg_vel_dc_013_Biker__scene__TrN_20__lr_0.0005__fusion_2.csv
    done 
done 


