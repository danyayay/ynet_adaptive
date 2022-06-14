list_train_seed=(1) 
batch_size=10
n_epoch=300
n_round=3

network=fusion
n_fusion=2


load_data=predefined

pretrained_ckpt=inD_ckpts/Seed_1__filter_agent_type_scene1_car_filter__train__fusion_2_weights.pt


list_n_train_batch=(2) 
list_lr=(0.00005)


list_train_net=(all)

ckpt_path=inD_ckpts
dataset_path=filter/agent_type/scene1/truck_bus_filter

for train_seed in ${list_train_seed[@]}; do
    for lr in ${list_lr[@]}; do 
        for n_train_batch in ${list_n_train_batch[@]}; do 
            for train_net in ${list_train_net[@]}; do 
                python inD_train.py --fine_tune --config_filename 'inD_shortterm_train.yaml' --seed $train_seed --batch_size $batch_size --n_epoch $n_epoch --dataset_path $dataset_path --network $network --load_data $load_data --pretrained_ckpt $pretrained_ckpt --train_net $train_net  --n_fusion $n_fusion --ckpt_path $ckpt_path --n_round $n_round --lr $lr --n_train_batch $n_train_batch
            done 
        done 
    done 
done