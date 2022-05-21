list_seed=(1)
dataset_name=sdd
dataset_path=filter/agent_type/
ckpts="ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt "
ckpts_name="OODG ET FT"
files=Biker.pkl
n_leftouts=500
n_limited=50

for seed in ${list_seed[@]}; do
    python -m saliency --seed $seed --dataset_name $dataset_name --dataset_path $dataset_path --files $files --n_leftouts $n_leftouts --ckpts $ckpts --ckpts_name $ckpts_name --n_limited $n_limited --decision map --VanillaGrad --SmoothGrad --GradCAM
done 

