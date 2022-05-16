import time 
from utils.parser import get_parser
from evaluator.write_files import write_csv, get_out_dir
from utils.dataset import set_random_seeds, dataset_split
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names


def main(args):
    # configuration
    tic = time.time()
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # prepare data 
    _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, [args.tuned_ckpt])
    print(ckpts, ckpts_name)
    if len(ckpts_name) == 1:
        model = restore_model(params, is_file_separated, ckpts_name[0], 
            ckpts[0] if not is_file_separated else args.pretrained_ckpt, 
            None if not is_file_separated else ckpts[0])
    elif len(ckpts_name) > 1:
        for ckpt, ckpt_name in zip(ckpts, ckpts_name):
            if ckpt_name != 'OODG':
                model = restore_model(params, is_file_separated, ckpt_name, 
                    ckpt if not is_file_separated else args.pretrained_ckpt, 
                    None if not is_file_separated else ckpt)
    
    # test
    print('############ Test model ##############')
    set_random_seeds(args.seed)
    ade, fde, _, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(False)
    args = parser.parse_args()
    main(args)


# python -m pdb test.py --seed 1 --batch_size 10 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_10.pt

# python -m pdb test.py --seed 1 --batch_size 10 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 10 --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --ckpts_name OODG


# python -m pdb test.py --seed 1 --batch_size 10 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_serial__0__TrN_40.pt