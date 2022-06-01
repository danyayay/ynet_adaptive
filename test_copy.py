import time 
from utils.parser import get_parser
from utils.dataset import set_random_seeds, dataset_split
from utils.util_copy import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names


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
        model = restore_model(params, is_file_separated,
            ckpts[0] if not is_file_separated else args.pretrained_ckpt, 
            None if not is_file_separated else ckpts[0])
    elif len(ckpts_name) > 1:
        for ckpt, ckpt_name in zip(ckpts, ckpts_name):
            if ckpt_name != 'OODG':
                model = restore_model(params, is_file_separated, 
                    ckpt if not is_file_separated else args.pretrained_ckpt, 
                    None if not is_file_separated else ckpt)
    
    # test
    print('############ Test model ##############')
    set_random_seeds(args.seed)
    ade, fde, _, _ = model.test(df_test, IMAGE_PATH, with_style=args.train_net=="modulator")

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(False)
    args = parser.parse_args()
    main(args)


# CUDA_VISIBLE_DEVICES=1 python -m pdb test_copy.py --seed 1 --batch_size 10 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train_pretrained.pt --tuned_ckpt ckpts/DC0__lora/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__lora_1__Pos_0_1_2_3_4__TrN_20__lr_0.0005.pt 