import os
import time
import pandas as pd

from models.trainer_embed import YNetTrainer
from utils.parser import get_parser
from utils.util_embed import get_experiment_name, get_params, get_image_and_data_path
from utils.dataset import set_random_seeds, limit_samples, dataset_split


def main(args):
    # config 
    tic = time.time()
    set_random_seeds(args.seed)
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # load data 
    if args.train_files == args.val_files:
        # train_files and val_files are fully overlapped 
        df_train, df_val, df_test = dataset_split(
            DATA_PATH, args.train_files, args.val_split, args.n_leftouts, 
            share_val_test=params['share_val_test'])
    else:
        # train_files and val_files are not fully overlapped
        df_train, _, df_test = dataset_split(
            DATA_PATH, args.train_files, 0, args.n_leftouts)
        df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in args.val_files])
    breakpoint()
    df_train = limit_samples(df_train, args.n_train_batch, args.batch_size)
    print(df_train.metaId.unique())
    print(f"df_train: {df_train.shape}; #={df_train.shape[0]/(params['obs_len']+params['pred_len'])}")
    if df_val is not None: print(f"df_val: {df_val.shape}; #={df_val.shape[0]/(params['obs_len']+params['pred_len'])}")
    if df_test is not None: print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.train_files).rstrip('.pkl')}"

    # experiment name 
    EXPERIMENT_NAME = get_experiment_name(args, df_train.shape[0])
    print(f"Experiment {EXPERIMENT_NAME} has started")

    # model
    model = YNetTrainer(params=params)
    if args.train_net == "modulator": model.model.initialize_style()
    if args.pretrained_ckpt is not None:
        model.load_params(args.pretrained_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt}")
    else:
        print("Training from scratch")

    # initialization check 
    if args.init_check:
        params_pretrained = params.copy()
        params_pretrained.update({'position': []})
        pretrained_model = YNetTrainer(params=params_pretrained)
        pretrained_model.load_params(args.pretrained_ckpt)
        set_random_seeds(args.seed)
        ade_pre, fde_pre, _, _ = pretrained_model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
        set_random_seeds(args.seed)
        ade_cur, fde_cur, _, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
        if ade_pre != ade_cur or fde_pre != fde_cur:
            raise RuntimeError('Wrong model initialization')
        else:
            print('Passed initialization check')

    # training
    print('############ Train model ##############')
    val_ade, val_fde = model.train(df_train, df_val, IMAGE_PATH, IMAGE_PATH, EXPERIMENT_NAME)

    # test for leftout data 
    if params['out_csv_dir'] and args.n_leftouts:
        print('############ Test leftout data ##############')
        set_random_seeds(args.seed)
        test_ade, test_fde, _, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(True)
    parser.add_argument('--init_check', action='store_true')
    args = parser.parse_args()

    main(args)

# CUDA_VISIBLE_DEVICES=1 python -m pdb train_embed.py --fine_tune --seed 1 --n_epoch 3 --batch_size 8 --dataset_path filter/agent_type/deathCircle_0/ --pretrained_ckpt ckpts/Seed_1__Tr_Pedestrian__Val_Pedestrian__ValRatio_0.1__filter_agent_type__train__original.pt --train_files Biker.pkl --val_files Biker.pkl --val_split 80 --n_leftouts 500 --lr 0.0005 --train_net lora_1 --position 0 1 2 3 4
