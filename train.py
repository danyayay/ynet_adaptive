import os
import yaml
import time
import torch
import numpy as np
import pandas as pd

from models.trainer import YNetTrainer
from utils.parser import get_parser
from utils.write_files import write_csv, get_out_dir
from utils.dataset import set_random_seeds, limit_samples, dataset_split


def main(args):
    # ## configuration
    tic = time.time()
    set_random_seeds(args.seed)
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(os.path.join('config', 'sdd_raw_train.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(params['data_dir'], params['dataset_name'], 'segmentation_model.pth')
    params['share_val_test'] = True
    params.update(vars(args))
    print(params)

    # ## set up data
    print('############ Prepare dataset ##############')
    IMAGE_PATH = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    DATA_PATH = os.path.join(params['data_dir'], params['dataset_name'], args.dataset_path)

    if args.train_files == args.val_files:
        # train_files and val_files are fully overlapped 
        df_train, df_val, df_test = dataset_split(
            DATA_PATH, args.train_files, args.val_ratio, args.n_leftouts, 
            share_val_test=params['share_val_test'])
    else:
        # train_files and val_files are not fully overlapped
        df_train, _, df_test = dataset_split(
            DATA_PATH, args.train_files, 0, args.n_leftouts)
        df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in args.val_files])
    df_train = limit_samples(df_train, args.n_train_batch, args.batch_size)
    print(df_train.metaId.unique())
    print(f"df_train: {df_train.shape}; #={df_train.shape[0]/(params['obs_len']+params['pred_len'])}")
    if df_val is not None: print(f"df_val: {df_val.shape}; #={df_val.shape[0]/(params['obs_len']+params['pred_len'])}")
    if df_test is not None: print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ## model
    model = YNetTrainer(params=params)

    if args.train_net == "modulator": model.model.initialize_style()

    if args.pretrained_ckpt is not None:
        model.load_params(args.pretrained_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt}")
    else:
        print("Training from scratch")

    # experiment name 
    EXPERIMENT_NAME = ""
    EXPERIMENT_NAME += f"Seed_{args.seed}_"
    EXPERIMENT_NAME += f"_Tr{'_'.join(['_'+f.split('.pkl')[0] for f in args.train_files])}_"
    EXPERIMENT_NAME += f"_Val{'_'.join(['_'+f.split('.pkl')[0] for f in args.val_files])}_"
    EXPERIMENT_NAME += f"_ValRatio_{args.val_ratio}_"
    EXPERIMENT_NAME += f"_{(args.dataset_path).replace('/', '_')}"
    EXPERIMENT_NAME += f"_{args.train_net}"
    if args.fine_tune:
        if args.train_net == 'all':
            EXPERIMENT_NAME += '_FT'
        elif args.train_net == 'adapter':
            EXPERIMENT_NAME += f'_{args.adapter_type}__{"_".join(map(str, args.adapter_position))}__TrN_{str(int((df_train.shape[0])/20))}'
        else:
            EXPERIMENT_NAME += f'__TrN_{str(int((df_train.shape[0])/20))}'
    print(f"Experiment {EXPERIMENT_NAME} has started")

    # ## training
    # print('############ Train model ##############')
    # val_ade, val_fde = model.train(df_train, df_val, IMAGE_PATH, IMAGE_PATH, EXPERIMENT_NAME)

    # # test for leftout data 
    # if params['out_csv_dir'] and args.n_leftouts:
    #     print('############ Test leftout data ##############')
    #     set_random_seeds(args.seed)
    #     # test
    #     test_ade, test_fde, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
    #     # save csv results
    #     out_dir = get_out_dir(params['out_csv_dir'], args.dataset_path, args.seed, args.train_net, args.val_files, args.train_files)
    #     write_csv(out_dir, 'fine_tune.csv', val_ade, val_fde, test_ade, test_fde)

    toc = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(train=True)
    args = parser.parse_args()

    main(args)

# python -m pdb train_adapter.py --fine_tune --seed 1 --batch_size 8 --n_epoch 10 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 100 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --lr 0.00005 --n_train_batch 1 --train_net adapter --adapter_type parallel --adapter_position 0

# python -m pdb train_adapter.py --fine_tune --seed 1 --batch_size 10 --n_epoch 10 --dataset_path filter/agent_type/deathCircle_0/ --train_files Biker.pkl --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 500 --n_train_batch 16 --ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --lr 0.00005 --train_net adapter --adapter_type serial --adapter_position 0 1 2 3 4
