import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd

from utils.dataset import set_random_seeds, dataset_split
from utils.parser import get_parser
from utils.util import get_ckpt_name, get_params, get_image_and_data_path, restore_model
from utils.visualize import plot_prediction


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.val_files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts 
    if args.pretrained_ckpt is not None: 
        ckpts = [args.pretrained_ckpt] + args.tuned_ckpts
        ckpts_name = ['OODG'] + [get_ckpt_name(tuned) for tuned in args.tuned_ckpts]
    else:
        raise ValueError('No checkpoint is found')

    # main  
    ckpts_trajs_dict = dict()
    for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
        print(f'====== Testing for {ckpt_name} ======')

        # load model 
        model = restore_model(params, ckpt_name, args.pretrained_ckpt, ckpt)

        # test 
        set_random_seeds(args.seed)
        _, _, list_metrics, list_trajs = model.test(df_test, IMAGE_PATH, False, True, False) 

        # store ade/fde comparison
        df_to_merge = list_metrics[0].rename({
            'ade': f'ade_{ckpt_name}', 'fde': f'fde_{ckpt_name}'}, axis=1)
        df_result = df_to_merge if i == 0 else df_result.merge(df_to_merge, on=['metaId', 'sceneId'])
        ckpts_trajs_dict[ckpt_name] = list_trajs[0]

    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}" 
    out_path = f"csv/comparison/{folder_name}/{'_'.join(ckpts_name)}.csv"
    df_result.to_csv(out_path, index=False)

    plot_prediction(IMAGE_PATH, ckpts_trajs_dict, f'figures/prediction/{folder_name}/{"_".join(ckpts_name)}')
        

if __name__ == '__main__':
    parser = get_parser(False)
    parser.add_argument('--tuned_ckpts', default=None, type=str, nargs='+')

    args=parser.parse_args()

    main(args)

# python -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0-1__TrN_20.pt
