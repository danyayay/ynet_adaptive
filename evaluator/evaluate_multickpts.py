import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd

from utils.dataset import set_random_seeds, dataset_split, get_meta_ids_focus
from utils.parser import get_parser
from utils.util import get_params, get_image_and_data_path, restore_model, get_ckpts_and_names
from evaluator.visualization import plot_prediction, plot_multiple_predictions, plot_goal_map_with_samples


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # prepare data 
    df_train, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    # get focused data 
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    meta_ids_focus = get_meta_ids_focus(df_test, 
        given_csv={'path': args.result_path, 'name': args.result_name, 'n_limited': args.result_limited}, 
        given_meta_ids=args.given_meta_ids, random_n=args.random_n)
    df_test = df_test[df_test.metaId.isin(meta_ids_focus)]
    print('meta_ids_focus: #=', len(meta_ids_focus))
    print(f"df_test_limited: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    
    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, args.tuned_ckpts)

    # main  
    ckpts_trajs_dict = dict()
    for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
        print(f'====== Testing for {ckpt_name} ======')

        # load model 
        model = restore_model(params, is_file_separated, ckpt_name, 
            ckpt if not is_file_separated else args.pretrained_ckpt, 
            None if not is_file_separated else ckpt)

        # test 
        set_random_seeds(args.seed)
        _, _, list_metrics, list_trajs = model.test(df_test, IMAGE_PATH, False, True, True) 

        # store ade/fde comparison
        df_to_merge = list_metrics[0].rename({
            'ade': f'ade_{ckpt_name}', 'fde': f'fde_{ckpt_name}'}, axis=1)
        df_result = df_to_merge if i == 0 else df_result.merge(df_to_merge, on=['metaId', 'sceneId'])
        if args.n_round == 1:
            ckpts_trajs_dict[ckpt_name] = list_trajs[0]
        else:
            ckpts_trajs_dict[ckpt_name] = list_trajs

    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}" 
    csv_name = f"{'_'.join(ckpts_name)}__N{'_'.join(str(n) for n in args.n_leftouts)}"
    out_path = f"csv/comparison/{folder_name}/{csv_name}.csv"
    df_result.to_csv(out_path, index=False)

    if args.n_round == 1:
        plot_prediction(IMAGE_PATH, ckpts_trajs_dict, 
            f'figures/prediction/{folder_name}/{"_".join(ckpts_name)}')
    else:
        plot_multiple_predictions(IMAGE_PATH, ckpts_trajs_dict, 
            f'figures/prediction_multiple/{folder_name}/{"_".join(ckpts_name)}')
        plot_goal_map_with_samples(IMAGE_PATH, ckpts_trajs_dict, 
            f'figures/goal_map_with_samples/{folder_name}')
        

def hook_store_output(module, input, output): 
    module.output = output


if __name__ == '__main__':
    parser = get_parser(False)
    # data
    parser.add_argument('--given_meta_ids', default=None, type=int, nargs='+')
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--result_name', default=None, type=str)
    parser.add_argument('--result_limited', default=None, type=int)
    parser.add_argument('--random_n', default=None, type=int)

    args=parser.parse_args()

    main(args)

# python -m pdb -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt --n_round 2 --given_meta_ids 6318

# python -m pdb -m evaluator.evaluate_multickpts --dataset_path filter/agent_type/deathCircle_0 --val_files Biker.pkl --n_leftouts 500 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt --result_path './csv/comparison/1__filter_agent_type_deathCircle_0__Biker/OODG_encoder_0(20)_encoder_0-1(20).csv' --result_name 'ade_OODG__ade_encoder_0(20)__diff' --result_limited 5 --n_round 10 