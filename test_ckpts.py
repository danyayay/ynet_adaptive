import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd

from model import YNetTrainer
from utils.dataset import set_random_seeds, dataset_split
from utils.visualize import plot_obs_pred_trajs, plot_feature_space


def main(args):
    # ## configuration
    with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(args.data_dir, args.dataset_name, 'segmentation_model.pth')
    params.update(vars(args))
    print(params)

    # ## set up data 
    print('############ Prepare dataset ##############')
    # image path 
    IMAGE_PATH = os.path.join(args.data_dir, args.dataset_name, 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    # data path 
    DATA_PATH = os.path.join(args.data_dir, args.dataset_name, args.dataset_path)

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.val_files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ## model 
    print('############ Load models ##############')
    if args.ckpts:
        dict_features, dict_trajs = dict(), dict()
        for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
            print(f'====== Testing for {ckpt_name} ======')
            set_random_seeds(args.seed)
            model = YNetTrainer(params=params)
            model.load(ckpt)
            val_ade, val_fde, list_metrics, list_features, list_trajs = \
                model.test(df_test, IMAGE_PATH, False, True, False) # True if not args.limit_by else False
            # store ade/fde comparison
            df_to_merge = list_metrics[0].rename({
                'ade': f'ade_{ckpt_name}', 'fde': f'fde_{ckpt_name}'}, axis=1)
            df_to_merge = pd.concat([df_to_merge, pd.DataFrame({
                'metaId': 0, 'sceneId': 'avg', f'ade_{ckpt_name}': val_ade, f'fde_{ckpt_name}': val_fde}, index=[0])], 
                ignore_index=True, axis=0)
            df_result = df_to_merge if i == 0 else df_result.merge(df_to_merge, on=['metaId', 'sceneId'])
            # store features and trajectories, taking only the first round results for now 
            dict_features[ckpt_name] = list_features[0]
            dict_trajs[ckpt_name] = list_trajs[0]
        name = str(args.seed) + '__' + args.ckpts[0].split('_filter_')[1].split('__')[0] + '__' + '_'.join(args.val_files).rstrip('.pkl')
        out_path = f"csv/comparison/{name}__{'_'.join(args.ckpts_name)}.csv"
        df_result.to_csv(out_path, index=False)

        # reduce df_test size if needed
        if args.limit_by is not None:
            print('Limiting df_test')
            df_result.loc[:, 'ade_OODG_FT'] = np.absolute(df_result.ade_OODG.values - df_result.ade_FT.values)
            df_result = df_result[df_result.sceneId != 'avg']
            if args.limit_by == 'n_viz':
                meta_ids_focus = df_result.sort_values(by='ade_OODG_FT', ascending=False).head(args.n_viz).metaId.values              
            else: # args.limit_by == 'threshold'
                meta_ids_focus = df_result[df_result.ade_OODG_FT >= args.threshold].metaId.values
                while meta_ids_focus.shape[0] == 0:
                    args.threshold /= 2
                    meta_ids_focus = df_result[df_result.ade_OODG_FT >= args.threshold].metaId.values
            df_test_focus = df_test[df_test.metaId.isin(meta_ids_focus)]
            # repeat the above process 
            dict_features, dict_trajs = dict(), dict()
            for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
                set_random_seeds(args.seed)
                model = YNetTrainer(params=params)
                model.load(ckpt)
                val_ade, val_fde, list_metrics, list_features, list_trajs = \
                    model.test(df_test_focus, IMAGE_PATH, False, True, True)
                # store ade/fde comparison
                df_to_merge = list_metrics[0].rename({
                    'ade': f'ade_{ckpt_name}', 'fde': f'fde_{ckpt_name}'}, axis=1)
                df_to_merge = pd.concat([df_to_merge, pd.DataFrame({
                    'metaId': 0, 'sceneId': 'avg', f'ade_{ckpt_name}': val_ade, f'fde_{ckpt_name}': val_fde}, index=[0])], 
                    ignore_index=True, axis=0)
                df_result = df_to_merge if i == 0 else df_result.merge(df_to_merge, on=['metaId', 'sceneId'])
                # store features and trajectories, taking only the first round results for now 
                dict_features[ckpt_name] = list_features[0]
                dict_trajs[ckpt_name] = list_trajs[0]
            if args.limit_by == 'n_viz':
                print(f'Visualize {meta_ids_focus.shape[0]} (n_viz={args.n_viz}) test samples')
            else:
                print(f'Visualize {meta_ids_focus.shape[0]} (threhold={args.threshold}) test samples')
        else:
            print(f'Visualize all n_test={args.n_leftouts} test samples')

        # visualize
        if args.viz:
            # plot_obs_pred_trajs(IMAGE_PATH, dict_trajs, f'figures/prediction/{name}')
            plot_feature_space(dict_features, f'figures/feature_space/{name}')
    else:
        raise ValueError('No checkpoint given!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--dataset_name', default='sdd', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_round', default=1, type=int)
    # files 
    parser.add_argument('--ckpts', default=None, type=str, nargs='+')
    parser.add_argument('--ckpts_name', default=None, type=str, nargs='+')
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--val_files', default=None, type=str, nargs='+')
    parser.add_argument('--n_leftouts', default=None, type=int, nargs='+')
    # visualization
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--depth', default=0, type=int)
    parser.add_argument('--limit_by', default=None, choices=[None, 'n_viz', 'threshold'])
    parser.add_argument('--n_viz', default=10, type=int)
    parser.add_argument('--threshold', default=4.0, type=float)

    args=parser.parse_args()

    main(args)
