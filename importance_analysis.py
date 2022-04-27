import os
import yaml
import time
import pathlib
import argparse
import numpy as np
import pandas as pd

from model import YNetTrainer
from utils.dataset import set_random_seeds, dataset_split, dataset_given_scenes
from utils.visualize import plot_importance_analysis


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
        _, _, df_test = dataset_split(DATA_PATH, args.files, 0, args.n_leftouts)
    elif args.scenes:
        df_test = dataset_given_scenes(DATA_PATH, args.files, args.scenes)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ## model 
    # pretrained model 
    pretrained_model = YNetTrainer(params=params)
    pretrained_model.load(args.pretrained)
    _, _, list_metrics = pretrained_model.test(df_test, IMAGE_PATH, False, False, False) 
    folder_name = f"{args.seed}__{args.pretrained.split('_filter_')[1].split('__')[0]}__{'_'.join(args.files).rstrip('.pkl')}" 
    out_dir = f"csv/importance_analysis/{folder_name}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    list_metrics[0].to_csv(
        f"{out_dir}/OODG__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
    print('Saved pretrained predictions')

    # tuned models 
    for tuned, tuned_name in zip(args.tuned, args.tuned_name):
        print(f'====== Testing for {tuned_name} ======')
        set_random_seeds(args.seed)
        tuned_model = YNetTrainer(params=params)
        tuned_model.load(tuned)
        _, _, list_metrics = tuned_model.test(df_test, IMAGE_PATH, False, False, False) 
        list_metrics[0].to_csv(
            f"{out_dir}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}.csv", index=False)
        # replace one/several layers in tuned model by pretrained model 
        for param_name, param in pretrained_model.model.named_parameters():
            if not param_name.startswith('semantic_segmentation'):
                tuned_model.load(tuned)
                print(f'Replacing {param_name}')
                tuned_model.model.load_state_dict({param_name: param}, strict=False)
                _, _, list_metrics = tuned_model.test(df_test, IMAGE_PATH, False, False, False) 
                # store ade/fde 
                out_path = f"{out_dir}/{tuned_name}__N{'_'.join(str(n) for n in args.n_leftouts)}__{param_name}.csv"
                list_metrics[0].to_csv(out_path, index=False)
    

def main(args):
    plot_importance_analysis(
        'csv/importance_analysis/1__agent_type__Biker', 
        'figures/importance_analysis/1__agent_type__Biker',
        n_test=args.n_leftouts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--dataset_name', default='sdd', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_round', default=1, type=int)
    # files 
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--tuned', default=None, type=str, nargs='+')
    parser.add_argument('--tuned_name', default=None, type=str, nargs='+')
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--files', default=None, type=str, nargs='+')
    parser.add_argument('--n_leftouts', default=None, type=int, nargs='+')
    parser.add_argument('--scenes', default=None, type=str, nargs='+')
    parser.add_argument('--depth', default=0, type=int)

    args=parser.parse_args()

    main(args)
