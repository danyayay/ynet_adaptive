import os
import yaml
import time
import numpy as np
import pandas as pd

from model import YNetTrainer
from utils.parser import get_parser
from utils.write_files import write_csv, get_out_dir
from utils.dataset import set_random_seeds, limit_samples, dataset_split

import warnings
warnings.filterwarnings("ignore")


# ## configuration
tic = time.time()
args = get_parser(train=True)
set_random_seeds(args.seed)
if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

with open(os.path.join('config', 'sdd_raw_train.yaml')) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

params['segmentation_model_fp'] = os.path.join(args.data_dir, args.dataset_name, 'segmentation_model.pth')
params['share_val_test'] = True
params.update(vars(args))
# set lr depending on the model
# if args.train_net == 'modulator': params['lr'] = 0.01
print(params)

# ## set up data
print('############ Prepare dataset ##############')
IMAGE_PATH = os.path.join(args.data_dir, args.dataset_name, 'raw', 'annotations')
assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
DATA_PATH = os.path.join(args.data_dir, args.dataset_name, args.dataset_path)

if args.train_files == args.val_files:
    # train_files and val_files are fully overlapped 
    df_train, df_val, df_test = dataset_split(
        DATA_PATH, args.train_files, args.val_ratio, args.n_leftouts, 
        share_val_test=params['share_val_test'])
else:
    # train_files and val_files are fully non-overlapped
    df_train, _, df_test = dataset_split(
        DATA_PATH, args.train_files, 0, args.n_leftouts)
    df_val = pd.concat([pd.read_pickle(os.path.join(DATA_PATH, val_file)) for val_file in args.val_files])
df_train = limit_samples(df_train, args.n_train_batch, args.batch_size)
print(f"df_train: {df_train.shape}; #={df_train.shape[0]/(params['obs_len']+params['pred_len'])}")
if df_val is not None: print(f"df_val: {df_val.shape}; #={df_val.shape[0]/(params['obs_len']+params['pred_len'])}")
if df_test is not None: print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

# ## model
model = YNetTrainer(params=params)

if args.train_net == "modulator": model.model.initialize_style()

if args.ckpt is not None:
    model.load(args.ckpt)
    print(f"Loaded checkpoint {args.ckpt}")
else:
    print("Training from scratch")

EXPERIMENT_NAME = ""
EXPERIMENT_NAME += f"Seed_{args.seed}"
EXPERIMENT_NAME += f"_Train_{'_'.join(['_'+f.split('.pkl')[0] for f in args.train_files])}_"
EXPERIMENT_NAME += f"_Val_{'_'.join(['_'+f.split('.pkl')[0] for f in args.val_files])}_"
EXPERIMENT_NAME += f"_Val_Ratio_{args.val_ratio}"
EXPERIMENT_NAME += f"_{(args.dataset_path).replace('/', '_')}"
EXPERIMENT_NAME += f"_train_{args.train_net}"
print(f"Experiment {EXPERIMENT_NAME} has started")
# TODO: modify experiment name 

# ## training
print('############ Train model ##############')
val_ade, val_fde = model.train(df_train, df_val, IMAGE_PATH, IMAGE_PATH, EXPERIMENT_NAME)

# test for leftout data 
if args.out_csv_dir and args.n_leftouts:
    print('############ Test leftout data ##############')
    set_random_seeds(args.seed)
    # test
    test_ade, test_fde, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
    # save csv results
    out_dir = get_out_dir(args.out_csv_dir, args.dataset_path, args.seed, args.train_net, args.val_files, args.train_files)
    write_csv(out_dir, 'fine_tune.csv', val_ade, val_fde, test_ade, test_fde)

# test for fine tuning
# if args.out_csv_dir and args.fine_tune:
#     print('############ Test finetune data ##############')
#     # test 
#     ade_final, fde_final = model.test(df_val, IMAGE_PATH, args.train_net == "modulator")
#     print(f'ade_final: {ade_final}, fde_final: {fde_final}')
#     # save csv results
#     n_train_batch = len(df_train) // ((params['OBS_LEN'] + params['PRED_LEN']) * args.batch_size)
#     out_dir = get_out_dir(args.out_csv_dir, args.dataset_path, args.seed, args.train_net, args.val_files, args.train_files)
#     write_csv(out_dir, f'{n_train_batch}.csv', val_ade, val_fde, ade_final, fde_final)

toc = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))
