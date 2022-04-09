import os
import yaml
import time
import pandas as pd

from model import YNetTrainer
from utils.parser import get_parser
from utils.write_files import write_csv, get_out_dir
from utils.dataset import set_random_seeds, dataset_split


# ## configuration
tic = time.time()
args = get_parser(train=False)
set_random_seeds(args.seed)
if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

params['segmentation_model_fp'] = os.path.join(args.data_dir, args.dataset_name, 'segmentation_model.pth')
params.update(vars(args))
print(params)

# ## set up data
print('############ Prepare dataset ##############')
IMAGE_PATH = os.path.join(args.data_dir, args.dataset_name, 'raw', 'annotations')
assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
DATA_PATH = os.path.join(args.data_dir, args.dataset_name, args.dataset_path)

if args.n_leftouts:
    _, _, df_test = dataset_split(DATA_PATH, args.val_files, args.val_ratio, args.n_leftouts)
else:
    _, df_test, _ = dataset_split(DATA_PATH, args.val_files, args.val_ratio)
print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

# ## model
print('############ Load model ##############')
model = YNetTrainer(params=params)
if args.ckpt is not None:
    if args.train_net == "modulator":
        model.model.initialize_style()
    model.load(args.ckpt)
    print(f"Loaded checkpoint {args.ckpt}")
else:
    raise ValueError("No checkpoint given!")

# ## test
print('############ Test model ##############')
ade, fde, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
if args.out_csv_dir is not None:
    out_dir = get_out_dir(args.out_csv_dir, args.dataset_path, args.seed, args.train_net, args.val_files)
    write_csv(out_dir, 'out-of-domain.csv', [ade], [fde])

# time
toc = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))