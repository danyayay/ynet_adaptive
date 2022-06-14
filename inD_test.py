import os
import time
import numpy as np 
import pandas as pd

from models.inD_trainer import YNetTrainer
from inD_utils.parser import get_parser
from inD_utils.util import get_experiment_name, get_params, get_image_and_data_path
from inD_utils.dataset import set_random_seeds, prepare_dataeset
from evaluator.visualization import plot_given_trajectories_scenes_overlay


def main(args):
    # config 
    tic = time.time()
    set_random_seeds(args.seed)
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)

    # load data 
    df_train, df_val, df_test = prepare_dataeset(
        DATA_PATH, args.load_data, args.batch_size, args.n_train_batch, 
        args.train_files, args.val_files, args.val_split, args.test_splits, 
        args.shuffle, args.share_val_test, 'train', args.hide_data_details)

    # model
    model = YNetTrainer(params=params)
    if args.train_net == "modulator": model.model.initialize_style()
    if args.pretrained_ckpt is not None:
        model.load_params(args.pretrained_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt}")
    else:
        print("Training from scratch")

    # test for leftout data 
    print('############ Test ##############')
    set_random_seeds(args.seed)
    test_ade, test_fde, _, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")

    toc = time.time()
    print('Time spent:', time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(True)
    args = parser.parse_args()

    main(args)
