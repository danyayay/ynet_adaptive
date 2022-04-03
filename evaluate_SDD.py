import os
import yaml
import time
import pandas as pd

from model import YNetTrainer
from utils.parser import get_parser
from utils.write_files import write_csv
from utils.dataset import set_random_seeds, split_df_ratio


def main():
    # ## arg
    tic = time.time()
    args = get_parser(train=False)
    set_random_seeds(args.seed)
    print(args)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ## config file
    # yaml config file containing all the hyperparameters
    CONFIG_FILE_PATH = os.path.join('config', 'sdd_raw_eval.yaml')
    with open(CONFIG_FILE_PATH) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(
        args.data_dir, 'ynet_additional_files', 'segmentation_models', 'SDD_segmentation.pth')

    # ## set up data
    TEST_IMAGE_PATH = os.path.join(
        args.data_dir, args.dataset_name, 'raw', 'annotations')
    assert os.path.isdir(TEST_IMAGE_PATH), 'raw data dir error'
    DATA_PATH = os.path.join(
        args.data_dir, args.dataset_name, args.dataset_path)

    df_test = pd.concat([pd.read_pickle(os.path.join(
        DATA_PATH, test_file)) for test_file in args.val_files])
    _, df_test = split_df_ratio(df_test, args.val_ratio)

    # ## model
    model = YNetTrainer(
        obs_len=params['OBS_LEN'], pred_len=params['PRED_LEN'], params=params)
    if args.ckpt is not None:
        if args.train_net == "modulator":
            model.model.initialize_style()
        model.load(args.ckpt)
        print(f"Loaded checkpoint {args.ckpt}")
    else:
        raise ValueError("No checkpoint given!")

    # ## evaluate
    ade, fde = model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
                              batch_size=args.batch_size, rounds=args.rounds,
                              num_goals=params['NUM_GOALS'], num_traj=params['NUM_TRAJ'], 
                              device=None, dataset_name=args.dataset_name,
                              use_raw_data=params['use_raw_data'], with_style=args.train_net == "modulator")

    # time
    toc = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))

    if args.out_csv_dir is not None:
        write_csv(args.out_csv_dir, args.seed, [ade], [
                  fde], 0, 0, "eval", args.dataset_path, args.val_files)


if '__name__' == '__main__':
    main()
