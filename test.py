import os
import yaml
import time

from models.trainer import YNetTrainer
from utils.parser import get_parser
from utils.util import get_adapter_info
from utils.write_files import write_csv, get_out_dir
from utils.dataset import set_random_seeds, dataset_split


def main(args):
    # ## configuration
    tic = time.time()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(params['data_dir'], params['dataset_name'], 'segmentation_model.pth')
    params.update(vars(args))
    print(params)

    # ## set up data
    print('############ Prepare dataset ##############')
    IMAGE_PATH = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    DATA_PATH = os.path.join(params['data_dir'], params['dataset_name'], args.dataset_path)

    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.val_files, args.val_ratio, args.n_leftouts)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.val_files, args.val_ratio)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ## model
    print('############ Load model ##############')
    if 'adapter' in args.tuned_ckpt:
        params = get_adapter_info(args.tuned_ckpt, params)
    model = YNetTrainer(params=params)

    if args.pretrained_ckpt is not None and args.tuned_ckpt is not None:
        model.load_separated_params(args.pretrained_ckpt, args.tuned_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt} and {args.tuned_ckpt}")
    elif args.pretrained_ckpt is not None:
        if args.train_net == "modulator":
            model.model.initialize_style()
        model.load_params(args.pretrained_ckpt)
        print(f"Loaded checkpoint {args.pretrained_ckpt}")
    else:
        raise ValueError("No checkpoint given!")
    
    # ## test
    print('############ Test model ##############')
    set_random_seeds(args.seed)
    ade, fde, _ = model.test(df_test, IMAGE_PATH, args.train_net == "modulator")
    if params['out_csv_dir'] is not None:
        out_dir = get_out_dir(params['out_csv_dir'], args.dataset_path, args.seed, args.train_net, args.val_files)
        write_csv(out_dir, 'out-of-domain.csv', [ade], [fde])

    # time
    toc = time.time()
    print(time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)))


if __name__ == '__main__':
    parser = get_parser(train=False)
    args = parser.parse_args()
    main(args)


# python -m pdb test.py --seed 1 --batch_size 10 --dataset_path filter/agent_type/deathCircle_0/ --val_files Biker.pkl --val_ratio 0.1 --n_leftouts 10 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type_deathCircle_0__train_adapter_parallel__0__TrN_8.pt