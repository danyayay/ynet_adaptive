import argparse

def get_general_parser():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gpu", default=None, type=int, help='gpu id to use')
    # dataset 
    parser.add_argument("--dataset_path", default='sherwin/dataset_ped_biker/gap', type=str)
    parser.add_argument("--val_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
    parser.add_argument("--val_ratio", default=0.1, type=float)
    # model
    parser.add_argument("--train_net", default="all", type=str, 
        help="train all parameters or only encoder or with modulator")
    parser.add_argument("--pretrained_ckpt", default=None, type=str)
    parser.add_argument("--tuned_ckpt", default=None, type=str)
    parser.add_argument("--n_round", default=1, type=int, help='number of rounds in stochastics eval process')  
    parser.add_argument("--n_leftouts", default=None, type=int, nargs='+', help='The number of data left for testing')\
    # adapter
    parser.add_argument('--adapter_type', default=None, type=str)
    parser.add_argument('--adapter_position', default=None, type=int, nargs='+')
    parser.add_argument('--adapter_initialization', default='zero', type=str)
    return parser


def get_parser(train):
    parser = get_general_parser()
    if train:
        parser.add_argument("--n_epoch", default=1, type=int)
        parser.add_argument("--train_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
        parser.add_argument("--n_train_batch", default=None, type=int, help="Limited number of batches for each training agent (fine-tuning), None means no limit (training)")
        parser.add_argument("--fine_tune", action="store_true")
        parser.add_argument("--lr", default=0.0001, type=float)
        parser.add_argument("--steps", default=[20], type=int, nargs='+')
        parser.add_argument("--lr_decay_ratio", default=0.1)
        parser.add_argument("--share_val_test", default=True, type=bool)
        parser.add_argument("--config_filename", default='sdd_raw_train.yaml', type=str)
    else:
        parser.add_argument("--config_filename", default='sdd_raw_eval.yaml', type=str)
    return parser