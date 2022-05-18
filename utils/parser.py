import argparse

__all__ = ['get_parser']


def get_data_args(parser):
    parser.add_argument("--dataset_path", default='sherwin/dataset_ped_biker/gap', type=str)
    parser.add_argument("--val_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--n_leftouts", default=None, type=int, nargs='+', help='The number of data left for testing')
    parser.add_argument("--share_val_test", default=True, type=bool)
    return parser 


def get_model_args(parser):
    # either
    parser.add_argument("--ckpts", default=None, type=str, nargs='+')
    parser.add_argument("--ckpts_name", default=None, type=str, nargs='+')
    # or 
    parser.add_argument("--pretrained_ckpt", default=None, type=str)
    parser.add_argument("--tuned_ckpt", default=None, type=str)
    parser.add_argument('--tuned_ckpts', default=None, type=str, nargs='+')
    
    # added net 
    parser.add_argument('--position', default=None, type=int, nargs='+')
    parser.add_argument("--train_net", default="all", type=str, 
        help="train all parameters or only encoder or with modulator")
    return parser


def get_general_args(parser):
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gpu", default=None, type=int, help='gpu id to use')
    parser.add_argument("--n_round", default=1, type=int, help='number of rounds in stochastics eval process')  
    return parser


def get_train_args(parser):
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--train_files", default=["0.25_0.75.pkl", "1.25_1.75.pkl", "2.25_2.75.pkl"], type=str, nargs="+")
    parser.add_argument("--n_epoch", default=1, type=int)
    parser.add_argument("--n_train_batch", default=None, type=int, help="Limited number of batches for each training agent (fine-tuning), None means no limit (training)")
    parser.add_argument("--is_augment_data", action='store_true')
    # learning rate 
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--steps", default=[20], type=int, nargs='+')
    parser.add_argument("--lr_decay_ratio", default=0.1)
    parser.add_argument("--config_filename", default='sdd_raw_train.yaml', type=str)
    return parser


def get_eval_args(parser):
    parser.add_argument("--config_filename", default='sdd_raw_eval.yaml', type=str)
    return parser


def get_parser(is_train):
    parser = argparse.ArgumentParser()
    parser = get_data_args(parser)
    parser = get_model_args(parser)
    parser = get_general_args(parser)
    if is_train:
        parser = get_train_args(parser)
    else:
        parser = get_eval_args(parser)
    return parser