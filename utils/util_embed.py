import os 
import yaml 
import numpy as np 
from models.trainer_embed import YNetTrainer


def get_experiment_name(args, n_train):
    experiment = ""
    experiment += f"Seed_{args.seed}_"
    experiment += f"_Tr{'_'.join(['_'+f.split('.pkl')[0] for f in args.train_files])}_"
    experiment += f"_Val{'_'.join(['_'+f.split('.pkl')[0] for f in args.val_files])}_"
    experiment += f"_ValRatio_{args.val_split}_"
    experiment += f"_{(args.dataset_path).replace('/', '_')}"
    experiment += f"_{args.train_net}"
    if args.position != []: experiment += f'__Pos_{"_".join(map(str, args.position))}' 
    experiment += f'__TrN_{str(int(n_train/20))}'
    if args.fine_tune: experiment += f'__lr_{np.format_float_positional(args.lr, trim="-")}'
    if args.is_augment_data: experiment += '__AUG'
    if args.ynet_bias: experiment += '__bias'
    if 'original' not in args.pretrained_ckpt: 
        base_arch = args.pretrained_ckpt.split('__')[-1].split('.')[0]
        experiment += f'__{base_arch}' 
    return experiment


def get_params(args):
    with open(os.path.join('config', args.config_filename)) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params['segmentation_model_fp'] = os.path.join(
        params['data_dir'], params['dataset_name'], 'segmentation_model.pth')
    params.update(vars(args))
    print(params)
    return params 


def get_image_and_data_path(params):
    # image path 
    image_path = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    assert os.path.isdir(image_path), 'image dir error'
    # data path 
    data_path = os.path.join(params['data_dir'], params['dataset_name'], params['dataset_path'])
    assert os.path.isdir(data_path), f'data dir error'
    return image_path, data_path 


def get_position(ckpt_path, return_list=True):
    if ckpt_path is not None:
        if 'Pos' in ckpt_path:
            pos = ckpt_path.split('__')[6].replace('Pos_', '')
            if not return_list:
                return pos
            else:
                pos_list = [i for i in pos.split('_')]
                return pos_list
        else:
            return None
    else:
        return None 


def get_ckpt_name(ckpt_path):
    ckpt_path = ckpt_path.split('/')[-1]
    train_net = ckpt_path.split('__')[5]
    if 'Pos' in ckpt_path:
        position = get_position(ckpt_path, return_list=False)
        n_train = ckpt_path.split('__')[7].split('_')[1]
        ckpt_name = f'{train_net}[{position}]({n_train})'
    else:
        n_train = ckpt_path.split('__')[6].split('_')[1]
        ckpt_name = f'{train_net}({n_train})'
    return ckpt_name 


def update_params(ckpt_path, params):
    ckpt_path = ckpt_path.split('/')[-1]
    updated_params = params.copy()
    # train net
    train_net = ckpt_path.split('__')[5].split('.')[0]
    updated_params.update({'train_net': train_net})
    # base 
    base_arch = params['pretrained_ckpt'].split('_')[-1].split('.')[0]
    if base_arch == 'embed':
        updated_params.update({'add_embedding': True})
    elif 'fusion' in base_arch:
        update_params.update({'n_fusion': int(base_arch.split('_')[-1])}) 
    # position     
    if 'Pos' in ckpt_path:
        position = get_position(ckpt_path)
        updated_params.update({'position': position})
    return updated_params


def get_ckpts_and_names(ckpts, ckpts_name, pretrained_ckpt, tuned_ckpts):
    if ckpts is not None:
        ckpts, ckpts_name = ckpts, ckpts_name
        is_file_separated = False
    elif pretrained_ckpt is not None:
        ckpts = [pretrained_ckpt] + tuned_ckpts
        ckpts_name = ['OODG'] + [get_ckpt_name(ckpt) for ckpt in tuned_ckpts]
        is_file_separated = True
    else:
        raise ValueError('No checkpoint provided')
    return ckpts, ckpts_name, is_file_separated


def restore_model(
    params, is_file_separated, 
    base_ckpt, separated_ckpt=None):
    if not is_file_separated:
        model = YNetTrainer(params=params)
        model.load_params(base_ckpt)
    else:  
        updated_params = update_params(separated_ckpt, params)
        model = YNetTrainer(params=updated_params)
        model.load_separated_params(base_ckpt, separated_ckpt)    
    return model 