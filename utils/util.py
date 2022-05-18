import os 
import yaml 
from models.trainer import YNetTrainer


def get_experiment_name(args, n_train):
    experiment = ""
    experiment += f"Seed_{args.seed}_"
    experiment += f"_Tr{'_'.join(['_'+f.split('.pkl')[0] for f in args.train_files])}_"
    experiment += f"_Val{'_'.join(['_'+f.split('.pkl')[0] for f in args.val_files])}_"
    experiment += f"_ValRatio_{args.val_ratio}_"
    experiment += f"_{(args.dataset_path).replace('/', '_')}"
    experiment += f"_{args.train_net}"
    if args.position != []: experiment += f'__Pos_{"_".join(map(str, args.position))}' 
    experiment += f'__TrN_{str(int(n_train/20))}'
    if args.fine_tune: experiment += f'__lr_{args.lr}'
    if args.is_augment_data: experiment += '__AUG'
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
    if 'Pos' in ckpt_path:
        position = [int(i) for i in ckpt_path.split('__')[6].split('_') if i != 'Pos']
        if return_list:
            return position
        else:
            return '_'.join(map(str, position))
    else:
        return None 


def get_ckpt_name(ckpt_path):
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
    train_net = ckpt_path.split('__')[5]
    updated_params = params.copy()
    updated_params.update({'train_net': train_net})
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
    if (not is_file_separated) or ('all' in base_ckpt) or ('train' in base_ckpt):
        model = YNetTrainer(params=params)
        model.load_params(base_ckpt)
    else:  
        updated_params = update_params(separated_ckpt, params)
        model = YNetTrainer(params=updated_params)
        model.load_separated_params(base_ckpt, separated_ckpt)    
    return model 