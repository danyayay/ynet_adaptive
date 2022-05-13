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
    if args.fine_tune:
        if args.train_net == 'all':
            experiment += '_FT'
        elif args.train_net == 'adapter':
            experiment += f'_{args.adapter_type}__{"_".join(map(str, args.adapter_position))}__TrN_{str(int(n_train/20))}'
        else:
            experiment += f'__TrN_{str(int(n_train/20))}'
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


def get_ckpt_name(ckpt_path):
    if 'adapter' in ckpt_path:
        train_net = ckpt_path.split('__')[5]
        adapter_position = ckpt_path.split('__')[6]
        n_train = ckpt_path.split('__')[7].split('_')[1].split('.')[0]
        ckpt_name = f'{train_net}[{adapter_position}]({n_train})'
    elif 'weight' in ckpt_path:
        train_net = ckpt_path.split('__')[5]
        n_train = ckpt_path.split('__')[6].split('_')[1]
        ckpt_name = f'{train_net}({n_train})'
    else:
        train_net = ckpt_path.split('__')[5]
        n_train = ckpt_path.split('__')[6].split('_')[1].split('.')[0]
        ckpt_name = f'{train_net}({n_train})'
    return ckpt_name 


def get_adapter_info(ckpt_path, params):
    if 'adapter' in ckpt_path:
        train_net, adapter_type = ckpt_path.split('__')[5].split('_')
        adapter_position = [int(i) for i in ckpt_path.split('__')[6].split('_')]
        updated_params = params.copy()
        updated_params.update({
            'train_net': train_net, 
            'adapter_type': adapter_type, 
            'adapter_position': adapter_position})
        return updated_params
    else:
        raise ValueError(f"{ckpt_path} is not an adapter's model")


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
    ckpt_name, base_ckpt, separated_ckpt=None):
    if (not is_file_separated) or (is_file_separated and ckpt_name == 'OODG'):
        model = YNetTrainer(params=params)
        model.load_params(base_ckpt)
    else:  
        if 'adapter' in ckpt_name:
            updated_params = get_adapter_info(separated_ckpt, params)
            model = YNetTrainer(params=updated_params)
            model.load_separated_params(base_ckpt, separated_ckpt)
        else:
            model = YNetTrainer(params=params)
            model.load_separated_params(base_ckpt, separated_ckpt)
        
    return model 