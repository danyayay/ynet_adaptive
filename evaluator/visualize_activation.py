import pandas as pd
from utils.dataset import set_random_seeds, dataset_split, dataset_given_scenes
from utils.parser import get_parser
from utils.util import get_params, get_image_and_data_path, get_ckpt_name, restore_model
from utils.visualize import plot_feature_space_compare


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)
    params.update({'decision': 'map'})

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    elif args.scenes:
        df_test = dataset_given_scenes(DATA_PATH, args.val_files, args.scenes)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.val_files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts
    if args.pretrained_ckpt is not None:
        ckpts = [args.pretrained_ckpt] + args.tuned_ckpts
        ckpts_name = ['OODG'] + [get_ckpt_name(ckpt) for ckpt in args.tuned_ckpts]
    else:
        raise ValueError('No checkpoints provided')

    # select most significant ones 
    if args.n_limited is not None:
        df_result = pd.read_csv(
            './csv/comparison/1__filter_agent_type_deathCircle_0__Biker/OODG_encoder_0(20)_encoder_0-1(20).csv')
        df_result.loc[:, 'diff_ade'] = df_result['ade_OODG'].values - df_result['ade_encoder_0(20)'].values
        meta_ids_limited = df_result.sort_values(
            by='diff_ade', ascending=False).head(args.n_limited).metaId.values
        df_test = df_test[df_test.metaId.isin(meta_ids_limited)]
        print(f"df_test_limited: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # main  
    ckpts_hook_dict = {}
    for ckpt, ckpt_name in zip(ckpts, ckpts_name):
        ckpts_hook_dict[ckpt_name] = {}
        print(f'====== {ckpt_name} ======')

        # model 
        model = restore_model(params, ckpt_name, args.pretrained_ckpt, ckpt)
        model.model.eval()

        # register layer 
        layers_dict = {
            'encoder.stages.0.0': model.model.encoder.stages[0][0],
            'encoder.stages.0.1': model.model.encoder.stages[0][1],
            'encoder.stages.1.4': model.model.encoder.stages[1][4],
            'encoder.stages.2.4': model.model.encoder.stages[2][4], 
            'encoder.stages.3.4': model.model.encoder.stages[3][4], 
            'encoder.stages.4.4': model.model.encoder.stages[4][4],
            'encoder.stages.5.0': model.model.encoder.stages[5][0],
        }
        layers_hook = {}
        for layer_name, layer in layers_dict.items():
            layer.register_forward_hook(hook_store_output)
            layers_hook[layer_name] = layer
        layer = model.model.encoder.stages[0][0]
        layer.register_forward_hook(hook_store_input)
        layers_hook['encoder.stages.0.0_input'] = layer
            
        # forward 
        pred_goal_map, pred_traj_map, raw_img = model.forward_test(
            df_test, IMAGE_PATH, require_input_grad=False, noisy_std_frac=None) 
        
        # store 
        for layer_name in layers_dict.keys():
            ckpts_hook_dict[ckpt_name][layer_name] = layers_hook[layer_name].output
        ckpts_hook_dict[ckpt_name]['encoder.stages.0.0_input'] = \
            layers_hook['encoder.stages.0.0_input'].input
        ckpts_hook_dict[ckpt_name]['goal_decoder.predictor'] = pred_goal_map
        ckpts_hook_dict[ckpt_name]['traj_decoder.predictor'] = pred_traj_map
        
        for key, value in ckpts_hook_dict[ckpt_name].items():
            print(key, value.shape)

    # plot 
    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}" 
    index = df_test.groupby(by=['metaId', 'sceneId']).count().index
    plot_feature_space_compare(ckpts_hook_dict, index, 
        f'figures/feature_space_compare/{folder_name}', 
        compare_raw=True, compare_diff=True, compare_overlay=True, raw_img=raw_img)
    

def hook_store_input(module, input, output):
    module.input = input[0]


def hook_store_output(module, input, output): 
    module.output = output


if __name__ == '__main__':
    parser = get_parser(False)
    # visualization
    parser.add_argument('--tuned_ckpts', default=None, type=str, nargs='+')
    parser.add_argument('--n_limited', default=None, type=int)
    args = parser.parse_args()
    main(args)


# python -m pdb -m evaluator.visualize_activation --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__encoder_0__TrN_20.pt --val_files Biker.pkl --n_leftouts 10 