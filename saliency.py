import os
import yaml
import torch
import numpy as np

import torch.nn.functional as F

from models.trainer import YNetTrainer
from utils.parser import get_parser
from utils.dataset import reduce_df_meta_ids, set_random_seeds, dataset_split, dataset_given_scenes
from utils.visualize import plot_saliency_maps, plot_prediction


def main(args):
    # ## configuration
    with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(
        params['data_dir'], params['dataset_name'], 'segmentation_model.pth')
    params.update(vars(args))
    print(params)

    resize_factor = params['resize_factor']

    # ## set up data 
    print('############ Prepare dataset ##############')
    IMAGE_PATH = os.path.join(params['data_dir'], params['dataset_name'], 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    DATA_PATH = os.path.join(params['data_dir'], params['dataset_name'], args.dataset_path)

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    elif args.scenes:
        df_test = dataset_given_scenes(DATA_PATH, args.val_files, args.scenes)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.val_files, 0)
    
    # select the most significant trajs
    if args.meta_ids is not None:
        meta_ids_focus = args.meta_ids
        df_test = reduce_df_meta_ids(df_test, np.array(meta_ids_focus))
    else:
        meta_ids_focus = df_test.metaId.unique()
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")

    # ckpts
    load_separated_files = False 
    if args.ckpts is not None:
        ckpts, ckpts_name = args.ckpts, args.ckpts_name
    elif args.pretrained_ckpt is not None:
        ckpts = [args.pretrained_ckpt] + args.tuned_ckpts
        ckpts_name = ['OODG'] + [get_ckpt_name(ckpt) for ckpt in args.tuned_ckpts]
        load_separated_files = True
    else:
        raise ValueError('No checkpoints provided')

    ckpt_trajs_dict = {}

    # folder name 
    folder_name = f"{str(args.seed)}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}"

    # plot
    if args.decision == 'loss':
        for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
            print(f'====== Testing for {ckpt_name} ======')
            set_random_seeds(args.seed)
            model = YNetTrainer(params=params)
            model.load_params(ckpt)
            model.model.eval()

            # select data
            for meta_id in meta_ids_focus:
                df_meta = df_test[df_test.metaId == meta_id]
                scene_id = df_meta.sceneId.values[0]
                folder_name = str(args.seed) + '__' + ckpt.split('_filter_')[1].split('__')[0] + '__' + '_'.join(args.val_files).rstrip('.pkl')

                if args.VanillaGrad:
                    goal_loss, traj_loss, input = model.forward_test(
                        df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=None)
                    loss = goal_loss + traj_loss
                    # get gradient 
                    grad_input_goal, = torch.autograd.grad(goal_loss, input, retain_graph=True)
                    grad_input_traj, = torch.autograd.grad(traj_loss, input, retain_graph=True)
                    grad_input_total, = torch.autograd.grad(loss, input)
                    plot_saliency_maps(input, grad_input_goal, 'vanilla_grad', 
                        f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__goal', 
                        f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')
                    plot_saliency_maps(input, grad_input_traj, 'vanilla_grad', 
                        f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__traj', 
                        f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')
                    plot_saliency_maps(input, grad_input_total, 'vanilla_grad', 
                        f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__total', 
                        f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')

                if args.SmoothGrad:
                    for std_frac in [0.1, 0.15, 0.2, 0.25]:
                        for i in range(args.n_smooth):
                            goal_loss, traj_loss, input, noisy_input = model.forward_test(
                                df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=std_frac)
                            loss = goal_loss + traj_loss 
                            # get gradient 
                            grad_input_goal, = torch.autograd.grad(goal_loss, noisy_input, retain_graph=True)
                            grad_input_traj, = torch.autograd.grad(traj_loss, noisy_input, retain_graph=True)
                            grad_input_total, = torch.autograd.grad(loss, noisy_input)
                            if i == 0:
                                smooth_grad_goal = grad_input_goal
                                smooth_grad_traj = grad_input_traj
                                smooth_grad_total = grad_input_total 
                            else: 
                                smooth_grad_goal += grad_input_goal
                                smooth_grad_traj += grad_input_traj
                                smooth_grad_total += grad_input_total
                        plot_saliency_maps(input, smooth_grad_goal, 'smooth_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__goal__std_{std_frac}', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')
                        plot_saliency_maps(input, smooth_grad_traj, 'smooth_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__traj__std_{std_frac}', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')
                        plot_saliency_maps(input, smooth_grad_total, 'smooth_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__total__std_{std_frac}', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')

                if args.GradCAM:                
                    # specify layer
                    layers_dict = {
                        'encoder.stages.0.0': model.model.encoder.stages[0][0],
                        'encoder.stages.1.3': model.model.encoder.stages[1][3],
                        'encoder.stages.2.3': model.model.encoder.stages[2][3], 
                        'encoder.stages.4.3': model.model.encoder.stages[4][3],
                        'goal_decoder.decoder.4.2': model.model.goal_decoder.decoder[4][2],
                        'goal_decoder.predictor': model.model.goal_decoder.predictor,
                        'traj_decoder.decoder.4.2': model.model.traj_decoder.decoder[4][2],
                        'traj_decoder.predictor': model.model.traj_decoder.predictor
                    }               

                    for layer_name, layer in layers_dict.items():
                        layer.register_forward_hook(hook_store_output) 
                        layer.register_full_backward_hook(hook_store_grad)

                        # forward and backward 
                        goal_loss, traj_loss, input = model.forward_test(
                            df_meta, IMAGE_PATH, require_input_grad=False, noisy_std_frac=None) 
                        loss = goal_loss + traj_loss 

                        # get gradient 
                        if 'traj_decoder' not in layer_name: 
                            goal_loss.backward(retain_graph=True)
                            L_goal = torch.relu((layer.grad * layer.output).sum(1, keepdim = True))
                            L_goal = F.interpolate(L_goal, size = (input.size(2), input.size(3)), 
                                mode = 'bilinear', align_corners = False)
                            plot_saliency_maps(input, L_goal, 'grad_cam', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__goal', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')

                        traj_loss.backward(retain_graph=True)
                        L_traj = torch.relu((layer.grad * layer.output).sum(1, keepdim = True))
                        L_traj = F.interpolate(L_traj, size = (input.size(2), input.size(3)), 
                            mode = 'bilinear', align_corners = False)
                        plot_saliency_maps(input, L_traj, 'grad_cam', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__traj', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')

                        loss.backward(retain_graph=True)
                        L_total = torch.relu((layer.grad * layer.output).sum(1, keepdim = True))
                        L_total = F.interpolate(L_total, size = (input.size(2), input.size(3)), 
                            mode = 'bilinear', align_corners = False)
                        plot_saliency_maps(input, L_total, 'grad_cam', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__total', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}')
    
    elif args.decision == 'map':
        for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
            print(f'====== Testing for {ckpt_name} ======')
            set_random_seeds(args.seed)
            # load parameters 
            if load_separated_files and ckpt_name != 'OODG':
                updated_params = get_adapter_info(ckpt, params)
                model = YNetTrainer(params=updated_params)
                model.load_separated_params(ckpts[0], ckpt)
            else:
                model = YNetTrainer(params=params)
                model.load_params(ckpt)
            model.model.eval()
            time_step = -1
            breakpoint()

            # test 
            _, _, _, list_trajs = model.test(df_test, IMAGE_PATH, False, True, False)
            trajs_dict = list_trajs[0]
            ckpt_trajs_dict[ckpt_name] = trajs_dict

            for meta_id in meta_ids_focus:
                # select df
                df_meta = df_test[df_test.metaId == meta_id]
                scene_id = df_meta.sceneId.values[0]
                
                # select trajs 
                indice_gt, indice_pred = get_gt_pred_indice(
                    trajs_dict, meta_id, time_step, resize_factor)

                if args.VanillaGrad:
                    method_name = 'vanilla_grad'
                    pred_goal_map, pred_traj_map, input = model.forward_test(
                        df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=None)
                    # find the decision of interest
                    point_goal_prob, indice_goal_prob = get_most_likely_point_and_indice(pred_goal_map, time_step)
                    point_traj_prob, indice_traj_prob = get_most_likely_point_and_indice(pred_traj_map, time_step)
                    point_goal_gt, point_goal_pred = get_gt_pred_point(pred_goal_map, indice_gt, indice_pred, time_step)
                    point_traj_gt, point_traj_pred = get_gt_pred_point(pred_traj_map, indice_gt, indice_pred, time_step)
                    # get gradient 
                    grad_goal_prob, = torch.autograd.grad(point_goal_prob, input, retain_graph=True)
                    grad_traj_prob, = torch.autograd.grad(point_traj_prob, input, retain_graph=True)
                    grad_goal_gt, = torch.autograd.grad(point_goal_gt, input, retain_graph=True)
                    grad_goal_pred, = torch.autograd.grad(point_traj_gt, input, retain_graph=True)
                    grad_traj_gt, = torch.autograd.grad(point_traj_gt, input, retain_graph=True)
                    grad_traj_pred, = torch.autograd.grad(point_traj_pred, input)
                    # plot
                    plot_dict = {
                        'prob__goal': {'grad': grad_goal_prob, 'best_point': indice_goal_prob},
                        'prob__traj': {'grad': grad_traj_prob, 'best_point': indice_traj_prob},
                        'gt__goal': {'grad': grad_goal_gt, 'best_point': indice_gt},
                        'gt__traj': {'grad': grad_traj_gt, 'best_point': indice_gt},
                        'pred__goal': {'grad': grad_goal_pred, 'best_point': indice_pred},
                        'pred__traj': {'grad': grad_traj_pred, 'best_point': indice_pred}
                    }
                    for name, v_dict in plot_dict.items():
                        plot_saliency_maps(input, v_dict['grad'], method_name, 
                            f'{ckpt_name}__{scene_id}__{meta_id}__{method_name}__{name}', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}', 
                            side_by_side=False, best_point=v_dict['best_point'])

                if args.SmoothGrad:
                    method = 'smooth_grad'
                    for std_frac in [0.1, 0.15, 0.2, 0.25]:
                        for i in range(args.n_smooth):
                            pred_goal_map, pred_traj_map, input, noisy_input = model.forward_test(
                                df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=std_frac)
                            # find the decision of interest
                            point_goal_gt, point_goal_pred = get_gt_pred_point(
                                pred_goal_map, indice_gt, indice_pred, time_step)
                            point_traj_gt, point_traj_pred = get_gt_pred_point(
                                pred_traj_map, indice_gt, indice_pred, time_step)
                            # get gradient 
                            grad_goal_gt, = torch.autograd.grad(point_goal_gt, noisy_input, retain_graph=True)
                            grad_goal_pred, = torch.autograd.grad(point_goal_pred, noisy_input, retain_graph=True)
                            grad_traj_gt, = torch.autograd.grad(point_traj_gt, noisy_input, retain_graph=True)
                            grad_traj_pred, = torch.autograd.grad(point_traj_pred, noisy_input)
                            if i == 0:
                                smooth_grad_goal_gt = grad_goal_gt
                                smooth_grad_goal_pred = grad_goal_pred
                                smooth_grad_traj_gt = grad_traj_gt
                                smooth_grad_traj_pred = grad_traj_pred
                            else: 
                                smooth_grad_goal_gt += grad_goal_gt
                                smooth_grad_goal_pred += grad_goal_pred
                                smooth_grad_traj_gt += grad_traj_gt
                                smooth_grad_traj_pred += grad_traj_pred
                        plot_dict = {
                            'gt__goal': {'grad': smooth_grad_goal_gt, 'best_point': indice_gt},
                            'gt__traj': {'grad': smooth_grad_traj_gt, 'best_point': indice_gt},
                            'pred__goal': {'grad': smooth_grad_goal_pred, 'best_point': indice_pred},
                            'pred__traj': {'grad': smooth_grad_traj_pred, 'best_point': indice_pred}
                        }
                        for name, v_dict in plot_dict.items():
                            plot_saliency_maps(input, v_dict['grad'], method, 
                                f'{ckpt_name}__{scene_id}__{meta_id}__{method}__{name}__std_{std_frac}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}', 
                                side_by_side=False, best_point=v_dict['best_point'])

                if args.GradCAM:
                    method = 'grad_cam'
                    # specify layer
                    layers_dict = {
                        'semantic_segmentation.encoder.conv1': model.model.semantic_segmentation.encoder.conv1,
                        'encoder.stages.0.0': model.model.encoder.stages[0][0],
                        'encoder.stages.1.3': model.model.encoder.stages[1][3],
                        'encoder.stages.2.3': model.model.encoder.stages[2][3], 
                        'encoder.stages.3.3': model.model.encoder.stages[3][3], 
                        'encoder.stages.4.3': model.model.encoder.stages[4][3],
                    }               

                    for layer_name, layer in layers_dict.items():
                        layer.register_forward_hook(hook_store_output) 
                        layer.register_full_backward_hook(hook_store_grad)

                        # forward and backward 
                        pred_goal_map, pred_traj_map, input = model.forward_test(
                            df_meta, IMAGE_PATH, require_input_grad=False, noisy_std_frac=None) 
                        # find the decision of interest
                        point_goal_prob, indice_goal_prob = get_most_likely_point_and_indice(pred_goal_map, time_step)
                        point_traj_prob, indice_traj_prob = get_most_likely_point_and_indice(pred_traj_map, time_step)
                        point_goal_gt, point_goal_pred = get_gt_pred_point(pred_goal_map, indice_gt, indice_pred, time_step)
                        point_traj_gt, point_traj_pred = get_gt_pred_point(pred_traj_map, indice_gt, indice_pred, time_step)
                        # get gradient 
                        if 'traj_decoder' not in layer_name: 
                            point_goal_prob.backward(retain_graph=True)
                            L_goal_prob = compute_grad_cam(layer, input)
                            point_goal_gt.backward(retain_graph=True)
                            L_goal_gt = compute_grad_cam(layer, input)
                            point_goal_pred.backward(retain_graph=True)
                            L_goal_pred = compute_grad_cam(layer, input)
                        point_traj_prob.backward(retain_graph=True)
                        L_traj_prob = compute_grad_cam(layer, input)
                        L_traj_gt = compute_grad_cam(layer, input)
                        point_traj_gt.backward(retain_graph=True)
                        point_traj_pred.backward(retain_graph=True)
                        L_traj_pred = compute_grad_cam(layer, input)
                        # plot
                        plot_dict = {
                            'prob__goal': {'L': L_goal_prob, 'best_point': indice_goal_prob},
                            'prob__traj': {'L': L_traj_prob, 'best_point': indice_traj_prob},
                            'gt__goal': {'L': L_goal_gt, 'best_point': indice_gt},
                            'gt__traj': {'L': L_traj_gt, 'best_point': indice_gt},
                            'pred__goal': {'L': L_goal_pred, 'best_point': indice_pred},
                            'pred__traj': {'L': L_traj_pred, 'best_point': indice_pred},
                        }
                        for name, v_dict in plot_dict.items():
                            plot_saliency_maps(input, v_dict['L'], method, 
                                f'{ckpt_name}__{scene_id}__{meta_id}__{method}__{name}__{layer_name}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}', 
                                side_by_side=False, best_point=v_dict['best_point'])

        plot_prediction(IMAGE_PATH, ckpt_trajs_dict, f'figures/prediction/{folder_name}/{"__".join(ckpts_name)}')


def hook_store_output(module, input, output): 
    module.output = output[0]


def hook_store_grad(module, grad_input, grad_output):
    module.grad = grad_output[0]


def get_most_likely_point_and_indice(map, time_step=-1):
    last_map = map[0, time_step]
    point_prob = last_map.max()
    indice_prob_inverse = (last_map==torch.max(last_map)).nonzero().squeeze().cpu().detach().numpy()
    indice_prob = indice_prob_inverse[::-1]
    return point_prob, indice_prob


def get_gt_pred_indice(trajs_dict, meta_id, time_step=-1, resize_factor=0.25):
    idx = np.where(trajs_dict['metaId'] == meta_id)[0][0]
    traj_gt = trajs_dict['groundtruth'][idx]
    traj_pred = trajs_dict['prediction'][idx] 
    indice_gt = np.around(traj_gt[time_step] * resize_factor).astype(int)
    indice_pred = np.around(traj_pred[time_step] * resize_factor).astype(int)
    return indice_gt, indice_pred


def get_gt_pred_point(map, indice_gt, indice_pred, time_step=-1):
    point_gt = map[0, time_step, indice_gt[0], indice_gt[1]]
    point_pred = map[0, time_step, indice_pred[0], indice_pred[1]]
    return point_gt, point_pred


def compute_grad_cam(layer, input):
    L = torch.relu((layer.grad * layer.output).sum(1, keepdim = True))
    L = F.interpolate(L, size=(input.size(2), input.size(3)), 
        mode='bilinear', align_corners=False)
    return L


def get_ckpt_name(ckpt_path):
    if 'adapter' in ckpt_path:
        train_net = ckpt_path.split('__')[5]
        adapter_position = ckpt_path.split('__')[6]
        n_train = ckpt_path.split('__')[7].split('_')[1].split('.')[0]
        ckpt_name = f'{train_net}[{adapter_position}]({n_train})'
    else:
        train_net = ckpt_path.split('__')[5]
        n_train = ckpt_path.split('__')[6].split('_')[1]
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


if __name__ == '__main__':
    
    parser = get_parser(False)
    # files 
    # TODO: think about which layer to plot (i.e., w.r.t.) (input scene, input trajs..., intermediate layers...)
    parser.add_argument('--ckpts', default=None, type=str, nargs='+')
    parser.add_argument('--ckpts_name', default=None, type=str, nargs='+')
    parser.add_argument('--tuned_ckpts', default=None, type=str, nargs='+')
    # focus samples  
    parser.add_argument('--meta_ids', default=None, type=int, nargs='+')
    parser.add_argument('--n_limited', default=10, type=int)
    # saliency map arguments 
    parser.add_argument('--decision', default=None, type=str, choices=['loss', 'map'])
    parser.add_argument('--VanillaGrad', action='store_true')
    parser.add_argument('--SmoothGrad', action='store_true')
    parser.add_argument('--n_smooth', default=4, type=int)
    parser.add_argument('--GradCAM', action='store_true')

    args=parser.parse_args()

    main(args)


# python -m pdb saliency.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt --ckpts_name OODG FT --val_files Biker.pkl --n_leftouts 10 --meta_id 22796 --decision map --VanillaGrad

# python -m pdb saliency.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --ckpts_name OODG FT ET --val_files Biker.pkl --n_leftouts 10 --meta_id 22796 --VanillaGrad --SmoothGrad --GradCAM

# python -m pdb saliency.py --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_serial__0__TrN_10.pt ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_serial__0__TrN_20.pt --val_files Biker.pkl --n_leftouts 10 --meta_id 22796 --decision map --VanillaGrad