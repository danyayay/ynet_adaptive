import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd

import PIL
import matplotlib.cm as cm
import torch.nn.functional as F

from model import YNetTrainer
from utils.dataset import set_random_seeds, dataset_split, dataset_given_scenes
from utils.visualize import plot_saliency_maps


def main(args):
    # ## configuration
    with open(os.path.join('config', 'sdd_raw_eval.yaml')) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params['segmentation_model_fp'] = os.path.join(args.data_dir, args.dataset_name, 'segmentation_model.pth')
    params.update(vars(args))
    print(params)

    # ## set up data 
    print('############ Prepare dataset ##############')
    # image path 
    IMAGE_PATH = os.path.join(args.data_dir, args.dataset_name, 'raw', 'annotations')
    assert os.path.isdir(IMAGE_PATH), 'raw data dir error'
    # data path 
    DATA_PATH = os.path.join(args.data_dir, args.dataset_name, args.dataset_path)

    # data 
    if args.n_leftouts:
        _, _, df_test = dataset_split(DATA_PATH, args.files, 0, args.n_leftouts)
    elif args.scenes:
        df_test = dataset_given_scenes(DATA_PATH, args.files, args.scenes)
    else:
        _, df_test, _ = dataset_split(DATA_PATH, args.files, 0)
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    
    # select the most significant trajs
    # metaId,sceneId,ade_OODG,fde_OODG,ade_FT,fde_FT,ade_ET,fde_ET
    if args.meta_ids is None:
        df_comparison = pd.read_csv('csv/comparison/1__agent_type__Biker__OODG_FT_ET.csv')
        df_comparison.loc[:, 'ade_OODG_ET'] = np.absolute(df_comparison.ade_OODG.values - df_comparison.ade_ET.values)
        df_comparison = df_comparison[df_comparison.sceneId != 'avg']
        meta_ids_focus = df_comparison.sort_values(by='ade_OODG_ET', ascending=False).head(args.n_limited).metaId.values
    else:
        meta_ids_focus = args.meta_ids

    # plot
    if args.decision == 'loss':
        if args.ckpts:
            for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
                print(f'====== Testing for {ckpt_name} ======')
                set_random_seeds(args.seed)
                model = YNetTrainer(params=params)
                model.load(ckpt)
                model.model.eval()

                # select data
                for meta_id in meta_ids_focus:
                    df_meta = df_test[df_test.metaId == meta_id]
                    scene_id = df_meta.sceneId.values[0]
                    folder_name = str(args.seed) + '__' + ckpt.split('_filter_')[1].split('__')[0] + '__' + '_'.join(args.files).rstrip('.pkl')

                    # TODO: add observations and predictions to visualization 
                    # TODO: add three kinds of saliency maps in one figures... be careful about the colorbar 

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
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')
                        plot_saliency_maps(input, grad_input_traj, 'vanilla_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__traj', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')
                        plot_saliency_maps(input, grad_input_total, 'vanilla_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__total', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')

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
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')
                            plot_saliency_maps(input, smooth_grad_traj, 'smooth_grad', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__traj__std_{std_frac}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')
                            plot_saliency_maps(input, smooth_grad_total, 'smooth_grad', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__total__std_{std_frac}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')

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
                            layer.register_forward_hook(hook_store_A) 
                            layer.register_full_backward_hook(hook_store_dydA)

                            # forward and backward 
                            goal_loss, traj_loss, input = model.forward_test(
                                df_meta, IMAGE_PATH, require_input_grad=False, noisy_std_frac=None) 
                            loss = goal_loss + traj_loss 

                            # get gradient 
                            if 'traj_decoder' not in layer_name: 
                                goal_loss.backward(retain_graph=True)
                                L_goal = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                                L_goal = F.interpolate(L_goal, size = (input.size(2), input.size(3)), 
                                    mode = 'bilinear', align_corners = False)
                                plot_saliency_maps(input, L_goal, 'grad_cam', 
                                    f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__goal', 
                                    f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')

                            traj_loss.backward(retain_graph=True)
                            L_traj = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                            L_traj = F.interpolate(L_traj, size = (input.size(2), input.size(3)), 
                                mode = 'bilinear', align_corners = False)
                            plot_saliency_maps(input, L_traj, 'grad_cam', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__traj', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')

                            loss.backward(retain_graph=True)
                            L_total = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                            L_total = F.interpolate(L_total, size = (input.size(2), input.size(3)), 
                                mode = 'bilinear', align_corners = False)
                            plot_saliency_maps(input, L_total, 'grad_cam', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__total', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}')
    
    elif args.decision == 'map':
        if args.ckpts:
            for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
                print(f'====== Testing for {ckpt_name} ======')
                set_random_seeds(args.seed)
                model = YNetTrainer(params=params)
                model.load(ckpt)
                model.model.eval()
                # TODO: for FT, traj-related plots mostly have errors when placing the most likely position 

                # select data
                for meta_id in meta_ids_focus:
                    df_meta = df_test[df_test.metaId == meta_id]
                    scene_id = df_meta.sceneId.values[0]
                    folder_name = str(args.seed) + '__' + ckpt.split('_filter_')[1].split('__')[0] + '__' + '_'.join(args.files).rstrip('.pkl')

                    if args.VanillaGrad:
                        pred_goal_map, pred_traj_map, input = model.forward_test(
                            df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=None)
                        # find the most likely position 
                        max_goal_point, max_goal_indice = get_most_likely_point(pred_goal_map)
                        max_traj_point, max_traj_indice = get_most_likely_point(pred_traj_map)
                        # get gradient 
                        grad_input_goal, = torch.autograd.grad(max_goal_point, input, retain_graph=True)
                        grad_input_traj, = torch.autograd.grad(max_traj_point, input)
                        plot_saliency_maps(input, grad_input_goal, 'vanilla_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__goal', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                            side_by_side=False, best_points=max_goal_indice)
                        plot_saliency_maps(input, grad_input_traj, 'vanilla_grad', 
                            f'{ckpt_name}__{scene_id}__{meta_id}__vanilla_grad__traj', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                            side_by_side=False, best_points=max_traj_indice)

                    if args.SmoothGrad:
                        # different std may lead to different most likely point 
                        for std_frac in [0.1, 0.15, 0.2, 0.25]:
                            for i in range(args.n_smooth):
                                pred_goal_map, pred_traj_map, input, noisy_input = model.forward_test(
                                    df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=std_frac)
                                # find the most likely position 
                                # TODO: the most likely points are changing for n_smooth
                                max_goal_point, max_goal_indice = get_most_likely_point(pred_goal_map)
                                max_traj_point, max_traj_indice = get_most_likely_point(pred_traj_map)
                                # get gradient 
                                grad_input_goal, = torch.autograd.grad(max_goal_point, noisy_input, retain_graph=True)
                                grad_input_traj, = torch.autograd.grad(max_traj_point, noisy_input)
                                if i == 0:
                                    smooth_grad_goal = grad_input_goal
                                    smooth_grad_traj = grad_input_traj
                                else: 
                                    smooth_grad_goal += grad_input_goal
                                    smooth_grad_traj += grad_input_traj
                            plot_saliency_maps(input, smooth_grad_goal, 'smooth_grad', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__goal__std_{std_frac}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                                side_by_side=False, best_points=max_goal_indice)
                            plot_saliency_maps(input, smooth_grad_traj, 'smooth_grad', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__smooth_grad__traj__std_{std_frac}', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                                side_by_side=False, best_points=max_traj_indice)

                    if args.GradCAM:
                        # TODO: not only the last time step, but also the former ones...
                        # TODO: relate somehow the groundtruth points / finally predicted points as our decision... 
                        # specify layer
                        layers_dict = {
                            'semantic_segmentation.encoder.conv1': model.model.semantic_segmentation.encoder.conv1,
                            'encoder.stages.0.0': model.model.encoder.stages[0][0],
                            'encoder.stages.1.3': model.model.encoder.stages[1][3],
                            'encoder.stages.2.1': model.model.encoder.stages[2][1], 
                            'encoder.stages.2.3': model.model.encoder.stages[2][3], 
                            'encoder.stages.3.3': model.model.encoder.stages[3][3], 
                            'encoder.stages.4.3': model.model.encoder.stages[4][3],
                            'goal_decoder.decoder.4.2': model.model.goal_decoder.decoder[4][2],
                            'goal_decoder.predictor': model.model.goal_decoder.predictor,
                            'traj_decoder.decoder.4.2': model.model.traj_decoder.decoder[4][2],
                            'traj_decoder.predictor': model.model.traj_decoder.predictor
                        }               

                        for layer_name, layer in layers_dict.items():
                            layer.register_forward_hook(hook_store_A) 
                            layer.register_full_backward_hook(hook_store_dydA)

                            # forward and backward 
                            pred_goal_map, pred_traj_map, input = model.forward_test(
                                df_meta, IMAGE_PATH, require_input_grad=False, noisy_std_frac=None) 
                            # find the most likely position 
                            max_goal_point, max_goal_indice = get_most_likely_point(pred_goal_map)
                            max_traj_point, max_traj_indice = get_most_likely_point(pred_traj_map)
                            # get gradient 
                            if 'traj_decoder' not in layer_name: 
                                max_goal_point.backward(retain_graph=True)
                                L_goal = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                                L_goal = F.interpolate(L_goal, size=(input.size(2), input.size(3)), 
                                    mode = 'bilinear', align_corners=False)
                                plot_saliency_maps(input, L_goal, 'grad_cam', 
                                    f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__goal', 
                                    f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                                    side_by_side=False, best_points=max_goal_indice)
                            max_traj_point.backward()
                            L_traj = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                            L_traj = F.interpolate(L_traj, size=(input.size(2), input.size(3)), 
                                mode = 'bilinear', align_corners=False)
                            plot_saliency_maps(input, L_traj, 'grad_cam', 
                                f'{ckpt_name}__{scene_id}__{meta_id}__grad_cam__{layer_name}__traj', 
                                f'figures/saliency_maps/{folder_name}/{args.decision}/{meta_id}', 
                                side_by_side=False, best_points=max_traj_indice)


def hook_store_A(module, input, output): 
    module.A = output[0]


def hook_store_dydA(module, grad_input, grad_output):
    module.dydA = grad_output[0]


def get_most_likely_point(map, time_step=-1):
    last_map = map[0, time_step]
    max_point = last_map.max()
    max_indice = (last_map==torch.max(last_map)).nonzero().squeeze().cpu().detach().numpy()
    return max_point, max_indice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--dataset_name', default='sdd', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_round', default=1, type=int)
    # files 
    parser.add_argument('--ckpts', default=None, type=str, nargs='+')
    parser.add_argument('--ckpts_name', default=None, type=str, nargs='+')
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--files', default=None, type=str, nargs='+')
    parser.add_argument('--n_leftouts', default=None, type=int, nargs='+')
    parser.add_argument('--scenes', default=None, type=str, nargs='+')
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


# python -m pdb saliency.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --ckpts_name OODG --files Biker.pkl --n_leftouts 10 --meta_id 22796 --decision map --VanillaGrad

# python -m pdb saliency.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_all_FT_weights.pt ckpts/Seed_1_Train__Biker__Val__Biker__Val_Ratio_0.1_filter_agent_type__train_encoder_weights.pt --ckpts_name OODG FT ET --files Biker.pkl --n_leftouts 10 --meta_id 22796 --VanillaGrad --SmoothGrad --GradCAM