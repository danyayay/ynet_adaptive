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

    # ## model 
    print('############ Load models ##############')
    if args.ckpts:
        for i, (ckpt, ckpt_name) in enumerate(zip(args.ckpts, args.ckpts_name)):
            print(f'====== Testing for {ckpt_name} ======')
            set_random_seeds(args.seed)
            model = YNetTrainer(params=params)
            model.load(ckpt)
            model.model.eval()

            # select data
            df_meta = df_test[df_test.metaId == args.meta_id]
            scene_id = df_meta.sceneId.values[0]
            folder_name = str(args.seed) + '__' + ckpt.split('_filter_')[1].split('__')[0] + '__' + '_'.join(args.files).rstrip('.pkl')

            # TODO: do the same thing with objective being the last goal prediction...? Or for each time step...
            # TODO: add observations and predictions to visualization 
            # TODO: do like a batch 
            # TODO: add three kinds of saliency maps in one figures... be careful about the colorbar 
            # TODO: do for different layers grad-cam

            # try to understand
            # TODO: difference between backpropagating goal_loss / traj_loss / goal_loss + traj_loss
            # TODO: how to explain the three methods in regression case

            if args.VanillaGrad:
                goal_loss, traj_loss, input = model.forward_test(
                    df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=None)
                loss = goal_loss + traj_loss
                # get gradient 
                grad_input_goal, = torch.autograd.grad(goal_loss, input, retain_graph=True)
                grad_input_traj, = torch.autograd.grad(traj_loss, input, retain_graph=True)
                grad_input_both, = torch.autograd.grad(loss, input)
                plot_saliency_maps(input, grad_input_goal, 'vanilla_grad', 
                    f'{ckpt_name}__{scene_id}__{args.meta_id}__vanilla_grad__goal', 
                    f'figures/saliency_maps/{folder_name}')
                plot_saliency_maps(input, grad_input_traj, 'vanilla_grad', 
                    f'{ckpt_name}__{scene_id}__{args.meta_id}__vanilla_grad__traj', 
                    f'figures/saliency_maps/{folder_name}')
                plot_saliency_maps(input, grad_input_both, 'vanilla_grad', 
                    f'{ckpt_name}__{scene_id}__{args.meta_id}__vanilla_grad__both', 
                    f'figures/saliency_maps/{folder_name}')

            if args.SmoothGrad:
                for std_frac in [0.1, 0.15, 0.2, 0.25]:
                    for i in range(args.n_smooth):
                        goal_loss, traj_loss, input, noisy_input = model.forward_test(
                            df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=std_frac)
                        loss = goal_loss + traj_loss 
                        # get gradient 
                        grad_input_goal, = torch.autograd.grad(goal_loss, noisy_input, retain_graph=True)
                        grad_input_traj, = torch.autograd.grad(traj_loss, noisy_input, retain_graph=True)
                        grad_input_both, = torch.autograd.grad(loss, noisy_input)
                        if i == 0:
                            smooth_grad_goal = grad_input_goal
                            smooth_grad_traj = grad_input_traj
                            smooth_grad_both = grad_input_both 
                        else: 
                            smooth_grad_goal += grad_input_goal
                            smooth_grad_traj += grad_input_traj
                            smooth_grad_both += grad_input_both
                    plot_saliency_maps(input, smooth_grad_goal, 'smooth_grad', 
                        f'{ckpt_name}__{scene_id}__{args.meta_id}__smooth_grad__goal__std_{std_frac}', 
                        f'figures/saliency_maps/{folder_name}')
                    plot_saliency_maps(input, smooth_grad_traj, 'smooth_grad', 
                        f'{ckpt_name}__{scene_id}__{args.meta_id}__smooth_grad__traj__std_{std_frac}', 
                        f'figures/saliency_maps/{folder_name}')
                    plot_saliency_maps(input, smooth_grad_both, 'smooth_grad', 
                        f'{ckpt_name}__{scene_id}__{args.meta_id}__smooth_grad__both__std_{std_frac}', 
                        f'figures/saliency_maps/{folder_name}')

            if args.GradCAM:                
                # specify layer
                layers_dict = {
                    'encoder.stages.0.0': model.model.encoder.stages[0][0],
                    'encoder.stages.4.3': model.model.encoder.stages[4][3],
                    'goal_decoder.predictor': model.model.goal_decoder.predictor,
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
                        print(L_goal.shape)
                        L_goal = F.interpolate(L_goal, size = (input.size(2), input.size(3)), 
                            mode = 'bilinear', align_corners = False)
                        print(L_goal.shape)
                        plot_saliency_maps(input, L_goal, 'grad_cam', 
                            f'{ckpt_name}__{scene_id}__{args.meta_id}__grad_cam__{layer_name}__goal', 
                            f'figures/saliency_maps/{folder_name}')

                    traj_loss.backward(retain_graph=True)
                    L_traj = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                    print(L_traj.shape)
                    L_traj = F.interpolate(L_traj, size = (input.size(2), input.size(3)), 
                        mode = 'bilinear', align_corners = False)
                    print(L_traj.shape)
                    plot_saliency_maps(input, L_traj, 'grad_cam', 
                        f'{ckpt_name}__{scene_id}__{args.meta_id}__grad_cam__{layer_name}__traj', 
                        f'figures/saliency_maps/{folder_name}')

                    loss.backward(retain_graph=True)
                    L_both = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                    L_both = F.interpolate(L_both, size = (input.size(2), input.size(3)), 
                        mode = 'bilinear', align_corners = False)
                    plot_saliency_maps(input, L_both, 'grad_cam', 
                        f'{ckpt_name}__{scene_id}__{args.meta_id}__grad_cam__{layer_name}__both', 
                        f'figures/saliency_maps/{folder_name}')

                    # goal_loss.backward(retain_graph=True)
                    # L = torch.relu((layer.dydA * layer.A).sum(1, keepdim = True))
                    # L = L / L.max() 
                    # L = F.interpolate(L, size = (input.size(2), input.size(3)), 
                    #     mode = 'bilinear', align_corners = False)
                    # l = L.view(L.size(2), L.size(3)).cpu().detach().numpy()
                    # PIL.Image.fromarray(np.uint8(cm.gist_earth(l) * 255)).save('result_goal.png')


def hook_store_A(module, input, output): 
    module.A = output[0]


def hook_store_dydA(module, grad_input, grad_output):
    module.dydA = grad_output[0]


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
    # others 
    parser.add_argument('--meta_id', default=None, type=int)
    parser.add_argument('--VanillaGrad', action='store_true')
    parser.add_argument('--SmoothGrad', action='store_true')
    parser.add_argument('--n_smooth', default=4, type=int)
    parser.add_argument('--GradCAM', action='store_true')

    args=parser.parse_args()

    main(args)


# python -m pdb saliency.py --dataset_path filter/agent_type/ --ckpts ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --ckpts_name OODG --files Biker.pkl --n_leftouts 10 --meta_id 22796 --VanillaGrad 