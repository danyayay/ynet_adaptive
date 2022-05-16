import torch
import numpy as np
import torch.nn.functional as F

from utils.parser import get_parser
from utils.dataset import get_meta_ids_focus, set_random_seeds, dataset_split, limit_samples
from evaluator.visualization import plot_saliency_maps, plot_prediction, plot_given_trajectories_scenes_overlay
from utils.util import get_image_and_data_path, get_params, restore_model, get_ckpts_and_names


def main(args):
    # configuration
    set_random_seeds(args.seed)
    params = get_params(args)
    IMAGE_PATH, DATA_PATH = get_image_and_data_path(params)
    resize_factor = params['resize_factor']

    # prepare data 
    df_train, _, df_test = dataset_split(DATA_PATH, args.val_files, 0, args.n_leftouts)
    # get focused data 
    print(f"df_test: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    meta_ids_focus = get_meta_ids_focus(df_test, 
        given_csv={'path': args.result_path, 'name': args.result_name, 'n_limited': args.result_limited}, 
        given_meta_ids=args.given_meta_ids, random_n=args.random_n)
    print(f"df_test_limited: {df_test.shape}; #={df_test.shape[0]/(params['obs_len']+params['pred_len'])}")
    print('meta_ids_focus: #=', len(meta_ids_focus))
    # plot 
    df_train = limit_samples(df_train, 2, 10)
    folder_name = f"{args.seed}__{'_'.join(args.dataset_path.split('/'))}__{'_'.join(args.val_files).rstrip('.pkl')}" 
    plot_given_trajectories_scenes_overlay(
        IMAGE_PATH, df_train, f'figures/scene_with_trajs_given/{folder_name}')

    # ckpts
    ckpts, ckpts_name, is_file_separated = get_ckpts_and_names(
        args.ckpts, args.ckpts_name, args.pretrained_ckpt, args.tuned_ckpts)

    ckpt_trajs_dict = {}

    # plot
    for i, (ckpt, ckpt_name) in enumerate(zip(ckpts, ckpts_name)):
        print(f'====== Testing for {ckpt_name} ======')
    
        # load parameters 
        model = restore_model(params, is_file_separated, ckpt_name, 
            ckpt if not is_file_separated else args.pretrained_ckpt, 
            None if not is_file_separated else ckpt)
        model.model.eval()

        # test 
        set_random_seeds(args.seed)
        _, _, _, list_trajs = model.test(df_test, IMAGE_PATH, False, True, False)
        trajs_dict = list_trajs[0]
        ckpt_trajs_dict[ckpt_name] = trajs_dict

        for meta_id in meta_ids_focus:
            # select df
            df_meta = df_test[df_test.metaId == meta_id]
            scene_id = df_meta.sceneId.values[0]
            
            # select trajs 
            indice_gt, indice_pred = get_gt_pred_indice(
                trajs_dict, meta_id, args.time_step, resize_factor)

            if args.VanillaGrad:
                method_name = 'vanilla_grad'
                set_input_list = ['scene', 'semantic', 'traj']
                pred_goal_map, pred_traj_map, scene_img, feature_input = model.forward_test(
                    df_meta, IMAGE_PATH, set_input=set_input_list, noisy_std_frac=None)
                # find the decision of interest
                point_goal_prob, indice_goal_prob = get_most_likely_point_and_indice(pred_goal_map, args.time_step)
                point_traj_prob, indice_traj_prob = get_most_likely_point_and_indice(pred_traj_map, args.time_step)
                point_goal_gt, point_goal_pred = get_gt_pred_point(
                    pred_goal_map, indice_gt, indice_pred, args.time_step, args.output_radius, args.output_relative)
                point_traj_gt, point_traj_pred = get_gt_pred_point(
                    pred_traj_map, indice_gt, indice_pred, args.time_step, args.output_radius, args.output_relative)
                # get gradient 
                set_input_dict = {'scene': scene_img, 'semantic_traj': feature_input}
                for input_name, input in set_input_dict.items():
                    grad_goal_prob, = torch.autograd.grad(point_goal_prob, input, retain_graph=True)
                    grad_traj_prob, = torch.autograd.grad(point_traj_prob, input, retain_graph=True)
                    grad_goal_gt, = torch.autograd.grad(point_goal_gt, input, retain_graph=True)
                    grad_goal_pred, = torch.autograd.grad(point_traj_gt, input, retain_graph=True)
                    grad_traj_gt, = torch.autograd.grad(point_traj_gt, input, retain_graph=True)
                    grad_traj_pred, = torch.autograd.grad(point_traj_pred, input, retain_graph=True)

                    # plot
                    def plot(
                        grad_goal_prob, grad_traj_prob, grad_goal_gt, grad_traj_gt, grad_goal_pred, grad_traj_pred, 
                        input_name, n_class=6):
                        if input_name == 'semantic': 
                            grad_goal_prob, grad_traj_prob, grad_goal_gt, grad_traj_gt, grad_goal_pred, grad_traj_pred = \
                                grad_goal_prob[:, :n_class], grad_traj_prob[:, :n_class], grad_goal_gt[:, :n_class], grad_goal_pred[:, :n_class], grad_traj_gt[:, :n_class], grad_traj_pred[:, :n_class]
                        elif input_name == 'traj':
                            grad_goal_prob, grad_traj_prob, grad_goal_gt, grad_traj_gt, grad_goal_pred, grad_traj_pred = \
                                grad_goal_prob[:, n_class:], grad_traj_prob[:, n_class:], grad_goal_gt[:, n_class:], grad_goal_pred[:, n_class:], grad_traj_gt[:, n_class:], grad_traj_pred[:, n_class:]
                        plot_dict = {
                            'prob__goal': {'grad': grad_goal_prob, 'best_point': indice_goal_prob},
                            'prob__traj': {'grad': grad_traj_prob, 'best_point': indice_traj_prob},
                            'gt__goal': {'grad': grad_goal_gt, 'best_point': indice_gt},
                            'gt__traj': {'grad': grad_traj_gt, 'best_point': indice_gt},
                            'pred__goal': {'grad': grad_goal_pred, 'best_point': indice_pred},
                            'pred__traj': {'grad': grad_traj_pred, 'best_point': indice_pred}
                        }
                        for name, v_dict in plot_dict.items():
                            plot_saliency_maps(scene_img, v_dict['grad'], method_name, 
                                f'{ckpt_name}__{method_name}__{name}', 
                                f'figures/saliency_maps/{folder_name}/{scene_id}__{meta_id}/{args.decision}__{args.time_step}__{args.output_radius}__{args.output_relative}/{input_name}', 
                                side_by_side=False, best_point=v_dict['best_point'])

                    if '_' in input_name:
                        plot(
                            grad_goal_prob, grad_traj_prob, grad_goal_gt, 
                            grad_traj_gt, grad_goal_pred, grad_traj_pred, input_name)
                        for input_name in input_name.split('_'):
                            plot(
                                grad_goal_prob, grad_traj_prob, grad_goal_gt, 
                                grad_traj_gt, grad_goal_pred, grad_traj_pred, input_name)
                    else:
                        plot(
                            grad_goal_prob, grad_traj_prob, grad_goal_gt, 
                            grad_traj_gt, grad_goal_pred, grad_traj_pred, input_name)

            if args.SmoothGrad:
                method = 'smooth_grad'
                for std_frac in [0.1, 0.15, 0.2, 0.25]:
                    for i in range(args.n_smooth):
                        pred_goal_map, pred_traj_map, input, noisy_input = model.forward_test(
                            df_meta, IMAGE_PATH, require_input_grad=True, noisy_std_frac=std_frac)
                        # find the decision of interest
                        point_goal_gt, point_goal_pred = get_gt_pred_point(
                            pred_goal_map, indice_gt, indice_pred, args.time_step)
                        point_traj_gt, point_traj_pred = get_gt_pred_point(
                            pred_traj_map, indice_gt, indice_pred, args.time_step)
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
                            f'{ckpt_name}__{method}__{name}__std_{std_frac}', 
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
                    point_goal_prob, indice_goal_prob = get_most_likely_point_and_indice(pred_goal_map, args.time_step)
                    point_traj_prob, indice_traj_prob = get_most_likely_point_and_indice(pred_traj_map, args.time_step)
                    point_goal_gt, point_goal_pred = get_gt_pred_point(pred_goal_map, indice_gt, indice_pred, args.time_step)
                    point_traj_gt, point_traj_pred = get_gt_pred_point(pred_traj_map, indice_gt, indice_pred, args.time_step)
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
                            f'{ckpt_name}__{method}__{name}__{layer_name}', 
                            f'figures/saliency_maps/{folder_name}/{args.decision}/{scene_id}__{meta_id}', 
                            side_by_side=False, best_point=v_dict['best_point'])

    ckpt_trajs_dict_focus = {}
    for ckpt_name in ckpts_name:
        ckpt_trajs_dict_focus[ckpt_name] = {}
        ckpt_trajs_dict_focus[ckpt_name]['metaId'] = meta_ids_focus
        mask = np.array(
            [1 if meta_id in meta_ids_focus else 0 for meta_id in ckpt_trajs_dict[ckpt_name]['metaId']]).astype(bool)
        ckpt_trajs_dict_focus[ckpt_name]['sceneId'] = np.array(ckpt_trajs_dict[ckpt_name]['sceneId'])[mask]
        ckpt_trajs_dict_focus[ckpt_name]['prediction'] = ckpt_trajs_dict[ckpt_name]['prediction'][mask]
        ckpt_trajs_dict_focus[ckpt_name]['groundtruth'] = ckpt_trajs_dict[ckpt_name]['groundtruth'][mask]
    plot_prediction(IMAGE_PATH, ckpt_trajs_dict_focus, f'figures/prediction/{folder_name}/{"__".join(ckpts_name)}')


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


def get_gt_pred_point(map, indice_gt, indice_pred, time_step=-1, radius=0, relative=False):
    point_gt = map[0, time_step, indice_gt[0], indice_gt[1]]
    point_pred = map[0, time_step, indice_pred[0], indice_pred[1]]
    # works at boundary 
    x_gt_min = indice_gt[0] - radius 
    y_gt_min = indice_gt[1] - radius 
    x_pred_min = indice_pred[0] - radius
    y_pred_min = indice_pred[1] - radius    
    if x_gt_min < 0: x_gt_min = 0
    if y_gt_min < 0: y_gt_min = 0
    if x_pred_min < 0: x_pred_min = 0
    if y_pred_min < 0: y_pred_min = 0
    # select neighboring region
    region_gt = map[0, time_step, 
        x_gt_min:(indice_gt[0]+radius+1), 
        y_gt_min:(indice_gt[1]+radius+1)].sum()
    region_pred = map[0, time_step, 
        x_pred_min:(indice_pred[0]+radius+1), 
        y_pred_min:(indice_pred[1]+radius+1)].sum()
    if relative:
        if radius != 0:
            n_neighbor = radius**2 - 1
        else:
            raise ValueError('Invalid radius(0) to compute relative output value')
        relative_gt = (region_gt - point_gt) / n_neighbor
        relative_pred = (region_pred - point_pred) / n_neighbor
        return relative_gt, relative_pred
    else:
        return region_gt, region_pred


def compute_grad_cam(layer, input):
    L = torch.relu((layer.grad * layer.output).sum(1, keepdim = True))
    L = F.interpolate(L, size=(input.size(2), input.size(3)), 
        mode='bilinear', align_corners=False)
    return L


if __name__ == '__main__':
    
    parser = get_parser(False)
    # data
    parser.add_argument('--given_meta_ids', default=None, type=int, nargs='+')
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--result_name', default=None, type=str)
    parser.add_argument('--result_limited', default=None, type=int)
    parser.add_argument('--random_n', default=None, type=int)
    # saliency map parameters  
    parser.add_argument('--decision', default=None, type=str, choices=['loss', 'map'])
    parser.add_argument('--output_radius', default=0, type=int)
    parser.add_argument('--output_relative', action='store_true')
    parser.add_argument('--time_step', default=-1, type=int)
    
    # saliency map type 
    parser.add_argument('--VanillaGrad', action='store_true')
    parser.add_argument('--SmoothGrad', action='store_true')
    parser.add_argument('--n_smooth', default=4, type=int)
    parser.add_argument('--GradCAM', action='store_true')

    args = parser.parse_args()

    main(args)

# TODO: make it work for other two methods 
# python -m pdb -m evaluator.analyze_saliency --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_serial__0__TrN_20.pt --val_files Biker.pkl --n_leftouts 500 --result_path './csv/comparison/1__filter_agent_type_deathCircle_0__Biker/OODG_encoder_0(20)_encoder_0-1(20).csv' --result_name 'ade_OODG__ade_encoder_0(20)__diff' --result_limited 5 --decision map --output_radius 2 --VanillaGrad

# python -m pdb -m evaluator.analyze_saliency --dataset_path filter/agent_type/deathCircle_0 --pretrained_ckpt ckpts/Seed_1_Train__Pedestrian__Val__Pedestrian__Val_Ratio_0.1_filter_agent_type__train_all_weights.pt --tuned_ckpts ckpts/Seed_1__Tr_Biker__Val_Biker__ValRatio_0.1__filter_agent_type_deathCircle_0__adapter_serial__0__TrN_20.pt --val_files Biker.pkl --n_leftouts 10 --decision map --output_radius 2 --VanillaGrad  --given_meta_ids 6318 