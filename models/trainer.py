import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils.train import train_epoch
from utils.dataset import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, get_patch, \
    preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from utils.evaluate import evaluate

from models.ynet import YNet


class YNetTrainer:
    def __init__(self, params, device=None):
        """
        Ynet class, following a sklearn similar class structure
        :param obs_len: observed timesteps
        :param pred_len: predicted timesteps
        :param params: dictionary with hyperparameters
        """
        self.params = params
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Working on {self.device}')

        self.division_factor = 2 ** len(params['encoder_channels'])
        self.template_size = int(4200 * params['resize_factor'])

        self.model = YNet(
            obs_len=params['obs_len'], pred_len=params['pred_len'],
            segmentation_model_fp=params['segmentation_model_fp'],
            use_features_only=params['use_features_only'],
            n_semantic_classes=params['n_semantic_classes'],
            encoder_channels=params['encoder_channels'],
            decoder_channels=params['decoder_channels'],
            n_waypoints=len(params['waypoints']),
            adapter_type=params['adapter_type'], 
            adapter_position=params['adapter_position']
        )
    
    def train(self, df_train, df_val, train_image_path, val_image_path, experiment_name):
        return self._train(df_train, df_val, train_image_path, val_image_path, experiment_name, **self.params)

    def _train(
        self, df_train, df_val, train_image_path, val_image_path, experiment_name, 
        dataset_name, resize_factor, obs_len, pred_len, batch_size, lr, n_epoch, 
        waypoints, n_goal, n_traj, kernlen, nsig, e_unfreeze, loss_scale, temperature,
        use_raw_data=False, save_every_n=10, train_net="all", fine_tune=False, 
        use_CWS=False, resl_thresh=0.002, CWS_params=None, n_early_stop=5, 
        steps=[20], lr_decay_ratio=0.1, **kwargs):
        """
        Train function
        :param df_train: pd.df, train data
        :param df_val: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param n_goal: int, number of goals per trajectory, K_e in paper
        :param n_traj: int, number of trajectory per goal, K_a in paper
        :return:
        """
        # get data
        train_images, train_loader, self.homo_mat = self.prepare_data(
            df_train, train_image_path, dataset_name, 'train', 
            obs_len, pred_len, resize_factor, use_raw_data, fine_tune)
        val_images, val_loader, _ = self.prepare_data(
            df_val, val_image_path, dataset_name, 'val', 
            obs_len, pred_len, resize_factor, use_raw_data, fine_tune)

        # model 
        model = self.model.to(self.device)

        # Freeze segmentation model
        for param in model.semantic_segmentation.parameters():
            param.requires_grad = False

        if train_net != 'all':
            for param in model.parameters():
                param.requires_grad = False
            if train_net == "encoder":
                for param in model.encoder.parameters():
                    param.requires_grad = True
            elif train_net == "modulator":
                for param in model.style_modulators.parameters():
                    param.requires_grad = True
            elif train_net == 'adapter':
                for param_name, param in model.encoder.named_parameters():
                    if 'adapter_layer' in param_name: 
                        param.requires_grad = True
            elif len(train_net.split('_')[-1].split('-')) == 1:
                layer_num = int(train_net.split('_')[-1])
                for param_name, param in model.encoder.named_parameters():
                    param_layer = int(param_name.split('.')[1])
                    if param_layer == layer_num:
                        param.requires_grad = True
            elif len(train_net.split('_')[-1].split('-')) == 2:
                layer_lower, layer_upper = train_net.split('_')[-1].split('-')
                layer_lower, layer_upper = int(layer_lower), int(layer_upper)
                for param_name, param in model.encoder.named_parameters():
                    param_layer = int(param_name.split('.')[1])
                    if (param_layer >= layer_lower) and (param_layer <= layer_upper):
                        param.requires_grad = True
            else:
                raise ValueError(f'No support for train_net={train_net}')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)

        print('The number of trainable parameters: {:d}'.format(
            sum(param.numel() for param in model.parameters() if param.requires_grad)))

        criterion = nn.BCEWithLogitsLoss()

        # Create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)
        gt_template = torch.Tensor(create_gaussian_heatmap_template(
            size=self.template_size, kernlen=kernlen, nsig=nsig, normalize=False)).to(self.device)

        # train 
        best_val_ADE = 99999999999999
        self.val_ADE = []
        self.val_FDE = []

        with_style = train_net == "modulator"
        print('Start training')
        for e in tqdm(range(n_epoch), desc='Epoch'):
            train_ADE, train_FDE, train_loss = train_epoch(
                model, train_loader, train_images, optimizer, criterion, loss_scale, self.device, 
                dataset_name, self.homo_mat, gt_template, input_template, waypoints,
                e, obs_len, pred_len, batch_size, e_unfreeze, resize_factor, with_style)

            # For faster inference, we don't use TTST and CWS here, only for the test set evaluation
            val_ADE, val_FDE, _ = evaluate(
                model, val_loader, val_images, self.device, 
                dataset_name, self.homo_mat, input_template, waypoints, 'val', 
                n_goal, n_traj, obs_len, batch_size, resize_factor, with_style,
                temperature, False, use_CWS, resl_thresh, CWS_params)

            
            print(
                f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Val (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}   lr={lr_scheduler.get_last_lr()[0]}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)
            lr_scheduler.step()

            if val_ADE < best_val_ADE:
                best_val_ADE = val_ADE
                best_state_dict = deepcopy(model.state_dict())

            if e % save_every_n == 0 and not fine_tune:
                torch.save(model.state_dict(), f'ckpts/{experiment_name}_weights_epoch_{e}.pt')

            # early stop in case of clear overfitting
            if best_val_ADE < min(self.val_ADE[-n_early_stop:]):
                print(f'Early stop at epoch {e}')
                break

        # Load the best model
        model.load_state_dict(best_state_dict, strict=True)

        # Save the best model
        pt_name = f'ckpts/{experiment_name}_weights.pt'
        torch.save(best_state_dict, pt_name)
        # TODO: for train_net == adapter, save only the adapters... 

        return self.val_ADE, self.val_FDE

    def test(self, df_test, image_path, with_style, return_features=False, viz_input=False):
        return self._test(df_test, image_path, with_style=with_style, 
            return_features=return_features, viz_input=viz_input, **self.params)

    def _test(
        self, df_test, image_path, dataset_name, resize_factor, 
        batch_size, n_round, obs_len, pred_len, 
        waypoints, n_goal, n_traj, temperature, rel_threshold, 
        use_TTST, use_CWS, CWS_params, use_raw_data=False, with_style=False, 
        return_features=False, viz_input=False, **kwargs):
        """
        Val function
        :param df_test: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param n_goal: int, number of goals per trajectory, K_e in paper
        :param n_traj: int, number of trajectory per goal, K_a in paper
        :param n_round: int, number of epochs to evaluate
        :return:
        """

        # get data 
        test_images, test_loader, self.homo_mat = self.prepare_data(
            df_test, image_path, dataset_name, 'test', 
            obs_len, pred_len, resize_factor, use_raw_data)

        # model 
        model = self.model.to(self.device)

        # Create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)

        self.eval_ADE = []
        self.eval_FDE = []
        list_metrics, list_features, list_trajs = [], [], []

        print("TTST setting:", use_TTST)
        print('Start testing')
        for e in tqdm(range(n_round), desc='Round'):
            if return_features:
                test_ADE, test_FDE, df_metrics, features_dict, trajs_dict = evaluate(
                    model, test_loader, test_images, self.device, 
                    dataset_name, self.homo_mat, input_template, waypoints, 'test',
                    n_goal, n_traj, obs_len, batch_size, resize_factor, with_style,
                    temperature, use_TTST, use_CWS, rel_threshold, CWS_params,
                    True, viz_input)
                list_features.append(features_dict)
                list_trajs.append(trajs_dict)
            else:
                test_ADE, test_FDE, df_metrics = evaluate(
                    model, test_loader, test_images, self.device, 
                    dataset_name, self.homo_mat, input_template, waypoints, 'test',
                    n_goal, n_traj, obs_len, batch_size, resize_factor, with_style,
                    temperature, use_TTST, use_CWS, rel_threshold, CWS_params,
                    False, viz_input)
            list_metrics.append(df_metrics)
            print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')
            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        avg_ade = sum(self.eval_ADE) / len(self.eval_ADE)
        avg_fde = sum(self.eval_FDE) / len(self.eval_FDE)
        print(
            f'\nAverage performance (by {n_round}): \nTest ADE: {avg_ade} \nTest FDE: {avg_fde}')

        if return_features:
            return avg_ade, avg_fde, list_metrics, list_features, list_trajs
        else:
            return avg_ade, avg_fde, list_metrics
    
    def forward_test(self, df_test, image_path, require_input_grad, noisy_std_frac):
        return self._forward_test(df_test, image_path, require_input_grad, noisy_std_frac, **self.params)

    def _forward_test(
        self, df_test, image_path, 
        require_input_grad, noisy_std_frac, decision,
        dataset_name, obs_len, pred_len, resize_factor, 
        use_raw_data, waypoints, kernlen, nsig, loss_scale, **kwargs):

        # get data 
        test_images, test_loader, self.homo_mat = self.prepare_data(
            df_test, image_path, dataset_name, 'test', 
            obs_len, pred_len, resize_factor, use_raw_data)

        # create template
        input_template = torch.Tensor(create_dist_mat(size=self.template_size)).to(self.device)
        gt_template = torch.Tensor(create_gaussian_heatmap_template(
            size=self.template_size, kernlen=kernlen, nsig=nsig, normalize=False)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()

        # test 
        if len(test_loader) == 1:
            for traj, _, scene_id in test_loader:
                scene_raw_img = test_images[scene_id].to(self.device).unsqueeze(0)
                if noisy_std_frac is not None: 
                    # noisy input
                    std = noisy_std_frac * (scene_raw_img.max() - scene_raw_img.min())
                    noisy_scene_img = scene_raw_img + scene_raw_img.new(scene_raw_img.size()).normal_(0, std)
                    noisy_scene_img.requires_grad = True
                    # forward 
                    if decision == 'loss':
                        goal_loss, traj_loss = self._forward_batch(
                            noisy_scene_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, False)
                    elif decision == 'map':
                        pred_goal_map, pred_traj_map = self._forward_batch(
                            noisy_scene_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, True)
                    else:
                        raise ValueError(f'No support for decision={decision}')
                else:
                    # require grad for input or not
                    scene_raw_img.requires_grad = False 
                    if require_input_grad: scene_raw_img.requires_grad = True
                    # forward 
                    if decision == 'loss':
                        goal_loss, traj_loss = self._forward_batch(
                            scene_raw_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, False)
                    elif decision == 'map':
                        pred_goal_map, pred_traj_map = self._forward_batch(
                            scene_raw_img, traj, input_template, gt_template, criterion, 
                            obs_len, pred_len, waypoints, loss_scale, self.device, True)
                    else:
                        raise ValueError(f'No support for decision={decision}')
        else:
            raise ValueError(f'Received more than 1 batch ({len(test_loader)})')
        
        if decision == 'loss':
            if noisy_std_frac is not None:
                return goal_loss, traj_loss, scene_raw_img, noisy_scene_img
            else:
                return goal_loss, traj_loss, scene_raw_img 
        elif decision == 'map':
            if noisy_std_frac is not None:
                return pred_goal_map, pred_traj_map, scene_raw_img, noisy_scene_img
            else:
                return pred_goal_map, pred_traj_map, scene_raw_img

    def _forward_batch(
        self, scene_raw_img, traj, input_template, gt_template, criterion, 
        obs_len, pred_len, waypoints, loss_scale, device, return_pred_map):
        
        # model 
        model = self.model.to(self.device)

        _, _, H, W = scene_raw_img.shape

        # create heatmap for observed trajectories 
        observed = traj[:, :obs_len, :].reshape(-1, 2).cpu().numpy() 
        observed_map = get_patch(input_template, observed, H, W)  
        observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W]) 

        # create heatmap for groundtruth future trajectories 
        gt_future = traj[:, obs_len:].to(device)  
        gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
        gt_future_map = torch.stack(gt_future_map).reshape([-1, pred_len, H, W])
        
        # create semantic segmentation map for all bacthes 
        scene_image = model.segmentation(scene_raw_img)
        semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)  

        # forward 
        feature_input = torch.cat([semantic_image, observed_map], dim=1) 
        features = model.pred_features(feature_input)
        pred_goal_map = model.pred_goal(features)

        # goal loss 
        goal_loss = criterion(pred_goal_map, gt_future_map) * loss_scale  
        pred_waypoint_map = pred_goal_map[:, waypoints] 
        
        # way points 
        gt_waypoints_maps_downsampled = [nn.AvgPool2d(
            kernel_size=2**i, stride=2**i)(pred_waypoint_map) for i in range(1, len(features))]
        gt_waypoints_maps_downsampled = [pred_waypoint_map] + gt_waypoints_maps_downsampled
        
        # traj loss
        traj_input = [torch.cat([feature, goal], dim=1) for feature, goal in zip(
            features, gt_waypoints_maps_downsampled)]
        pred_traj_map = model.pred_traj(traj_input)
        traj_loss = criterion(pred_traj_map, gt_future_map) * loss_scale  
        
        if return_pred_map:
            return pred_goal_map, pred_traj_map
        return goal_loss, traj_loss

    def prepare_data(
        self, df, image_path, dataset_name, mode, obs_len, pred_len, 
        resize_factor, use_raw_data, fine_tune=False):
        """
        Prepare dataset for training, validation, and testing. 

        Args:
            df (pd.DataFrame): df_train / df_val / df_test 
            image_path (str): path storing scene images 
            dataset_name (str): name of the dataset
            mode (str): choices=[train, val, test]
            resize_factor (float): _description_
            use_raw_data (bool): _description_
            fine_tune (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # get image filename 
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = 'reference.jpg'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported') 

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(self.device)
            seg_mask = True
        else:
            homo_mat = None
            seg_mask = False
        # Load scene images 
        if (fine_tune & (mode == 'train')) | (mode == 'val') | (mode == 'test'):
            images_dict = create_images_dict(
                df.sceneId.unique(), image_path=image_path, 
                image_file=image_file_name, use_raw_data=use_raw_data)
        else: # mode == 'train' & not fine_tune
            # augment train data and images
            df, images_dict = augment_data(
                df, image_path=image_path, image_file=image_file_name,
                seg_mask=seg_mask, use_raw_data=use_raw_data)

        # Initialize dataloaders
        dataset = SceneDataset(df, resize=resize_factor, total_len=obs_len+pred_len)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=scene_collate, 
            shuffle=True if mode=='train' else False)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(images_dict, factor=resize_factor, seg_mask=seg_mask)
        # make sure that image shape is divisible by 32, for UNet segmentation
        pad(images_dict, division_factor=self.division_factor)
        preprocess_image_for_segmentation(images_dict, seg_mask=seg_mask)

        return images_dict, dataloader, homo_mat

    def load(self, path):
        if self.device == torch.device('cuda'):
            print(self.model.load_state_dict(torch.load(path), strict=False))
            print('Loaded ynet model to GPU')
        else:  # self.device == torch.device('cpu')
            print(self.model.load_state_dict(torch.load(path, map_location='cpu'), strict=False))
            print('Loaded ynet model to CPU')

    def save(self, path):
        torch.save(self.model.state_dict(), path)
