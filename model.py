from cv2 import dft
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from utils.train import train_epoch
from utils.softargmax import SoftArgmax2D, create_meshgrid
from utils.dataset import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, get_patch, \
    preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from utils.evaluate import evaluate


class StyleModulator(nn.Module):
    def __init__(self, sizes):
        """
        Additional style modulator for efficient fine-tuning
        """
        from ddf import DDFPack
        super(StyleModulator, self).__init__()
        tau = 0.5
        self.modulators = nn.ModuleList(
            [DDFPack(s) for s in sizes + [sizes[-1]]]
        )

    def forward(self, x):
        stylized = []
        for xi, layer in zip(x, self.modulators):
            stylized.append(layer(xi))
        return stylized


class YNetEncoder(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
        """
        Encoder model
        :param in_channels: int, n_semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()
        self.stages = nn.ModuleList()

        # First block
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
            )
        )

        # Subsequent blocks, each starting with MaxPool
        for i in range(len(channels)-1):
            self.stages.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True)
                )
            )

        # Last MaxPool layer before passing the features into decoder
        self.stages.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                )
            )

    def forward(self, x, depth=1):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        x_copy = x.clone()
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # for detailed feature visualization
        if depth == 2:
            details = []
            x = x_copy
            for stage in self.stages:
                for layer in stage:
                    x = layer(x)
                    details.append(x)
            return features, details
        else:
            return features 


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(YNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel + traj for channel in encoder_channels]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]
        center_channels = encoder_channels[0]

        decoder_channels = decoder_channels

        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
            nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
        upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = nn.ModuleList([
            nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
                for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)])

        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
        out_channels = decoder_channels

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True))
            for in_channels_, out_channels_ in zip(in_channels, out_channels)]
        )

        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(
            in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

    def forward(self, features, depth=0):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        # reverse the order of encoded features, as the decoder starts from the smallest image
        features = features[::-1]
        details = []
        # decoder: layer 1
        center_feature = features[0]
        x = self.center(center_feature)
        if depth == 1:
            details.append(x)
        elif depth == 2:
            x = center_feature
            for layer in self.center:
                x = layer(x)
                details.append(x)
        # decoder: layer 2-6
        for f, d, up in zip(features[1:], self.decoder, self.upsample_conv):
            # bilinear interpolation for upsampling
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = up(x)  # 3x3 conv for upsampling
            if depth == 2: details.append(x)
            # concat encoder and decoder features
            x = torch.cat([x, f], dim=1)
            if depth == 0:
                x = d(x)
            elif depth == 1:
                x = d(x)
                details.append(x)   
            elif depth == 2:
                for layer in d:
                    x = layer(x)
                    details.append(x)
        # decoder: layer 7 (last predictor layer)
        x = self.predictor(x) 
    
        if depth > 0: 
            details.append(x)
            return x, details
        else:
            return x


class YNet(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, n_semantic_classes=6,
                 encoder_channels=[], decoder_channels=[], n_waypoints=1):
        """
        Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model_fp: str, filepath to pretrained segmentation model
        :param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
        :param n_semantic_classes: int, number of semantic classes
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param n_waypoints: int, number of waypoints
        """
        super(YNet, self).__init__()

        if segmentation_model_fp is not None:
            if torch.cuda.is_available():
                self.semantic_segmentation = torch.load(segmentation_model_fp)
                print('Loaded segmentation model to GPU')
            else:
                self.semantic_segmentation = torch.load(
                    segmentation_model_fp, map_location=torch.device('cpu'))
                print('Loaded segmentation model to CPU')
            if use_features_only:
                self.semantic_segmentation.segmentation_head = nn.Identity()
                n_semantic_classes = 16  # instead of classes use number of feature_dim
        else:
            self.semantic_segmentation = nn.Identity()

        self.encoder = YNetEncoder(
            in_channels=n_semantic_classes + obs_len, channels=encoder_channels)

        self.goal_decoder = YNetDecoder(
            encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(
            encoder_channels, decoder_channels, output_len=pred_len, traj=n_waypoints)

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

        self.encoder_channels = encoder_channels

    def initialize_style(self):
        self.style_modulators = nn.ModuleList(
            [StyleModulator(self.encoder_channels) for _ in range(3)])

    def segmentation(self, image):
        return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def pred_goal(self, features, depth=0):
        return self.goal_decoder(features, depth)

    # Forward pass for trajectory decoder
    def pred_traj(self, features, depth=0):
        return self.traj_decoder(features, depth)

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x, depth=1):
        return self.encoder(x, depth)

    # Used for style encoding
    def stylize_features(self, x, style_class):
        return self.style_modulators[style_class](x)

    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

    def softargmax_on_softmax_map(self, x):
        """ Softargmax: As input a batched image where softmax is already performed (not logits) """
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        x = x.flatten(2)

        estimated_x = pos_x * x
        estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
        estimated_y = pos_y * x
        estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
        softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
        return softargmax_coords


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
            n_waypoints=len(params['waypoints'])
        )
    
    def train(self, df_train, df_val, train_image_path, val_image_path, experiment_name):
        return self._train(df_train, df_val, train_image_path, val_image_path, experiment_name, **self.params)

    def _train(
        self, df_train, df_val, train_image_path, val_image_path, experiment_name, 
        dataset_name, resize_factor, obs_len, pred_len, batch_size, lr, n_epoch, 
        waypoints, n_goal, n_traj, kernlen, nsig, e_unfreeze, loss_scale, temperature,
        use_raw_data=False, save_every_n=10, train_net="all", fine_tune=False, 
        use_CWS=False, resl_thresh=0.002, CWS_params=None, **kwargs):
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
                f'Epoch {e}: 	Train (Top-1) ADE: {train_ADE:.2f} FDE: {train_FDE:.2f} 		Val (Top-k) ADE: {val_ADE:.2f} FDE: {val_FDE:.2f}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)

            if val_ADE < best_val_ADE:
                best_val_ADE = val_ADE
                best_state_dict = deepcopy(model.state_dict())

            if e % save_every_n == 0 and not fine_tune:
                torch.save(model.state_dict(), 'ckpts/' +
                           experiment_name + f'_weights_epoch_{e}.pt')

            # early stop in case of clear overfitting
            if best_val_ADE < min(self.val_ADE[-5:]):
                print(f'Early stop at epoch {e}')
                break

        # Load best model
        model.load_state_dict(best_state_dict, strict=True)

        # # Save best model
        if fine_tune:
            if train_net == 'all':
                torch.save(best_state_dict, f'ckpts/{experiment_name}_FT__TN_{str(int((df_train.shape[0])/20))}_weights.pt')
            else:
                torch.save(best_state_dict, f'ckpts/{experiment_name}__TN_{str(int((df_train.shape[0])/20))}_weights.pt')
        else:
            torch.save(best_state_dict, f'ckpts/{experiment_name}_weights.pt')

        return self.val_ADE, self.val_FDE

    def test(self, df_test, image_path, with_style, return_features=False, viz_input=False):
        return self._test(df_test, image_path, with_style=with_style, 
            return_features=return_features, viz_input=viz_input, **self.params)

    def _test(
        self, df_test, image_path, dataset_name, resize_factor, 
        batch_size, n_round, obs_len, pred_len, 
        waypoints, n_goal, n_traj, temperature, rel_threshold, 
        use_TTST, use_CWS, CWS_params, use_raw_data=False, with_style=False, 
        return_features=False, viz_input=False, depth=0, **kwargs):
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
                    True, viz_input, depth)
                list_features.append(features_dict)
                list_trajs.append(trajs_dict)
            else:
                test_ADE, test_FDE, df_metrics = evaluate(
                    model, test_loader, test_images, self.device, 
                    dataset_name, self.homo_mat, input_template, waypoints, 'test',
                    n_goal, n_traj, obs_len, batch_size, resize_factor, with_style,
                    temperature, use_TTST, use_CWS, rel_threshold, CWS_params,
                    False, viz_input, depth)
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
        if self.device == 'cuda':
            print(self.model.load_state_dict(torch.load(path), strict=False))
        else:  # self.device == 'cpu'
            print(self.model.load_state_dict(torch.load(path, map_location='cpu'), strict=False))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
