import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from utils.softargmax import SoftArgmax2D, create_meshgrid


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


def conv2d(in_planes, out_planes=None, kernel_size=1, stride=1, padding=None, bias=False):
    if padding is None: padding = kernel_size // 2
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class Adapter(nn.Module):
    def __init__(self, planes, adapter_type, out_planes=None, stride=1, is_bias=False):
        super(Adapter, self).__init__()
        self.adapter_type = adapter_type
        self.is_bias = is_bias 
        self.is_multiple_conv = False 
        # default serial adapter 
        if adapter_type == 'serial':
            self.adapter_layer = nn.Sequential(nn.BatchNorm2d(planes), conv2d(planes, bias=is_bias))
        # default parallel adapter 
        elif adapter_type == 'parallel':
            self.adapter_layer = conv2d(planes, out_planes, 1, stride, bias=is_bias)
        # parallel adapter with changed filter size 
        elif ('parallel' in adapter_type) and (len(adapter_type.split('_')) <= 2):
            kernel_size = int(adapter_type.split('_')[1].split('x')[0])
            self.adapter_layer = conv2d(planes, out_planes, kernel_size, stride, bias=is_bias)
        # multiple parallel adapter 
        elif ('parallel' in adapter_type) and (len(adapter_type.split('_')) > 2):
            sizes = adapter_type.split('_')[1:]
            self.is_multiple_conv = True
            self.adapter_layer = nn.ModuleList()
            for size in sizes:
                kernel_size = int(size.split('x')[0])
                self.adapter_layer.append(conv2d(planes, out_planes, kernel_size, stride, bias=is_bias))       
        else:
            self.adapter_layer = conv2d(planes, bias=is_bias)
        
        # initialize parameters
        self.initialize()

    def initialize(self):
        if self.adapter_type == 'serial':
            nn.init.zeros_(self.adapter_layer[1].weight)
            if self.is_bias: nn.init.zeros_(self.adapter_layer[1].bias)
        elif 'parallel' in self.adapter_type:
            for p in self.adapter_layer.parameters():
                nn.init.zeros_(p)

    def forward(self, x):
        if self.is_multiple_conv:
            y = 0
            for layer in self.adapter_layer:
                x_ = layer(x)
                y += x_
        elif 'serial' in self.adapter_type:
            y = self.adapter_layer(x)
            y += x
        else:
            y = self.adapter_layer(x)
        return y


class YNetEncoder(nn.Module):
    def __init__(
        self, in_channels, channels=(64, 128, 256, 512, 512), 
        train_net=None, position=[]):
        """
        Encoder model
        :param in_channels: int, n_semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()

        self.train_net = train_net
        self.position = position
        if 'lora' in self.train_net: 
            rank = int(self.train_net.split('_')[1]) if len(self.train_net.split('_')) > 1 else 1

        self.stages = nn.ModuleList()

        # First block
        modules = []
        if 'lora' in self.train_net and 0 in self.position:
            modules.append(lora.Conv2d(in_channels, channels[0], 
                kernel_size=3, r=rank, stride=(1, 1), padding=(1, 1)))
        else:
            modules.append(nn.Conv2d(in_channels, channels[0], 
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        modules.append(nn.ReLU(inplace=False))
        self.stages.append(nn.Sequential(*modules))

        # Subsequent blocks, each starts with MaxPool
        for i in range(len(channels) - 1):
            modules = [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)]
            if 'lora' in self.train_net and i+1 in self.position:
                modules.append(lora.Conv2d(channels[i], channels[i+1], 
                    kernel_size=3, r=rank, stride=(1, 1), padding=(1, 1)))
                modules.append(nn.ReLU(inplace=False))
                modules.append(lora.Conv2d(channels[i+1], channels[i+1], 
                    kernel_size=3, r=rank, stride=(1, 1), padding=(1, 1)))
                modules.append(nn.ReLU(inplace=False))
            else:
                modules.append(nn.Conv2d(channels[i], channels[i+1], 
                    kernel_size=3, stride=(1, 1), padding=(1, 1)))
                modules.append(nn.ReLU(inplace=False))
                modules.append(nn.Conv2d(channels[i+1], channels[i+1], 
                    kernel_size=3, stride=(1, 1), padding=(1, 1)))
                modules.append(nn.ReLU(inplace=False))
            self.stages.append(nn.Sequential(*modules))

        # Last MaxPool layer before passing the features into decoder
        self.stages.append(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                )
            )

        # adapter
        par_channels_in = [in_channels] + channels[:-1]
        if 'serial' in self.train_net:
            self.adapters = nn.ModuleList([
                Adapter(channels[i], train_net) for i in position])
        elif 'parallel' in self.train_net:
            self.adapters = nn.ModuleList([
                Adapter(par_channels_in[i], train_net, channels[i]) for i in position])

    def forward(self, x):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        j = 0
        for i, stage in enumerate(self.stages):
            if 'serial' in self.train_net:
                x = stage(x) 
                if i in self.position:
                    x = self.adapters[j](x)
                    j += 1
            elif 'parallel' in self.train_net:
                if isinstance(stage[0], nn.MaxPool2d):
                    y = stage[0](x)
                    x = stage(x)
                    if i in self.position:
                        x = x + self.adapters[j](y)
                        j += 1
                else:
                    y = stage(x)
                    if i in self.position:
                        y = y + self.adapters[j](x)
                        j += 1
                    x = y
            else:
                x = stage(x)
            features.append(x)
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
            nn.ReLU(inplace=False),
            nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False)
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
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=False))
            for in_channels_, out_channels_ in zip(in_channels, out_channels)]
        )

        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(
            in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        # reverse the order of encoded features, as the decoder starts from the smallest image
        features = features[::-1]
        # decoder: layer 1
        center_feature = features[0]
        x = self.center(center_feature)
        # decoder: layer 2-6
        for f, d, up in zip(features[1:], self.decoder, self.upsample_conv):
            # bilinear interpolation for upsampling
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = up(x)  # 3x3 conv for upsampling
            # concat encoder and decoder features
            x = torch.cat([x, f], dim=1)
            x = d(x)
        # decoder: layer 7 (last predictor layer)
        x = self.predictor(x) 
    
        return x


class YNet(nn.Module):
    def __init__(
        self, obs_len, pred_len, segmentation_model_fp, 
        use_features_only=False, n_semantic_classes=6,
        encoder_channels=[], decoder_channels=[], n_waypoints=1, 
        train_net=None, position=[]):
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
            in_channels=n_semantic_classes + obs_len, channels=encoder_channels,
            train_net=train_net, position=position)

        self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=n_waypoints)

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)

        self.encoder_channels = encoder_channels

    def initialize_style(self):
        self.style_modulators = nn.ModuleList([StyleModulator(self.encoder_channels) for _ in range(3)])

    def segmentation(self, image):
        return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def pred_goal(self, features):
        return self.goal_decoder(features)

    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        return self.traj_decoder(features)

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x):
        return self.encoder(x)

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
