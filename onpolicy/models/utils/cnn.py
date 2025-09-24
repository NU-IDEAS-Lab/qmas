import torch.nn as nn
from .util import init
from .coordconv import CoordConv
import segmentation_models_pytorch as smp
from .attention import SelfAttention

"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x

import torch
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class EncoderSkipConnectionsLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):
        super(EncoderSkipConnectionsLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]

        self.encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=input_channel,
            depth=5,
            weights=None,
        )

        self.hidden_size = hidden_size
        self.init_ = init_
        self.active_func = active_func

        self.post = nn.Sequential(
            init_(nn.Linear(len(self.encoder.out_channels) * hidden_size, hidden_size)),
            active_func
        )

    def forward(self, x):
        # Get encoder outputs (list of feature maps)
        features = self.encoder(x)
        skip_features = []
        # Dynamically create projection layers for each feature map shape
        for i, feature in enumerate(features):
            batch_size = feature.size(0)
            out_channels = feature.size(1)
            h = feature.size(2)
            w = feature.size(3)
            flatten_size = out_channels * h * w
            # Create projection layer on the fly if not already created
            proj = nn.Sequential(
                Flatten(),
                self.init_(nn.Linear(flatten_size, self.hidden_size)),
                self.active_func
            ).to(feature.device)
            skip_features.append(proj(feature))
        x = torch.cat(skip_features, dim=1)
        x = self.post(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):
        super(EncoderLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]

        self.encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=input_channel,
            depth=5,
            weights=None,
        )

        self.hidden_size = hidden_size
        self.init_ = init_
        self.active_func = active_func

        self.post = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            init_(nn.Linear(self.encoder.out_channels[-1], hidden_size)),
            active_func,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.post(x[-1])
        return x


class UNetLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(UNetLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.sequence = nn.Sequential(
            smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=input_channel,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
                activation="sigmoid",          # activation function
            ),
            Flatten(),
            init_(nn.Linear(input_width * input_height, hidden_size)),
            active_func,
        )


    def forward(self, x):
        x = self.sequence(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, mode='unet'):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        MODES = {
            'cnn': CNNLayer,
            'coordconv': CoordConv,
            'unet': UNetLayer,
            'encoder': EncoderLayer,
            'encoder_skip': EncoderSkipConnectionsLayer,
        }
        assert mode in MODES.keys(), f"mode should be one of {MODES.keys()}"
        base_class = MODES[mode]
        self.cnn = base_class(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x
