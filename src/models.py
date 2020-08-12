import torch
from taggle.models import BaseModel
from taggle.models.backbones import backbones, get_backbone
from taggle.models.common import (
    GeM,
    convert_model_ReLU2Mish,
    convert_model_ReLU2Swish
)
from taggle.models.heads import get_head, heads
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class AttModel(BaseModel):
    def __init__(self, backbone, heads: dict):
        super().__init__(backbone, heads)
        self.initialize()
        self.att = AttBlock

    def forward(self, x):
        y = {}
        x = self.backbone(x)
        for key in self.heads:
            y.update({key: self.heads[key](x)})
        return y


class BirdSongModelProvider:
    def __init__(self):
        self.backbones = backbones
        self.heads = heads

    def get_model(self, model_config: dict):
        if "type" in model_config:
            if model_config["type"] == "AttModel":
                Model = AttModel
        else:
            Model = BaseModel
        backbone_cfg = model_config["backbone"]
        heads_cfg = model_config["heads"]
        backbone = get_backbone(
            self.backbones, backbone_cfg["type"], **backbone_cfg["params"])
        heads = {}
        for name in heads_cfg:
            heads.update({name: get_head(
                self.heads, heads_cfg[name]["type"], heads_cfg[name]["params"], backbone.out_shapes)})
        model = Model(backbone, heads)
        if "mid_activation" in model_config:
            if model_config["mid_activation"] == "Mish":
                model = convert_model_ReLU2Mish(model)
            elif model_config["mid_activation"] == "Swish":
                model = convert_model_ReLU2Swish(model)
            elif model_config["mid_activation"] == "ReLU":
                pass
            else:
                raise NotImplementedError
        if "audio_pooling" in model_config:
            if model_config["audio_pooling"] == "backbone":
                model.backbone = convert_to_audio_pooling(model.backbone)
            elif model_config["audio_pooling"] == "heads":
                model.heads = convert_to_audio_pooling(model.heads)
            elif model_config["audio_pooling"] == "all":
                model.heads = convert_to_audio_pooling(model)
        return model


class AudioPooling2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, input):
        if isinstance(self.kernel_size, int):
            kernel_max = self.kernel_size
            kernel_avg = self.kernel_size
        elif isinstance(self.kernel_size, list):
            kernel_avg = self.kernel_size[0]
            kernel_max = self.kernel_size[1]
        elif self.kernel_size is None:
            kernel_avg = input.size(-2)
            kernel_max = input.size(-1)
        if isinstance(self.stride, int):
            stride_max = self.stride
            stride_avg = self.stride
        elif isinstance(self.stride, list):
            stride_avg = self.stride[0]
            stride_max = self.stride[1]
        elif self.stride is None:
            stride_avg = input.size(-2)
            stride_max = input.size(-1)
        x = F.max_pool2d(
            input, (1, kernel_max), (1, stride_max), self.padding)
        x = F.avg_pool2d(
            x, (kernel_avg, 1), (stride_avg, 1), self.padding)
        return x


class AudioGeMPooling2D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = p
        self.p1 = Parameter(torch.ones(1) * self.p)
        self.p2 = Parameter(torch.ones(1) * self.p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p1=3, p2=3, eps=1e-6):
        x = F.avg_pool2d(x.clamp(min=eps).pow(
            p1), (x.size(-2), 1)).pow(1. / p1)
        x = F.max_pool2d(x.clamp(min=eps).pow(
            p2), (1, x.size(-1))).pow(1. / p2)
        return x


def convert_to_audio_pooling(module: nn.Module):
    mod = module
    for pth_module, audio_module in zip([nn.MaxPool2D,
                                         nn.AvgPool2D],
                                        [AudioPooling2D,
                                         AudioPooling2D]):
        if isinstance(module, pth_module):
            mod = pth_module(module.kernel_size, module.stride, module.padding)
    if isinstance(module, GeM):
        mod = AudioGeMPooling2D(module.p, module.eps)
    for name, child in module.named_children():
        mod.add_module(name, convert_to_audio_pooling(child))
    return mod
