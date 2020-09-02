import torch
from taggle.models import BaseModel
from taggle.models.backbones import backbones, get_backbone
from taggle.models.common import (
    GeM,
    convert_model_ReLU2Mish,
    convert_model_ReLU2Swish
)
from taggle.models.heads import get_head, heads
from taggle.models.sync_batchnorm.replicate import DataParallelWithCallback
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from heads import AttentionHead
from torchlibrosa import LogmelFilterBank, SpecAugmentation, Spectrogram

heads.update({"AttentionHead": AttentionHead})


class WaveToSpec(nn.Module):
    def __init__(self, sample_rate: int = 32000,
                 window_size: int = 2048,
                 hop_size: int = 512,
                 mel_bins: int = 128,
                 fmin: int = 50,
                 fmax: int = 14000,
                 img_size: int = 224,
                 aug_prob: float = 0.3):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
            prob=aug_prob)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.mono_to_color = MonoToColor(img_size)
        self.init_bn(self.bn0)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def forward(self, input):
        # t1 = time.time()
        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)
        x = x.transpose(2, 3)
        # (batch_size, 1, mel_bins, time_steps)
        x = self.mono_to_color(x)
        # (batch_size, 3, mel_bins, time_steps)
        frames_num = x.shape[3]
        return x, frames_num


class MonoToColor(nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.size = size
        # self.bn = torch.nn.BatchNorm2d(3)
        # self.bn.bias.data.fill_(0.)
        # self.bn.weight.data.fill_(1.0)

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        # (batch_size, 3, mel_bins, time_steps)
        input_width, input_height = x.size(3), x.size(2)
        x = F.interpolate(
            x, (self.size, int(input_width * self.size / input_height)), mode='nearest')
        # x = self.bn(x)
        return x


class WaveInputModel(BaseModel):
    # wave -> self.preprocess -> self.backbone -> self.heads -> logits
    def __init__(self, backbone, heads: dict, aug_prob=0.):
        super().__init__(backbone, heads)
        self.initialize()
        self.preprocess = WaveToSpec(sample_rate=32000,
                                     window_size=2048,
                                     hop_size=512,
                                     mel_bins=128,
                                     fmin=50, fmax=14000, aug_prob=aug_prob)
        self.mono_to_color = MonoToColor()

    def forward(self, x):
        y = {}
        x, frames_num = self.preprocess(x)
        # print("backbone input:", x.shape)
        x = self.backbone(x)
        # print("backbone output:", x[0].shape)
        for key in self.heads:
            if isinstance(self.heads[key], AttentionHead):
                y.update({key: self.heads[key](x, frames_num)})
            else:
                y.update({key: self.heads[key](x)})
        return y


class BirdSongModelProvider:
    def __init__(self):
        self.backbones = backbones
        self.heads = heads

    def get_model(self, model_config: dict):
        if "type" in model_config:
            if model_config["type"] == "WaveInputModel":
                Model = WaveInputModel
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
                model = convert_to_audio_pooling(model)
        # print(model)
        return model


class AudioPooling2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=True, count_include_pad=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

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
        if isinstance(self.padding, int):
            pad_max = self.padding
            pad_avg = self.padding
        elif isinstance(self.padding, list):
            pad_max = self.padding[0]
            pad_avg = self.padding[1]
        elif self.padding is None:
            pad_max = 0
            pad_avg = 0
        x = F.max_pool2d(
            input, (1, kernel_max), (1, stride_max), (0, pad_max), ceil_mode=self.ceil_mode)
        x = F.avg_pool2d(x, (kernel_avg, 1), (stride_avg, 1), (pad_avg, 0),
                         ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad)
        return x


class AudioGeMPooling2d(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = p
        self.p1 = Parameter(torch.ones(1) * self.p)
        self.p2 = Parameter(torch.ones(1) * self.p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p1=self.p1, p2=self.p2, eps=self.eps)

    def gem(self, x, p1=3, p2=3, eps=1e-6):
        x = F.avg_pool2d(x.clamp(min=eps).pow(
            p1), (x.size(-2), 1)).pow(1. / p1)
        x = F.max_pool2d(x.clamp(min=eps).pow(
            p2), (1, x.size(-1))).pow(1. / p2)
        return x


def convert_to_audio_pooling(module: nn.Module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_to_audio_pooling(mod)
        mod = DataParallelWithCallback(mod)
        return mod
    mod = module

    for pth_module, audio_module in zip([nn.MaxPool2d,
                                         nn.AvgPool2d],
                                        [AudioPooling2d,
                                         AudioPooling2d]):
        if isinstance(module, pth_module):
            mod = audio_module(module.kernel_size, module.stride,
                               module.padding, module.ceil_mode)
    if isinstance(module, GeM):
        mod = AudioGeMPooling2d(module.p, module.eps)
    for name, child in module.named_children():
        mod.add_module(name, convert_to_audio_pooling(child))
    return mod
