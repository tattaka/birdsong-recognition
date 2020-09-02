import torch
from torch import nn
from torch.nn import functional as F


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    # print(framewise_output.size(), frames_num)
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 norm_type="clamp",  # or tanh
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.norm_type = norm_type
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
        # print(x.size())
        if self.norm_type == "clamp":
            norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        elif self.norm_type == "tanh":
            norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        # x = torch.sum(norm_att * cla, dim=2)
        # x = torch.sum(norm_att * cla + cla, dim=2)
        # return x, norm_att, norm_att * cla
        x = torch.sum(cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


att_blocks = {"v1": AttBlock}


class AttentionHead(nn.Module):
    __name__ = 'AttentionHead'

    def __init__(self,
                 encoder_channels,
                 p=0.2,
                 att_type="v1",
                 num_class=10):
        super().__init__()
        self.p = p
        dense_input = encoder_channels[0]
        self.fc1 = nn.Linear(dense_input * 2, dense_input, bias=True)
        self.att_block = att_blocks[att_type](
            dense_input, num_class, norm_type="tanh", activation='linear')

    def forward(self, feats):
        x = torch.cat([torch.mean(feats[0], dim=2),
                       torch.max(feats[0], dim=2)[0]], dim=1)
        x = F.dropout(x, p=self.p, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.p, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        output_dict = {
            # 'framewise_output': framewise_output,
            'framewise_output': segmentwise_output,
            'clipwise_output': clipwise_output
        }
        return output_dict
