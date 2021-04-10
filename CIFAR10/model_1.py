import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups)


def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=dilation, groups=groups)


class Bottleneck(nn.Module):
    expansion = 4  # 降采样的系数，表示conv3和conv1的比值

    def __init__(self, in_planes, planes, stride=1, down_sample=None,
                 groups=1, base_weight=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_weight / 64.)) * groups

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

        self.conv4 = conv1x1(in_planes, planes * self.expansion)
        self.bn4 = norm_layer(planes * self.expansion)

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.conv4(identity)
        identity = self.bn4(identity)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out
