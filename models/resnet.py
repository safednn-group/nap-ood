"""
ResNet source code with additional functions for OOD methods

Origin url: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet as TorchResNet


def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    v = v[:, -1].unsqueeze(dim=1).expand(t.shape).reshape(x.shape)
    fill = fill.unsqueeze(dim=1).expand(t.shape).reshape(x.shape)
    y = torch.where(x >= v, fill, torch.zeros(1, device=x.device, dtype=x.dtype))
    return y


class ResNet(TorchResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.maxpools = [nn.Identity(), nn.AdaptiveMaxPool2d(1), nn.AdaptiveMaxPool2d(2), nn.AdaptiveMaxPool2d(3),
                         nn.AdaptiveMaxPool2d(4)]
        self.avgpools = [nn.Identity(), nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool2d(2), nn.AdaptiveAvgPool2d(3),
                         nn.AdaptiveAvgPool2d(4)]
        self.pools = {"avg": self.avgpools, "max": self.maxpools}

    def forward_threshold(self, x, threshold=1e10):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.clamp(max=threshold)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_binarize(self, x, percentile=65):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = ash_b(x, percentile)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_nap(self, x, nap_params):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer_counter = 0
        prev = torch.Tensor([])
        shapes = []
        zero_tensor = torch.zeros(1, device="cuda")

        for name, layer in chain(self.layer1.named_children(), self.layer2.named_children(),
                                 self.layer3.named_children(), self.layer4.named_children()):
            x = layer.forward(x)
            if layer_counter in nap_params:
                intermediate = torch.flatten(
                    self.pools[nap_params[layer_counter]["pool_type"]][nap_params[layer_counter]["pool_size"]](
                        x), 1)
                intermediate = torch.where(
                    intermediate > torch.quantile(intermediate, nap_params[layer_counter]["quantile"],
                                                  dim=1).unsqueeze(1), intermediate, zero_tensor)
                shapes.append(intermediate.shape[-1])
                if prev.numel():
                    intermediate = torch.cat((intermediate, prev), dim=1)
                prev = intermediate
            layer_counter += 1

        shapes.reverse()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, prev, shapes

    def feature_list(self, x):
        out_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out_list.append(x)
        x = self.layer1(x)
        out_list.append(x)
        x = self.layer2(x)
        out_list.append(x)
        x = self.layer3(x)
        out_list.append(x)
        x = self.layer4(x)
        out_list.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, out_list

    def intermediate_forward(self, x, layer_index):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if layer_index == 1:
            x = self.layer1(x)
        elif layer_index == 2:
            x = self.layer1(x)
            x = self.layer2(x)
        elif layer_index == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif layer_index == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x
