"""
VGG source code with additional functions for OOD methods

Origin url: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models.vgg import VGG as TorchVGG


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


class VGG(TorchVGG):

    def __init__(self, *args, relu_indices=None, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.relu_indices = relu_indices
        self.maxpools = [nn.Identity(), nn.AdaptiveMaxPool2d(1), nn.AdaptiveMaxPool2d(2), nn.AdaptiveMaxPool2d(3),
                         nn.AdaptiveMaxPool2d(4)]
        self.avgpools = [nn.Identity(), nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool2d(2), nn.AdaptiveAvgPool2d(3),
                         nn.AdaptiveAvgPool2d(4)]
        self.pools = {"avg": self.avgpools, "max": self.maxpools}

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_threshold(self, x, threshold=1e10):
        x = self.features(x)
        x = x.clamp(max=threshold)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_binarize(self, x, percentile=65):
        x = self.features(x)
        x = ash_b(x, percentile)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_nap(self, x, nap_params=None):
        self.classifier.eval()
        self.features.eval()
        layer_counter = 0
        prev = torch.Tensor([])
        shapes = []
        new_nap_params = dict()
        zero_tensor = torch.zeros(1, device="cuda")
        for k in nap_params.copy():
            new_nap_params[self.relu_indices[int(k)]] = nap_params.get(k)

        for _, layer in self.features.named_children():
            x = layer.forward(x)
            if layer_counter in new_nap_params:
                intermediate = torch.flatten(
                    self.pools[new_nap_params[layer_counter]["pool_type"]][new_nap_params[layer_counter]["pool_size"]](
                        x), 1)
                intermediate = torch.where(
                    intermediate > torch.quantile(intermediate, new_nap_params[layer_counter]["quantile"],
                                                  dim=1).unsqueeze(1), intermediate, zero_tensor)

                shapes.append(intermediate.shape[-1])
                if prev.numel():
                    intermediate = torch.cat((intermediate, prev), dim=1)
                prev = intermediate
            layer_counter += 1

        x = torch.flatten(x, 1)
        for _, layer in self.classifier.named_children():
            x = layer.forward(x)
            if layer_counter in new_nap_params:
                intermediate = torch.where(
                    x > torch.quantile(x, new_nap_params[layer_counter]["quantile"], dim=1).unsqueeze(1), x,
                    zero_tensor)
                shapes.append(intermediate.shape[-1])
                if prev.numel():
                    intermediate = torch.cat((intermediate, prev), dim=1)
                prev = intermediate
            layer_counter += 1
        shapes.reverse()
        return x, prev, shapes

    def feature_list(self, x):
        out_list = []
        counter = 0
        for _, layer in self.features.named_children():
            x = layer.forward(x)
            if counter in self.relu_indices.values():
                out_list.append(x)
            counter += 1
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, out_list

    def intermediate_forward(self, x, layer_index):
        counter = 0
        for _, layer in self.features.named_children():
            x = layer.forward(x)
            if counter == self.relu_indices[layer_index]:
                return x
            counter += 1
