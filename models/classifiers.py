import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.newvgg as VGG
import models.nap_resnet as Resnet
# import torchvision.models.resnet as Resnet
import torchvision.models.vgg


class MNIST_VGG(nn.Module):
    """
        VGG-style MNIST.
    """

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def __init__(self):
        super(MNIST_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 2, 1: 6, 2: 9, 3: 13, 4: 16, 5: 20, 6: 23, 7: 26, 8: 29}
        # Reduced VGG16.
        self.cfg = [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.model = VGG.VGG(self.make_layers(self.cfg, batch_norm=True), num_classes=10, relu_indices=self.relu_indices)
        # MNIST would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config

class MNIST_Resnet(nn.Module):
    """
        MNIST_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of MNIST.
    """
    def __init__(self):
        super(MNIST_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 6, 1: 14, 2: 21, 3: 29, 4: 36, 5: 43, 6: 51, 7: 58, 8: 65, 9: 72, 10: 79, 11: 87}
        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [2, 3, 5, 2], num_classes=10, relu_indices=self.relu_indices)

        # MNIST would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params=nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config

class CIFAR10_VGG(nn.Module):
    """
        CIFAR_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR10_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 2, 1: 5, 2: 9, 3: 12, 4: 16, 5: 19, 6: 22, 7: 26, 8: 29, 9: 32, 10: 36, 11: 39, 12: 42, 13: 44, 14: 47}
        # VGG16 minus last maxpool.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=10, relu_indices=self.relu_indices)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config
        
class CIFAR10_Resnet(nn.Module):
    """
        CIFAR_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR10_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 6, 1: 14, 2: 21, 3: 28, 4: 36, 5: 43, 6: 50, 7: 57, 8: 65, 9: 72, 10: 79, 11: 86, 12: 93,
                             13: 100, 14: 108, 15: 115}
        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, relu_indices=self.relu_indices)

        # Cifar 10 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params=nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config
        
class CIFAR100_VGG(nn.Module):
    """
        CIFAR_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR100_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 2, 1: 5, 2: 9, 3: 12, 4: 16, 5: 19, 6: 22, 7: 26, 8: 29, 9: 32, 10: 36, 11: 39, 12: 42, 13: 44, 14: 47}
        # VGG16 minus last maxpool.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=100, relu_indices=self.relu_indices)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 100),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 100])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config
        
class CIFAR100_Resnet(nn.Module):
    """
        CIFAR_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR100_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 6, 1: 14, 2: 21, 3: 28, 4: 36, 5: 43, 6: 50, 7: 57, 8: 65, 9: 72, 10: 79, 11: 86, 12: 93,
                             13: 100, 14: 108, 15: 115}
        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=100, relu_indices=self.relu_indices)

        # Cifar 100 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params=nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes


    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def output_size(self):
        return torch.LongTensor([1, 100])
        
    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class STL10_VGG(nn.Module):
    """
        STL10_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of STL10.
    """
    def __init__(self):
        super(STL10_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 2, 1: 5, 2: 9, 3: 12, 4: 16, 5: 19, 6: 22, 7: 26, 8: 29, 9: 32, 10: 36, 11: 39, 12: 42, 13: 45, 14: 48}
        # VGG16.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=10, relu_indices=self.relu_indices)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class STL10_Resnet(nn.Module):
    """
        STL10_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of STL10.
    """
    def __init__(self):
        super(STL10_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 6, 1: 14, 2: 21, 3: 28, 4: 36, 5: 43, 6: 50, 7: 57, 8: 65, 9: 72, 10: 79, 11: 86, 12: 93,
                             13: 100, 14: 108, 15: 115}
        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, relu_indices=self.relu_indices)

        # STL10 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=None):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params=nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class TinyImagenet_VGG(nn.Module):
    """
        TinyImagenet_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of TinyImagenet.
    """
    def __init__(self):
        super(TinyImagenet_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 2, 1: 5, 2: 9, 3: 12, 4: 16, 5: 19, 6: 22, 7: 26, 8: 29, 9: 32, 10: 36, 11: 39, 12: 42, 13: 45, 14: 48}
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=200, relu_indices=self.relu_indices)
        # TinyImagenet would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 200),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 200])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class TinyImagenet_Resnet(nn.Module):
    """
        TinyImagenet_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of TinyImagenet.
    """
    def __init__(self):
        super(TinyImagenet_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477
        self.relu_indices = {0: 6, 1: 14, 2: 21, 3: 28, 4: 36, 5: 43, 6: 50, 7: 57, 8: 65, 9: 72, 10: 79, 11: 86, 12: 93,
                             13: 100, 14: 108, 15: 115}
        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=200, relu_indices=self.relu_indices)

        # TinyImagenet would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        # del self.model.maxpool
        # self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def forward_nap(self, x, softmax=True, nap_params=0.):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output, intermediate, sizes = self.model.forward_nap(x, nap_params=nap_params)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, intermediate, sizes

    def forward_threshold(self, x, softmax=True, threshold=1e10):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.forward_threshold(x, threshold)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def feature_list(self, x, softmax=True):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output, out_list = self.model.feature_list(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output, out_list

    def intermediate_forward(self, x, softmax=True, layer_index=0):
        # Perform late normalization.
        x = (x - self.offset) * self.multiplier

        output = self.model.intermediate_forward(x, layer_index)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output


    def output_size(self):
        return torch.LongTensor([1, 200])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config



class MNIST_Simple(nn.Module):
    """
        Simple CNN for MNIST.
    """

    def __init__(self, num_classes=10, sizeOfNeuronsToMonitor=84):
        super(MNIST_Simple, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.conv1_bn = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 20, 5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 4 * 4, 240)
        self.fc2 = nn.Linear(240, sizeOfNeuronsToMonitor)
        self.fc3 = nn.Linear(sizeOfNeuronsToMonitor, num_classes)
        self.dr1 = nn.Dropout()
        self.dr2 = nn.Dropout()

    def forward(self, x, softmax=True):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn((self.conv2(x)))))
        # Flatten it to an array of inputs
        #print(x.size())
        x = x.view(-1,  20 * 4 * 4)
        #x = x.reshape(-1, 20 * 4 * 4)
        #print(x.size())
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.dr1(F.relu(self.fc1(x)))  # ReLU(fc(240))
        x = self.dr2(F.relu(self.fc2(x)))  # ReLU(fc(84))
        output = self.fc3(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim'] = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2,
                                                                   min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 40
        return config

class GTSRB_Simple(nn.Module):
    """
        Simple CNN for CIFAR.
    """

    def __init__(self, num_classes=43, sizeOfNeuronsToMonitor=84):
        super(GTSRB_Simple, self).__init__()
        self.conv1 = nn.Conv2d(3, 40, 5)
        self.conv1_bn = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 20, 5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, sizeOfNeuronsToMonitor)
        self.fc3 = nn.Linear(sizeOfNeuronsToMonitor, num_classes)
        self.dr1 = nn.Dropout()
        self.dr2 = nn.Dropout()

    def forward(self, x, softmax=True):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn((self.conv2(x)))))
        # Flatten it to an array of inputs
        #print(x.size())
        x = x.view(-1,  20 * 5 * 5)
        #x = x.reshape(-1, 20 * 4 * 4)
        #print(x.size())
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.dr1(F.relu(self.fc1(x)))  # ReLU(fc(240))
        x = self.dr2(F.relu(self.fc2(x)))  # ReLU(fc(84))
        output = self.fc3(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 43])

    def train_config(self):
        config = {}
        config['optim'] = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2,
                                                                   min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 20
        return config

class GTSRB_VGG(nn.Module):
    """
        CIFAR_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(GTSRB_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # VGG16 minus last maxpool.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=43)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 43),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 43])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 20
        return config

class GTSRB_Resnet(nn.Module):
    """
        TinyImagenet_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of TinyImagenet.
    """
    def __init__(self):
        super(GTSRB_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=43)

        # TinyImagenet would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        # del self.model.maxpool
        # self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 43])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 20
        return config