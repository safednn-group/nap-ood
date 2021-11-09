import os.path

import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, num_classes=43, sizeOfNeuronsToMonitor=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 40, 5)
        self.conv1_bn = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 20, 5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 240)
        self.fc2 = nn.Linear(240, sizeOfNeuronsToMonitor)
        self.fc3 = nn.Linear(sizeOfNeuronsToMonitor, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn((self.conv2(x)))))
        # Flatten it to an array of inputs
        # x = x.view(-1, 20 * 5 * 5)
        x = x.reshape(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_nap(self, x, nap_params):
        shapes = []
        intermediate0 = intermediate1 = intermediate2 = intermediate3 = None
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        if str(0) in nap_params:
            intermediate0 = torch.flatten(x, 1)
            intermediate0 = torch.tensor(
                np.where(intermediate0.cpu().numpy() > np.quantile(intermediate0.cpu().numpy(),
                                                                   nap_params[str(0)]["quantile"]),
                         intermediate0.cpu(), 0))
            shapes.append(intermediate0.shape[-1])
        x = self.pool(F.relu(self.conv2_bn((self.conv2(x)))))
        if str(1) in nap_params:
            intermediate1 = torch.flatten(x, 1)
            intermediate1 = torch.tensor(
                np.where(intermediate1.cpu().numpy() > np.quantile(intermediate1.cpu().numpy(),
                                                                   nap_params[str(1)]["quantile"]),
                         intermediate1.cpu(), 0))
            shapes.append(intermediate1.shape[-1])
        # Flatten it to an array of inputs
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        if str(2) in nap_params:
            intermediate2 = x
            intermediate2 = torch.tensor(
                np.where(intermediate2.cpu().numpy() > np.quantile(intermediate2.cpu().numpy(),
                                                                   nap_params[str(2)]["quantile"]),
                         intermediate2.cpu(), 0))
            shapes.append(intermediate2.shape[-1])
        x = F.relu(self.fc2(x))
        if str(3) in nap_params:
            intermediate3 = x
            intermediate3 = torch.tensor(
                np.where(intermediate3.cpu().numpy() > np.quantile(intermediate3.cpu().numpy(),
                                                                   nap_params[str(3)]["quantile"]),
                         intermediate3.cpu(), 0))
            shapes.append(intermediate3.shape[-1])
        x = self.fc3(x)
        intermediates = tuple({intermediate3, intermediate2, intermediate1, intermediate0} - {None})
        return x, torch.cat(intermediates, dim=1), shapes


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("3_model_GTSRB_CNN_27k_train99.ckpt"))
    model.eval()
    data_path = 'data/1425'
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root=data_path, transform=standard_transform)
    original = Image.open('data/1425/orig.png')
    original = to_tensor(original)
    original.unsqueeze_(0)
    with torch.no_grad():
        for layer in range(4):
            for quantile in np.linspace(0., 0.9, num=7):
                params = {
                    str(layer): {
                        "quantile": quantile
                    }
                }
                _, original_pattern, shapes = model.forward_nap(original, params)
                mat_torch = torch.zeros(original_pattern.shape, device=original_pattern.device)
                original_pattern = original_pattern.gt(mat_torch).type(torch.uint8).cuda()

                train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
                lvls = []
                for batch_idx, (data, targets) in enumerate(train_loader):

                    _, patterns, _ = model.forward_nap(data, params)
                    mat_torch = torch.zeros(patterns.shape, device=patterns.device)
                    patterns = patterns.gt(mat_torch).type(torch.uint8).cuda()

                    for i in range(patterns.shape[0]):
                        lvl = (original_pattern ^ patterns[i]).sum()
                        lvls.append(lvl.item())

                df = pd.DataFrame(lvls, columns=["hamming_distance"])
                df.to_csv(os.path.join("results", "attacks" + str(layer) + "_" + str(quantile)))