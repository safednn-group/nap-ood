"""
GradNorm algorithm integrated with OD-test benchmark.
Origin url: https://github.com/deeplearning-wisc/gradnorm_ood
"""
from __future__ import print_function
import numpy as np
import tqdm
from torch.autograd import Variable

import global_vars as Global
from datasets import MirroredDataset
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
import torch
import os
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from methods import AbstractMethodInterface


class GradNorm(AbstractMethodInterface):
    def __init__(self, args):
        super(GradNorm, self).__init__()
        self.base_model = None
        self.args = args
        self.class_count = 0
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.seed = 1
        self.model_name = ""
        self.workspace_dir = "workspace/grad_norm"

    def propose_H(self, dataset, mirror=True):
        config = self.get_H_config(dataset, mirror)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        self.best_h_path = os.path.join(h_path, 'model.best.pth')

        if not os.path.isfile(self.best_h_path):
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s' % self.best_h_path, 'red'))
            config.model.load_state_dict(torch.load(self.best_h_path))

        self.base_model = config.model
        self.base_model.eval()
        self.class_count = self.base_model.output_size()[1].item()
        self.add_identifier = self.base_model.__class__.__name__
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else "Resnet"
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def method_identifier(self):
        output = "GradNorm"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, mirror):
        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True, shuffle=True)
        # Set up the model
        model = Global.get_ref_classifier(self.args.D1)[self.default_model]().to(self.args.device)
        # model.forward()

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s->%s)' % (self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = self.train_loader
        config.model = model
        config.logger = Logger()
        return config

    def train_H(self, dataset):
        self.known_loader = DataLoader(dataset.datasets[0], batch_size=1, shuffle=True,
                                       num_workers=self.args.workers,
                                       pin_memory=True)
        self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=1, shuffle=False,
                                         num_workers=self.args.workers,
                                         pin_memory=True)

        self.base_model.eval()
        return self._find_threshold()

    def get_ood_score(self, input):
        scores = []
        with torch.enable_grad():
            for img in input:
                logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
                inputs = Variable(img.unsqueeze(0).cuda(), requires_grad=True)
                self.base_model.train()
                self.base_model.zero_grad()
                outputs = self.base_model(inputs, softmax=False)
                targets = torch.ones((inputs.shape[0], self.class_count)).cuda()
                loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

                loss.backward()
                if self.model_name == "VGG":
                    layer_grad = self.base_model.model.classifier[-1].weight.grad.data
                else:
                    layer_grad = self.base_model.model.layer4[-1].conv3.weight.grad.data

                self.base_model.eval()
                score = torch.sum(torch.abs(layer_grad)).cpu().numpy()
                scores.append(score)
        return np.array(scores)
