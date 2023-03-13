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

class ASHB(AbstractMethodInterface):
    def __init__(self, args):
        super(ASHB, self).__init__()
        self.base_model = None
        self.args = args
        self.class_count = 0
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.binarization_percentile = 0.65
        self.seed = 1
        self.model_name = ""
        self.workspace_dir = "workspace/ashb"

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
        output = "ASHB"
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
        self.known_loader = DataLoader(dataset.datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.workers,
                                       pin_memory=True)
        self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=self.args.batch_size, shuffle=False,
                                         num_workers=self.args.workers,
                                         pin_memory=True)

        self.valid_dataset_name = dataset.datasets[1].name
        self.valid_dataset_length = len(dataset.datasets[0])
        return self._find_threshold()


    def test_H(self, dataset):
        self.base_model.eval()

        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():

                correct = 0.0
                all_probs = np.array([])
                labels = np.array([])
                dataset_iter = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.workers, pin_memory=True)
                counter = 0
                for i, (image, label) in enumerate(dataset_iter):
                    pbar.update()
                    counter += 1
                    # Get and prepare data.
                    input, target = image.to(self.args.device), label.to(self.args.device)
                    logits = self.base_model.forward_binarize(input, softmax=False, threshold=self.binarization_percentile)
                    scores = self._get_energy_score(logits)
                    classification = np.where(scores > self.threshold, 1, 0)
                    correct += (classification == label.numpy()).sum()
                    if all_probs.size:
                        labels = np.concatenate((labels, label))
                        all_probs = np.concatenate((all_probs, scores))
                    else:
                        labels = label
                        all_probs = scores

                auroc = roc_auc_score(labels, all_probs)
                p, r, _ = precision_recall_curve(labels, all_probs)
                aupr = auc(r, p)
                print("Final Test average accuracy %s" % (
                    colored('%.4f%%' % (correct / labels.shape[0] * 100), 'red')))
        return correct / labels.shape[0], auroc, aupr


    def _find_threshold(self):
        scores_known = np.array([])
        scores_unknown = np.array([])
        with torch.no_grad():
            for i, (image, label) in enumerate(self.known_loader):

                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model.forward_binarize(input, softmax=False, percentile=self.binarization_percentile)
                scores = self._get_energy_score(logits)
                if scores_known.size:
                    scores_known = np.concatenate((scores_known, scores))
                else:
                    scores_known = scores

            for i, (image, label) in enumerate(self.unknown_loader):
                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model.forward_binarize(input, softmax=False, percentile=self.binarization_percentile)
                scores = self._get_energy_score(logits)
                if scores_unknown.size:
                    scores_unknown = np.concatenate((scores_unknown, scores))
                else:
                    scores_unknown = scores

        min = np.max([scores_unknown.min(), scores_known.min()])
        max = np.min([scores_unknown.max(), scores_known.max()])
        cut_threshold = np.quantile(scores_known, .95)
        cut_correct_count = (scores_unknown > cut_threshold).sum()
        cut_correct_count += (scores_known <= cut_threshold).sum()
        best_correct_count = 0
        best_threshold = 0
        for i in np.linspace(min, max, num=1000):
            correct_count = 0
            correct_count += (scores_unknown > i).sum()
            correct_count += (scores_known <= i).sum()
            if best_correct_count < correct_count:
                best_correct_count = correct_count
                best_threshold = i
        if best_threshold > cut_threshold:
            best_correct_count = cut_correct_count
            best_threshold = cut_threshold
        self.threshold = best_threshold
        acc = best_correct_count / (scores_known.shape[0] * 2)
        return acc


    def _get_energy_score(self, logits, temperature=1):
        scores = -(temperature * torch.logsumexp(logits.data.cpu() / temperature, dim=1).numpy())
        return scores

    def _generate_execution_times(self, loader):
        import time
        import numpy as np
        n_times = 1000
        exec_times = np.ones(n_times)

        trainiter = iter(loader)
        x = trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            for i in range(n_times):
                start_time = time.time()
                logits = self.base_model.forward_binarize(x, softmax=False, percentile=self.binarization_percentile)
                scores = self._get_energy_score(logits)

                _ = np.where(scores > self.threshold, 1, 0)
                exec_times[i] = time.time() - start_time

        exec_times = exec_times.mean()
        print(exec_times)