from __future__ import print_function

import copy
import time

import faiss
import numpy as np
import tqdm

import global_vars as Global
from datasets import MirroredDataset
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from termcolor import colored
from torch.utils.data.dataloader import DataLoader
import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from methods import AbstractMethodInterface
import torchvision.transforms as transforms
import random
from PIL import ImageFilter


class MSAD(AbstractMethodInterface):
    def __init__(self, args):
        super(MSAD, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.class_count = 0
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.train_loader_clean = None
        self.train_dataset_name = ""
        self.valid_dataset_name = ""
        self.test_dataset_name = ""
        self.train_dataset_length = 0
        self.seed = 1
        self.model_name = ""
        self.workspace_dir = "workspace/msad"

    def propose_H(self, dataset, mirror=True):
        config = self.get_H_config(dataset, mirror)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        self.best_h_path = os.path.join(h_path, 'model.best.pth')

        # trainer = IterativeTrainer(config, self.args)

        if not os.path.isfile(self.best_h_path):
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s' % self.best_h_path, 'red'))
            config.model.load_state_dict(torch.load(self.best_h_path))

        self.base_model = config.model
        self.base_model.eval()
        self.class_count = self.base_model.output_size()[1].item()
        self.add_identifier = self.base_model.__class__.__name__
        self.train_dataset_name = dataset.name
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else "Resnet"
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def method_identifier(self):
        output = "MeanShiftedAD"
        # if len(self.add_identifier) > 0:
        #     output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, mirror):
        # if self.args.D1 in Global.mirror_augment and mirror:
        #     print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
        #     new_train_ds = dataset + MirroredDataset(dataset)
        #     dataset = new_train_ds

        dataset2 = copy.deepcopy(dataset)
        dataset2.transform = Transform()
        self.train_loader_clean = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True, shuffle=True)

        self.train_loader = DataLoader(dataset2, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True, shuffle=True)
        self.train_dataset_length = len(dataset)
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
        config.visualize = not self.args.no_visualize
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
        epochs = 20
        self._fine_tune_model(epochs=epochs)
        _ = self._find_threshold()

        model_path = os.path.join(os.path.join(self.workspace_dir,
                                               self.train_dataset_name + '_' + self.model_name + '_s' + str(
                                                   self.seed) + '_epoch_' + str(epochs - 1) + '.pt'))
        if os.path.exists(model_path):
            self.base_model.load_state_dict(torch.load(model_path))
            return

    def test_H(self, dataset):
        self.base_model.eval()
        train_feature_space = []
        with torch.no_grad():
            for (imgs, _) in tqdm.tqdm(self.train_loader_clean, desc='Train set feature extracting'):
                imgs = imgs.to(self.args.device)
                features = self.base_model(imgs)
                train_feature_space.append(features)
            train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():

                correct = 0.0
                all_probs = np.array([])
                labels = np.array([])
                dataset_iter = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                          num_workers=self.args.workers, pin_memory=True)

                self._generate_execution_times(dataset_iter, train_feature_space)
                return 0, 0, 0
                counter = 0
                for i, (image, label) in enumerate(dataset_iter):
                    pbar.update()
                    counter += 1
                    # Get and prepare data.
                    input, target = image.to(self.args.device), label.to(self.args.device)
                    logits = self.base_model(input).cpu().numpy()
                    scores = knn_score(train_feature_space, logits)
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
                print(f"Auroc: {auroc} aupr: {aupr}")
                print(counter)
        return correct / labels.shape[0], auroc, aupr

    def _fine_tune_model(self, epochs):
        model_path = os.path.join(os.path.join(self.workspace_dir,
                                               self.train_dataset_name + '_' + self.model_name + '_s' + str(
                                                   self.seed) + '_epoch_' + str(epochs - 1) + '.pt'))
        if os.path.exists(model_path):
            self.base_model = Model(self.base_model, self.model_name)
            self.base_model.load_state_dict(torch.load(model_path))
            return
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)
        if not os.path.isdir(self.workspace_dir):
            raise Exception('%s is not a dir' % self.workspace_dir)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


        print('Beginning Training\n')

        self.base_model = Model(self.base_model, self.model_name)
        self.base_model = self.base_model.to(self.args.device)
        self.base_model.eval()
        optimizer = torch.optim.SGD(self.base_model.parameters(), lr=1e-5,
                                    weight_decay=0.00005)
        auc, feature_space = get_score(self.base_model, self.args.device, self.known_loader, self.unknown_loader)
        print('Epoch: {}, AUROC is: {}'.format(0, auc.sum()))
        center = torch.FloatTensor(feature_space).mean(dim=0)

        center = F.normalize(center, dim=-1)
        center = center.to(self.args.device)

        for epoch in range(epochs):
            running_loss = run_epoch(self.base_model, self.train_loader, optimizer, center, self.args.device, True)
            print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
            auc, _ = get_score(self.base_model, self.args.device, self.known_loader, self.unknown_loader)
            print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc.sum()))

            # Save model
            torch.save(self.base_model.state_dict(),
                       os.path.join(os.path.join(self.workspace_dir,
                                                 self.train_dataset_name + '_' + self.model_name + '_s' + str(
                                                     self.seed) + '_epoch_' + str(epoch) + '.pt')))

            # Let us not waste space and delete the previous model
            prev_path = os.path.join(os.path.join(self.workspace_dir,
                                                  self.train_dataset_name + '_' + self.model_name + '_s' + str(
                                                      self.seed) + '_epoch_' + str(epoch - 1) + '.pt'))
            if os.path.exists(prev_path): os.remove(prev_path)

    def _find_threshold(self):
        scores_known, _ = get_score(self.base_model, self.args.device, self.train_loader_clean, self.known_loader)
        scores_unknown, _ = get_score(self.base_model, self.args.device, self.train_loader_clean, self.unknown_loader)

        # with torch.no_grad():
        #     for i, (image, label) in enumerate(self.known_loader):
        #
        #         # Get and prepare data.
        #         input, target = image.to(self.args.device), label.to(self.args.device)
        #         logits = self.base_model(input)
        #         smax = F.softmax(logits, dim=1).cpu().numpy()
        #         scores = -np.max(smax, axis=1)
        #         if scores_known.size:
        #             scores_known = np.concatenate((scores_known, scores))
        #         else:
        #             scores_known = scores
        #
        #     for i, (image, label) in enumerate(self.unknown_loader):
        #         # Get and prepare data.
        #         input, target = image.to(self.args.device), label.to(self.args.device)
        #         logits = self.base_model(input)
        #         smax = F.softmax(logits, dim=1).cpu().numpy()
        #         scores = -np.max(smax, axis=1)
        #         if scores_unknown.size:
        #             scores_unknown = np.concatenate((scores_unknown, scores))
        #         else:
        #             scores_unknown = scores

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
        print(f"Best th: {best_threshold} acc: {acc}")
        return acc

    def _generate_execution_times(self, loader, train_feature_space):
        import time
        import numpy as np
        n_times = 1000
        exec_times = np.ones(n_times)

        trainiter = iter(loader)
        x = trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            for i in range(n_times):
                start_time = time.time()

                logits = self.base_model(x).cpu().numpy()
                scores = knn_score(train_feature_space, logits)
                _ = np.where(scores > self.threshold, 1, 0)
                exec_times[i] = time.time() - start_time

        exec_times = exec_times.mean()
        np.savez("results/article_plots/execution_times/" + self.method_identifier() + "_" + self.model_name + "_" + self.train_dataset_name, exec_times=exec_times)


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm.tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    # test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm.tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            # test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        # test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = knn_score(train_feature_space, test_feature_space)

    # auc = roc_auc_score(test_labels, distances)

    return distances, train_feature_space


def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm.tqdm(train_loader, desc='Train...'):

        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone, model_name):
        super().__init__()
        self.backbone = backbone
        if model_name == "Resnet":
            self.backbone.model.fc = torch.nn.Identity()
            for p in self.backbone.model.fc.parameters():
                p.requires_grad = False

        else:
            self.backbone.model.classifier = torch.nn.Identity()
            for p in self.backbone.model.classifier.parameters():
                p.requires_grad = False

    def forward(self, x):
        z1 = self.backbone(x, softmax=False)
        z_n = F.normalize(z1, dim=-1)
        return z_n
