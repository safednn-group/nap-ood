from __future__ import print_function

import time

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
import torch.nn as nn


class NTOM(AbstractMethodInterface):
    def __init__(self, args):
        super(NTOM, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.class_count = 0
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.train_dataset_name = ""
        self.valid_dataset_name = ""
        self.test_dataset_name = ""
        self.train_dataset_length = 0
        self.seed = 1
        self.quantile = 0.125
        self.model_name = ""
        self.workspace_dir = "workspace/ntom"

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
        output = "NTOM"
        # if len(self.add_identifier) > 0:
        #     output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, mirror):
        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True, shuffle=True)
        self.train_dataset_length = len(dataset)
        self.input_shape = iter(dataset).__next__()[0].shape
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
        epochs = 10
        if self.model_name == "VGG":
            self.base_model.model.classifier = nn.Sequential(
                nn.Linear(self.base_model.model.classifier[0].in_features, self.base_model.model.classifier[0].out_features),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(self.base_model.model.classifier[3].in_features, self.base_model.model.classifier[3].out_features),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(self.base_model.model.classifier[6].in_features, self.base_model.model.classifier[6].out_features + 1), )
            for m in self.base_model.model.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        else:
            self.base_model.model.fc = nn.Linear(512 * 4, self.class_count + 1)
        self._fine_tune_model(epochs=epochs)
        _ = self._find_threshold()

        model_path = os.path.join(os.path.join(self.workspace_dir,
                                               self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                                   self.seed) + '_epoch_' + str(epochs - 1) + '.pt'))
        if os.path.exists(model_path):
            self.base_model.load_state_dict(torch.load(model_path))
            return

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
                    logits = self.base_model(input, softmax=False)
                    smax = F.softmax(logits, dim=1).cpu().numpy()
                    scores = np.max(smax, axis=1)
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
                                               self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                                   self.seed) + '_epoch_' + str(epochs - 1) + '.pt'))
        # if os.path.exists(model_path):
        #     self.base_model.load_state_dict(torch.load(model_path))
        #     self.base_model = self.base_model.to(self.args.device)
        #     return
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)
        if not os.path.isdir(self.workspace_dir):
            raise Exception('%s is not a dir' % self.workspace_dir)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        with open(os.path.join(self.workspace_dir,
                               self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                   self.seed) + '_training_results.csv'), 'w') as f:
            f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

        print('Beginning Training\n')
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.base_model = self.base_model.to(self.args.device)
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), 0.00001,
                                         momentum=0.9,
                                         nesterov=True,
                                         weight_decay=0.0001)


        # Main loop
        for epoch in range(0, epochs):
            self.epoch = epoch

            begin_epoch = time.time()
            selected_ood_loader = self._select_ood()
            self._train_epoch(selected_ood_loader)
            self._eval_model()

            # Save model
            torch.save(self.base_model.state_dict(),
                       os.path.join(os.path.join(self.workspace_dir,
                                                 self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                                     self.seed) + '_epoch_' + str(epoch) + '.pt')))

            # Let us not waste space and delete the previous model
            prev_path = os.path.join(os.path.join(self.workspace_dir,
                                                  self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                                      self.seed) + '_epoch_' + str(epoch - 1) + '.pt'))
            if os.path.exists(prev_path): os.remove(prev_path)


    def _train_epoch(self, selected_loader):
        self.base_model.train()  # enter train mode
        batch_time = AverageMeter()

        out_confs = AverageMeter()
        in_confs = AverageMeter()

        in_losses = AverageMeter()
        out_losses = AverageMeter()
        nat_top1 = AverageMeter()

        end = time.time()
        for in_set, out_set in zip(self.train_loader, selected_loader):
            in_len = len(in_set[0])
            out_len = len(out_set[0])

            in_input = in_set[0].cuda()
            in_target = in_set[1]
            in_target = in_target.cuda()

            out_input = out_set[0].cuda()

            out_target = out_set[1]
            out_target = out_target.cuda()

            cat_input = torch.cat((in_input, out_input), 0)
            cat_output = self.base_model(cat_input, softmax=False)

            in_output = cat_output[:in_len]
            in_conf = F.softmax(in_output, dim=1)[:, -1].mean()
            in_confs.update(in_conf.data, in_len)
            in_loss = self.criterion(in_output, in_target)

            out_output = cat_output[in_len:]
            out_conf = F.softmax(out_output, dim=1)[:, -1].mean()
            out_confs.update(out_conf.data, out_len)
            out_loss = self.criterion(out_output, out_target)

            in_losses.update(in_loss.data, in_len)
            out_losses.update(out_loss.data, out_len)

            nat_prec1 = self._accuracy(in_output[:, :self.class_count].data, in_target, topk=(1,))[0]
            nat_top1.update(nat_prec1, in_len)

            loss = in_loss + out_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    def _eval_model(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.base_model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.known_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = self.base_model(input, softmax=False)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1 = self._accuracy(output[:, :self.class_count].data, target, topk=(1,))[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            batch_time=batch_time, loss=losses,
            top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    def _find_threshold(self):
        scores_known = np.array([])
        scores_unknown = np.array([])
        with torch.no_grad():
            for i, (image, label) in enumerate(self.known_loader):

                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model(input, softmax=False)
                smax = F.softmax(logits, dim=1).cpu().numpy()
                scores = np.max(smax, axis=1)
                if scores_known.size:
                    scores_known = np.concatenate((scores_known, scores))
                else:
                    scores_known = scores

            for i, (image, label) in enumerate(self.unknown_loader):
                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model(input, softmax=False)
                smax = F.softmax(logits, dim=1).cpu().numpy()
                scores = np.max(smax, axis=1)
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
        print(f"Best th: {best_threshold} acc: {acc}")
        return acc

    def _adjust_learning_rate(self, epoch, lr_schedule):
        """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
        lr = 0.00001
        if epoch >= lr_schedule[0]:
            lr *= 0.1
        if epoch >= lr_schedule[1]:
            lr *= 0.1
        if epoch >= lr_schedule[2]:
            lr *= 0.1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _select_ood(self):

        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        offset = np.random.randint(len(self.unknown_loader.dataset))

        self.unknown_loader.dataset.offset = offset

        out_iter = iter(self.unknown_loader)
        print('Start selecting OOD samples...')

        start = time.time()
        # select ood samples
        self.base_model.eval()
        with torch.no_grad():
            all_ood_input = []
            all_ood_conf = []
            for k in range(1000):

                try:
                    out_set = next(out_iter)
                except StopIteration:
                    offset = np.random.randint(len(self.unknown_loader.dataset))
                    self.unknown_loader.dataset.offset = offset
                    out_iter = iter(self.unknown_loader)
                    out_set = next(out_iter)

                input = out_set[0]
                output = self.base_model(input.cuda())
                conf = F.softmax(output, dim=1)[:, -1]

                all_ood_input.append(input)
                all_ood_conf.extend(conf.detach().cpu().numpy())

        all_ood_input = torch.cat(all_ood_input, 0)[:self.valid_dataset_length * 4]
        all_ood_conf = np.array(all_ood_conf)[:self.valid_dataset_length * 4]
        indices = np.argsort(all_ood_conf)

        N = all_ood_input.shape[0]
        selected_indices = indices[int(self.quantile * N):int(self.quantile * N) + self.valid_dataset_length]

        print('Total OOD samples: ', len(all_ood_conf))
        print('Max OOD Conf: ', np.max(all_ood_conf), 'Min OOD Conf: ', np.min(all_ood_conf), 'Average OOD Conf: ',
              np.mean(all_ood_conf))
        selected_ood_conf = all_ood_conf[selected_indices]
        print('Selected Max OOD Conf: ', np.max(selected_ood_conf), 'Selected Min OOD Conf: ',
              np.min(selected_ood_conf), 'Selected Average OOD Conf: ', np.mean(selected_ood_conf))

        ood_images = all_ood_input[selected_indices]
        ood_labels = (torch.ones(self.valid_dataset_length) * self.class_count).long()

        ood_train_loader = torch.utils.data.DataLoader(
            OODDataset(ood_images, ood_labels),
            batch_size=self.args.batch_size, shuffle=True)

        print('Time: ', time.time() - start)

        return ood_train_loader

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class OODDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
