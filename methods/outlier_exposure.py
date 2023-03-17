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


class OutlierExposure(AbstractMethodInterface):
    def __init__(self, args):
        super(OutlierExposure, self).__init__()
        self.base_model = None
        self.args = args
        self.default_model = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.train_dataset_name = ""
        self.valid_dataset_name = ""
        self.train_dataset_length = 0
        self.seed = 1
        self.model_name = ""
        self.workspace_dir = "workspace/outlier_exposure"

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
        self.add_identifier = self.base_model.__class__.__name__
        self.train_dataset_name = dataset.name
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else "Resnet"
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def method_identifier(self):
        output = "OutlierExposure"
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
        self._fine_tune_model(epochs=epochs)
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
                    logits = self.base_model(input, softmax=False)
                    smax = F.softmax(logits, dim=1).cpu().numpy()
                    scores = -np.max(smax, axis=1)
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

    def _cosine_annealing(self, step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    def _fine_tune_model(self, epochs):
        model_path = os.path.join(os.path.join(self.workspace_dir,
                                               self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                                   self.seed) + '_epoch_' + str(epochs - 1) + '.pt'))
        if os.path.exists(model_path):
            self.base_model.load_state_dict(torch.load(model_path))
            return
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
        self.optimizer = torch.optim.SGD(
            self.base_model.parameters(), 0.001, momentum=0.9,
            weight_decay=0.0005, nesterov=True)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lambda step: self._cosine_annealing(step,
                                                                                                         10 * self.valid_dataset_length,
                                                                                                         1,
                                                                                                         1e-6 / 0.001))
        # Main loop
        for epoch in range(0, epochs):
            self.epoch = epoch

            begin_epoch = time.time()

            self._train_epoch()
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

            # Show results
            with open(
                    os.path.join(self.workspace_dir,
                                 self.train_dataset_name + '_' + self.valid_dataset_name + '_' + self.model_name + '_s' + str(
                                     self.seed) + '_training_results.csv'), 'a') as f:
                f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                    (epoch + 1),
                    time.time() - begin_epoch,
                    self._train_loss,
                    self._test_loss,
                    100 - 100. * self._test_accuracy,
                ))

            # # print state with rounded decimals
            # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

            print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                self._train_loss,
                self._test_loss,
                100 - 100. * self._test_accuracy,
            ))

    def _train_epoch(self):
        self.base_model.train()  # enter train mode
        loss_avg = 0.0

        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        self.unknown_loader.dataset.offset = np.random.randint(self.valid_dataset_length)
        for in_set, out_set in zip(self.train_loader, self.unknown_loader):
            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]

            data, target = data.cuda(), target.cuda()

            # forward
            x = self.base_model(data, softmax=False)

            # backward
            self.scheduler.step()
            self.optimizer.zero_grad()

            loss = F.cross_entropy(x[:len(in_set[0])], target)
            # cross-entropy from softmax distribution to uniform distribution

            loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

            loss.backward()
            self.optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        self._train_loss = loss_avg

    def _eval_model(self):
        self.base_model.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.cuda(), target.cuda()

                # forward
                output = self.base_model(data, softmax=False)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]

                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        self._test_loss = loss_avg / self.train_dataset_length
        self._test_accuracy = correct / self.train_dataset_length

    def _find_threshold(self):
        scores_known = np.array([])
        scores_unknown = np.array([])
        with torch.no_grad():
            for i, (image, label) in enumerate(self.known_loader):

                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model(input, softmax=False)
                smax = F.softmax(logits, dim=1).cpu().numpy()
                scores = -np.max(smax, axis=1)
                if scores_known.size:
                    scores_known = np.concatenate((scores_known, scores))
                else:
                    scores_known = scores

            for i, (image, label) in enumerate(self.unknown_loader):
                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)
                logits = self.base_model(input, softmax=False)
                smax = F.softmax(logits, dim=1).cpu().numpy()
                scores = -np.max(smax, axis=1)
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

    def _generate_execution_times(self, loader):
        assert self.args.batch_size == 1
        import time
        import numpy as np
        n_times = 1000
        exec_times = np.ones(n_times)

        trainiter = iter(loader)
        x = trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            for i in range(n_times):
                start_time = time.time()
                logits = self.base_model(x, softmax=False)
                smax = F.softmax(logits, dim=1).cpu().numpy()
                scores = -np.max(smax, axis=1)
                _ = np.where(scores > self.threshold, 1, 0)

                exec_times[i] = time.time() - start_time

        exec_times = exec_times.mean()
        print(exec_times)