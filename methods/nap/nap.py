import csv
import os
from os import path

import numpy as np
import pandas as pd
import torch
import tqdm
from termcolor import colored
from torch import nn
from torch.utils.data.dataloader import DataLoader

import global_vars as Global
from datasets import MirroredDataset
from methods import AbstractModelWrapper, AbstractMethodInterface
from methods.nap.monitor import Monitor, RangeMonitor
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger

def CIFAR100sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]

def write_csv(data, filename, write_header=False):
    fieldnames = ['class', 'comfort_level', 'correct']
    # rows = [[klass, x[0], x[1]] for x in data]
    # print(f"rows: {data}")
    # for row in data:
    #     print(row)
    # print("write")
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(fieldnames)
        writer.writerows(data)
        f.flush()
        os.fsync(f.fileno())


class NeuronActivationPatterns(AbstractMethodInterface):
    def __init__(self, args):
        super(NeuronActivationPatterns, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.monitor = None
        self.class_count = 0
        self.last_layer_size = 0
        self.default_model = 0
        self.threshold = 0
        self.best_monitored_count = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None

    def propose_H(self, dataset, mirror=True):
        config = self.get_H_config(dataset, mirror)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        best_h_path = path.join(h_path, 'model.best.pth')

        # trainer = IterativeTrainer(config, self.args)

        if not path.isfile(best_h_path):
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s' % best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))

        self.base_model = config.model
        self.base_model.eval()
        self.class_count = self.base_model.output_size()[1].item()
        # self.monitor = Monitor(self.class_count)

    def train_H(self, dataset):
        # print(int(len(dataset) / 2))
        # print(type(dataset))
        # print(len(dataset[0]))
        # print(len(dataset[1]))
        # print(len(dataset.datasets[1]))
        # exit(0)
        print("c")
        self.known_loader = DataLoader(dataset.datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.workers,
                                       pin_memory=True)
        print("d")
        self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=self.args.batch_size, shuffle=True,
                                         num_workers=self.args.workers,
                                         pin_memory=True)
        print("e")
        self._find_best_neurons_count()
        print("f")

    def test_H(self, dataset):
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers,
                             pin_memory=True)
        correct = 0.0
        total_count = 0
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for i, (image, label) in enumerate(dataset):
                pbar.update()

                # Get and prepare data.
                input, target = image.to(self.args.device), label.to(self.args.device)

                outputs, intermediate_values = self.base_model.forward_nap(input)
                _, predicted = torch.max(outputs.data, 1)
                for example_index in range(intermediate_values.shape[0]):
                    lvl = self.monitor.get_comfort_level(intermediate_values.cpu().detach().numpy()[example_index, :],
                                                         predicted.cpu().detach().numpy()[example_index], omit=True)
                    if lvl <= self.threshold:
                        classification = 0
                    else:
                        classification = 1
                    correct += classification == label
                total_count += len(input)

                message = 'Accuracy %.4f' % (correct / total_count)
                pbar.set_description(message)

        test_average_acc = correct / total_count
        print("Final Test average accuracy %s" % (colored('%.4f%%' % (test_average_acc * 100), 'red')))
        return test_average_acc.item()

    def method_identifier(self):
        output = "NeuronActivationPatterns"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, mirror):

        # train_ds, valid_ds = dataset.split_dataset(0.8)

        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        # Initialize the multi-threaded loaders.
        # self.train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,
        #                           num_workers=self.args.workers, pin_memory=True, drop_last=True)
        # self.valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, num_workers=self.args.workers,
        #                           pin_memory=True)
        print("a")
        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True)
        print("b")
        # Set up the model
        model = Global.get_ref_classifier(self.args.D1)[self.default_model]().to(self.args.device)

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

    def _choose_neurons_to_monitor(self, neurons_to_monitor_count: int):
        neurons_to_monitor = {}
        # n = {}
        for klass in range(self.class_count):
            class_weights = None
            for name, param in self.base_model.named_parameters():
                if name == "model.classifier.6.weight" or name == "model.fc.weight":
                    # print(name, param.data[klass])
                    class_weights = param.data[klass].cpu().numpy()

            abs_weights = np.absolute(class_weights)

            neurons_to_monitor[klass] = abs_weights.argsort()[::-1][:neurons_to_monitor_count]
            # n[klass] = absWeight.argsort()[::-1][:neurons_to_monitor_count]

        # print("neurons omitted for monitoring: " + str(len(neuronIndicesToBeOmitted[0])))
        # print(f"neurons omitted for monitoring: {neuronIndicesToBeOmitted[0]}")
        # print(f"neurons for monitoring: {n[0]}")

        return neurons_to_monitor

    def _add_class_patterns_to_monitor(self, quantile=None):
        dataiter = iter(self.train_loader)
        for img, label in tqdm.tqdm(dataiter):
            label = label.to(self.args.device)
            img = img.to(self.args.device)

            _, intermediate_values, _ = self.base_model.forward_nap(img, quantile=quantile)

            # self.monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), CIFAR100sparse2coarse(label.cpu().numpy()))
            self.monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), label.cpu().numpy())
            # self.monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), np.zeros(intermediate_values.cpu().numpy().shape[0]))

    def _find_threshold(self, filename_test, filename_cifar, integers=True, steps=1000):
        df_cifar = pd.read_csv(filename_cifar)
        df_test = pd.read_csv(filename_test)
        min = df_cifar["comfort_level"].min() if df_cifar["comfort_level"].min() > df_test["comfort_level"].min() else \
            df_test["comfort_level"].min()
        max = df_cifar["comfort_level"].max() if df_cifar["comfort_level"].max() > df_test["comfort_level"].max() else \
            df_test["comfort_level"].max()
        best_acc = 0
        best = 0
        for i in range(min - 1, max + 1) if integers else np.linspace(min, max, num=steps):
            curr = 0
            curr += (df_cifar["comfort_level"] > i).sum()
            curr += (df_test["comfort_level"] <= i).sum()
            if best_acc < curr:
                best_acc = curr
                best = i
        print(f" best threshold: {best}")
        print(f" accuracy: {best_acc / (len(df_cifar.index) + len(df_test.index))}")
        return best, best_acc / (len(df_cifar.index) + len(df_test.index))

    def _get_last_layer_size(self, quantiles):
        trainiter = iter(self.train_loader)
        with torch.no_grad():
            self.last_layer_size = \
            self.base_model.forward_nap(trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device),
                                        quantile=quantiles)[1].shape[-1]
            self.monitored_layers_shapes = \
            self.base_model.forward_nap(trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device),
                                        quantile=quantiles)[2]

    def _find_best_neurons_count(self):
        best_acc = 0
        results = []
        with torch.no_grad():
            for layer in range(5):
                for maxpool in range(1, 4):
                    # for q0 in np.concatenate((np.linspace(0.3, 0.5, num=3), np.linspace(0.81, 0.99, num=7))):
                    for q0 in np.linspace(0.01, 0.99, num=5):
                        for q1 in np.linspace(0.97, 0.99, num=1):
                            for q2 in np.linspace(0.97, 0.99, num=1):
                                # quantiles = [layer, maxpool, q0, q1, q2]
                                quantiles = [q0, q1, q2]
                                self._get_last_layer_size(quantiles)
                                self.monitor = Monitor(self.class_count, layers_shapes=self.monitored_layers_shapes)
                                self._add_class_patterns_to_monitor(quantile=quantiles)
                                for i in tqdm.tqdm(np.linspace(int(self.monitored_layers_shapes[0]), 100, num=1)):

                                    print(
                                        f" quantile0: {q0} quantile1: {q1} quantile2: {q2} lastlayer: {i}")
                                    i = int(i)
                                    # neurons_to_monitor = self._choose_neurons_to_monitor(i)
                                    # self.monitor.set_neurons_to_monitor(neurons_to_monitor)
                                    print("g" + str(i))
                                    filename_a = "a" + str(i) + "q0" + str(q0) + "q1" + str(q1) + "q2" + str(q2) + ".csv"
                                    filename_b = "b" + str(i) + "q0" + str(q0) + "q1" + str(q1) + "q2" + str(q2) + ".csv"
                                    # self._process_dataset(filename_a, self.known_loader, quantile=j, monitored_class=k, omit=False)
                                    # print("h" + str(i))
                                    # self._process_dataset(filename_b, self.unknown_loader, quantile=j, monitored_class=k, omit=False)
                                    self._process_dataset(filename_a, self.known_loader, quantile=quantiles, omit=False)
                                    print("h" + str(i))
                                    self._process_dataset(filename_b, self.unknown_loader, quantile=quantiles, omit=False)
                                    print(f"n to monitor: {i}")
                                    threshold, acc = self._find_threshold(filename_a, filename_b, integers=True)
                                    quantiles.append(threshold)
                                    quantiles.append(acc)
                                    results.append(quantiles)
                                    if acc >= best_acc:
                                        self.threshold = threshold
                                        best_acc = acc
                                        self.best_monitored_count = i
                                    os.remove(filename_a)
                                    os.remove(filename_b)
            for i in results:
                print()

    def _process_dataset(self, result_filename, testloader, omit=True, quantile=None):
        comfort_level_data = []
        testiter = iter(testloader)

        for imgs, label in testiter:
            label = label.to(self.args.device)
            imgs = imgs.to(self.args.device)
            outputs, intermediate_values, _ = self.base_model.forward_nap(imgs, quantile=quantile)
            _, predicted = torch.max(outputs.data, 1)
            correct_bitmap = (predicted == label)

            for example_index in range(intermediate_values.shape[0]):
                # lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
                #                                      CIFAR100sparse2coarse(predicted.cpu().numpy()[example_index]), omit=omit,
                #                                      ignore_minor_values=True)
                lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
                                                     predicted.cpu().numpy()[example_index],
                                                     omit=omit,
                                                     ignore_minor_values=True)
                # lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
                #                                      0, omit=omit, monitored_class=predicted.cpu().numpy()[example_index])
                comfort_level_data.append(
                    (label.cpu().numpy()[example_index], lvl,
                     correct_bitmap.cpu().numpy()[example_index]))

        write_csv(comfort_level_data, result_filename, write_header=True)



