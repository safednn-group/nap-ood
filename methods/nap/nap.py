import json
from os import path
import numpy as np
import pandas as pd
import torch
import tqdm
from termcolor import colored
from torch.utils.data.dataloader import DataLoader

import global_vars as Global
from datasets import MirroredDataset
from methods import AbstractMethodInterface
from methods.nap.monitor import Monitor, EuclideanMonitor
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from .utils import CIFAR100sparse2coarse, get_nap_params


class NeuronActivationPatterns(AbstractMethodInterface):
    def __init__(self, args):
        super(NeuronActivationPatterns, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.monitor = None
        self.class_count = 0
        self.default_model = 0
        self.threshold = 0
        self.best_monitored_count = 0
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.nap_params = None
        self.train_dataset_name = ""
        self.train_dataset_length = 0
        self.model_name = ""
        self.nap_cfg = None
        self.nap_cfg_path = "nap_cfgs/default.json"

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
        self.add_identifier = self.base_model.__class__.__name__
        self.train_dataset_name = dataset.name
        self.model_name = "VGG" if self.add_identifier.find("VGG") >= 0 else "Resnet"
        with open(self.nap_cfg_path) as cf:
            self.nap_cfg = json.load(cf)
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def train_H(self, dataset):
        self.known_loader = DataLoader(dataset.datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.workers,
                                       pin_memory=True)
        self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=self.args.batch_size, shuffle=True,
                                         num_workers=self.args.workers,
                                         pin_memory=True)
        # self.nap_params = get_nap_params(self.nap_cfg, self.model_name, self.train_dataset_name)
        self.nap_params = self.nap_cfg[self.model_name][self.train_dataset_name]
        return self._find_only_threshold()
        # return self._find_best_layer_to_monitor()

    def test_H(self, dataset):
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                             num_workers=self.args.workers, pin_memory=True)

        correct = 0.0
        total_count = 0
        print(f"quantiles {self.nap_params}")
        print(f"threshold {self.threshold}")

        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():
                for i, (image, label) in enumerate(dataset):
                    pbar.update()

                    # Get and prepare data.
                    input, target = image.to(self.args.device), label.to(self.args.device)

                    outputs, intermediate_values, _ = self.base_model.forward_nap(input, nap_params=self.nap_params)
                    _, predicted = torch.max(outputs.data, 1)
                    lvl = self.monitor.get_comfort_level(intermediate_values.cpu(),
                                                         predicted, omit=self.omit)

                    classification = np.where(lvl <= self.threshold, 0, 1)

                    correct += (classification == label.numpy()).sum()
                    # for example_index in range(intermediate_values.shape[0]):
                    #     lvl = self.monitor.get_comfort_level(
                    #         intermediate_values.cpu().detach().numpy()[example_index, :],
                    #         predicted.cpu().detach().numpy()[example_index], omit=self.omit)
                    #     if lvl <= self.threshold:
                    #         classification = 0
                    #     else:
                    #         classification = 1
                    #     # print(f"index {example_index} lvl {lvl} label {label[example_index]} class {classification}")
                    #     correct += classification == label[example_index]

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

        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True)
        self.train_dataset_length = len(dataset)
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

    def _find_only_threshold(self):
        with torch.no_grad():
            last_layer_fraction = self.nap_params.pop("last_layer_fraction", None)
            self._get_layers_shapes(self.nap_params)
            self.monitor = Monitor(self.class_count, self.train_dataset_length,
                                   layers_shapes=self.monitored_layers_shapes)
            if not last_layer_fraction:
                self.omit = False
            else:
                self.omit = True
                neurons_to_monitor = self._choose_neurons_to_monitor(
                    int(self.monitored_layers_shapes[0] * last_layer_fraction))
                self.monitor.set_neurons_to_monitor(neurons_to_monitor)

            self._add_class_patterns_to_monitor(self.train_loader, nap_params=self.nap_params)
            df_known = self._process_dataset(self.known_loader, nap_params=self.nap_params,
                                             omit=self.omit)
            df_unknown = self._process_dataset(self.unknown_loader, nap_params=self.nap_params,
                                               omit=self.omit)
            self.threshold, acc = self._find_threshold(df_known, df_unknown, integers=True)
            print(f"threshold: {self.threshold}, accuracy: {acc}")
            return acc

    def _find_best_layer_to_monitor(self):
        best_acc = 0
        results = []
        self.omit = False
        with torch.no_grad():
            for layer in range(0, 13, 2):
                for pool in range(1, 4):
                    # for q0 in np.concatenate((np.linspace(0.3, 0.5, num=3), np.linspace(0.81, 0.99, num=7))):
                    for q0 in np.linspace(0.1, 0.9, num=5):
                        for q1 in np.linspace(0.2, 0.2, num=1):
                            for q2 in np.linspace(0.2, 0.3, num=1):
                                for q3 in np.linspace(0.95, 0.97, num=1):
                                    for q4 in np.linspace(0.85, 0.9, num=1):
                                        # nap_params = [layer, pool, q0, q1, q2]
                                        # resnet_nap_params = [q0, q1, q2, q3, q4]
                                        nap_params = [layer, pool, q0]
                                        self._get_layers_shapes(nap_params)
                                        self.monitor = Monitor(self.class_count, self.train_dataset_length,
                                                               layers_shapes=self.monitored_layers_shapes)
                                        self._add_class_patterns_to_monitor(self.train_loader, nap_params=nap_params)
                                        for i in tqdm.tqdm(np.linspace(int(self.monitored_layers_shapes[0]),
                                                                       self.monitored_layers_shapes[0] -
                                                                       self.monitored_layers_shapes[0] / 4, num=1)):

                                            # print(
                                            #     f" quantile0: {q0} quantile1: {q1} quantile2: {q2} quantile3: {q3} quantile4: {q4} lastlayer: {i}")
                                            print(f" quantile0: {q0} layer: {layer} pool: {pool} lastlayer: {i}")
                                            i = int(i)
                                            if self.omit:
                                                neurons_to_monitor = self._choose_neurons_to_monitor(i)
                                                self.monitor.set_neurons_to_monitor(neurons_to_monitor)
                                            df_known = self._process_dataset(self.known_loader, nap_params=nap_params,
                                                                             omit=self.omit)
                                            df_unknown = self._process_dataset(self.unknown_loader,
                                                                               nap_params=nap_params,
                                                                               omit=self.omit)
                                            threshold, acc = self._find_threshold(df_known, df_unknown, integers=True)
                                            results.append([q0, q1, q2, q3, q4, i, threshold, acc])
                                            if acc > best_acc + 0.01:
                                                self.threshold = threshold
                                                best_acc = acc
                                                self.best_monitored_count = i
                                                self.nap_params = nap_params
            for layer in [13, 14]:
                for q0 in np.linspace(0.1, 0.9, num=5):
                    for q1 in np.linspace(0.2, 0.2, num=1):
                        for q2 in np.linspace(0.2, 0.3, num=1):
                            for q3 in np.linspace(0.95, 0.97, num=1):
                                for q4 in np.linspace(0.85, 0.9, num=1):
                                    pool = 0
                                    nap_params = [layer, pool, q0, q1, q2]
                                    # nap_params = [q0, q1, q2, q3, q4]
                                    self._get_layers_shapes(nap_params)
                                    self.monitor = Monitor(self.class_count, self.train_dataset_length,
                                                           layers_shapes=self.monitored_layers_shapes)
                                    self._add_class_patterns_to_monitor(self.train_loader, nap_params=nap_params)
                                    for i in tqdm.tqdm(np.linspace(int(self.monitored_layers_shapes[0]),
                                                                   self.monitored_layers_shapes[0] -
                                                                   self.monitored_layers_shapes[0] / 4, num=1)):

                                        # print(
                                        #     f" quantile0: {q0} quantile1: {q1} quantile2: {q2} quantile3: {q3} quantile4: {q4} lastlayer: {i}")
                                        print(f" quantile0: {q0} layer: {layer} pool: {pool} lastlayer: {i}")
                                        i = int(i)
                                        if self.omit:
                                            neurons_to_monitor = self._choose_neurons_to_monitor(i)
                                            self.monitor.set_neurons_to_monitor(neurons_to_monitor)
                                        df_known = self._process_dataset(self.known_loader, nap_params=nap_params,
                                                                         omit=self.omit)
                                        df_unknown = self._process_dataset(self.unknown_loader, nap_params=nap_params,
                                                                           omit=self.omit)
                                        threshold, acc = self._find_threshold(df_known, df_unknown, integers=True)
                                        results.append([q0, q1, q2, q3, q4, i, threshold, acc])
                                        if acc > best_acc + 0.01:
                                            self.threshold = threshold
                                            best_acc = acc
                                            self.best_monitored_count = i
                                            self.nap_params = nap_params
            for i in results:
                print(i)
        return best_acc

    def _process_dataset(self, testloader, omit=True, nap_params=None):
        comfort_level_data = np.array([])
        testiter = iter(testloader)

        for imgs, label in testiter:
            label = label.to(self.args.device)
            imgs = imgs.to(self.args.device)
            outputs, intermediate_values, _ = self.base_model.forward_nap(imgs, nap_params=nap_params)
            _, predicted = torch.max(outputs.data, 1)
            correct_bitmap = (predicted == label)
            lvl = self.monitor.get_comfort_level(intermediate_values.cpu(),
                                                 predicted.cpu().detach().numpy(), omit=self.omit)
            stacked = np.stack((label.cpu().numpy(), lvl, correct_bitmap.cpu().numpy()), axis=1)
            if comfort_level_data.size:
                comfort_level_data = np.concatenate((comfort_level_data, stacked))
            else:
                comfort_level_data = stacked
            # for example_index in range(intermediate_values.shape[0]):
            #     # lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
            #     #                                      CIFAR100sparse2coarse(predicted.cpu().numpy()[example_index]), omit=omit,
            #     #                                      ignore_minor_values=True)
            #     lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
            #                                          predicted.cpu().numpy()[example_index],
            #                                          omit=omit)
            #     # lvl = self.monitor.get_comfort_level(intermediate_values.cpu().numpy()[example_index, :],
            #     #                                      0, omit=omit, monitored_class=predicted.cpu().numpy()[example_index])
            #     comfort_level_data.append(
            #         (label.cpu().numpy()[example_index], lvl,
            #          correct_bitmap.cpu().numpy()[example_index]))

        return pd.DataFrame(comfort_level_data, columns=['class', 'comfort_level', 'correct'], dtype=np.int32)

    def _find_threshold(self, df_known, df_unknown, integers=True, steps=1000):
        min = df_unknown["comfort_level"].min() if df_unknown["comfort_level"].min() > df_known[
            "comfort_level"].min() else \
            df_known["comfort_level"].min()
        max = df_unknown["comfort_level"].max() if df_unknown["comfort_level"].max() > df_known[
            "comfort_level"].max() else \
            df_known["comfort_level"].max()
        best_correct_count = 0
        best_threshold = 0
        for i in range(min - 1, max + 1) if integers else np.linspace(min, max, num=steps):
            correct_count = 0
            correct_count += (df_unknown["comfort_level"] > i).sum()
            correct_count += (df_known["comfort_level"] <= i).sum()
            if best_correct_count < correct_count:
                best_correct_count = correct_count
                best_threshold = i
        print(f" best threshold: {best_threshold}")
        print(f" accuracy: {best_correct_count / (len(df_unknown.index) + len(df_known.index))}")
        return best_threshold, best_correct_count / (len(df_unknown.index) + len(df_known.index))

    def _get_layers_shapes(self, nap_params):
        trainiter = iter(self.train_loader)
        with torch.no_grad():
            self.monitored_layers_shapes = \
                self.base_model.forward_nap(trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device),
                                            nap_params=nap_params)[2]

    def _add_class_patterns_to_monitor(self, loader, nap_params=None):
        dataiter = iter(loader)
        for img, label in tqdm.tqdm(dataiter):
            label = label.to(self.args.device)
            img = img.to(self.args.device)
            _, intermediate_values, _ = self.base_model.forward_nap(img, nap_params=nap_params)
            # self.monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), CIFAR100sparse2coarse(label.cpu().numpy()))
            self.monitor.add_neuron_pattern(intermediate_values.cpu(), label.cpu().numpy())
            # self.monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), np.zeros(intermediate_values.cpu().numpy().shape[0]))

    def _choose_neurons_to_monitor(self, neurons_to_monitor_count: int):
        neurons_to_monitor = {}
        for klass in range(self.class_count):
            class_weights = None
            for name, param in self.base_model.named_parameters():
                if name == "model.classifier.6.weight" or name == "model.fc.weight":
                    class_weights = param.data[klass].cpu().numpy()

            abs_weights = np.absolute(class_weights)
            neurons_to_monitor[klass] = abs_weights.argsort()[::-1][:neurons_to_monitor_count]

        return neurons_to_monitor
