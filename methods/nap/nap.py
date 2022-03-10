import copy
import json
import os
from os import path
import numpy as np
import pandas as pd
import torch
import pickle
import tqdm
from scipy import stats
from termcolor import colored
from torch.utils.data.dataloader import DataLoader

import global_vars as Global
from datasets import MirroredDataset
from methods import AbstractMethodInterface
from methods.nap.monitor import FullNetMonitor
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


class NeuronActivationPatterns(AbstractMethodInterface):
    def __init__(self, args):
        super(NeuronActivationPatterns, self).__init__()
        self.base_model = None
        self.H_class = None
        self.args = args
        self.monitor = None
        self.class_count = 0
        self.default_model = 0
        self.accuracies = None
        self.add_identifier = ""
        self.known_loader = None
        self.unknown_loader = None
        self.train_loader = None
        self.nap_params = None
        self.train_dataset_name = ""
        self.model_name = ""
        self.nap_cfg = None
        self.nap_cfg_path = "methods/nap/cfg/strategies.json"
        self.nap_device = "cuda"

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
        # with open(self.nap_params_path) as cf:
        #     cfg = json.load(cf)
        #     self.nap_params = cfg[self.model_name][self.train_dataset_name]
        with open(self.nap_cfg_path) as cf:
            self.nap_cfg = json.load(cf)
        self._make_nap_params()
        if hasattr(self.base_model, 'preferred_name'):
            self.add_identifier = self.base_model.preferred_name()

    def _make_nap_params(self):
        self.nap_params = dict()
        for i in self.base_model.relu_indices:
            self.nap_params[i] = {
                "pool_type": "max",
                "pool_size": 1,
                "quantile": 0.5
            }

    def method_identifier(self):
        output = "NeuronActivationPatterns"
        if len(self.model_name) > 0:
            output = output + "/" + self.model_name
        return output

    def get_H_config(self, dataset, mirror):
        if self.args.D1 in Global.mirror_augment and mirror:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                       pin_memory=True)
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

    def train_H(self, dataset):
        h_path = path.join(self.args.experiment_path, '%s' % (self.__class__.__name__),
                           '%d' % (self.default_model),
                           '%s->%s.pth' % (self.args.D1, self.args.D2))
        h_parent = path.dirname(h_path)
        if not path.isdir(h_parent):
            os.makedirs(h_parent)

        done_path = h_path + '.nap_state'
        will_train = self.args.force_train_h or not path.isfile(done_path)
        if will_train:
            self.known_loader = DataLoader(dataset.datasets[0], batch_size=self.args.batch_size, shuffle=True,
                                           num_workers=self.args.workers,
                                           pin_memory=True)
            self.unknown_loader = DataLoader(dataset.datasets[1], batch_size=self.args.batch_size, shuffle=True,
                                             num_workers=self.args.workers,
                                             pin_memory=True)

            self._find_thresolds_for_every_layer()
            acc = self._compute_valid_acc(dataset)
            with open(h_path + ".nap_state", 'wb') as f:
                if self.nap_cfg["store_monitor"]:
                    pickle.dump(
                        [self.monitor, self.nap_params, self.scaled_thresholds, self.thresholds, self.chosen_layers,
                         self.add_factor, self.multiplier, acc], f, protocol=-1)
                else:
                    pickle.dump(
                        [False, self.nap_params, self.scaled_thresholds, self.thresholds, self.chosen_layers,
                         self.add_factor, self.multiplier, acc], f, protocol=-1)
            return acc
        else:
            with open(h_path + ".nap_state", 'rb') as f:
                self.monitor, self.nap_params, self.scaled_thresholds, self.thresholds, self.chosen_layers, \
                self.add_factor, self.multiplier, acc = pickle.load(f)
                if not self.monitor:
                    self.monitor = FullNetMonitor(self.class_count, self.nap_device,
                                                  layers_shapes=self.monitored_layers_shapes)
                    self._add_class_patterns_to_monitor(self.train_loader, nap_params=self.nap_params)
                if self.nap_cfg["n_votes"] != self.chosen_layers.shape[0]:
                    raise ValueError("Config n_votes should be equal to the number of layers chosen during training")

                if self.nap_cfg["use_tree"]:
                    self.monitor.make_forest()

            return acc

    def test_H(self, dataset):
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                             num_workers=self.args.workers, pin_memory=True)
        correct = 0.0
        total_count = 0
        concat_distances = np.array([])
        concat_labels = np.array([])
        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():
                for i, (image, label) in enumerate(dataset):
                    pbar.update()

                    input, target = image.to(self.args.device), label.to(self.args.device)

                    outputs, intermediate_values, _ = self.base_model.forward_nap(input, nap_params=self.nap_params)
                    _, predicted = torch.max(outputs.data, 1)
                    distance = self.monitor.compute_hamming_distance(intermediate_values,
                                                                     predicted.cpu().detach().numpy(),
                                                                     tree=self.nap_cfg["use_tree"])
                    score = (distance[:, :] + self.add_factor) * self.multiplier - self.scaled_thresholds

                    score = score.sum(axis=1)
                    if self.nap_cfg["binary_voting"]:
                        classification = stats.mode(
                            np.where(distance <= self.thresholds, 0, 1),
                            axis=1)[0].squeeze()
                    else:
                        classification = (score > 0).astype(np.int)
                    compared = classification == label.numpy()

                    if concat_distances.size:
                        concat_distances = np.concatenate((concat_distances, score))
                        concat_labels = np.concatenate((concat_labels, label.cpu().numpy()))
                    else:
                        concat_distances = score
                        concat_labels = label.cpu().numpy()

                    correct += compared.sum(axis=0)
                    total_count += len(input)
                    message = 'Accuracy: ' + str(correct / total_count)
                    pbar.set_description(message)

        test_average_acc = correct / total_count
        auroc = roc_auc_score(concat_labels, concat_distances)
        p, r, _ = precision_recall_curve(concat_labels, concat_distances)
        aupr = auc(r, p)
        print("Final Test average accuracy %s" % (colored(str(correct / total_count * 100), 'red')))
        return test_average_acc, auroc, aupr

    def _compute_valid_acc(self, dataset):
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                             num_workers=self.args.workers, pin_memory=True)
        correct = 0.0
        total_count = 0
        concat_distances = np.array([])
        concat_labels = np.array([])
        print(colored(f"Computing final validation metrics", 'green'))
        with tqdm.tqdm(total=len(dataset)) as pbar:
            with torch.no_grad():
                for i, (image, label) in enumerate(dataset):
                    pbar.update()

                    input, target = image.to(self.args.device), label.to(self.args.device)
                    outputs, intermediate_values, _ = self.base_model.forward_nap(input, nap_params=self.nap_params)
                    _, predicted = torch.max(outputs.data, 1)
                    distance = self.monitor.compute_hamming_distance(intermediate_values,
                                                                     predicted.cpu().detach().numpy(),
                                                                     tree=self.nap_cfg["use_tree"])
                    score = (distance[:, :] + self.add_factor) * self.multiplier - self.scaled_thresholds
                    score = score.sum(axis=1)
                    if self.nap_cfg["binary_voting"]:
                        classification = stats.mode(
                            np.where(distance <= self.thresholds, 0, 1),
                            axis=1)[0].squeeze()
                    else:
                        classification = (score > 0).astype(np.int)
                    compared = classification == label.numpy()

                    if concat_distances.size:
                        concat_distances = np.concatenate((concat_distances, score))
                        concat_labels = np.concatenate((concat_labels, label.cpu().numpy()))
                    else:
                        concat_distances = score
                        concat_labels = label.cpu().numpy()

                    correct += compared.sum(axis=0)
                    total_count += len(input)
                    message = 'Accuracy: ' + str(correct / total_count)
                    pbar.set_description(message)

        average_acc = correct / total_count
        print("Final valid average accuracy %s" % (colored(str(correct / total_count * 100), 'red')))
        return average_acc

    def _find_thresolds_for_every_layer(self, n_steps=5):
        with torch.no_grad():
            self._get_layers_shapes(self.nap_params)
            self.linspace = np.linspace(0.1, 0.9, num=n_steps)
            self.thresholds, self.accuracies = self._generate_thresholds_for_every_configuration()
            scores = self._compute_scores_for_configurations()
            max_score_ids = np.argmax(scores, axis=2)[:, :, np.newaxis]
            self.thresholds = np.take_along_axis(self.thresholds, max_score_ids, axis=2).squeeze()
            self.accuracies = np.take_along_axis(self.accuracies, max_score_ids, axis=2).squeeze()
            self.thresholds, accuracies = self._pick_best_parameters(max_score_ids)
            self.scaled_thresholds = (self.thresholds + self.add_factor) * self.multiplier
            self.chosen_layers = np.sort(accuracies.argsort()[::-1][:self.nap_cfg["n_votes"]])
            new_params = copy.deepcopy(self.nap_params)
            for l in self.nap_params:
                layer_id = len(self.monitored_layers_shapes) - int(l) - 1
                if int(layer_id) not in self.chosen_layers:
                    new_params.pop(str(l))
            self.nap_params = new_params
            self._get_layers_shapes(self.nap_params)
            self.scaled_thresholds = self.scaled_thresholds[self.chosen_layers]
            self.thresholds = self.thresholds[self.chosen_layers]
            self.add_factor = self.add_factor[self.chosen_layers]
            self.multiplier = self.multiplier[self.chosen_layers]
            self.monitor = FullNetMonitor(self.class_count, self.nap_device,
                                          layers_shapes=self.monitored_layers_shapes)
            self._add_class_patterns_to_monitor(self.train_loader, nap_params=self.nap_params)
            if self.nap_cfg["use_tree"]:
                self.monitor.make_forest()
            return accuracies

    def _generate_thresholds_for_every_configuration(self):
        thresholds = np.zeros((2, len(self.monitored_layers_shapes), self.linspace.shape[0]))
        accuracies = np.zeros(thresholds.shape)
        counter = 0
        for pool_type_id, pool_type in enumerate(["max", "avg"]):
            for i, q in enumerate(self.linspace):
                for k in self.nap_params:
                    self.nap_params[k]["quantile"] = q
                    self.nap_params[k]["pool_type"] = pool_type
                counter += 1
                print(colored(f"Evaluating NAP configuration no.{counter} out of {2 * len(self.linspace)}", 'red'))
                self.monitor = FullNetMonitor(self.class_count, self.nap_device,
                                              layers_shapes=self.monitored_layers_shapes)
                self._add_class_patterns_to_monitor(self.train_loader, nap_params=self.nap_params)
                print(colored(f"Generating Hamming distances for known validation samples", 'green'))
                df_known = self._process_dataset(self.known_loader, nap_params=self.nap_params)
                print(colored(f"Generating Hamming distances for unknown validation samples", 'green'))
                df_unknown = self._process_dataset(self.unknown_loader, nap_params=self.nap_params)
                print(colored(f"Finding the best threshold for every layer", 'green'))
                thresholds[pool_type_id, :, i], accuracies[pool_type_id, :, i] = self._find_threshold(df_known,
                                                                                                      df_unknown)
        return thresholds, accuracies

    def _compute_scores_for_configurations(self, tf=0.1):
        quantile_factors = np.sqrt(1. / np.abs(self.linspace - np.rint(self.linspace)))
        max_threshold = np.max((self.thresholds + quantile_factors) * quantile_factors, axis=2)[:, :, np.newaxis]
        scores = (self.accuracies - 0.5) * (tf + np.abs(
            ((self.thresholds + quantile_factors) * quantile_factors - max_threshold) / max_threshold))
        return scores

    def _get_criterion_array(self):
        if self.nap_cfg["accuracy_criterion"]:
            return self.accuracies
        else:
            return self.thresholds * (-1)

    def _pick_best_parameters(self, max_score_ids):
        quantile_factors = np.sqrt(1. / np.abs(self.linspace - np.rint(self.linspace)))
        new_th = np.zeros(len(self.monitored_layers_shapes))
        new_acc = np.zeros(len(self.monitored_layers_shapes))
        self.multiplier = np.zeros(len(self.monitored_layers_shapes))
        self.add_factor = np.zeros(len(self.monitored_layers_shapes))
        criterion_array = self._get_criterion_array()
        for k in self.nap_params:
            layer_id = len(self.monitored_layers_shapes) - int(k) - 1
            if criterion_array[0, layer_id] > criterion_array[1, layer_id]:
                self.nap_params[k]["quantile"] = self.linspace[max_score_ids[0, layer_id, :]].item()
                self.nap_params[k]["pool_type"] = "max"
                self.add_factor[layer_id] = quantile_factors[max_score_ids[0, layer_id, :]] + self.shape_factors[
                    layer_id]
                self.multiplier[layer_id] = quantile_factors[max_score_ids[0, layer_id, :]] * (
                        self.max_factor / self.shape_factors[layer_id])
                new_th[layer_id] = self.thresholds[0, layer_id]
                new_acc[layer_id] = self.accuracies[0, layer_id]
            else:
                self.nap_params[k]["quantile"] = self.linspace[max_score_ids[1, layer_id, :]].item()
                self.nap_params[k]["pool_type"] = "avg"
                self.add_factor[layer_id] = quantile_factors[max_score_ids[1, layer_id, :]] + self.shape_factors[
                    layer_id]
                self.multiplier[layer_id] = quantile_factors[max_score_ids[1, layer_id, :]] * (
                        self.max_factor / self.shape_factors[layer_id])
                new_th[layer_id] = self.thresholds[1, layer_id]
                new_acc[layer_id] = self.accuracies[1, layer_id]
        return new_th, new_acc

    def _process_dataset(self, testloader, nap_params=None):
        hamming_distance = np.array([])
        labels = np.array([])
        testiter = iter(testloader)

        for imgs, label in tqdm.tqdm(testiter):
            label = label.to(self.args.device)
            imgs = imgs.to(self.args.device)
            outputs, intermediate_values, _ = self.base_model.forward_nap(imgs, nap_params=nap_params)
            _, predicted = torch.max(outputs.data, 1)
            distance = self.monitor.compute_hamming_distance(intermediate_values,
                                                             predicted.cpu().detach().numpy(),
                                                             tree=self.nap_cfg["use_tree"])

            if hamming_distance.size:
                hamming_distance = np.concatenate((hamming_distance, distance))
                labels = np.concatenate((labels, label.unsqueeze(1).cpu().numpy()))
            else:
                hamming_distance = distance
                labels = label.unsqueeze(1).cpu().numpy()
        frames = []
        for i in range(hamming_distance.shape[1]):
            df = pd.DataFrame({"class": labels.flatten(), "hamming_distance": hamming_distance[:, i]})
            frames.append(df)
        return frames

    def _find_threshold(self, dfs_known, dfs_unknown, cut_tail=True):
        thresholds = []
        accuracies = []
        for j, (df_known, df_unknown) in enumerate(zip(dfs_known, dfs_unknown)):
            min = df_unknown["hamming_distance"].min() if df_unknown["hamming_distance"].min() > df_known[
                "hamming_distance"].min() else \
                df_known["hamming_distance"].min()
            max = df_unknown["hamming_distance"].max() if df_unknown["hamming_distance"].max() > df_known[
                "hamming_distance"].max() else \
                df_known["hamming_distance"].max()
            if cut_tail:
                cut_threshold = int(df_known["hamming_distance"].quantile(.95))
                cut_correct_count = (df_unknown["hamming_distance"] > cut_threshold).sum()
                cut_correct_count += (df_known["hamming_distance"] <= cut_threshold).sum()
            best_correct_count = 0
            best_threshold = 0
            for i in range(int(min) - 1, int(max) + 1):
                correct_count = 0
                correct_count += (df_unknown["hamming_distance"] > i).sum()
                correct_count += (df_known["hamming_distance"] <= i).sum()
                if best_correct_count < correct_count:
                    best_correct_count = correct_count
                    best_threshold = i
            if cut_tail:
                if best_threshold > cut_threshold:
                    best_correct_count = cut_correct_count
                    best_threshold = cut_threshold
            acc = best_correct_count / (len(df_unknown.index) + len(df_known.index))
            thresholds.append(best_threshold)
            accuracies.append(acc)
        return np.array(thresholds), accuracies

    def _get_layers_shapes(self, nap_params):
        trainiter = iter(self.train_loader)
        with torch.no_grad():
            self.monitored_layers_shapes = \
                self.base_model.forward_nap(trainiter.__next__()[0][0].unsqueeze(0).to(self.args.device),
                                            nap_params=nap_params)[2]
            shapes = np.array(self.monitored_layers_shapes)
            self.shape_factors = shapes / shapes.min()
            self.max_factor = self.shape_factors.max()

    def _count_classes(self, loader):
        dataiter = iter(loader)
        count_class = dict()
        for _, label in dataiter:
            for i in range(label.shape[0]):
                if count_class.get(label[i].item()):
                    count_class[label[i].item()] += 1
                else:
                    count_class[label[i].item()] = 1
        return count_class

    def _add_class_patterns_to_monitor(self, loader, nap_params=None, monitor=None):
        count_class = self._count_classes(loader)
        if not monitor:
            monitor = self.monitor
        monitor.set_class_patterns_count(count_class)
        dataiter = iter(loader)
        print(colored(f"Generating known activation patterns", 'green'))
        for img, label in tqdm.tqdm(dataiter):
            label = label.to(self.args.device)
            img = img.to(self.args.device)
            _, intermediate_values, shapes = self.base_model.forward_nap(img, nap_params=nap_params)

            monitor.add_neuron_pattern(intermediate_values, label.cpu().numpy())
        monitor.cut_duplicates()
